from __future__ import annotations
import atexit
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import weakref
import httpx
import psutil
from app.files import read_json, write_json

log = logging.getLogger(__name__)


def process_identity(pid: int) -> dict:
    proc = psutil.Process(pid)
    return {'pid': pid, 'created': proc.create_time(), 'command': proc.cmdline(), 'uid': proc.uids().real}


def owned(record: dict) -> bool:
    try:
        current = process_identity(record['pid'])
        return all(current[k] == record[k] for k in ('created', 'command', 'uid')) and current['uid'] == os.getuid()
    except (psutil.Error, KeyError, OSError):
        return False


def terminate(record: dict) -> None:
    if not owned(record):
        return
    pid = record['pid']
    if os.getpgid(pid) == pid:
        os.killpg(pid, signal.SIGTERM)
    else:
        psutil.Process(pid).terminate()
    try:
        psutil.Process(pid).wait(timeout=10)
    except psutil.TimeoutExpired:
        if owned(record):
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
            else:
                psutil.Process(pid).kill()
            psutil.Process(pid).wait(timeout=5)
    except psutil.NoSuchProcess:
        pass


class VRAMMonitor:
    def __init__(self):
        self.available = False
        self.error = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(self.name, bytes):
                self.name = self.name.decode()
            self.available = True
        except Exception as error:
            self.error = str(error)
            self.name = None
        self.peak = 0
        self.stop_event = threading.Event()

    def memory(self) -> dict:
        if not self.available:
            return {'used': None, 'total': None}
        info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {'used': info.used, 'total': info.total}

    def sample(self):
        while not self.stop_event.is_set():
            self.peak = max(self.peak, self.memory()['used'] or 0)
            self.stop_event.wait(.5)

    @contextmanager
    def stage(self, result: dict, tolerance_mb: int):
        baseline = self.memory()['used']
        self.peak = baseline or 0
        self.stop_event.clear()
        thread = threading.Thread(target=self.sample, daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            yield
        finally:
            self.stop_event.set()
            thread.join(timeout=2)
            end = time.monotonic() + 15
            after = self.memory()['used']
            while baseline is not None and after > baseline + tolerance_mb * 1024**2 and time.monotonic() < end:
                time.sleep(.5)
                after = self.memory()['used']
            released = None if baseline is None else after <= baseline + tolerance_mb * 1024**2
            result.update(seconds=time.monotonic()-started, total_vram_bytes=self.memory()['total'],
                          peak_vram_bytes=self.peak if self.available else None,
                          vram_before_bytes=baseline, vram_after_bytes=after, memory_released=released)
            if released is False:
                raise RuntimeError('VRAM_NOT_RELEASED: memory remains above stage baseline; inspect nvidia-smi')


class GPUProcessManager:
    instances = weakref.WeakSet()

    @classmethod
    def close_all(cls):
        for manager in list(cls.instances):
            manager.close()

    def __init__(self, output_dir: Path):
        self.registry = output_dir / '.gpu-procs'
        self.registry.mkdir(parents=True, exist_ok=True)
        self.active: dict[str, dict] = {}
        self.logs = []
        self.stop_event = threading.Event()
        self.cleanup_registered()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        self.instances.add(self)
        atexit.register(self.close)

    def cleanup_registered(self):
        for path in self.registry.glob('*.json'):
            try:
                record = read_json(path)
            except (OSError, ValueError):
                log.warning('Unreadable process record; no process will be terminated: %s', path)
                continue
            if not owned(record):
                path.unlink(missing_ok=True)
            elif record.get('component') == 'comfy' or time.time() - record.get('heartbeat', 0) > 3600:
                terminate(record)
                path.unlink(missing_ok=True)
            else:
                # A live registered model must be stopped before another component can acquire VRAM.
                terminate(record)
                path.unlink(missing_ok=True)

    def heartbeat(self):
        while not self.stop_event.wait(10):
            for name, record in list(self.active.items()):
                if owned(record):
                    record['heartbeat'] = time.time()
                    write_json(self.registry/f'{name}.json', record)

    def start(self, name: str, args: list[str], cwd: Path | None = None):
        if self.active:
            raise RuntimeError('GPU component already active')
        stream = (self.registry/f'{name}.log').open('a')
        self.logs.append(stream)
        process = subprocess.Popen(args, cwd=cwd, stdout=stream, stderr=stream, start_new_session=True)
        record = {**process_identity(process.pid), 'component': name, 'heartbeat': time.time()}
        self.active[name] = record
        write_json(self.registry/f'{name}.json', record)
        return process

    def stop(self, name: str):
        record = self.active.pop(name, None)
        if record:
            terminate(record)
        (self.registry/f'{name}.json').unlink(missing_ok=True)

    @contextmanager
    def server(self, name: str, args: list[str], url: str, readiness: str,
               timeout: float, cwd: Path | None = None, model: str | None = None):
        client = httpx.Client(base_url=url, timeout=5)
        try:
            # Do not attach to an unregistered local service: its model lifecycle is not owned by this app.
            try:
                client.get(readiness).raise_for_status()
            except httpx.HTTPError:
                pass
            else:
                raise RuntimeError(f'PORT_IN_USE: {url}; stop the unrelated service first')
            for attempt in range(2):
                process = self.start(name, args, cwd)
                end = time.monotonic() + timeout
                ready = False
                while time.monotonic() < end and process.poll() is None:
                    try:
                        client.get(readiness).raise_for_status()
                        ready = True
                        break
                    except httpx.HTTPError:
                        time.sleep(.5)
                if ready:
                    break
                self.stop(name)
            else:
                raise TimeoutError(f'{name} readiness timeout after two starts; see {self.registry}/{name}.log')
            yield process
        finally:
            if name in self.active:
                try:
                    if name == 'ollama' and model:
                        client.post('/api/generate', json={'model': model, 'keep_alive': 0}, timeout=30).raise_for_status()
                    elif name == 'comfy':
                        client.post('/free', json={'unload_models': True, 'free_memory': True}).raise_for_status()
                except httpx.HTTPError:
                    log.exception('Graceful GPU unload failed; terminating owned process')
                finally:
                    self.stop(name)
            client.close()

    def worker(self, module: str, input_path: Path, output_path: Path, timeout: float, progress=lambda message: None):
        import sys
        output_path.unlink(missing_ok=True)
        finished = threading.Event()
        timed_out = threading.Event()
        progress_file = input_path.parent / 'progress.json'
        progress_file.unlink(missing_ok=True)
        def poll_progress():
            previous = None
            while not finished.wait(.5):
                try:
                    message = read_json(progress_file)['message']
                    if message != previous:
                        progress(message)
                        previous = message
                except (OSError, ValueError, KeyError):
                    pass
        def enforce_deadline():
            if not finished.wait(timeout):
                timed_out.set()
                for path in self.registry.glob('worker-*.json'):
                    try:
                        terminate(read_json(path))
                    except (OSError, ValueError, psutil.Error):
                        log.exception('Worker deadline cleanup failed: %s', path)
        watchdog = threading.Thread(target=enforce_deadline, daemon=True)
        watchdog.start()
        poller = threading.Thread(target=poll_progress, daemon=True)
        poller.start()
        try:
            result = subprocess.run([sys.executable, '-m', module, '--input', str(input_path), '--output', str(output_path)],
                                    timeout=timeout+15, capture_output=True, text=True, start_new_session=True)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f'{module} timeout: {error.stderr}') from error
        finally:
            finished.set()
            poller.join(timeout=2)
            watchdog.join(timeout=12)
            # Workers register themselves before loading torch. Verify identity before orphan cleanup.
            for path in self.registry.glob('worker-*.json'):
                record = read_json(path)
                terminate(record)
                path.unlink(missing_ok=True)
        if timed_out.is_set():
            raise TimeoutError(f'{module} timeout after {timeout}s: {result.stderr[-16000:]}')
        if result.returncode:
            detail = result.stderr[-16000:]
            raise RuntimeError(f'{module} failed ({result.returncode}): {detail}')
        if not output_path.is_file():
            raise RuntimeError(f'{module} missing JSON output')
        return read_json(output_path)

    def diagnostics(self) -> str:
        try:
            return subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'],
                                  timeout=10, capture_output=True, text=True).stdout
        except (OSError, subprocess.SubprocessError) as error:
            return str(error)

    def close(self):
        self.stop_event.set()
        for name in list(self.active):
            self.stop(name)
        for stream in self.logs:
            stream.close()
        for path in self.registry.glob('worker-*.json'):
            terminate(read_json(path))
            path.unlink(missing_ok=True)
        self.instances.discard(self)
        atexit.unregister(self.close)
