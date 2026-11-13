from __future__ import annotations
from contextlib import contextmanager
import fcntl
import logging
from pathlib import Path
import signal
import sys
import time
import uuid
import wave
import httpx
import yaml
from app.comfy import ComfyClient
from app.config import ROOT, Settings
from app.content import ManualPromptProvider, require_news_text
from app.files import Manifest, file_hash, read_json, valid_png, valid_video, wav_duration, write_json
from app.gpu import GPUProcessManager, VRAMMonitor
from app.llm import ScriptGenerator
from app.render import Renderer
from app.schema import AlignmentOutput, Params, Script, TTSOutput, WorkerInput
from app.subtitles import write_subtitles

log = logging.getLogger(__name__)


def valid_audio(result: TTSOutput, languages: list[str], count: int) -> bool:
    try:
        if set(result.languages) != set(languages):
            return False
        for audio in result.languages.values():
            if len(audio.segments) != count:
                return False
            total = wav_duration(audio.audio, audio.sample_rate)
            cursor = 0
            for segment in audio.segments:
                length = wav_duration(segment.path, audio.sample_rate)
                if abs(segment.start-cursor) > .002 or abs((segment.end-segment.start)-length) > .002:
                    return False
                cursor = segment.end
            if abs(total-cursor) > .01:
                return False
        return True
    except (OSError, ValueError, EOFError, wave.Error):
        return False


class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.load()

    def run(self, params: Params, run_id: str | None = None, progress=lambda message: None) -> dict[str, Path]:
        settings = self.settings
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        # flock also serializes separate CLI processes and the Gradio process.
        with (settings.output_dir/'.generation.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('GENERATION_BUSY: another generation is running') from error
            return self._run(params, run_id, progress)

    def _run(self, params: Params, run_id: str | None, progress) -> dict[str, Path]:
        s = self.settings
        if run_id is not None and (not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id)):
            raise ValueError('Invalid run ID')
        run_id = run_id or time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
        directory = (s.output_dir/run_id).resolve()
        for name in ('images', 'audio', 'subtitles', 'tmp'):
            (directory/name).mkdir(parents=True, exist_ok=True)
        versions = yaml.safe_load((ROOT/'models.lock.yaml').read_text())
        params_data = params.model_dump(mode='json')
        music_hash = file_hash(params.music) if params.music else None
        signature = {'params': params_data, 'music': music_hash,
                     'settings': s.model_dump(mode='json'),
                     'voices': (ROOT/'config/voices.yaml').read_text(),
                     'prompts': {p.name: p.read_text() for p in (ROOT/'config/prompts').glob('*.txt')},
                     'workflow': read_json(ROOT/'workflows/flux_schnell_api.json'),
                     'code': {p.name: file_hash(p) for p in (ROOT/'app').glob('*.py')}}
        manifest = Manifest(directory, signature, versions)
        report = {'run_id': run_id, 'models': {'llm': s.llm_model, 'image': 'flux1-schnell',
                  'tts': versions['huggingface']['qwen3_tts']['repo'], 'subtitle': 'whisperx', 'versions': versions},
                  'params': params_data, 'status': 'running',
                  'timings': {}, 'per_language': {}, 'errors': [], 'retries': {}, 'stages': {}}
        started = time.monotonic()
        manager = GPUProcessManager(s.output_dir)
        vram = VRAMMonitor()
        report['gpu'] = {'name': vram.name, 'total_vram_bytes': vram.memory()['total'], 'monitor_available': vram.available, 'monitor_error': vram.error,
                         'processes': manager.diagnostics()}
        write_json(directory/'generation_report.json', report)
        @contextmanager
        def stage(name):
            progress(name.replace('_', ' '))
            metric = {}
            try:
                with vram.stage(metric, s.vram_release_tolerance_mb):
                    yield
            finally:
                report['stages'][name] = metric
                report['timings'][name+'_s'] = metric.get('seconds', 0)
                write_json(directory/'generation_report.json', report)
        old_handlers = {}
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        import threading
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_handlers[sig] = signal.signal(sig, interrupted)
        try:
            research = ManualPromptProvider().fetch(params.prompt, params.source_urls)
            if params.content_type == 'news':
                require_news_text(research)
            script_path = directory/'script.json'
            scripts = None
            if manifest.reusable('script', signature, [script_path]):
                try:
                    scripts = {lang: Script.model_validate(value) for lang, value in read_json(script_path)['scripts'].items()}
                    if set(scripts) != set(params.languages) or any(len(x.scenes) != params.image_count for x in scripts.values()):
                        scripts = None
                except (ValueError, KeyError):
                    scripts = None
            if scripts is None:
                with stage('script'):
                    with manager.server('ollama', ['ollama', 'serve'], s.ollama_url, '/api/tags',
                                        s.timeouts.readiness, model=s.llm_model):
                        generator = ScriptGenerator(s)
                        try:
                            scripts = generator.generate(params, research, directory)
                            report['retries']['script'] = generator.retries
                            report['script_language_seconds'] = generator.timings
                        finally:
                            generator.client.close()
                    write_json(script_path, {'research': research.model_dump(),
                                              'scripts': {lang: value.model_dump() for lang, value in scripts.items()}})
                    manifest.complete('script', signature, [script_path])
            primary = scripts[params.languages[0]]
            images = [directory/f'images/{i:02}.png' for i in range(params.image_count)]
            image_keys = [{'prompt': scene.visual_prompt, 'style': params.style, 'seed': params.seed+i,
                           'size': [s.width,s.height], 'versions': versions,
                           'workflow': signature['workflow']} for i, scene in enumerate(primary.scenes)]
            missing = [i for i, image in enumerate(images)
                       if not (manifest.reusable(f'image_{i}', image_keys[i], [image])
                               and valid_png(image,s.width,s.height,s.pixel_std_min))]
            if missing:
                with stage('images'):
                    deadline = time.monotonic()+s.timeouts.images
                    restarts = 0
                    consecutive_timeouts = 0
                    while missing:
                        try:
                            with manager.server('comfy', [sys.executable, 'main.py', '--listen', '127.0.0.1',
                                    '--port', '8188', '--lowvram', '--disable-auto-launch',
                                    '--front-end-root', str(ROOT/'workflows')], s.comfy_url,
                                    '/object_info', s.timeouts.readiness, cwd=s.comfy_dir):
                                client = ComfyClient(s)
                                try:
                                    client.check_nodes()
                                    for i in list(missing):
                                        if time.monotonic() >= deadline:
                                            raise TimeoutError('Images stage timeout')
                                        progress(f'images {i+1}/{len(images)}')
                                        try:
                                            client.image(params.style+', '+primary.scenes[i].visual_prompt,
                                                         params.seed+i, images[i], min(s.timeouts.image, deadline-time.monotonic()))
                                        except TimeoutError:
                                            consecutive_timeouts += 1
                                            if consecutive_timeouts >= 2:
                                                raise RuntimeError('TWO_CONSECUTIVE_IMAGE_TIMEOUTS')
                                            raise
                                        consecutive_timeouts = 0
                                        manifest.complete(f'image_{i}', image_keys[i], [images[i]])
                                        missing.remove(i)
                                finally:
                                    client.client.close()
                        except (httpx.HTTPError, TimeoutError, RuntimeError, ValueError) as error:
                            details = str(error)
                            logfile = manager.registry/'comfy.log'
                            if logfile.exists():
                                details += logfile.read_text(errors='replace')[-8000:]
                            if 'out of memory' in details.lower() or 'cuda oom' in details.lower():
                                raise RuntimeError('CUDA OOM in FLUX; stop stage, reduce resolution/offloading preset') from error
                            if restarts >= 1 or 'TWO_CONSECUTIVE' in details or time.monotonic() >= deadline:
                                raise
                            restarts += 1
                            report['retries']['images_restart'] = restarts
            script_hash = file_hash(script_path)
            tts_output = directory/'tmp/tts_output.json'
            tts_key = {'script': script_hash, 'voices': signature['voices'], 'models': versions['huggingface']['qwen3_tts']}
            tts = None
            if tts_output.exists():
                try:
                    candidate = TTSOutput.model_validate(read_json(tts_output))
                    paths = [tts_output]+[a.audio for a in candidate.languages.values()]+[seg.path for a in candidate.languages.values() for seg in a.segments]
                    if manifest.reusable('tts', tts_key, paths) and valid_audio(candidate,params.languages,params.image_count):
                        tts = candidate
                except (ValueError, KeyError):
                    pass
            if tts is None:
                with stage('tts'):
                    input_path = directory/'tmp/tts_input.json'
                    write_json(input_path, WorkerInput(run_dir=directory,languages=params.languages,timeout=s.timeouts.tts).model_dump(mode='json'))
                    tts = TTSOutput.model_validate(manager.worker('app.tts_worker', input_path, tts_output, s.timeouts.tts, progress))
                    if not valid_audio(tts,params.languages,params.image_count):
                        raise ValueError('Invalid TTS output artifacts')
                    paths = [tts_output]+[a.audio for a in tts.languages.values()]+[seg.path for a in tts.languages.values() for seg in a.segments]
                    manifest.complete('tts',tts_key,paths)
                    report['retries']['tts_segments'] = tts.retries
            for lang, audio in tts.languages.items():
                report['per_language'][lang] = {'scene_timings': [seg.model_dump(mode='json') for seg in audio.segments],
                                                'audio_duration_s': audio.segments[-1].end,
                                                'tts_s': tts.timings.get(lang),
                                                'script_s': report.get('script_language_seconds',{}).get(lang)}
                manifest.data['timings'][lang] = report['per_language'][lang]
            align_output = directory/'tmp/whisperx_output.json'
            align_key = {'tts':file_hash(tts_output), 'models':versions['huggingface']}
            if not manifest.reusable('alignment',align_key,[align_output]):
                with stage('subtitles'):
                    input_path = directory/'tmp/whisperx_input.json'
                    write_json(input_path, WorkerInput(run_dir=directory,languages=params.languages,timeout=s.timeouts.subtitles).model_dump(mode='json'))
                    aligned = AlignmentOutput.model_validate(manager.worker('app.whisperx_worker',input_path,align_output,s.timeouts.subtitles,progress))
                    if set(aligned.languages) != set(params.languages):
                        raise ValueError('Alignment languages mismatch')
                    manifest.complete('alignment',align_key,[align_output])
            aligned = AlignmentOutput.model_validate(read_json(align_output))
            outputs = {}
            for lang in params.languages:
                alignment = aligned.languages[lang]
                report['per_language'][lang]['alignment_s'] = aligned.timings.get(lang)
                if alignment.fallback:
                    report['per_language'][lang]['alignment_fallback'] = alignment.fallback
                    report['per_language'][lang]['alignment_reason'] = alignment.reason
                subs_key = {'alignment':file_hash(align_output),'tts':file_hash(tts_output)}
                subs_paths = [directory/f'subtitles/{lang}.srt', directory/f'subtitles/{lang}.ass']
                if not manifest.reusable(f'subtitles_{lang}',subs_key,subs_paths):
                    write_subtitles(directory/'subtitles',lang,tts.languages[lang],alignment)
                    manifest.complete(f'subtitles_{lang}',subs_key,subs_paths)
                video = directory/f'video_{lang}.mp4'
                render_key = {'images':[file_hash(p) for p in images], 'audio':file_hash(tts.languages[lang].audio),
                              'ass':file_hash(subs_paths[1]), 'music':music_hash}
                if not (manifest.reusable(f'render_{lang}',render_key,[video]) and valid_video(video,tts.languages[lang].segments[-1].end)):
                    with stage(f'render_{lang}'):
                        renderer = Renderer(s.timeouts.render)
                        render_started = time.monotonic()
                        details = renderer.render(images,tts.languages[lang],subs_paths[1],video,params.music)
                        report['per_language'][lang]['render_s'] = time.monotonic()-render_started
                        report['per_language'][lang]['render'] = details
                        manifest.complete(f'render_{lang}',render_key,[video])
                outputs[lang] = video
            for lang in {'ru','en'} - set(params.languages):
                previous = directory/f'video_{lang}.mp4'
                if previous.exists():
                    archive = directory/'tmp/archive'
                    archive.mkdir(exist_ok=True)
                    previous.replace(archive/f'{uuid.uuid4().hex[:8]}_{previous.name}')
            report['status'] = 'complete'
            report['timings']['render_s'] = sum(v for k,v in report['timings'].items() if k.startswith('render_') and k != 'render_s')
            return outputs
        except BaseException as error:
            report['status'] = 'failed'
            report['errors'].append({'type':type(error).__name__, 'message':str(error)})
            raise
        finally:
            manager.close()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            report['total_s'] = time.monotonic()-started
            for name in ('script_s','images_s','tts_s','subtitles_s','render_s'):
                report['timings'].setdefault(name,0)
            manifest.data['status'] = report['status']
            manifest.save()
            write_json(directory/'generation_report.json',report)
