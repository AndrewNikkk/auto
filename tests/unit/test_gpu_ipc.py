import json
from app.schema import WorkerInput
import subprocess
import sys
import time
from unittest.mock import Mock
import pytest
from app.files import write_json
from app.gpu import GPUProcessManager, process_identity, owned, VRAMMonitor


def test_real_process_lifecycle(tmp_path):
    manager=GPUProcessManager(tmp_path)
    process=manager.start('test',[sys.executable,'-c','import time; time.sleep(60)'])
    record=process_identity(process.pid)
    assert owned(record) and (manager.registry/'test.json').is_file()
    with pytest.raises(RuntimeError): manager.start('another',[sys.executable,'-c','pass'])
    manager.stop('test')
    assert process.poll() is not None
    assert not (manager.registry/'test.json').exists()
    manager.close()


def test_orphan_cleanup_identity(tmp_path):
    foreign=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
    orphan=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
    registry=tmp_path/'.gpu-procs';registry.mkdir()
    identity=process_identity(foreign.pid);identity['created']-=1
    write_json(registry/'foreign.json',dict(identity,component='comfy',heartbeat=0))
    write_json(registry/'orphan.json',dict(process_identity(orphan.pid),component='comfy',heartbeat=0))
    try:
        manager=GPUProcessManager(tmp_path)
        assert foreign.poll() is None and orphan.poll() is not None
        assert not list(registry.glob('*.json'))
        manager.close()
    finally:
        foreign.terminate();foreign.wait(timeout=5)
        if orphan.poll() is None: orphan.terminate();orphan.wait(timeout=5)


def test_worker_errors(tmp_path,monkeypatch):
    manager=GPUProcessManager(tmp_path)
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,1,'','CUDA out of memory'))
    with pytest.raises(RuntimeError,match='CUDA out of memory'): manager.worker('app.tts_worker',tmp_path/'in',tmp_path/'out',1)
    def timeout(*a,**k): raise subprocess.TimeoutExpired(a,1,stderr='timeout details')
    monkeypatch.setattr(subprocess,'run',timeout)
    with pytest.raises(TimeoutError): manager.worker('app.tts_worker',tmp_path/'in',tmp_path/'out',1)
    manager.close()


def test_worker_success_schema(tmp_path,monkeypatch):
    manager=GPUProcessManager(tmp_path)
    destination=tmp_path/'out'
    def execute(args,**kwargs):
        assert kwargs=={'timeout':18,'capture_output':True,'text':True,'start_new_session':True}
        assert args[1:3]==['-m','app.tts_worker']
        write_json(destination,{'languages':{}})
        return subprocess.CompletedProcess(args,0,'','')
    monkeypatch.setattr(subprocess,'run',execute)
    assert manager.worker('app.tts_worker',tmp_path/'in',destination,3)=={'languages':{}}
    manager.close()


def test_readiness_retry(tmp_path,monkeypatch):
    import httpx
    manager=GPUProcessManager(tmp_path)
    starts=[];stops=[]
    def start(*a,**k): starts.append(a);return Mock(poll=lambda:1)
    monkeypatch.setattr(manager,'start',start)
    monkeypatch.setattr(manager,'stop',lambda name:stops.append(name))
    def fail(*a,**k): raise httpx.ConnectError('offline')
    monkeypatch.setattr(httpx.Client,'get',fail)
    with pytest.raises(TimeoutError):
        with manager.server('comfy',['fake'],'http://test','/object_info',.01): pass
    assert len(starts)==2 and len(stops)==2
    manager.close()


def test_vram_samples(monkeypatch):
    monitor=VRAMMonitor()
    monkeypatch.setattr(monitor,'available',True)
    monkeypatch.setattr(monitor,'memory',lambda:{'used':12345,'total':99999})
    metrics={}
    with monitor.stage(metrics,512):
        time.sleep(.02)
    assert metrics['peak_vram_bytes']==12345 and metrics['memory_released'] is True


def test_actual_worker_json_error_exit(tmp_path):
    import os
    directory=tmp_path/'run'
    (directory/'tmp').mkdir(parents=True)
    input_path=directory/'tmp/tts_input.json';output_path=directory/'tmp/tts_output.json'
    write_json(input_path,WorkerInput(run_dir=directory,languages=['ru'],timeout=10).model_dump(mode='json'))
    env=dict(os.environ,HF_HOME=str(tmp_path/'missing-weights'))
    result=subprocess.run([sys.executable,'-m','app.tts_worker','--input',str(input_path),'--output',str(output_path)],
                          timeout=30,capture_output=True,text=True,env=env)
    assert result.returncode==1
    payload=json.loads(output_path.read_text())
    assert 'TTS weights missing' in payload['error']
    assert not list((tmp_path/'.gpu-procs').glob('*.json'))


def test_malformed_registry_does_not_kill(tmp_path):
    registry=tmp_path/'.gpu-procs';registry.mkdir()
    (registry/'bad.json').write_text('not json')
    manager=GPUProcessManager(tmp_path)
    assert (registry/'bad.json').exists()
    manager.close()


def test_deadline_kills_owned_worker_group_and_child(tmp_path,monkeypatch):
    import os
    import psutil
    from app.config import ROOT
    module=tmp_path/'test_short_worker.py'
    module.write_text('''import sys,time,subprocess,os
from pathlib import Path
from app.gpu import process_identity
from app.files import write_json
root=Path(sys.argv[sys.argv.index('--input')+1]).parent
write_json(root/'.gpu-procs'/f'worker-{os.getpid()}.json',dict(process_identity(os.getpid()),heartbeat=time.time(),component='worker'))
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
(root/'child.pid').write_text(str(child.pid))
time.sleep(60)
''')
    monkeypatch.setenv('PYTHONPATH',str(tmp_path)+os.pathsep+str(ROOT))
    manager=GPUProcessManager(tmp_path)
    try:
        with pytest.raises(TimeoutError,match='timeout'):
            manager.worker('test_short_worker',tmp_path/'input.json',tmp_path/'out.json',1)
        pid=int((tmp_path/'child.pid').read_text())
        try:assert psutil.Process(pid).status()==psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:pass
        assert not list(manager.registry.glob('worker-*.json'))
    finally:manager.close()


def test_application_shutdown_closes_active_managers(tmp_path):
    manager=GPUProcessManager(tmp_path)
    process=manager.start('test',[sys.executable,'-c','import time;time.sleep(60)'])
    GPUProcessManager.close_all()
    assert process.poll() is not None
    assert manager not in GPUProcessManager.instances
