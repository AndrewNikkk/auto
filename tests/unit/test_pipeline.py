from contextlib import contextmanager
from pathlib import Path
import wave
import pytest
from app.files import read_json,write_json
from app.schema import Params,AudioResult,TTSOutput,AlignmentOutput,AlignmentResult
from app.pipeline import Pipeline

@pytest.fixture
def fake_services(monkeypatch,settings,script):
    import app.pipeline as module
    import numpy as np
    from PIL import Image
    calls={'llm':0,'images':[],'tts':0,'align':0,'render':[]}
    class Manager:
        def __init__(self,*a):self.registry=settings.output_dir/'.gpu-procs'
        @contextmanager
        def server(self,*a,**k):yield
        def diagnostics(self):return 'mock'
        def close(self):pass
        def worker(self,name,inp,out,timeout,progress):
            request=read_json(inp);directory=Path(request['run_dir'])
            if name=='app.tts_worker':
                calls['tts']+=1
                results={}
                for language in request['languages']:
                    chunks=[];position=0;length=.1 if language=='ru' else .15
                    for i in range(8):
                        path=directory/f'audio/{language}_{i}.wav'
                        with wave.open(str(path),'wb') as stream:
                            stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(24000)
                            stream.writeframes(b'\x00\x01'*round(24000*length))
                        chunks.append(dict(text='text',start=position,end=position+length,path=path));position+=length
                    audio=directory/f'audio/{language}.wav'
                    with wave.open(str(audio),'wb') as stream:
                        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(24000)
                        stream.writeframes(b'\x00\x01'*round(24000*position))
                    results[language]=AudioResult(audio=audio,sample_rate=24000,segments=chunks)
                result=TTSOutput(languages=results)
            else:
                calls['align']+=1
                result=AlignmentOutput(languages={lang:AlignmentResult(fallback='segment_level') for lang in request['languages']})
            write_json(out,result.model_dump(mode='json'));return read_json(out)
    class Generator:
        def __init__(self,*a):self.retries=0;self.client=type('Client',(),{'close':lambda self:None})();self.timings={}
        def generate(self,params,*a):
            calls['llm']+=1
            return {lang:script.model_copy(update={'language':lang}) for lang in params.languages}
    class Comfy:
        def __init__(self,*a):self.client=type('Client',(),{'close':lambda self:None})()
        def check_nodes(self):pass
        def image(self,prompt,seed,target,timeout):
            calls['images'].append(target.name)
            Image.fromarray(np.random.default_rng(seed).integers(0,256,(1216,832,3),dtype=np.uint8)).save(target)
    class Renderer:
        def __init__(self,*a):pass
        def render(self,images,audio,ass,target,music):
            calls['render'].append(target.name);target.write_bytes(b'mockvideo')
            return {'duration':audio.segments[-1].end}
    monkeypatch.setattr(module,'GPUProcessManager',Manager)
    monkeypatch.setattr(module,'ScriptGenerator',Generator)
    monkeypatch.setattr(module,'ComfyClient',Comfy)
    monkeypatch.setattr(module,'Renderer',Renderer)
    monkeypatch.setattr(module,'valid_video',lambda p,*a:p.is_file() and p.read_bytes()==b'mockvideo')
    return calls

@pytest.mark.parametrize('languages',[['ru'],['en'],['ru','en']])
def test_pipeline_resume_and_language_timings(settings,fake_services,languages):
    params=Params(prompt='test',duration=30,content_type='facts',languages=languages)
    pipeline=Pipeline(settings)
    outputs=pipeline.run(params,'resume')
    assert set(outputs)==set(languages)
    assert len(fake_services['images'])==8
    report=read_json(settings.output_dir/'resume/generation_report.json')
    assert report['status']=='complete' and set(report['per_language'])==set(languages)
    if len(languages)==2:
        assert report['per_language']['ru']['audio_duration_s'] != report['per_language']['en']['audio_duration_s']
    snapshot={key:list(value) if isinstance(value,list) else value for key,value in fake_services.items()}
    pipeline.run(params,'resume')
    assert fake_services==snapshot
    (settings.output_dir/'resume/images/03.png').write_bytes(b'corrupt')
    pipeline.run(params,'resume')
    assert fake_services['images']==snapshot['images']+['03.png']
    assert fake_services['tts']==1


def test_news_rejected_before_models(settings,fake_services):
    with pytest.raises(ValueError,match='NEWS_CONTENT_REQUIRED'):
        Pipeline(settings).run(Params(prompt='news please',content_type='news'),'news')
    assert fake_services['llm']==0
    report=read_json(settings.output_dir/'news/generation_report.json')
    assert report['status']=='failed'


def test_path_and_lock(settings):
    import fcntl
    settings.output_dir.mkdir(exist_ok=True)
    with (settings.output_dir/'.generation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='GENERATION_BUSY'):Pipeline(settings).run(Params(prompt='x'))
    with pytest.raises(ValueError,match='Invalid run ID'):Pipeline(settings).run(Params(prompt='x'),'../escape')


def test_changing_language_archives_previous_final(settings,fake_services):
    pipeline=Pipeline(settings)
    pipeline.run(Params(prompt='test',duration=30,content_type='facts',languages=['ru','en']),'switch')
    pipeline.run(Params(prompt='test',duration=30,content_type='facts',languages=['ru']),'switch')
    run=settings.output_dir/'switch'
    assert (run/'video_ru.mp4').is_file() and not (run/'video_en.mp4').exists()
    assert len(list((run/'tmp/archive').glob('*_video_en.mp4')))==1
