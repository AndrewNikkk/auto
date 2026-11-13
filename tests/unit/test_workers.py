import sys
import types
import numpy as np
import pytest
from app.files import write_json
from app.schema import AudioResult, TTSOutput, WorkerInput


def test_tts_segments_retry_and_cache(tmp_path,settings,script,monkeypatch):
    from app import tts_worker
    settings.model_path('qwen3_tts').mkdir(parents=True)
    monkeypatch.setattr(tts_worker.Settings,'load',lambda:settings)
    calls=[]
    class Model:
        def get_supported_speakers(self):return ['ryan']
        def get_supported_languages(self):return ['russian','english']
        def generate_custom_voice(self,**kwargs):
            calls.append(kwargs)
            if len(calls)==1:return [np.zeros(2400,dtype=np.float32)],24000
            return [np.sin(np.arange(4800,dtype=np.float32)*.1)*.2],24000
    class Factory:
        @staticmethod
        def from_pretrained(*args,**kwargs):return Model()
    monkeypatch.setitem(sys.modules,'qwen_tts',types.SimpleNamespace(Qwen3TTSModel=Factory))
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(bfloat16='bf16'))
    run=tmp_path/'run';(run/'tmp').mkdir(parents=True)
    write_json(run/'script.json',{'scripts':{'ru':script.model_dump(),'en':script.model_copy(update={'language':'en'}).model_dump()}})
    request=WorkerInput(run_dir=run,languages=['ru','en'],timeout=30)
    result=tts_worker.generate(request)
    assert result.retries==1 and len(calls)==17
    assert set(result.timings)=={'ru','en'}
    assert all(len(audio.segments)==8 for audio in result.languages.values())
    second=tts_worker.generate(request)
    assert len(calls)==17 and second.retries==0
    result.languages['ru'].segments[2].path.write_bytes(b'broken')
    tts_worker.generate(request)
    assert len(calls)==18

@pytest.mark.parametrize('mode',['success','incomplete','error','oom'])
def test_alignment_worker_policy(tmp_path,settings,monkeypatch,mode):
    from app import whisperx_worker
    monkeypatch.setattr(whisperx_worker.Settings,'load',lambda:settings)
    module=types.ModuleType('whisperx.alignment')
    module.nltk_load=lambda _: (_ for _ in ()).throw(AssertionError('pretrained tokenizer must not load'))
    module.load_align_model=lambda **kwargs:(object(),{})
    def align(*args,**kwargs):
        if mode=='error':raise ValueError('no alignment path')
        if mode=='oom':raise RuntimeError('CUDA out of memory')
        return {'word_segments':[{'word':'Hello','start':.1,'end':.4} if mode=='success' else {'word':'Hello'}]}
    module.align=align
    root=types.ModuleType('whisperx');root.alignment=module
    monkeypatch.setitem(sys.modules,'whisperx',root)
    monkeypatch.setitem(sys.modules,'whisperx.alignment',module)
    monkeypatch.setitem(sys.modules,'whisperx.audio',types.SimpleNamespace(load_audio=lambda p:np.zeros(16000)))
    run=tmp_path/'run';(run/'tmp').mkdir(parents=True)
    audio=AudioResult(audio=run/'en.wav',sample_rate=24000,segments=[dict(text='Hello',start=0,end=1,path=run/'0.wav')])
    write_json(run/'tmp/tts_output.json',TTSOutput(languages={'en':audio}).model_dump(mode='json'))
    request=WorkerInput(run_dir=run,languages=['en'],timeout=10)
    if mode=='oom':
        with pytest.raises(RuntimeError,match='CUDA OOM'):whisperx_worker.generate(request)
    else:
        result=whisperx_worker.generate(request).languages['en']
        assert bool(result.words)==(mode=='success')
        assert result.fallback==(None if mode=='success' else 'segment_level')
