import json
import httpx
import pytest
from pydantic import ValidationError
from PIL import Image
from app.config import ROOT, Settings
from app.content import ManualPromptProvider, require_news_text
from app.files import Manifest, valid_png
from app.llm import ScriptGenerator
from app.schema import Params, Script, Classification
from app.tts_worker import validate_voice
from app.ui import music_choices

@pytest.mark.parametrize('duration,count',[(30,8),(45,9),(60,12)])
def test_scene_count(duration,count):
    assert Params(prompt='test',duration=duration).image_count == count

@pytest.mark.parametrize('languages',[['ru'],['en'],['ru','en']])
def test_language_modes(languages):
    assert Params(prompt='a',languages=languages).languages==languages

@pytest.mark.parametrize('values',[{'languages':['fr']},{'languages':['ru','ru']},{'duration':31},{'seed':-1},{'prompt':''}])
def test_invalid_params(values):
    with pytest.raises(ValidationError): Params(**({'prompt':'test'} | values))


def test_settings(settings,monkeypatch,tmp_path):
    monkeypatch.setenv('OUTPUT_DIR',str(tmp_path))
    monkeypatch.setenv('TIMEOUT_IMAGES','25')
    loaded=Settings.load()
    assert loaded.output_dir==tmp_path and loaded.timeouts.images==25
    with pytest.raises(ValidationError): Settings(width=833)


def test_content_no_network():
    value=ManualPromptProvider().fetch('some text',['https://example.invalid'])
    assert value.text=='some text' and not value.verified
    assert value.source_urls==['https://example.invalid']
    with pytest.raises(ValueError,match='NEWS_CONTENT_REQUIRED'): require_news_text(value)
    require_news_text(ManualPromptProvider().fetch('Подробный текст новости с известными фактами. '*10,[]))
    with pytest.raises(ValueError): require_news_text(ManualPromptProvider().fetch('https://example.org/'+'a'*200,[]))

@pytest.mark.parametrize('name',['classify','what_if','science','facts','story','news'])
def test_templates(name):
    assert (ROOT/f'config/prompts/{name}.txt').stat().st_size>100


def test_schema_retry(settings,tmp_path,script):
    calls=[]
    responses=['bad json',json.dumps({'title':'missing scenes'}),script.model_dump_json()]
    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200,json={'message':{'content':responses[len(calls)-1]}})
    gen=ScriptGenerator(settings,httpx.Client(transport=httpx.MockTransport(handle),base_url='http://test'))
    assert gen.structured(Script,'system','prompt',tmp_path/'last.txt')==script
    assert gen.retries==2 and len(calls)==3
    assert calls[0]['format']==Script.model_json_schema()
    assert calls[0]['options']['temperature']==.7 and 'seed' not in calls[0]['options']
    assert 'Correct validation errors' in calls[1]['messages'][-1]['content']


def test_invalid_llm_stops(settings,tmp_path):
    client=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'message':{'content':'bad'}})),base_url='http://test')
    gen=ScriptGenerator(settings,client)
    with pytest.raises(ValueError,match='LLM_INVALID_JSON'): gen.structured(Classification,'s','p',tmp_path/'last.txt')
    assert gen.retries==2 and (tmp_path/'last.txt').read_text()=='bad'


def test_separate_localization(settings,tmp_path,script):
    (tmp_path/'tmp').mkdir()
    calls=[]
    def handle(request):
        value=json.loads(request.content);calls.append(value)
        result=script.model_dump();result['language']='en' if len(calls)==2 else 'ru'
        return httpx.Response(200,json={'message':{'content':json.dumps(result)}})
    gen=ScriptGenerator(settings,httpx.Client(transport=httpx.MockTransport(handle),base_url='http://test'))
    scripts=gen.generate(Params(prompt='test',languages=['ru','en'],duration=30,content_type='facts'),ManualPromptProvider().fetch('test',[]),tmp_path)
    assert set(scripts)=={'ru','en'} and len(calls)==2
    assert calls[0]['messages'][0]['content'] != calls[1]['messages'][0]['content']
    assert json.loads(calls[1]['messages'][1]['content'])['shared_outline'][0]['key_fact']=='Fact 0'


def test_manifest_invalidation(tmp_path):
    artifact=tmp_path/'artifact';artifact.write_text('valid')
    manifest=Manifest(tmp_path,{'seed':1},{})
    manifest.complete('images',{'script':'a'},[artifact])
    assert manifest.reusable('images',{'script':'a'},[artifact])
    assert not manifest.reusable('images',{'script':'b'},[artifact])
    artifact.write_text('changed')
    assert not manifest.reusable('images',{'script':'a'},[artifact])
    assert not Manifest(tmp_path,{'seed':2},{}).reusable('images',{'script':'a'},[artifact])


def test_png_validation(tmp_path):
    path=tmp_path/'image.png'
    assert not valid_png(path,832,1216)
    Image.new('RGB',(832,1216),'white').save(path)
    assert not valid_png(path,832,1216)
    import numpy as np
    image=Image.fromarray(np.random.default_rng(42).integers(0,256,(1216,832,3),dtype=np.uint8))
    image.save(path)
    assert valid_png(path,832,1216)
    assert not valid_png(path,1216,832)
    path.write_bytes(b'corrupt'*2000)
    assert not valid_png(path,832,1216)


def test_voices():
    class Model:
        def get_supported_speakers(self): return ['ryan']
        def get_supported_languages(self): return ['Russian','English']
    validate_voice(Model(),{'speaker':'Ryan','language':'Russian'})
    with pytest.raises(ValueError,match='UNSUPPORTED_SPEAKER'): validate_voice(Model(),{'speaker':'bad','language':'Russian'})
    with pytest.raises(ValueError,match='UNSUPPORTED_LANGUAGE'): validate_voice(Model(),{'speaker':'Ryan','language':'bad'})


def test_music(tmp_path):
    for filename in ['a.mp3','b.wav','c.m4a','d.ogg','no.txt']:
        (tmp_path/filename).touch()
    choices=music_choices(tmp_path)
    assert choices[0]==('Без музыки','') and len(choices)==5


def test_wrong_scene_count_retries(settings,tmp_path,script):
    (tmp_path/'tmp').mkdir()
    calls=[]
    def handle(request):
        calls.append(request)
        return httpx.Response(200,json={'message':{'content':script.model_dump_json()}})
    gen=ScriptGenerator(settings,httpx.Client(transport=httpx.MockTransport(handle),base_url='http://test'))
    with pytest.raises(ValueError,match='exactly 9 scenes'):
        gen.generate(Params(prompt='test',duration=45,content_type='facts'),ManualPromptProvider().fetch('test',[]),tmp_path)
    assert len(calls)==3


def test_auto_cache_and_explicit_news_skip_classification(settings,tmp_path,script):
    (tmp_path/'tmp').mkdir()
    calls=[]
    def handle(request):
        value=json.loads(request.content);calls.append(value)
        content=json.dumps({'content_type':'facts'}) if value['format']['title']=='Classification' else script.model_dump_json()
        return httpx.Response(200,json={'message':{'content':content}})
    client=httpx.Client(transport=httpx.MockTransport(handle),base_url='http://test')
    research=ManualPromptProvider().fetch('Detailed supplied news text with enough factual content and words to qualify as a source document. '*3,[])
    for _ in range(2):
        ScriptGenerator(settings,client).generate(Params(prompt=research.text,duration=30),research,tmp_path)
    assert [c['format']['title'] for c in calls].count('Classification')==1
    calls.clear()
    ScriptGenerator(settings,client).generate(Params(prompt=research.text,duration=30,content_type='news'),research,tmp_path)
    assert len(calls)==1 and calls[0]['format']['title']=='Script'


def test_download_integrity(tmp_path):
    import hashlib
    from app.download import verify_file
    path=tmp_path/'model';path.write_bytes(b'weights')
    entry={'size':7,'sha256':hashlib.sha256(b'weights').hexdigest()}
    assert verify_file(path,entry)
    path.write_bytes(b'corrupt')
    assert not verify_file(path,entry)
    path.unlink()
    assert not verify_file(path,entry)
