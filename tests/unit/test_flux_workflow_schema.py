import json
import httpx
import pytest
from app.comfy import workflow, ComfyClient
from app.config import ROOT


def test_workflow_structure():
    data=json.loads((ROOT/'workflows/flux_schnell_api.json').read_text())
    assert {'UNETLoader','DualCLIPLoader','VAELoader','KSampler','VAEDecode','SaveImage'} <= {n['class_type'] for n in data.values()}
    for node in data.values():
        for value in node['inputs'].values():
            if isinstance(value,list):
                assert value[0] in data and isinstance(value[1],int)
    assert data['7']['inputs']['steps']==4
    assert data['7']['inputs']['cfg']==1.0
    assert data['7']['inputs']['sampler_name']=='euler'
    assert data['7']['inputs']['scheduler']=='simple'
    assert data['1']['inputs']=={'unet_name':'flux1-schnell.safetensors','weight_dtype':'fp8_e4m3fn'}
    assert data['2']['inputs']['clip_name1']=='clip_l.safetensors'
    assert data['2']['inputs']['clip_name2']=='t5xxl_fp8_e4m3fn.safetensors'
    assert data['3']['inputs']['vae_name']=='ae.safetensors'
    assert data['9']['inputs']['images']==['8',0]


def test_substitution():
    value=workflow('new prompt',1234,640,960)
    assert value['4']['inputs']['text']=='new prompt'
    assert value['7']['inputs']['seed']==1234
    assert value['6']['inputs']['width']==640 and value['6']['inputs']['height']==960
    assert workflow('x',1,832,1216)['7']['inputs']['seed']==1


def test_http_mock(settings,tmp_path):
    from PIL import Image
    import numpy as np
    from io import BytesIO
    buffer=BytesIO()
    Image.fromarray(np.random.default_rng(1).integers(0,256,(1216,832,3),dtype=np.uint8)).save(buffer,format='PNG')
    paths=[]
    def handle(request):
        paths.append(request.url.path)
        if request.url.path=='/object_info': return httpx.Response(200,json={n['class_type']:{} for n in workflow('',1,832,1216).values()})
        if request.url.path=='/prompt': return httpx.Response(200,json={'prompt_id':'abc'})
        if request.url.path=='/history/abc': return httpx.Response(200,json={'abc':{'outputs':{'9':{'images':[{'filename':'x.png','subfolder':'','type':'output'}]}}}})
        return httpx.Response(200,content=buffer.getvalue())
    client=ComfyClient(settings,httpx.Client(transport=httpx.MockTransport(handle),base_url='http://test'))
    client.check_nodes();client.image('p',1,tmp_path/'result.png',5)
    assert paths==['/object_info','/prompt','/history/abc','/view']


def test_http_timeout(settings,tmp_path,monkeypatch):
    calls=[]
    def handle(request):
        calls.append(request.url.path)
        return httpx.Response(200,json={'prompt_id':'abc'} if request.url.path=='/prompt' else {})
    client=ComfyClient(settings,httpx.Client(transport=httpx.MockTransport(handle),base_url='http://test'))
    with pytest.raises(TimeoutError): client.image('p',1,tmp_path/'out.png',.01)
    assert calls[-1]=='/interrupt'


def test_comfy_execution_error_no_silent_success(settings,tmp_path):
    def handle(request):
        if request.url.path=='/prompt':return httpx.Response(200,json={'prompt_id':'abc'})
        return httpx.Response(200,json={'abc':{'status':{'status_str':'error','messages':['CUDA out of memory']}}})
    client=ComfyClient(settings,httpx.Client(transport=httpx.MockTransport(handle),base_url='http://test'))
    with pytest.raises(RuntimeError,match='out of memory'):client.image('p',1,tmp_path/'image.png',1)
