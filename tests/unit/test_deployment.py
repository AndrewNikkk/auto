import yaml
from app.config import ROOT


def test_single_service_private_gpu_compose():
    compose=yaml.safe_load((ROOT/'docker-compose.yml').read_text())
    assert set(compose['services'])=={'app'}
    app=compose['services']['app']
    assert '127.0.0.1' in app['ports'][0]
    assert app['environment']['GRADIO_BIND']=='0.0.0.0'
    assert app['environment']['NVIDIA_DRIVER_CAPABILITIES']=='compute,utility,video'
    assert app['deploy']['resources']['reservations']['devices'][0]['capabilities']==['gpu']
    assert all('docker.sock' not in volume for volume in app['volumes'])
    lock=yaml.safe_load((ROOT/'models.lock.yaml').read_text())
    assert app['build']['args']['COMFY_COMMIT']==lock['comfyui']['git_commit']
    assert len(lock['comfyui']['git_commit'])==40
    for model in lock['huggingface'].values():
        assert len(model['revision'])==40
        assert model['license']=='apache-2.0' and not model['requires_manual_verification']
        assert all(file['size']>0 and len(file['sha256'])==64 for file in model['files'])


def test_no_database_in_dependency_lock():
    lock=(ROOT/'requirements.lock').read_text().lower()
    for dependency in ['alembic==','sqlalchemy==','redis==','celery==']:
        assert dependency not in lock
