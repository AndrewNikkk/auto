import argparse
import os
from pathlib import Path
import shutil
import httpx
import yaml
from huggingface_hub import HfApi, hf_hub_download
from app.config import ROOT, Settings
from app.files import file_hash
from app.gpu import GPUProcessManager


def verify_file(path: Path, entry: dict):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    return not entry.get('sha256') or file_hash(path) == entry['sha256']


def check_models(lock: dict, settings: Settings) -> list[str]:
    missing = []
    comfy = Path(os.getenv('COMFYUI_MODELS','/data/models/comfyui'))
    for key,model in lock['huggingface'].items():
        if not model.get('revision') or model.get('requires_manual_verification'):
            missing.append(f'{key}: requires_manual_verification')
            continue
        for entry in model['files']:
            path = comfy/entry['destination'] if 'destination' in entry else settings.model_path(key)/entry['path']
            if not verify_file(path,entry):
                missing.append(str(path))
    ollama = Path(os.getenv('OLLAMA_MODELS','/data/models/ollama'))
    manifest = ollama/'manifests/registry.ollama.ai/library/qwen3/8b-q4_K_M'
    if not manifest.is_file() or 'sha256:'+file_hash(manifest) != lock['ollama']['digest']:
        missing.append('Ollama model manifest/digest')
    else:
        import json
        content=json.loads(manifest.read_text())
        for blob in [content['config'],*content['layers']]:
            path=ollama/'blobs'/blob['digest'].replace(':','-')
            if not path.is_file() or path.stat().st_size != blob['size'] or 'sha256:'+file_hash(path) != blob['digest']:
                missing.append(str(path))
    return missing


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    settings=Settings.load()
    lock=yaml.safe_load((ROOT/'models.lock.yaml').read_text())
    if args.check_only:
        missing=check_models(lock,settings)
        if missing:
            parser.exit(1,'MODEL_CHECK_FAILED:\n'+'\n'.join(missing)+'\n')
        print('VERIFIED: model files and SHA256')
        return
    api=HfApi()
    comfy=Path(os.getenv('COMFYUI_MODELS','/data/models/comfyui'))
    for key,model in lock['huggingface'].items():
        if not model.get('revision') or model.get('requires_manual_verification'):
            raise RuntimeError(f'MODEL_UNVERIFIED: {key}')
        info=api.model_info(model['repo'],revision=model['revision'],files_metadata=True,timeout=30)
        if info.sha != model['revision'] or info.card_data.get('license') != model['license']:
            raise RuntimeError(f'Model revision/license differs from lock: {key}')
        for entry in model['files']:
            target=comfy/entry['destination'] if 'destination' in entry else settings.model_path(key)/entry['path']
            if verify_file(target,entry):
                continue
            cached=Path(hf_hub_download(model['repo'],entry['path'],revision=model['revision']))
            if not verify_file(cached,entry):
                cached = Path(hf_hub_download(model['repo'], entry['path'], revision=model['revision'], force_download=True))
            if not verify_file(cached,entry):
                raise RuntimeError(f'MODEL_CHECKSUM_FAILED: {cached}')
            target.parent.mkdir(parents=True,exist_ok=True)
            # Cache and snapshots share one persistent HF volume. Hardlinks avoid a second weight copy.
            temporary=target.with_suffix(target.suffix+'.partial')
            temporary.unlink(missing_ok=True)
            try:
                os.link(cached.resolve(),temporary)
            except OSError:
                shutil.copyfile(cached,temporary)
            temporary.replace(target)
            print(f'VERIFIED: {key}/{entry["path"]}',flush=True)
    settings.output_dir.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (settings.output_dir/'.generation.lock').open('a') as lock_file:
        fcntl.flock(lock_file,fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager=GPUProcessManager(settings.output_dir)
        try:
            with manager.server('ollama',['ollama','serve'],settings.ollama_url,'/api/tags',settings.timeouts.readiness,
                                model=lock['ollama']['model']):
                with httpx.Client(base_url=settings.ollama_url,timeout=3600) as client:
                    response=client.post('/api/pull',json={'model':lock['ollama']['model'],'stream':False})
                    response.raise_for_status()
                    if response.json().get('error'):
                        raise RuntimeError(response.json()['error'])
                    tags=client.get('/api/tags').json()['models']
                    match=next(m for m in tags if m['name']==lock['ollama']['model'])
                    if match['digest'] != lock['ollama']['digest']:
                        raise RuntimeError('OLLAMA_DIGEST_CHANGED: upstream tag changed; review lock explicitly')
        finally:
            manager.close()
    missing=check_models(lock,settings)
    if missing:
        raise RuntimeError('MODEL_CHECK_FAILED: '+str(missing))

if __name__=='__main__':
    main()
