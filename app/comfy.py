import copy
import time
from pathlib import Path
import httpx
from app.config import ROOT, Settings
from app.files import read_json, valid_png


def workflow(prompt: str, seed: int, width: int, height: int) -> dict:
    data = copy.deepcopy(read_json(ROOT / 'workflows/flux_schnell_api.json'))
    data['4']['inputs']['text'] = prompt
    data['7']['inputs']['seed'] = seed
    data['6']['inputs'].update(width=width, height=height)
    return data


class ComfyClient:
    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self.settings = settings
        self.client = client or httpx.Client(base_url=settings.comfy_url, timeout=15)

    def check_nodes(self):
        response = self.client.get('/object_info')
        response.raise_for_status()
        info = response.json()
        for node in workflow('', 1, self.settings.width, self.settings.height).values():
            if node['class_type'] not in info:
                raise RuntimeError(f'COMFY_WORKFLOW_INCOMPATIBLE: missing {node["class_type"]}')

    def image(self, prompt: str, seed: int, target: Path, timeout: float):
        end = time.monotonic() + timeout
        response = self.client.post('/prompt', json={'prompt': workflow(prompt, seed, self.settings.width, self.settings.height)})
        response.raise_for_status()
        payload = response.json()
        if payload.get('node_errors') or 'prompt_id' not in payload:
            raise RuntimeError(f'ComfyUI prompt validation failed: {payload}')
        prompt_id = payload['prompt_id']
        while time.monotonic() < end:
            response = self.client.get(f'/history/{prompt_id}')
            response.raise_for_status()
            history = response.json().get(prompt_id, {})
            status = history.get('status', {})
            if status.get('status_str') == 'error':
                raise RuntimeError(f'ComfyUI execution error: {status}')
            images = history.get('outputs', {}).get('9', {}).get('images', [])
            if images:
                response = self.client.get('/view', params=images[0])
                response.raise_for_status()
                target.write_bytes(response.content)
                if not valid_png(target, self.settings.width, self.settings.height, self.settings.pixel_std_min):
                    raise ValueError(f'INVALID_IMAGE: {target}')
                return
            time.sleep(.25)
        self.client.post('/interrupt')
        raise TimeoutError(f'IMAGE_TIMEOUT: {target.name}')
