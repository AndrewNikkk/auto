import os
from pathlib import Path
import yaml
from pydantic import Field
from app.schema import StrictModel

ROOT = Path(__file__).resolve().parent.parent

class Timeouts(StrictModel):
    script: float = Field(default=120, gt=0)
    images: float = Field(default=600, gt=0)
    image: float = Field(default=90, gt=0)
    tts: float = Field(default=300, gt=0)
    subtitles: float = Field(default=180, gt=0)
    render: float = Field(default=300, gt=0)
    readiness: float = Field(default=90, gt=0)

class Settings(StrictModel):
    width: int = Field(default=832, ge=512, multiple_of=16)
    height: int = Field(default=1216, ge=512, multiple_of=16)
    pixel_std_min: float = Field(default=5, ge=0)
    llm_model: str = 'qwen3:8b-q4_K_M'
    ollama_url: str = 'http://127.0.0.1:11434'
    comfy_url: str = 'http://127.0.0.1:8188'
    comfy_dir: Path = Path('/opt/ComfyUI')
    output_dir: Path = Path('outputs')
    music_dir: Path = Path('assets/music')
    hf_home: Path = Path('data/models/huggingface')
    timeouts: Timeouts = Field(default_factory=Timeouts)
    vram_release_tolerance_mb: int = Field(default=512, ge=0)

    @classmethod
    def load(cls, path: Path = ROOT / 'config/settings.yaml'):
        raw = yaml.safe_load(path.read_text())
        for name in ('output_dir', 'music_dir', 'hf_home', 'comfy_dir'):
            if name.upper() in os.environ:
                raw[name] = os.environ[name.upper()]
        for name in Timeouts.model_fields:
            if f'TIMEOUT_{name.upper()}' in os.environ:
                raw['timeouts'][name] = float(os.environ[f'TIMEOUT_{name.upper()}'])
        return cls.model_validate(raw)

    def model_path(self, key: str) -> Path:
        lock = yaml.safe_load((ROOT / 'models.lock.yaml').read_text())
        entry = lock['huggingface'][key]
        if entry.get('requires_manual_verification') or not entry.get('revision'):
            raise RuntimeError(f'MODEL_UNVERIFIED: {key}; see models.lock.yaml')
        return self.hf_home / 'snapshots' / key / entry['revision']
