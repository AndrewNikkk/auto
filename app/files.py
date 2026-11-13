import hashlib
import json
import os
import subprocess
import wave
from pathlib import Path
from PIL import Image, ImageStat


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def fingerprint(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def valid_png(path: Path, width: int, height: int, threshold: float = 5) -> bool:
    try:
        if path.stat().st_size <= 10240:
            return False
        with Image.open(path) as im:
            im.load()
            if im.format != 'PNG' or im.size != (width, height):
                return False
            return max(ImageStat.Stat(im.convert('RGB')).stddev) >= threshold
    except (OSError, ValueError):
        return False


def wav_duration(path: Path, sample_rate: int | None = None) -> float:
    with wave.open(str(path)) as audio:
        if audio.getnchannels() != 1 or audio.getnframes() < 1:
            raise ValueError(f'Invalid mono WAV: {path}')
        if sample_rate and audio.getframerate() != sample_rate:
            raise ValueError('WAV sample rate mismatch')
        return audio.getnframes() / audio.getframerate()


def probe(path: Path) -> dict:
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)],
                            timeout=20, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def valid_video(path: Path, expected_duration: float | None = None) -> bool:
    try:
        data = probe(path)
        video = next(s for s in data['streams'] if s['codec_type'] == 'video')
        audio = next(s for s in data['streams'] if s['codec_type'] == 'audio')
        return (video['codec_name'] == 'h264' and (video['width'], video['height']) == (1080, 1920)
                and video['r_frame_rate'] == '30/1' and audio['codec_name'] == 'aac'
                and (expected_duration is None or abs(float(data['format']['duration']) - expected_duration) < .25))
    except (OSError, ValueError, KeyError, StopIteration, subprocess.SubprocessError):
        return False


class Manifest:
    def __init__(self, run_dir: Path, params: dict, versions: dict):
        self.path = run_dir / 'manifest.json'
        self.key = fingerprint({'params': params, 'versions': versions})
        try:
            old = read_json(self.path) if self.path.exists() else {}
        except (OSError, ValueError):
            old = {}
        self.data = old if old.get('input_hash') == self.key else {
            'input_hash': self.key, 'params': params, 'models': versions, 'stages': {}, 'timings': {}}
        self.save()

    def reusable(self, stage: str, inputs, paths: list[Path]) -> bool:
        old = self.data['stages'].get(stage, {})
        return (old.get('key') == fingerprint(inputs) and bool(paths)
                and all(p.is_file() and old.get('files', {}).get(str(p)) == file_hash(p) for p in paths))

    def complete(self, stage: str, inputs, paths: list[Path]):
        self.data['stages'][stage] = {'key': fingerprint(inputs), 'files': {str(p): file_hash(p) for p in paths}}
        self.save()

    def save(self):
        write_json(self.path, self.data)
