import time
import wave
import yaml
from app.config import ROOT, Settings
from app.files import file_hash, fingerprint, read_json, wav_duration, write_json
from app.schema import AudioResult, Script, Segment, TTSOutput, WorkerInput
from app.worker_common import worker_main


def validate_voice(model, voice: dict):
    speakers = model.get_supported_speakers()
    languages = model.get_supported_languages()
    if not speakers or voice['speaker'].lower() not in {str(x).lower() for x in speakers}:
        raise ValueError(f'UNSUPPORTED_SPEAKER: {voice["speaker"]}; available={speakers}')
    if not languages or voice['language'].lower() not in {str(x).lower() for x in languages}:
        raise ValueError(f'UNSUPPORTED_LANGUAGE: {voice["language"]}; available={languages}')


def generate(request: WorkerInput) -> TTSOutput:
    settings = Settings.load()
    path = settings.model_path('qwen3_tts')
    if not path.is_dir():
        raise FileNotFoundError(f'TTS weights missing: {path}; run download_models.sh')
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(str(path), device_map='cuda:0', dtype=torch.bfloat16,
                                          attn_implementation='sdpa')
    voices = yaml.safe_load((ROOT/'config/voices.yaml').read_text())
    for language in request.languages:
        validate_voice(model, voices[language])
    scripts = read_json(request.run_dir/'script.json')['scripts']
    result = TTSOutput(languages={})
    deadline = time.monotonic() + request.timeout
    for language in request.languages:
        language_started = time.monotonic()
        script = Script.model_validate(scripts[language])
        directory = request.run_dir / 'audio' / language
        directory.mkdir(parents=True, exist_ok=True)
        segments = []
        position = 0.0
        rate = None
        chunks = []
        for i, scene in enumerate(script.scenes):
            target = directory / f'{i:02d}.wav'
            meta = target.with_suffix('.json')
            key = fingerprint({'text': scene.narration, 'voice': voices[language], 'model': str(path)})
            cached = False
            if target.exists() and meta.exists():
                try:
                    metadata = read_json(meta)
                    duration = wav_duration(target)
                    cached = (metadata.get('key') == key and duration > .1
                              and metadata.get('sha256') == file_hash(target))
                except (OSError, ValueError, EOFError, wave.Error):
                    pass
            if not cached:
                for attempt in range(3):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('TTS stage timeout')
                    try:
                        wavs, sr = model.generate_custom_voice(text=scene.narration,
                            language=voices[language]['language'], speaker=voices[language]['speaker'],
                            instruct='Clear, engaging narration at a brisk natural pace. No long pauses.', max_new_tokens=2048)
                        audio = np.asarray(wavs[0], dtype=np.float32)
                        if sr < 16000 or not np.isfinite(audio).all() or len(audio)/sr < .1 or np.max(np.abs(audio)) < .0001:
                            raise ValueError('Invalid or silent TTS audio')
                        # Consistent segment RMS; final mix uses measured two-pass LUFS normalization.
                        rms = np.sqrt(np.mean(audio**2))
                        audio *= min(.12 / max(rms, 1e-8), .95 / max(float(np.max(np.abs(audio))), 1e-8))
                        sf.write(target, audio, sr, subtype='PCM_16')
                        wav_duration(target, sr)
                        write_json(meta, {'key': key, 'sha256': file_hash(target)})
                        break
                    except Exception as error:
                        if 'out of memory' in str(error).lower() or attempt == 2:
                            raise
                        result.retries += 1
            audio, sr = sf.read(target, dtype='float32')
            if rate is not None and rate != sr:
                raise ValueError('TTS sample rates differ between segments')
            rate = sr
            duration = len(audio)/sr
            segments.append(Segment(text=scene.narration, start=position, end=position+duration, path=target))
            position += duration
            chunks.append(audio)
            write_json(request.run_dir/'tmp/progress.json', {'message': f'tts {language} {i+1}/{len(script.scenes)}'})
        destination = request.run_dir / f'audio/{language}.wav'
        sf.write(destination, np.concatenate(chunks), rate, subtype='PCM_16')
        result.timings[language] = time.monotonic() - language_started
        result.languages[language] = AudioResult(audio=destination, sample_rate=rate, segments=segments)
    return result

if __name__ == '__main__':
    worker_main(generate)
