import time
from app.config import Settings
from app.files import read_json
from app.schema import AlignmentOutput, AlignmentResult, TTSOutput, Word, WorkerInput
from app.worker_common import worker_main


def generate(request: WorkerInput) -> AlignmentOutput:
    import whisperx.alignment as alignment_module
    from app.alignment import scene_segmentation
    from whisperx.alignment import load_align_model, align
    from whisperx.audio import load_audio
    settings = Settings.load()
    tts = TTSOutput.model_validate(read_json(request.run_dir/'tmp/tts_output.json'))
    result = AlignmentOutput(languages={})
    for language in request.languages:
        language_started = time.monotonic()
        audio = tts.languages[language]
        try:
            model, metadata = load_align_model(language_code=language, device='cuda',
                                               model_name=str(settings.model_path(f'whisperx_align_{language}')))
            transcript = [{'start': s.start, 'end': s.end, 'text': s.text} for s in audio.segments]
            with scene_segmentation(alignment_module):
                aligned = align(transcript, model, metadata, load_audio(str(audio.audio)), 'cuda',
                                return_char_alignments=False)
            words = [Word(text=w['word'], start=w['start'], end=w['end']) for w in aligned['word_segments']
                     if 'start' in w and 'end' in w and w['end'] > w['start']]
            total = len(aligned['word_segments'])
            if not total or len(words) != total:
                raise ValueError('Incomplete word alignment; using real TTS segment boundaries')
            if any(w.end > audio.segments[-1].end + .05 for w in words):
                raise ValueError('Alignment exceeds audio duration')
            result.languages[language] = AlignmentResult(words=words)
            del model
        except Exception as error:
            if 'out of memory' in str(error).lower():
                raise RuntimeError('CUDA OOM during alignment') from error
            result.languages[language] = AlignmentResult(fallback='segment_level', reason=str(error))
        result.timings[language] = time.monotonic() - language_started
    return result

if __name__ == '__main__':
    worker_main(generate)
