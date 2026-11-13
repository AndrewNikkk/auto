import sys
from app.config import Settings
from app.files import read_json, valid_video, valid_png
from app.schema import TTSOutput, AlignmentOutput
from app.pipeline import valid_audio


def main():
    s=Settings.load()
    run=s.output_dir/sys.argv[1]
    report=read_json(run/'generation_report.json')
    assert report['status']=='complete',report
    assert report['gpu']['monitor_available'],'NVML required for smoke verification'
    assert len(list((run/'images').glob('*.png')))==8
    assert all(valid_png(p,s.width,s.height,s.pixel_std_min) for p in (run/'images').glob('*.png'))
    audio=TTSOutput.model_validate(read_json(run/'tmp/tts_output.json'))
    assert valid_audio(audio,['ru','en'],8)
    alignment=AlignmentOutput.model_validate(read_json(run/'tmp/whisperx_output.json'))
    assert all(a.words and not a.fallback for a in alignment.languages.values()),'Forced alignment fell back; inspect report'
    for lang in ['ru','en']:
        assert valid_video(run/f'video_{lang}.mp4',audio.languages[lang].segments[-1].end)
    assert all(v['memory_released'] for v in report['stages'].values()),'VRAM release failed'
    print('VERIFIED: GPU end-to-end smoke test')
    print(report['timings'])
    print('GPU:',report['gpu']['name'])
    print('Report:',run/'generation_report.json')

if __name__=='__main__':
    main()
