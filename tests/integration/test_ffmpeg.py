import wave
import shutil
import pytest
from PIL import Image
from app.files import valid_video, probe
from app.render import Renderer, ffmpeg
from app.schema import AudioResult,AlignmentResult
from app.subtitles import write_subtitles

pytestmark=[pytest.mark.integration,pytest.mark.skipif(not shutil.which('ffmpeg'),reason='FFmpeg unavailable')]


def fixtures(tmp_path,language):
    images=[];segments=[];time=0
    lengths=[.6,.8] if language=='ru' else [.8,1.0]
    for i,length in enumerate(lengths):
        image=tmp_path/f'{language}_{i}.png'
        ffmpeg(['-f','lavfi','-i','testsrc=s=832x1216:d=0.1','-frames:v','1',str(image)])
        images.append(image)
        wav=tmp_path/f'{language}_{i}.wav'
        ffmpeg(['-f','lavfi','-i',f'sine=frequency={400+i*100}:sample_rate=24000:duration={length}',
                '-ac','1','-c:a','pcm_s16le',str(wav)])
        segments.append({'text':'Привет, мир!' if language=='ru' else 'Hello world!', 'start':time,'end':time+length,'path':wav})
        time+=length
    audio=tmp_path/f'{language}.wav'
    with wave.open(str(audio),'wb') as out:
        out.setnchannels(1);out.setsampwidth(2);out.setframerate(24000)
        for segment in segments:
            with wave.open(str(segment['path'])) as inp:out.writeframes(inp.readframes(inp.getnframes()))
    return images,AudioResult(audio=audio,sample_rate=24000,segments=segments)

@pytest.mark.parametrize('languages',[['ru'],['en'],['ru','en']])
def test_real_mp4_language_modes(tmp_path,languages):
    durations={}
    for language in languages:
        images,audio=fixtures(tmp_path,language)
        _,ass=write_subtitles(tmp_path/'subtitles',language,audio,AlignmentResult(fallback='segment_level'))
        target=tmp_path/f'video_{language}.mp4'
        renderer=Renderer(timeout=120)
        music = None
        if languages == ['ru']:
            music = tmp_path/'short_music.wav'
            ffmpeg(['-f','lavfi','-i','sine=frequency=120:duration=0.2',str(music)])
        result=renderer.render(images,audio,ass,target,music)
        assert valid_video(target,audio.segments[-1].end)
        assert float(result['final_audio_measurements']['input_tp']) <= -1
        assert abs(float(result['final_audio_measurements']['input_i']) + 14) < 1
        assert result['scene_durations']==pytest.approx([s.end-s.start for s in audio.segments])
        durations[language]=float(probe(target)['format']['duration'])
    assert set(p.name for p in tmp_path.glob('video_*.mp4'))=={f'video_{lang}.mp4' for lang in languages}
    if len(languages)==2:assert abs(durations['ru']-durations['en'])>.25


def test_rendered_safe_zones(tmp_path):
    audio=AudioResult(audio=tmp_path/'unused.wav',sample_rate=24000,segments=[{'text':'Кириллица и English subtitle','start':0,'end':1,'path':tmp_path/'0.wav'}])
    _,ass=write_subtitles(tmp_path,'ru',audio,AlignmentResult(fallback='segment_level'))
    png=tmp_path/'caption.png'
    ffmpeg(['-f','lavfi','-i','color=black:s=1080x1920:r=30:d=1','-vf',f'ass=filename={ass}',
            '-ss','0.2','-frames:v','1',str(png)])
    with Image.open(png) as image:
        gray=image.convert('L').point(lambda p:255 if p>60 else 0)
        box=gray.getbbox()
    assert box and box[1]>=150 and box[3]<=1520 and box[0]>=80 and box[2]<=1000,box

@pytest.mark.parametrize('text',['Очень длинный субтитр. '*25,'W'*550])
def test_long_subtitle_safe_zones(tmp_path,text):
    audio=AudioResult(audio=tmp_path/'unused.wav',sample_rate=24000,segments=[{'text':text,'start':0,'end':1,'path':tmp_path/'0.wav'}])
    _,ass=write_subtitles(tmp_path,'ru',audio,AlignmentResult(fallback='segment_level'))
    png=tmp_path/'long.png'
    ffmpeg(['-f','lavfi','-i','color=black:s=1080x1920:r=30:d=1','-vf',f'ass=filename={ass}',
            '-ss','0.2','-frames:v','1',str(png)])
    with Image.open(png) as image:
        box=image.convert('L').point(lambda p:255 if p>60 else 0).getbbox()
    assert box and box[1]>=150 and box[3]<=1520 and box[0]>=80 and box[2]<=1000,box
