from app.schema import AudioResult, AlignmentResult, Word
from app.subtitles import write_subtitles, timestamp


def test_fallback_real_bounds(tmp_path):
    audio=AudioResult(audio=tmp_path/'ru.wav',sample_rate=24000,segments=[{'text':'Привет, мир!','start':0,'end':1.75,'path':tmp_path/'0.wav'}])
    srt,ass=write_subtitles(tmp_path,'ru',audio,AlignmentResult(fallback='segment_level'))
    assert '00:00:01,750' in srt.read_text() and 'Привет' in ass.read_text()
    assert r'\c&H0000D7FF&' not in ass.read_text()
    assert ',90,90,420,1' in ass.read_text()


def test_word_highlight(tmp_path):
    audio=AudioResult(audio=tmp_path/'en.wav',sample_rate=24000,segments=[{'text':'Hello world','start':0,'end':2,'path':tmp_path/'0.wav'}])
    alignment=AlignmentResult(words=[Word(text='Hello',start=.2,end=.7),Word(text='world',start=.8,end=1.3)])
    _,ass=write_subtitles(tmp_path,'en',audio,alignment)
    text=ass.read_text()
    assert '0:00:00.20,0:00:00.70' in text
    assert r'{\c&H0000D7FF&}Hello' in text
    assert len([line for line in text.splitlines() if line.startswith('Dialogue')])==3


def test_timestamp_carry():
    assert timestamp(59.9999)=='00:01:00,000'
    assert timestamp(59.9999,True)=='0:01:00.00'


def test_existing_scene_segmentation_no_pretrained_tokenizer():
    from types import SimpleNamespace
    from app.alignment import scene_segmentation
    original=lambda path: 'original'
    module=SimpleNamespace(nltk_load=original)
    with scene_segmentation(module):
        assert list(module.nltk_load('unused').span_tokenize('Привет. Hello!'))==[(0,14)]
    assert module.nltk_load is original
