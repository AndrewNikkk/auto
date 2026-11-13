from pathlib import Path
import re
from PIL import ImageFont
from app.schema import AlignmentResult, AudioResult

ASS_HEADER = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,54,&H00FFFFFF,&H0000D7FF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,90,90,420,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''


def timestamp(seconds: float, ass=False):
    ticks = round(seconds * (100 if ass else 1000))
    scale = 100 if ass else 1000
    whole, fraction = divmod(ticks, scale)
    hours, remainder = divmod(whole, 3600)
    minutes, second = divmod(remainder, 60)
    return f'{hours}:{minutes:02}:{second:02}.{fraction:02}' if ass else f'{hours:02}:{minutes:02}:{second:02},{fraction:03}'


def clean(text: str) -> str:
    return re.sub(r'[{}\\\r\n]', ' ', text).strip()


def text_width(text: str, size: int = 54) -> float:
    font = ImageFont.truetype('DejaVuSans-Bold.ttf', size)
    return font.getlength(text)


def wrapped(text: str) -> str:
    lines = []
    current = ''
    for word in clean(text).split():
        if current and text_width(current + ' ' + word) > 840:
            lines.append(current)
            current = ''
        for character in ((' ' if current else '') + word):
            if current and text_width(current + character) > 840:
                lines.append(current)
                current = ''
            current += character
    if current:
        lines.append(current)
    return r'\N'.join(lines)


def write_subtitles(directory: Path, language: str, audio: AudioResult, alignment: AlignmentResult) -> tuple[Path, Path]:
    srt = []
    events = []
    if alignment.fallback or not alignment.words:
        items = [(s.start, s.end, s.text, None) for s in audio.segments]
    else:
        # Small groups prevent captions from crossing the conservative safe area.
        groups = []
        group = []
        for word in alignment.words:
            if group and (len(' '.join(w.text for w in group+[word])) > 26 or len(group) >= 5):
                groups.append(group)
                group = []
            group.append(word)
        if group:
            groups.append(group)
        items = [(g[0].start, g[-1].end, ' '.join(w.text for w in g), g) for g in groups]
    for index, (start, end, text, words) in enumerate(items, 1):
        srt.append(f'{index}\n{timestamp(start)} --> {timestamp(end)}\n{text}\n')
        if words:
            size = min(54, max(20, int(54 * 840 / max(840, text_width(text)))))
            base = rf'{{\fs{size}}}' + clean(text)
            events.append(f'Dialogue: 0,{timestamp(start, True)},{timestamp(end, True)},Default,,0,0,0,,{base}')
            for current in words:
                caption = ' '.join((r'{\c&H0000D7FF&}' + clean(w.text) + r'{\c&H00FFFFFF&}')
                                   if w is current else clean(w.text) for w in words)
                caption = rf'{{\fs{size}}}' + caption
                events.append(f'Dialogue: 1,{timestamp(current.start, True)},{timestamp(current.end, True)},Default,,0,0,0,,{caption}')
        else:
            # TTS segments have measured boundaries. Long segments use smaller type, never fabricated word times.
            lines = wrapped(text).split(r'\N')
            font = max(20, min(54, int(1000 / max(len(lines)*1.3, 1))))
            caption = rf'{{\fs{font}}}' + r'\N'.join(lines)
            events.append(f'Dialogue: 0,{timestamp(start, True)},{timestamp(end, True)},Default,,0,0,0,,{caption}')
    directory.mkdir(parents=True, exist_ok=True)
    srt_path, ass_path = directory/f'{language}.srt', directory/f'{language}.ass'
    srt_path.write_text('\n'.join(srt), encoding='utf-8')
    ass_path.write_text(ASS_HEADER + '\n'.join(events) + '\n', encoding='utf-8')
    return srt_path, ass_path
