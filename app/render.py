from __future__ import annotations
from functools import lru_cache
import json
import logging
import math
from pathlib import Path
import re
import shutil
import subprocess
import time
from app.files import valid_video
from app.schema import AudioResult

log = logging.getLogger(__name__)


def ffmpeg(args: list[str], timeout: float = 300, cwd: Path | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(['ffmpeg', '-hide_banner', '-nostdin', '-y', '-filter_threads', '1',
                             '-filter_complex_threads', '1', *args], cwd=cwd,
                            timeout=max(.1, timeout), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'FFmpeg failed: {result.stderr[-12000:]}')
    return result


@lru_cache(maxsize=1)
def encoder() -> str:
    encoders = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], timeout=15,
                              capture_output=True, text=True, check=True).stdout
    selected = 'libx264'
    if 'h264_nvenc' in encoders:
        try:
            ffmpeg(['-f', 'lavfi', '-i', 'color=s=1080x1920:r=30:d=0.1', '-c:v', 'h264_nvenc',
                    '-f', 'null', '-'], timeout=20)
            selected = 'h264_nvenc'
        except (RuntimeError, subprocess.SubprocessError):
            log.warning('NVENC probe failed; falling back to libx264')
    log.info('Video encoder: %s', selected)
    return selected


def codec_args(name: str) -> list[str]:
    return ['-c:v', name, '-preset', 'p4' if name == 'h264_nvenc' else 'veryfast',
            '-b:v', '8M', '-pix_fmt', 'yuv420p', '-threads', '2']


class Renderer:
    def __init__(self, timeout: float = 300):
        self.timeout = timeout
        self.encoder = encoder()

    def render(self, images: list[Path], audio: AudioResult, ass: Path, output: Path,
               music: Path | None = None) -> dict:
        deadline = time.monotonic() + self.timeout
        directory = output.parent / 'tmp' / output.stem
        directory.mkdir(parents=True, exist_ok=True)
        def run(args):
            if time.monotonic() >= deadline:
                raise TimeoutError('Render stage timeout')
            return ffmpeg(args, timeout=deadline-time.monotonic(), cwd=directory)
        if len(images) != len(audio.segments):
            raise ValueError('Image/segment count mismatch')
        durations = [s.end-s.start for s in audio.segments]
        transition = min(.25, min(durations) / 3)
        clips = []
        for i, (image, duration) in enumerate(zip(images, durations)):
            clip = directory/f'scene_{i:02}.mp4'
            length = duration + (transition if i < len(images)-1 else 0)
            frames = math.ceil(length*30)
            pan = "(iw-iw/zoom)*on/"+str(max(frames-1, 1)) if i%2 else '(iw-iw/zoom)/2'
            vf = ("scale=1200:2134:force_original_aspect_ratio=increase:flags=lanczos,crop=1200:2134,"
                  f"zoompan=z='1.02+0.06*on/{max(frames-1,1)}':x='{pan}':y='(ih-ih/zoom)/2':"
                  f'd={frames}:s=1080x1920:fps=30,setsar=1,format=yuv420p')
            run(['-i', str(image.resolve()), '-vf', vf, '-t', str(length), *codec_args(self.encoder), str(clip.resolve())])
            clips.append(clip)
        joined = clips[0]
        elapsed = durations[0]+(transition if len(images)>1 else 0)
        # Two clips per graph bounds FFmpeg graph memory regardless of scene count.
        for i, clip in enumerate(clips[1:], 1):
            target = directory/f'join_{i:02}.mp4'
            offset = sum(durations[:i])
            graph = (f'[0:v]settb=AVTB,setpts=PTS-STARTPTS[a];[1:v]settb=AVTB,setpts=PTS-STARTPTS[b];'
                     f'[a][b]xfade=transition=fade:duration={transition}:offset={offset},format=yuv420p[v]')
            run(['-i', str(joined.resolve()), '-i', str(clip.resolve()), '-filter_complex', graph,
                 '-map', '[v]', '-r', '30', *codec_args(self.encoder), str(target.resolve())])
            if joined not in clips:
                joined.unlink(missing_ok=True)
            joined = target
            elapsed += durations[i]
        total = audio.segments[-1].end
        mixed = directory/'mix.wav'
        base = ['-i', str(audio.audio.resolve())]
        if music:
            if not music.is_file():
                raise FileNotFoundError(music)
            run([*base, '-stream_loop', '-1', '-i', str(music.resolve()), '-filter_complex',
                 f'[1:a]volume=0.10,atrim=duration={total},afade=t=out:st={max(0,total-1.5)}:d={min(1.5,total)}[m];'
                 '[0:a][m]amix=inputs=2:duration=first:normalize=0[a]', '-map', '[a]', '-t', str(total),
                 '-ar', '48000', '-ac', '1', str(mixed.resolve())])
        else:
            run([*base, '-ar', '48000', '-ac', '1', str(mixed.resolve())])
        measurement = run(['-i', str(mixed.resolve()), '-af', 'loudnorm=I=-14:TP=-1:LRA=11:print_format=json',
                           '-f', 'null', '-'])
        def parse_loudness(result):
            matches = re.findall(r'\{[^{}]*"input_i"[^{}]*\}', result.stderr, flags=re.S)
            if not matches:
                raise ValueError('FFmpeg loudnorm measurements missing')
            data = json.loads(matches[-1])
            for key in ('input_i', 'input_tp', 'input_lra', 'input_thresh', 'target_offset'):
                if not math.isfinite(float(data[key])):
                    raise ValueError('Audio is silent or invalid for loudnorm')
            return data
        loud = parse_loudness(measurement)
        af = (f'loudnorm=I=-14:TP=-1:LRA=11:measured_I={loud["input_i"]}:'
              f'measured_TP={loud["input_tp"]}:measured_LRA={loud["input_lra"]}:'
              f'measured_thresh={loud["input_thresh"]}:offset={loud["target_offset"]}:linear=true,'
              'alimiter=limit=0.89125:level=false')
        shutil.copyfile(ass, directory/'captions.ass')
        pending = directory/'final.mp4'
        run(['-i', str(joined.resolve()), '-i', str(mixed.resolve()), '-vf', 'ass=filename=captions.ass',
             '-af', af, '-map', '0:v:0', '-map', '1:a:0', '-t', str(total),
             *codec_args(self.encoder), '-r', '30', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
             '-movflags', '+faststart', str(pending.resolve())])
        def final_loudness(path):
            return parse_loudness(run(['-i', str(path.resolve()), '-af',
                'loudnorm=I=-14:TP=-1:LRA=11:print_format=json', '-f', 'null', '-']))
        final_audio = final_loudness(pending)
        if float(final_audio['input_tp']) > -1:
            corrected = directory/'final_peak_corrected.mp4'
            reduction = -1.25 - float(final_audio['input_tp'])
            run(['-i', str(pending.resolve()), '-c:v', 'copy', '-af', f'volume={reduction}dB',
                 '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(corrected.resolve())])
            pending = corrected
            final_audio = final_loudness(pending)
        if float(final_audio['input_tp']) > -1:
            raise ValueError(f'AAC_TRUE_PEAK_EXCEEDED: {final_audio["input_tp"]} dBTP')
        if not valid_video(pending, total):
            raise ValueError('Final MP4 failed ffprobe validation')
        pending.replace(output)
        return {'encoder': self.encoder, 'duration': total, 'scene_durations': durations, 'loudnorm': loud, 'final_audio_measurements': final_audio}
