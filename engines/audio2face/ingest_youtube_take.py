#!/usr/bin/env python3
"""A YouTube URL -> a take laid out exactly like the original studio capture takes.

    python ingest_youtube_take.py URL                          # plan + disk estimate only
    python ingest_youtube_take.py URL --run                    # download, extract first 60 s
    python ingest_youtube_take.py URL --run --start 12:30 --duration 90
    python ingest_youtube_take.py URL --run --full             # whole video (READ THE ESTIMATE)
    python ingest_youtube_take.py URL --run --no-video         # audio.wav only, the A2F path

Output layout, same shape the Capture Manager ingest writes:

    <out>/take.cparch                       manifest, Version 1
    <out>/Video/Video/frame_%06d.jpg        -q:v 2, yuvj444p, starts at 000000
    <out>/Audio/Audio/audio.wav             pcm_s16le 48 kHz mono
    <out>/source.json                       provenance: url, id, format, clip window

WHAT THIS TAKE DOES NOT HAVE. No Depth/ and no Calibration/ -- a monocular YouTube
stream carries neither, and nothing can synthesise them. So this take feeds the AUDIO
branch only. Every depth-solve stage in prepare_stavatar_take.py (S3 rigid, S5 matte,
S7 verify) reads Depth/Depth/*.exr or calibration.json and will fail on it by design.
Video/ here is reference footage for eyeballing the result, not a solve input: a
podcast feed cuts between cameras, so there is no single static camera and no
head-locked framing like the iPhone takes have.

IDEMPOTENT, in the house style: each stage checks its own output and skips. --force
redoes them. The download is cached in <out>/../_yt_cache so re-clipping never refetches.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
# Deliberately NOT the Capture Manager archive: prepare_take.py globs the studio takes' names
# and every consumer assumes depth exists. A depthless take goes in its own root.
DEFAULT_ROOT = P / 'takes'
# 0.077 measured over a full 308,853-frame 1080p extraction (158 KB/frame, -q:v 2).
# The iPhone takes hit 0.25 (229 KB at 720x1280) because a sensor-native frame carries
# far more high-frequency detail than a re-encoded stream, so a YouTube source estimated
# at the capture rate comes out ~3x pessimistic.
BYTES_PER_PIXEL_JPEG = 0.077


def run(cmd, **kw):
    print('  $', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def probe(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    # Audio-only corpus files have no video stream at all, so this has to degrade rather
    # than raise: everything downstream that needs w/h/fps is gated on with_video.
    v = next((s for s in d['streams'] if s['codec_type'] == 'video'), None)
    num, den = (int(x) for x in v['r_frame_rate'].split('/')) if v else (0, 1)
    return {
        'width': int(v['width']) if v else 0,
        'height': int(v['height']) if v else 0,
        'fps': num / den,
        'duration': float(d['format']['duration']),
    }


def hms(sec):
    return f'{int(sec // 3600):02d}:{int(sec // 60) % 60:02d}:{sec % 60:06.3f}'


def meta(url):
    out = subprocess.run(['yt-dlp', *js_runtime(), '-J', '--no-warnings', url],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    return {k: d.get(k) for k in ('id', 'title', 'uploader', 'duration', 'license',
                                  'webpage_url', 'upload_date')}


def slug(title, vid):
    """Readable dir name that still collides with nothing: sanitised title + video id."""
    words, out = (''.join(c.lower() if c.isalnum() else ' ' for c in title or '')).split(), []
    for w in words:                      # cut on a word boundary: a hard [:50] slice
        if len('_'.join(out + [w])) > 50:  # turns "...huberman" into "...huberma"
            break
        out.append(w)
    return ('_'.join(out) or 'take') + '_' + vid


def js_runtime():
    """yt-dlp deprecated YouTube extraction without a JS runtime: with none enabled it
    falls back to the android_vr client, whose media URLs 403. Only deno is on by
    default, so point it at whatever is actually installed."""
    for name in ('deno', 'node', 'bun'):
        if shutil.which(name):
            return ['--js-runtimes', name]
    print('  WARNING: no JS runtime (deno/node/bun) -- expect HTTP 403 on the media URLs')
    return []


def subtitles(url, out, langs, force):
    """Both caption tracks, under distinct names, because they are not interchangeable.

    AUTO captions carry per-word timing: on a 35 min episode, 1993 cues / 6156 segments
    of which 5159 have tOffsetMs, i.e. a word-level onset grid -- what a viseme aligner
    wants. MANUAL captions are human-punctuated but phrase-level: 787 cues, 787 segments,
    zero tOffsetMs, ~2.6 s apart. yt-dlp prefers manual when both exist and writes them
    to the same filename, so fetching them in one pass silently loses the word timings.

    json3 as well as vtt because only json3 exposes tOffsetMs; vtt flattens to cues.
    Only the requested langs -- YouTube offers 100+ machine translations of the English
    auto-captions and none of them are useful here.
    """
    sub = out / 'Subtitles'
    if sub.is_dir() and any(sub.iterdir()) and not force:
        print('  subtitles exist, skip')
        return
    sub.mkdir(parents=True, exist_ok=True)
    for flag, tag in (('--write-auto-subs', 'auto'), ('--write-subs', 'manual')):
        for fmt in ('json3', 'vtt'):
            run(['yt-dlp', *js_runtime(), '--skip-download', flag,
                 '--sub-langs', langs, '--sub-format', fmt,
                 '-o', f'%(id)s.{tag}.%(ext)s', url], cwd=str(sub))
    got = sorted(f.name for f in sub.iterdir())
    print('  ' + (', '.join(got) if got else 'none available'))


def download(url, cache, fmt, force, merge='mp4'):
    cache.mkdir(parents=True, exist_ok=True)
    hits = sorted(cache.glob('source.*'))
    if hits and not force:
        print(f'  cached {hits[0].name}')
        return hits[0]
    for h in hits:
        h.unlink()
    # merge=None for audio-only: nothing to mux, and m4a is not a legal merge container
    # (yt-dlp accepts avi/flv/mkv/mov/mp4/webm there and exits 2 on anything else).
    run(['yt-dlp', *js_runtime(), '-f', fmt,
         *(['--merge-output-format', merge] if merge else []),
         '-o', str(cache / 'source.%(ext)s'), '--no-playlist', url])
    return sorted(cache.glob('source.*'))[0]


def extract_audio(src, dst, start, dur, force):
    if dst.exists() and not force:
        print('  audio.wav exists, skip')
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-v', 'warning', '-stats', '-y']
    cmd += ['-ss', hms(start)] if start else []
    cmd += ['-i', str(src)]
    cmd += ['-t', hms(dur)] if dur else []
    # 48 kHz mono s16 is what the iPhone takes carry, and what A2F wants.
    cmd += ['-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '1', str(dst)]
    run(cmd)


def extract_frames(src, dst, start, dur, fps, scale, force):
    n_have = len(list(dst.glob('frame_*.jpg'))) if dst.is_dir() else 0
    if n_have and not force:
        print(f'  {n_have} frames exist, skip')
        return
    if force and dst.is_dir():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    vf = []
    if fps:
        vf.append(f'fps={fps}')
    if scale:
        vf.append(f'scale={scale.replace("x", ":")}:flags=lanczos')
    cmd = ['ffmpeg', '-v', 'warning', '-stats', '-y']
    cmd += ['-ss', hms(start)] if start else []
    cmd += ['-i', str(src)]
    cmd += ['-t', hms(dur)] if dur else []
    cmd += ['-vf', ','.join(vf)] if vf else []
    # -q:v 2 + yuvj444p matches the Capture Manager frames byte-for-byte in format,
    # so anything downstream that opens frame_%06d.jpg sees the same thing.
    cmd += ['-q:v', '2', '-pix_fmt', 'yuvj444p', '-start_number', '0',
            str(dst / 'frame_%06d.jpg')]
    run(cmd)


def write_manifest(out, info, fps, dur, slate, take_no, with_video, device):
    m = {'Version': 1, 'DeviceModel': device, 'Slate': slate, 'TakeNumber': take_no}
    if with_video:
        m['Video'] = [{
            'Name': 'Video',
            'Path': 'Video/Video',
            'FrameRate': round(fps, 3),
            'FrameWidth': info['width'],
            'FrameHeight': info['height'],
            'TimecodeStart': '00:00:00:00',
        }]
    m['Audio'] = [{
        'Name': 'Audio',
        'Path': 'Audio/Audio/audio.wav',
        'TimecodeStart': '00:00:00:00',
        'TimecodeRate': 30,
    }]
    # Depth and Calibration are absent, not empty: an empty list would read as "a depth
    # stream with zero frames" to anything walking the manifest.
    (out / 'take.cparch').write_text(json.dumps(m, indent='\t'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url', help='YouTube URL, or a local media file with --local')
    ap.add_argument('--local', action='store_true',
                    help='treat the positional arg as a file on disk, skip yt-dlp entirely')
    ap.add_argument('--out', type=Path, help='take dir (default takes/yt_<video id>)')
    ap.add_argument('--root', type=Path,
                    help='batch root; take dir becomes <root>/<title slug>_<id> and the '
                         'download cache is shared at <root>/_yt_cache')
    ap.add_argument('--start', default='0', help='clip start, s or HH:MM:SS')
    ap.add_argument('--duration', default='60', help='clip length in s (ignored with --full)')
    ap.add_argument('--full', action='store_true', help='whole video')
    ap.add_argument('--fps', type=float, default=0, help='resample frames (0 = source fps)')
    ap.add_argument('--scale', help='e.g. 1280x720; default keeps source size')
    ap.add_argument('--no-video', action='store_true', help='audio.wav only')
    ap.add_argument('--a2f', action='store_true',
                    help='also write Audio/Audio/audio_16k.wav, the 16 kHz mono s16 that '
                         'a2f/run_a2f.py asserts on (16000-sample windows)')
    ap.add_argument('--subs', nargs='?', const='en', default=None, metavar='LANGS',
                    help='also fetch subtitles into <take>/Subtitles (default en)')
    ap.add_argument('--format', default='bv*[height<=1080][vcodec^=avc1]+ba[acodec^=mp4a]'
                                        '/bv*[height<=1080]+ba/b[height<=1080]')
    # Above 1080p YouTube ships VP9/AV1 only -- there is no avc1 ladder up there -- and
    # mp4 is a poor host for those, so --best also switches the container to mkv.
    ap.add_argument('--best', action='store_true',
                    help='highest available resolution, any codec, muxed into mkv')
    # Facts-corpus mode. At 733 h a wav per take is 253 GB and buys nothing: ASR reads
    # the compressed stream directly, and only the voice/avatar subject ever needs a
    # 16 kHz wav. Compressed audio + captions is ~20 GB for the same 733 h.
    ap.add_argument('--corpus', action='store_true',
                    help='bestaudio only, no wav, no frames, subtitles on: the cheap '
                         'shape for a retrieval corpus')
    ap.add_argument('--slate', default='ingest')
    ap.add_argument('--take-number', type=int, default=0)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='redo extraction stages (keeps the cached download)')
    ap.add_argument('--refetch', action='store_true',
                    help='also discard the cached mux and download again')
    a = ap.parse_args()

    for tool in ('yt-dlp', 'ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            sys.exit(f'{tool} not on PATH')

    def tosec(s):
        parts = [float(x) for x in str(s).split(':')]
        return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))

    print('== source')
    if a.corpus:
        a.no_video, a.full = True, True
        a.subs = a.subs or 'en'
    if a.local:
        local = Path(a.url).expanduser().resolve()
        if not local.is_file():
            sys.exit(f'no such file: {local}')
        v0 = probe(local)
        info_yt = {'id': local.stem, 'title': local.name, 'uploader': None,
                   'duration': v0['duration'], 'license': None,
                   'webpage_url': str(local), 'upload_date': None}
        print(f'  file      {local}')
        print(f'  media     {v0["width"]}x{v0["height"]} @ {v0["fps"]:g} fps  '
              f'{v0["duration"]:.1f}s')
    else:
        local = None
        info_yt = meta(a.url)
    for k in ('id', 'title', 'uploader', 'duration', 'license'):
        print(f'  {k:9s} {info_yt[k]}')

    if a.out:
        out = a.out
    elif a.root:
        out = a.root / slug(info_yt['title'], info_yt['id'])
    else:
        out = DEFAULT_ROOT / (f'local_{info_yt["id"]}' if a.local
                              else f'yt_{info_yt["id"]}')
    cache = out.parent / '_yt_cache' / info_yt['id']
    start = tosec(a.start)
    dur = None if a.full else tosec(a.duration)
    span = (info_yt['duration'] or 0) - start if dur is None else dur

    print('== plan')
    print(f'  out       {out}')
    print(f'  cache     {cache}')
    print(f'  clip      {hms(start)} + {span:.1f}s' + ('  (FULL)' if a.full else ''))
    src = [local] if local else sorted(cache.glob('source.*'))
    if src:
        v = probe(src[0])
        fps = a.fps or v['fps']
        w, h = (int(x) for x in a.scale.split('x')) if a.scale else (v['width'], v['height'])
        nf = int(span * fps)
        gb = nf * w * h * BYTES_PER_PIXEL_JPEG / 1e9
        print(f'  video     {w}x{h} @ {fps:g} fps -> {nf} frames, ~{gb:.1f} GB')
    else:
        print('  video     unknown until downloaded (~1.2 GB for a 1080p mux)')
        print(f'  estimate  {int(span * 30)} frames at 30 fps, '
              f'~{span * 30 * 1920 * 1080 * BYTES_PER_PIXEL_JPEG / 1e9:.0f} GB at 1080p')
    print(f'  audio     ~{span * 48000 * 2 / 1e9:.2f} GB wav')
    free = shutil.disk_usage(out.parent.parent if out.parent.exists() else P).free
    print(f'  free      {free / 1e9:.0f} GB')
    if not a.run:
        print('\n(dry run -- add --run to execute)')
        return

    out.mkdir(parents=True, exist_ok=True)
    if local:
        src = local
    else:
        print('== download')
        fmt = ('ba[abr<=70]/ba/b' if a.corpus
               else 'bv*+ba/b' if a.best else a.format)
        merge = None if a.corpus else 'mkv' if a.best else 'mp4'
        src = download(a.url, cache, fmt, a.refetch, merge)
    v = probe(src)
    shape = (f'{v["width"]}x{v["height"]} @ {v["fps"]:g} fps' if v['width']
             else 'audio only')
    print(f'  {src.name}  {shape}  {v["duration"]:.1f}s')

    if a.corpus:
        print('== audio  (corpus mode: compressed stream kept, no wav)')
    else:
        print('== audio')
        extract_audio(src, out / 'Audio/Audio/audio.wav', start, dur, a.force)

    if a.a2f:
        w16 = out / 'Audio/Audio/audio_16k.wav'
        if not w16.exists() or a.force:
            run(['ffmpeg', '-v', 'warning', '-y', '-i', str(out / 'Audio/Audio/audio.wav'),
                 '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', str(w16)])

    if a.subs and not a.local:
        print('== subtitles')
        subtitles(a.url, out, a.subs, a.force)

    fps = a.fps or v['fps']
    if not a.no_video:
        print('== frames')
        extract_frames(src, out / 'Video/Video', start, dur, a.fps or None, a.scale, a.force)

    w, h = (int(x) for x in a.scale.split('x')) if a.scale else (v['width'], v['height'])
    write_manifest(out, {'width': w, 'height': h}, fps, dur, a.slate, a.take_number,
                   not a.no_video, 'LocalFile' if local else 'YouTube')
    (out / 'source.json').write_text(json.dumps({
        **info_yt, 'format_selector': a.format, 'clip_start_s': start,
        'clip_duration_s': dur, 'extract_fps': fps, 'scale': a.scale,
        'source_file': str(src),
    }, indent=2))

    # The mux lives in the shared cache so re-clipping never refetches, but a take dir
    # with no playable video in it reads as "the video failed". Link it in.
    link = out / f'source{src.suffix}'
    if not link.exists():
        link.symlink_to(src.resolve())

    n = len(list((out / 'Video/Video').glob('*.jpg'))) if not a.no_video else 0
    wav = out / 'Audio/Audio/audio.wav'
    sz = (wav if wav.exists() else src).stat().st_size / 1e6
    print(f'== done\n  {out}\n  frames {n}  audio {sz:.0f} MB')


if __name__ == '__main__':
    main()
