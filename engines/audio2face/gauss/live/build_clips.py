#!/usr/bin/env python3
"""The clip shelf the demo page offers, cut once so clicking one costs nothing.

Three kinds, and the kind is the whole point of the shelf -- a demo that only ever
plays training audio proves nothing:

  held-out      one of the 26 TEST chunks. Neither model has seen it. The source
                video beside the render is the real answer, frame for frame.
  unseen take   the same person, a recording nothing in this project has touched.
  novel voice   a different person entirely. The audio model was trained on eight
                other people plus a correction fitted to this one, so this is the
                case where it has to generalise across a voice it has never heard.

    python pipeline/gauss/live/build_clips.py --profile <name>
"""
import json
import pathlib
import random
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent.parent                   # the face engine
TAKES = PIPE / "takes"
OUT = HERE / "clips"
MAX_S = 9.0

sys.path.insert(0, str(PIPE / "gauss/recipes/mesh-to-render/tools"))
from _profile import load                                             # noqa: E402


def ffprobe_duration(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(p)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def cut_wav(mp4, dst):
    """The 16 kHz mono the audio model asserts on, taken FROM THE CLIP, not from the
    source again.

    Cutting it from the source with a second seek looked equivalent and is not: `-ss`
    lands on a keyframe for the picture and on a sample for the sound, so the mp4 the
    page plays and the wav the model hears could begin tens of milliseconds apart. That
    difference then reads as the model's lip-sync being off, which is the one thing this
    page exists to let someone judge. Derived from the clip, the two cannot disagree.
    """
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vn",
                    "-ac", "1", "-ar", "16000", str(dst)], check=True)


def cut(src, t0, dur, dst, crop_square=True):
    """A clip with its own audio, re-encoded so the browser can seek it.

    The source frames are 16:9 and the render is square, so the source is cropped to
    a centred square too -- otherwise the two panels are different shapes and the eye
    reads that difference before it reads the face.
    """
    # the commas inside min() are escaped: ffmpeg splits a filtergraph on commas and
    # an unescaped one turns the second half of the expression into a filter name
    vf = (r"crop='min(iw\,ih)':'min(iw\,ih)',scale=512:512" if crop_square
          else "scale=512:-2")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.3f}",
                    "-i", str(src), "-t", f"{dur:.3f}",
                    "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "2",
                    "-movflags", "+faststart", str(dst)], check=True)


def loud_enough(src, t0, dur):
    """Refuse a window that is silence or music. Speech sits well above -40 LUFS."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{t0:.3f}",
                        "-i", str(src), "-t", f"{dur:.3f}", "-af", "loudnorm=print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    try:
        j = json.loads(r.stderr[r.stderr.rindex("{"):r.stderr.rindex("}") + 1])
        return float(j["input_i"]) > -35.0
    except Exception:
        return True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True,
                    help="recipes/mesh-to-render/profiles/<name>.py: whose clips")
    p = load(ap.parse_args().profile)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*"):
        f.unlink()
    rng = random.Random(7)
    shelf = []

    sources = {t.parent.name: t for t in TAKES.glob("*/*/source.*")
               if t.suffix in (".mkv", ".mp4")}

    # ---- held-out test chunks ------------------------------------------------
    split = json.load(open(p.SPLIT))
    cand = []
    for c in split["test"]:
        z = np.load(PIPE / f"rigfit/cache/targets_{p.SUBJECT}/{c}.npz")
        t = z["t"]
        if t[-1] - t[0] >= 5.0:
            cand.append((c, float(t[0]), min(float(t[-1] - t[0]), MAX_S)))
    for c, t0, dur in rng.sample(cand, min(3, len(cand))):
        rec = next(v for v in p.RECORDINGS if c.startswith(v))
        src = sources.get(rec)
        if src is None:
            print(f"  no source video for {rec}, skipping {c}")
            continue
        name = f"heldout_{c[-4:]}"
        cut(src, t0, dur, OUT / f"{name}.mp4")
        cut_wav(OUT / f"{name}.mp4", OUT / f"{name}.wav")
        shelf.append(dict(id=name, kind="held-out test clip", seconds=round(dur, 2),
                          note="neither model has seen this clip; the panel on the left "
                               "is the real answer", chunk=c))
        print(f"  {name}  {dur:.1f}s  from {rec[:40]} @ {t0:.0f}s")

    # ---- the same person, recordings nothing has touched ----------------------
    trained_on = set(p.RECORDINGS)
    # this creator's own folder of takes is the one holding the recordings it trained on;
    # every other folder under takes/ is somebody else
    own = {sources[k].parent.parent for k in sources if k in trained_on}
    unseen = sorted(k for k in sources if k not in trained_on and sources[k].parent.parent in own)
    for rec in rng.sample(unseen, min(3, len(unseen))):
        src = sources[rec]
        d = ffprobe_duration(src)
        for _ in range(12):
            t0 = rng.uniform(d * 0.2, d * 0.8)
            if loud_enough(src, t0, MAX_S):
                break
        name = f"unseen_{rec[:22]}"
        cut(src, t0, MAX_S, OUT / f"{name}.mp4")
        cut_wav(OUT / f"{name}.mp4", OUT / f"{name}.wav")
        shelf.append(dict(id=name, kind="unseen recording, same person",
                          seconds=MAX_S,
                          note="his voice, a session no stage of this project has read",
                          chunk=None))
        print(f"  {name}  {MAX_S:.1f}s  @ {t0:.0f}s of {d:.0f}s")

    # ---- a different person entirely -----------------------------------------
    other = sorted(k for k in sources if sources[k].parent.parent not in own)
    for rec in rng.sample(other, min(2, len(other))):
        src = sources[rec]
        d = ffprobe_duration(src)
        for _ in range(12):
            t0 = rng.uniform(d * 0.2, d * 0.8)
            if loud_enough(src, t0, MAX_S):
                break
        name = f"novel_{rec[:22]}"
        cut(src, t0, MAX_S, OUT / f"{name}.mp4")
        cut_wav(OUT / f"{name}.mp4", OUT / f"{name}.wav")
        shelf.append(dict(id=name, kind="novel voice", seconds=MAX_S,
                          note="a different person; only the shipped encoder has any "
                               "claim on this voice", chunk=None))
        print(f"  {name}  {MAX_S:.1f}s  @ {t0:.0f}s of {d:.0f}s")

    (OUT / "shelf.json").write_text(json.dumps(shelf, indent=1))
    print(f"\n{len(shelf)} clips in {OUT}")


if __name__ == "__main__":
    main()
