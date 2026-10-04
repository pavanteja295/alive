#!/usr/bin/env python3
"""Corpus video -> a VHAP `data/monocular/<seq>` directory. Replaces preprocess_video.py's
frame extraction, which cannot read these files at all:

  * `if input.suffix in ['.mov', '.mp4']` -- the corpus is `.mkv`, so it falls through to
    the is_dir() branch and raises ValueError before doing anything.
  * `image_dir = input.parent / input.stem / 'images'` -- puts frames next to the source
    video, not under data/monocular/<seq>/ where the tracker looks for them.
  * `int(probe['streams'][0]['nb_frames'])` -- ffprobe reports N/A for these mkv files, so
    this raises TypeError even once the suffix check passes.

Matting is NOT reimplemented: it calls VHAP's own robust_video_matting so the alpha maps
are identical to what the tracker was tuned against.

FRAME RATE
    The corpus is heterogeneous -- 23.976, 30.002 and 60 fps across takes. Extracting
    everything at a fixed 30 would DUPLICATE frames on the 23.976 take, which fabricates
    zero-motion frames and hands the tracker fake temporal smoothness. So the target is
    min(native, --max-fps): 23.976 stays native, 60 halves to 30. The rate actually used is
    written to meta.json, because downstream anything time-indexed needs it.
"""
import argparse, json, os, pathlib, subprocess, sys, time

# --- creator profile (opt-in) --------------------------------------------------
# Thresholds below are healthygamer's, read off that creator's histograms. Pass
# --profile <creator> to take them from profiles/<creator>.py instead. An
# explicitly passed flag always wins over the profile.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "recipes/face-clips/tools"))
try:
    from _profile import load as _load_profile, resolve as _resolve_profile, add_arg as _profile_arg
except Exception:                                   # flat use without the recipe folder
    _load_profile = lambda n: None
    _resolve_profile = lambda a, p, m: a
    _profile_arg = lambda ap: ap



def probe(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,avg_frame_rate,width,height,codec_name",
         "-show_entries", "format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s, f = d["streams"][0], d["format"]

    def rate(x):
        if not x or x in ("0/0", "N/A"):
            return 0.0
        n, _, den = x.partition("/")
        return float(n) / float(den or 1)

    fps = rate(s.get("r_frame_rate")) or rate(s.get("avg_frame_rate"))
    if not fps:
        raise SystemExit(f"cannot determine fps for {video}")
    return dict(fps=fps, w=int(s["width"]), h=int(s["height"]),
                codec=s.get("codec_name", "?"), duration=float(f["duration"]))


def main():
    ap = argparse.ArgumentParser()
    _profile_arg(ap)
    ap.add_argument("--video", required=True)
    ap.add_argument("--seq-dir", required=True, help="data/monocular/<seq>")
    ap.add_argument("--max-fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1280,
                    help="output width; height follows the source aspect, rounded even")
    ap.add_argument("--skip-matting", action="store_true")
    a = ap.parse_args()
    a = _resolve_profile(a, _load_profile(a.profile), {'max_fps': 'MAX_FPS', 'width': 'WIDTH'})

    seq = pathlib.Path(a.seq_dir).resolve()
    images, alphas = seq / "images", seq / "alpha_maps"
    info = probe(a.video)
    fps = min(info["fps"], a.max_fps)
    w = min(a.width, info["w"])
    h = int(round(info["h"] * w / info["w"])) // 2 * 2
    print(f"source: {info['w']}x{info['h']} {info['codec']} {info['fps']:.3f} fps "
          f"{info['duration']:.0f}s -> {w}x{h} @ {fps:.3f} fps", flush=True)

    # --- frames ---------------------------------------------------------------------
    # Extract into a .part dir and rename, so an interrupted extraction is never mistaken
    # for a complete one by the `if not images.exists()` check on the next attempt.
    t0 = time.time()
    if images.exists():
        print(f"images/ exists ({len(list(images.iterdir()))} files), skipping extraction")
    else:
        part = seq / "images.part"
        subprocess.run(["rm", "-rf", str(part)], check=False)
        part.mkdir(parents=True)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(a.video),
             "-vf", f"fps={fps},scale={w}:{h}", "-qscale:v", "1",
             "-start_number", "0", str(part / "%06d.jpg")], check=True)
        os.replace(part, images)
    n = len(list(images.iterdir()))
    t_extract = time.time() - t0
    print(f"[timing] extract: {t_extract:.1f}s for {n} frames "
          f"({t_extract/max(n,1)*1000:.1f} ms/frame)", flush=True)

    # --- matting --------------------------------------------------------------------
    t0 = time.time()
    if a.skip_matting:
        print("matting skipped")
        t_matte = 0.0
    elif alphas.exists() and len(list(alphas.iterdir())) == n:
        print("alpha_maps/ complete, skipping matting")
        t_matte = 0.0
    else:
        vhap = os.environ.get("VHAP_ROOT",
                              str(pathlib.Path(__file__).resolve().parents[1] / "vhap"))
        sys.path.insert(0, vhap)
        os.chdir(vhap)
        from vhap.preprocess_video import robust_video_matting
        robust_video_matting(images)
        t_matte = time.time() - t0
        print(f"[timing] matting: {t_matte:.1f}s for {n} frames "
              f"({t_matte/max(n,1)*1000:.1f} ms/frame)", flush=True)

    meta = dict(video=str(a.video), frames=n, fps=fps, width=w, height=h,
                source_fps=info["fps"], source_w=info["w"], source_h=info["h"],
                codec=info["codec"], duration=info["duration"],
                extract_s=round(t_extract, 1), matting_s=round(t_matte, 1))
    (seq / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


main()
