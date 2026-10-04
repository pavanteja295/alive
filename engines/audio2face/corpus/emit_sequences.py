#!/usr/bin/env python3
"""Chunks -> VHAP sequence directories, ready to track.

    python emit_sequences.py --seq <seq>                 # plan only
    python emit_sequences.py --seq <seq> --write         # crop frames
    python emit_sequences.py --seq <seq> --write --matte # + alpha maps

Writes one directory per chunk under VHAP/data/monocular/, named `<seq>__cNNN`, each holding
`images/` and `alpha_maps/` renumbered from 0. That is all VHAP needs.

LANDMARKS ARE NOT WRITTEN HERE, DELIBERATELY
    tracker.py:1271-1283 runs STAR itself when `landmark2d/STAR` is absent. Letting it do so
    ON THE CROPPED FRAMES is what fixes the wrong-face problem, and it fixes it by
    construction rather than by filtering: the crop is 3.2x the face, so on take 1 another
    face falls inside it in 0.00% of kept frames and on take 2 in 0.15%. There is nothing
    else for STAR to lock onto. Copying the original landmarks across would instead import
    the exact bug -- boxes pointing at a bodybuilder -- and in the wrong coordinate frame.

WHY THE CROP MATTERS TO THE TRACKER, not just to the filter
    VHAP hardcodes the principal point at the image centre (tracker.py:153) and optimises
    only focal_length. An off-centre face is therefore inexplicable to its camera model and
    gets absorbed into head pose: measured 30.93 deg mean step and a 2.6x depth jump at a
    centred->panel cut, while the 2D fit stays perfect. Cropping puts the face back on the
    optical axis, which is the condition VHAP's model already assumes.

MATTING
    Calls VHAP's own robust_video_matting so the alpha maps are identical in kind to what
    the tracker was tuned against. Run it on the CROP, not by cropping the full-frame matte:
    RVM sometimes latches onto illustrated people in a graphic (a take-2 matte spans
    x 0.098-0.999 because of a cartoon panel), and cropping first removes the graphic.
"""
import argparse, json, os, pathlib, shutil, subprocess, sys, time
import numpy as np
from PIL import Image

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


VHAP = pathlib.Path(os.environ.get("VHAP_ROOT",
    str(pathlib.Path(__file__).resolve().parents[1] / "vhap")))


def write_chunk(src_imgs, s, dest, size):
    x, y, w, h = s["crop"]
    imgs = dest / "images"
    part = dest / "images.part"
    if imgs.exists() and len(list(imgs.iterdir())) == s["frames"]:
        return False
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    for j, i in enumerate(range(s["start"], s["end"] + 1)):
        im = Image.open(src_imgs / f"{i:06d}.jpg").convert("RGB")
        pl, pt = max(0, -x), max(0, -y)
        pr, pb = max(0, x + w - im.width), max(0, y + h - im.height)
        if pl or pt or pr or pb:
            im = Image.fromarray(np.pad(np.asarray(im), ((pt, pb), (pl, pr), (0, 0)),
                                        mode="edge"))
        im.crop((x + pl, y + pt, x + pl + w, y + pt + h)).resize(
            (size, size), Image.LANCZOS).save(part / f"{j:06d}.jpg", quality=95)
    shutil.rmtree(imgs, ignore_errors=True)
    os.replace(part, imgs)
    return True


def main():
    ap = argparse.ArgumentParser()
    _profile_arg(ap)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--only", type=int, nargs="+", help="chunk ids")
    ap.add_argument("--min-frames", type=int, default=90,
                    help="skip chunks shorter than this: too little to solve an identity from")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--matte", action="store_true")
    a = ap.parse_args()
    a = _resolve_profile(a, _load_profile(a.profile), {'min_frames': 'MIN_SEQUENCE', 'size': 'CROP_SIZE'})

    root = pathlib.Path(__file__).resolve().parent
    outdir = root / "chunks" / a.seq
    sh = json.load(open(outdir / "kept_shots.json"))
    if a.only is not None:
        sh = [s for s in sh if s["shot"] in a.only]
    sh = [s for s in sh if s["frames"] >= a.min_frames]

    src = VHAP / "data/monocular" / a.seq / "images"
    print(f"{a.seq}: {len(sh)} chunks, {sum(s['frames'] for s in sh)} frames "
          f"-> {a.size}x{a.size}")
    rows = []
    for s in sh:
        name = f"{a.seq}__c{s['shot']:03d}"
        dest = VHAP / "data/monocular" / name
        rows.append(dict(chunk=s["shot"], name=name, kind=s["kind"],
                         frames=s["frames"], src_range=[s["start"], s["end"]],
                         crop=s["crop"], face_frac=s.get("face_frac")))
        if not a.write:
            continue
        t0 = time.time()
        fresh = write_chunk(src, s, dest, a.size)
        print(f"  {name}  {s['frames']:5d} fr  {s['kind']:<8} "
              f"{'wrote' if fresh else 'exists'} {time.time()-t0:.0f}s", flush=True)

    if a.write and a.matte:
        sys.path.insert(0, str(VHAP))
        os.chdir(VHAP)
        from vhap.preprocess_video import robust_video_matting
        for r in rows:
            dest = VHAP / "data/monocular" / r["name"]
            al = dest / "alpha_maps"
            if al.exists() and len(list(al.iterdir())) == r["frames"]:
                print(f"  {r['name']} alpha ok"); continue
            t0 = time.time()
            robust_video_matting(dest / "images")
            print(f"  {r['name']} matted {time.time()-t0:.0f}s", flush=True)

    json.dump(rows, open(outdir / "sequences.json", "w"), indent=1)
    print(f"\nwrote {outdir/'sequences.json'}")
    if a.write:
        print("track one with:")
        print(f"  cd {VHAP} && python vhap/track.py --data.root_folder $PWD/data/monocular \\")
        print(f"      --data.sequence {rows[0]['name']} --exp.output_folder output/chunks/{rows[0]['name']} \\")
        print(f"      --log.interval-media 100000")


main()
