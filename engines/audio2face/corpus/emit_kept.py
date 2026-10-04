#!/usr/bin/env python3
"""Turn the kept frames into cropped, face-centred clips -- one per shot.

    python emit_kept.py --seq <name>            # list the shots
    python emit_kept.py --seq <name> --emit     # write the mp4s

Runs off clusters.npy + verdicts.json, so what gets cut is exactly what was labelled by
looking at the montages. Nothing here re-derives which frames are good.

HOW SHOTS ARE FORMED
    A shot is a contiguous run of kept frames sharing one KIND (centred / panel), not one
    cluster id. Cluster ids change constantly inside a single continuous take -- on take 2
    the centred fireplace footage alternates between c05, c23, c24 and c28 as he leans in
    and out -- and splitting on every id change would shred a 30 s shot into fragments that
    the minimum-length filter then deletes. Kind is what actually matters downstream,
    because it is what decides the crop.

    A guard band is dropped either side of every boundary: a cut has blend frames and the
    detector lags a frame or two, so the frames adjacent to a boundary are the dirtiest in
    the take and the cheapest to lose.

THE CROP
    One FIXED box per shot, from the median face over that shot, k x face width, square,
    symmetric about the face, edge-padded where it runs off the source.

      fixed, not per-frame  -- a tracking box adds apparent camera motion, which VHAP has
                               no camera model for and would absorb into head pose
      symmetric             -- an off-centre box re-creates the very offset being removed:
                               VHAP hardcodes cx=w/2 (tracker.py:153), so a face at x=0.85
                               costs a measured 30.93 deg of false head rotation at the cut
      padded, not clamped   -- clamping makes panel crops tighter than centred ones, and
                               unequal face scale reads as unequal depth
"""
import argparse, json, pathlib, subprocess, tempfile, shutil
import os
import sys
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


def identity_boxes(outdir, ref_mask):
    """Per-frame box of the face that MATCHES HIM, plus the match score.

    Not STAR's box, for two reasons. STAR returns one face and does not check which -- on
    take 2 it lands on a bodybuilder in a screenshot for 2,602 frames. And its box is
    quantised to detector pyramid scales: 3-5 distinct widths across a whole video, in
    steps of ~20%, which is larger than the camera changes we are trying to detect.
    InsightFace gives 112 distinct widths on the same video, and every face in the frame.

    The reference is accumulated, not supplied: average the embedding over frames already
    known clean (single-face, centred, kept). Order matters -- a reference built over
    frames containing strangers poisons itself and every later match degrades.
    """
    z = np.load(outdir / "faces.npz")
    nf, bx, em = z["nfaces"], z["boxes"], z["emb"].astype(np.float32)
    R = em[ref_mask & (nf == 1), 0]
    R = R[np.linalg.norm(R, axis=1) > 0.5]
    ref = R.mean(0)
    ref /= np.linalg.norm(ref)
    cols = np.arange(bx.shape[1])[None, :]
    S = np.where(cols < nf[:, None], em @ ref, -9.0)
    bi, score = S.argmax(1), S.max(1)
    take = lambda a: np.take_along_axis(a, bi[:, None], 1)[:, 0]
    return dict(w=take(bx[:, :, 2] - bx[:, :, 0]),
                cx=take((bx[:, :, 0] + bx[:, :, 2]) / 2),
                cy=take((bx[:, :, 1] + bx[:, :, 3]) / 2),
                score=score, nfaces=nf, nref=len(R))


def camera_splits(idb, s, thresh=0.15, win=10, smooth=15, min_sim=0.5):
    """Frames inside a shot where the CAMERA changed, not where he moved.

    VHAP solves one focal_length per sequence and assumes a static camera, and the crop box
    is a median over the shot -- so a framing change mid-shot makes the crop wrong for part
    of it, and the face-scale change reads as depth change. Same disease as an uncropped
    panel cut, which cost a measured 30.93 deg of false head rotation.

    A STEP, not a spread. Median within-shot face-width spread is 8.4% on take 2, which is
    him leaning in and out -- real head motion that VHAP should solve, not camera movement.
    Splitting on spread would cut good shots. A zoom or angle change is a step: compare the
    median of the next `win` frames against the previous `win`. Position is folded in at 3x
    because a re-frame moves the face across the frame far less than a zoom changes its size.
    """
    a, b = s["start"], s["end"]
    m = idb["score"][a:b + 1] >= min_sim
    if m.sum() < 4 * win:
        return []
    idx = np.arange(a, b + 1)[m]
    med = lambda x: np.array([np.median(x[max(0, i - smooth // 2):i + smooth // 2 + 1])
                              for i in range(len(x))])
    w, cx = med(idb["w"][a:b + 1][m]), med(idb["cx"][a:b + 1][m])
    n, base = len(w), np.median(w)
    step = np.zeros(n)
    for i in range(win, n - win):
        step[i] = (abs(np.median(w[i:i + win]) - np.median(w[i - win:i])) / base
                   + abs(np.median(cx[i:i + win]) - np.median(cx[i - win:i])) * 3)
    hits = np.where(step > thresh)[0]
    if not len(hits):
        return []
    # one split per run of consecutive hits, at its peak
    out, run = [], [hits[0]]
    for i in hits[1:]:
        if i - run[-1] > win * 2:
            out.append(run)
            run = []
        run.append(i)
    out.append(run)
    return [int(idx[r[int(np.argmax(step[r]))]]) for r in out if len(r)]


def identity_trim(sh, idb, min_sim, guard, minlen):
    """Cut frames out of a shot where the matched face does not match his identity.

    camera_splits already excludes these frames when measuring whether the camera
    moved -- correct for that question, since a contaminated frame carries no camera
    information. But excluding a frame from a measurement is not the same as removing
    it from the shot, and nothing else did that. A short run of contamination that
    survives clustering (a graphic mixed into an otherwise-good chunk, not large enough
    or step-shaped enough to read as a camera change) can sit inside a shot's frame
    range indefinitely unless something acts on the score directly rather than only
    reading it. Found on huberman take 1: 13 frames of an intro thumbnail-grid montage
    inside an otherwise-clean 75-frame shot, score 0.34-0.50 against a p50 of 0.93 for
    the rest of the corpus -- below IDENTITY_MIN_SIM but not a camera step, so it
    passed camera_splits unnoticed and reached kept_shots.json.
    """
    score = idb["score"]
    out = []
    for s in sh:
        a, b = s["start"], s["end"]
        good = score[a:b + 1] >= min_sim
        if good.all():
            out.append(s)
            continue
        idx = np.arange(a, b + 1)
        d = np.diff(good.astype(int))
        starts = list(np.where(d == 1)[0] + 1)
        ends = list(np.where(d == -1)[0])
        if good[0]:
            starts = [0] + starts
        if good[-1]:
            ends = ends + [len(good) - 1]
        for gs, ge in zip(starts, ends):
            x = idx[gs] + (guard if gs != 0 else 0)
            y = idx[ge] - (guard if ge != len(good) - 1 else 0)
            if y - x + 1 >= minlen:
                out.append(dict(start=int(x), end=int(y), kind=s["kind"],
                                frames=int(y - x + 1)))
    return out


def shots(keep, lab, kind_of, guard, minlen):
    n = len(keep)
    kind = np.array([kind_of.get(int(c), "?") for c in lab])
    out, st = [], None
    for i in range(n + 1):
        cur = (keep[i], kind[i]) if i < n else (False, "?")
        if st is None:
            if cur[0]:
                st = i
        elif not cur[0] or cur[1] != kind[st]:
            a, b = st + guard, i - 1 - guard
            if b - a + 1 >= minlen:
                out.append(dict(start=int(a), end=int(b), kind=kind[st],
                                frames=int(b - a + 1)))
            st = i if cur[0] else None
    return out


def crop_box(bb, s, W, H, k):
    f = bb[s["start"]:s["end"] + 1]
    good = f[:, 4] > 0
    if good.sum() < 3:
        return None
    f = f[good]
    fw = np.median(f[:, 2] - f[:, 0]) * W
    cx = np.median((f[:, 0] + f[:, 2]) / 2) * W
    cy = np.median((f[:, 1] + f[:, 3]) / 2) * H
    half = k * fw / 2
    return int(cx - half), int(cy - half), int(2 * half), int(2 * half), fw


def main():
    ap = argparse.ArgumentParser()
    _profile_arg(ap)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--guard", type=int, default=5)
    ap.add_argument("--minlen", type=int, default=60)
    ap.add_argument("--k", type=float, default=3.2)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--cam-thresh", type=float, default=0.15,
                    help="step size in the identity-matched face box that counts as a camera change")
    ap.add_argument("--no-camera-split", action="store_true")
    ap.add_argument("--no-dissolve", action="store_true")
    ap.add_argument("--identity-min-sim", type=float, default=0.5,
                    help="cosine similarity to the accumulated identity below which a "
                         "face is not him. Gap is large (good p05 0.84, graphics p50 0.05)")
    ap.add_argument("--grf-frac", type=float, default=0.15,
                    help="fraction of pixels a dissolving graphic must hold; see chunk_take.py")
    a = ap.parse_args()
    a = _resolve_profile(a, _load_profile(a.profile), {'guard': 'GUARD', 'minlen': 'MIN_SHOT', 'k': 'CROP_K', 'size': 'CROP_SIZE', 'cam_thresh': 'CAMERA_STEP', 'grf_frac': 'DISSOLVE_FRAC', 'identity_min_sim': 'IDENTITY_MIN_SIM'})

    outdir = pathlib.Path(__file__).resolve().parent / "chunks" / a.seq
    lab = np.load(outdir / "clusters.npy")
    v = json.load(open(outdir / "verdicts.json"))
    kind_of = {int(j): d["kind"] for j, d in v.items()
               if j.lstrip("-").isdigit() and isinstance(d, dict)
               and d["verdict"] == "keep"}
    keep = np.isin(lab, list(kind_of))

    imgdir = VHAP / "data/monocular" / a.seq / "images"
    W, H = Image.open(imgdir / "000000.jpg").size
    bb = np.load(VHAP / "data/monocular" / a.seq / "landmark2d/STAR.npz",
                 allow_pickle=True)["bounding_box"]

    sh = shots(keep, lab, kind_of, a.guard, a.minlen)

    # Dissolve pass. It lives in chunk_take.py because that is where the RGB thumbnail cache
    # is built, but it MUST be applied here: this function, not chunk_take's, produces the
    # final shot list. It was orphaned once for exactly that reason -- built, calibrated,
    # written into the recipe, and then silently skipped when the pipeline moved from
    # thresholds to clustering. Caught by the step-9 overlay showing a title card inside a
    # kept chunk (c005 f1767-1777), not by any number.
    if not a.no_dissolve and (outdir / "signals.npz").exists():
        from chunk_take import graphic_transitions
        z = np.load(outdir / "signals.npz")
        before = len(sh)
        sh = graphic_transitions(sh, z["thumb"], z["athumb"], a.grf_frac, minlen=a.minlen)
        for s in sh:
            s.setdefault("kind", "?")
        print(f"  dissolve pass: {before} -> {len(sh)} shots")

    if not a.no_camera_split and (outdir / "faces.npz").exists():
        cen = np.isin(lab, [j for j, k in kind_of.items() if k == "centred"])
        idb = identity_boxes(outdir, cen)
        print(f"  identity reference from {idb['nref']} clean single-face centred frames")
        out, nsplit = [], 0
        for s in sh:
            cuts = camera_splits(idb, s, a.cam_thresh, min_sim=a.identity_min_sim)
            if not cuts:
                out.append(s)
                continue
            nsplit += 1
            edges = [s["start"]] + cuts + [s["end"] + 1]
            for lo, hi in zip(edges[:-1], edges[1:]):
                x, y = lo + (a.guard if lo != s["start"] else 0), hi - 1 - a.guard
                if y - x + 1 >= a.minlen:
                    out.append(dict(start=int(x), end=int(y), kind=s["kind"],
                                    frames=int(y - x + 1)))
        print(f"  camera-change split: {nsplit} shots split, {len(sh)} -> {len(out)} pieces")
        sh = sorted(out, key=lambda s: s["start"])

        before_frames = sum(s["frames"] for s in sh)
        before_shots = len(sh)
        sh = identity_trim(sh, idb, a.identity_min_sim, a.guard, a.minlen)
        cut = before_frames - sum(s["frames"] for s in sh)
        if cut:
            print(f"  identity trim: {cut} contaminated frames cut, "
                  f"{before_shots} -> {len(sh)} shots")
    elif not (outdir / "faces.npz").exists():
        print("  (no faces.npz -- run face_scan.py to enable camera-change splitting)")

    tot = sum(s["frames"] for s in sh)
    print(f"{a.seq}: {len(lab)} frames, {keep.sum()} kept by verdict")
    print(f"  shots >= {a.minlen/30:.0f}s : {len(sh)}   frames in shots: {tot} "
          f"({tot/len(lab)*100:.1f}% of source, {tot/max(keep.sum(),1)*100:.1f}% of kept)")
    for k in sorted({s["kind"] for s in sh}):
        f = sum(s["frames"] for s in sh if s["kind"] == k)
        print(f"    {k:<10} {sum(1 for s in sh if s['kind']==k):>3} shots  {f:>6} frames")

    rows = []
    vdir = outdir / "clips"
    if a.emit:
        # Clear first. Shot boundaries change whenever a verdict or a threshold changes, so
        # a stale clip from an earlier run is a chunk that no longer exists -- and it sits
        # in the directory looking exactly as valid as the real ones. Found this the hard
        # way: 101 clips in a directory that should have held 66.
        if vdir.exists():
            shutil.rmtree(vdir)
        vdir.mkdir()
    for idx, s in enumerate(sh):
        box = crop_box(bb, s, W, H, a.k)
        if box is None:
            continue
        x, y, w, h, fw = box
        name = f"{idx:03d}_{s['kind']}_f{s['start']}-{s['end']}_{s['frames']//30}s"
        rows.append(dict(shot=idx, name=name, crop=[x, y, w, h],
                         face_px=round(float(fw), 1),
                         face_frac=round(float(fw / w), 3), **s))
        if not a.emit:
            continue
        tmp = pathlib.Path(tempfile.mkdtemp())
        for j, i in enumerate(range(s["start"], s["end"] + 1)):
            im = Image.open(imgdir / f"{i:06d}.jpg").convert("RGB")
            pl, pt = max(0, -x), max(0, -y)
            pr, pb = max(0, x + w - im.width), max(0, y + h - im.height)
            if pl or pt or pr or pb:
                im = Image.fromarray(np.pad(np.asarray(im),
                                            ((pt, pb), (pl, pr), (0, 0)), mode="edge"))
            im.crop((x + pl, y + pt, x + pl + w, y + pt + h)).resize(
                (a.size, a.size), Image.LANCZOS).save(tmp / f"{j:06d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
                        "-i", str(tmp / "%06d.png"), "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", "-crf", "18", str(vdir / (name + ".mp4"))], check=True)
        shutil.rmtree(tmp)
        print(f"  {name}  face {fw:.0f}px -> {fw/w*100:.0f}% of clip", flush=True)

    json.dump(rows, open(outdir / "kept_shots.json", "w"), indent=1)
    print(f"\nwrote {outdir/'kept_shots.json'}" + (f"\nclips -> {vdir}" if a.emit else ""))


if __name__ == "__main__":
    main()
