#!/usr/bin/env python3
"""Cut an edited talking-head video into single-layout shots VHAP can actually track.

    python chunk_take.py --seq <name> --propose          # detect shots, write contact sheet
    python chunk_take.py --seq <name> --emit             # write per-shot VHAP sequences

WHY THIS EXISTS
    VHAP assumes one continuous shot from one static camera. These videos are edited. Two
    things break, and they break differently:

    1. GRAPHIC FRAMES (no face on screen). STAR writes landmarks (-1,-1) with confidence
       -1, and tracker.py:382 is `lmk_loss = norm(diff) * confidence` with no sign guard.
       A negative confidence flips the term, so minimising it MAXIMISES distance from the
       landmark -- an unbounded repulsion. Measured on take 1: over an 18.4 s graphic the
       head walks 0 -> 138 deg, and after the face returns it never recovers, sitting at
       130-168 deg for the next 28 s until the shot ends.

    2. LAYOUT CUTS (full-frame <-> side-by-side panel). The face is fine -- 186 px wide,
       sharp -- but it is no longer near the optical axis, and tracker.py:153 hardcodes
       `cx, cy = w/2, h/2`. Only focal_length is optimisable; the principal point is not.
       So VHAP cannot say "the camera is looking left of him" and instead explains the
       new screen position with 3D pose: measured 30.93 deg mean step and a 2.6x depth
       jump (|t| 0.123 -> 0.317 m) at the cut, while the 2D fit stays perfect.

    Hence: drop graphic shots, keep face shots, CROP side-by-side panels back to centre,
    and hand each shot to VHAP as its own sequence so nothing seeds across a cut.

SIGNALS, cheapest first. All but the cut detector are already on disk.
    face confidence   STAR.npz face_landmark_2d[:,:,2] < 0   -> no face at all
    alpha coverage    mean of alpha_maps/*.jpg               -> cartoon faces the detector
                                                                fires on (0.056-0.071 vs
                                                                0.248 p05 for real frames)
    face centre-x     STAR.npz bounding_box                  -> layout, cleanly bimodal
                                                                (~0.55 vs ~0.85)
    frame difference  computed here                          -> hard cuts WITHIN a layout,
                                                                which the above cannot see
"""
import argparse, json, pathlib, subprocess, sys
import os
import numpy as np
from PIL import Image, ImageDraw

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
GRAPHIC, CENTRED, PANEL = 0, 1, 2
NAMES = {GRAPHIC: "graphic", CENTRED: "centred", PANEL: "panel"}


def load_signals(seq, cache):
    d = VHAP / "data/monocular" / seq
    lm = np.load(d / "landmark2d/STAR.npz", allow_pickle=True)
    bb, l2 = lm["bounding_box"], lm["face_landmark_2d"]
    n = len(bb)

    if cache.exists():
        z = np.load(cache)
        alpha, diff = z["alpha"], z["diff"]
        thumb, athumb = z["thumb"], z["athumb"]
    else:
        alpha = np.empty(n, np.float32)
        diff = np.zeros(n, np.float32)
        # Keep the thumbnails, not just the scalar. A hard cut shows up in a single
        # frame-to-frame difference; a DISSOLVE does not -- it spreads the same total
        # change over 20-40 frames, so each individual step looks like ordinary motion.
        # Catching it needs frames compared against a background model, which needs the
        # stack. 22635 x 36 x 64 x 2 bytes is 104 MB, so just keep it.
        # RGB, not grayscale. The overlays in this footage are TRANSLUCENT COLOUR washes:
        # a blue gradient card barely moves luminance but moves chroma hard. Measured on
        # the known card at f1757-1788, deviation from the clean part of its own shot:
        #   grayscale  frac 0.066 clean -> 0.092 card   (1.4x, unusable)
        #   RGB        frac 0.121 clean -> 0.235 card   (1.9x, separable)
        # Grayscale threw away the only channel carrying the signal.
        thumb = np.empty((n, 36, 64, 3), np.uint8)
        athumb = np.empty((n, 36, 64), np.uint8)
        prev = None
        for i in range(n):
            a = np.asarray(Image.open(d / f"alpha_maps/{i:06d}.jpg").convert("L"), np.uint8)
            alpha[i] = a.mean() / 255.0
            athumb[i] = np.asarray(Image.fromarray(a).resize((64, 36)), np.uint8)
            # Downscale hard before differencing: a cut changes the whole frame, while
            # motion, grain and compression do not survive a 64x36 box filter.
            c = np.asarray(Image.open(d / f"images/{i:06d}.jpg").convert("RGB").resize((64, 36)), np.float32)
            thumb[i] = c.astype(np.uint8)
            g = c.mean(2)
            if prev is not None:
                diff[i] = np.abs(g - prev).mean()
            prev = g
            if i % 2000 == 0:
                print(f"  scan {i}/{n}", flush=True)
        np.savez_compressed(cache, alpha=alpha, diff=diff, thumb=thumb, athumb=athumb)
    return bb, l2, alpha, diff, thumb, athumb, n


def label(bb, l2, alpha, n, alpha_th=0.15, cx_th=0.62):
    noface = l2[:, 0, 2] < 0
    cx = (bb[:, 0] + bb[:, 2]) / 2
    lab = np.where(noface | (alpha < alpha_th), GRAPHIC,
                   np.where(cx > cx_th, PANEL, CENTRED))
    return lab.astype(np.int8), cx


def shots(lab, diff, n, guard, minlen, cut_z):
    """Boundaries = layout changes UNION hard cuts. A cut inside one layout still has to
    split the sequence: VHAP would otherwise seed the first frame of the new shot from the
    last frame of the old one, which is the same propagation that ruins a graphic."""
    b = set(np.where(np.diff(lab) != 0)[0] + 1)
    med, mad = np.median(diff), np.median(np.abs(diff - np.median(diff))) + 1e-6
    hard = np.where((diff - med) / (1.4826 * mad) > cut_z)[0]
    b |= set(hard.tolist())
    bounds = np.array(sorted(b | {0, n}))

    bad = np.zeros(n, bool)
    for c in bounds:
        bad[max(0, c - guard):min(n, c + guard)] = True

    out = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        a, z = s, e - 1
        while a <= z and bad[a]:
            a += 1
        while z >= a and bad[z]:
            z -= 1
        if z - a + 1 >= minlen and lab[a] != GRAPHIC:
            out.append(dict(start=int(a), end=int(z), layout=int(lab[a]),
                            frames=int(z - a + 1)))
    return out, len(hard)


def graphic_transitions(sh, thumb, athumb, frac_th=0.15, dev_th=25.0,
                        persist=5, min_run=8, pad=6, minlen=90):
    """Trim or split shots where a graphic DISSOLVES or SLIDES in.

    The hard-cut detector thresholds frame-to-frame difference, so it is structurally blind
    to any transition taking more than one frame. A card fading in over 30 frames raises
    each individual step by a thirtieth of the total -- indistinguishable from head motion.

    WHAT IS MEASURED. Build a per-pixel temporal median over the shot, then ask what
    FRACTION of background pixels deviate from it by more than `dev_th`. Background means
    the matte says so, dilated first (below).

    THREE THINGS HAD TO BE RIGHT, and each was wrong in an earlier version:

      COLOUR, not luminance. The overlays here are translucent colour washes -- a blue
      gradient card barely moves grey level. Deviation of the known card from the clean
      part of its own shot: grayscale 0.066 -> 0.092 (1.4x, unusable), RGB 0.121 -> 0.235.

      PERSISTENCE IN PLACE, not magnitude. Mean deviation measures whatever moved, and what
      moves most in this footage is his hands: that version flagged 2.7% of a clean centred
      shot and the fragmentation cost 15% of the take. A card holds the SAME pixels for 30+
      frames; a hand holds any given pixel for 2-3. Eroding the deviation mask over a
      +/-`persist` window keeps the first and erases the second.

      NO PERSON MASK. Restricting to background was meant to reject hand motion. It failed
      both ways -- the matte lags on motion blur so hand pixels leak in as background
      anyway, and the known card sits over his CHEST, so masking the person masked the
      evidence. Persistence needs no mask.

    Scored per shot, never globally: every shot has its own background, and one global
    model would flag every panel shot wholesale.
    """
    out, nsplit = [], 0
    for s_ in sh:
        a, b = s_["start"], s_["end"]
        T = thumb[a:b + 1].astype(np.float32)                     # (t, 36, 64, 3)
        med = np.median(T, axis=0)
        # worst channel, so a pure chroma shift counts as much as a luminance one
        dev = np.abs(T - med).max(3) > dev_th
        # Temporal erosion: a pixel counts only if it deviated across the whole +/-persist
        # window. THIS is what separates a graphic from a moving limb, and it is measured
        # over the WHOLE frame with no person mask.
        #
        # Restricting to background was the previous attempt and it fails both ways: it is
        # what let hand motion through (the matte lags on motion blur, so leaked hand
        # pixels read as background), and it is what hid the known card at f1757-1788,
        # which sits over his CHEST and so was masked out as person. Persistence needs
        # neither mask: a card holds the same pixels for 30+ frames, a hand holds any given
        # pixel for 2-3. Measured, at this threshold:
        #     card shot     27.6% of frames flagged  (= the last 43 of 156, exactly right)
        #     gesticulation  0.0%
        #     clean panel    max frac 0.060
        keep = dev.copy()
        for sft in range(1, persist + 1):
            keep[sft:] &= dev[:-sft]
            keep[:-sft] &= dev[sft:]
        frac = keep.reshape(len(keep), -1).mean(1)

        bad = frac > frac_th
        # persistence: drop any flagged run shorter than min_run
        i = 0
        while i < len(bad):
            if bad[i]:
                j = i
                while j < len(bad) and bad[j]:
                    j += 1
                if j - i < min_run:
                    bad[i:j] = False
                i = j
            else:
                i += 1

        if not bad.any():
            out.append(s_)
            continue
        for i in np.where(bad)[0]:
            bad[max(0, i - pad):min(len(bad), i + pad + 1)] = True
        runs, st = [], None
        for i, v in enumerate(bad):
            if not v and st is None:
                st = i
            elif v and st is not None:
                runs.append((st, i - 1))
                st = None
        if st is not None:
            runs.append((st, len(bad) - 1))
        runs = [r for r in runs if r[1] - r[0] + 1 >= minlen]
        if not runs:
            nsplit += 1
            print(f"  dissolve: DROP f{a}-{b} ({bad.mean()*100:.0f}% flagged, nothing clean left)")
            continue
        if len(runs) > 1:
            nsplit += 1
        for r0, r1 in runs:
            # carry every other key through unchanged. This function is called from two
            # places whose shot dicts differ -- chunk_take uses "layout", emit_kept uses
            # "kind" and "shot" -- and hardcoding one of them silently breaks the other.
            piece = {k: v for k, v in s_.items() if k not in ("start", "end", "frames")}
            piece.update(start=a + r0, end=a + r1, frames=r1 - r0 + 1)
            out.append(piece)
        if runs != [(0, len(bad) - 1)]:
            print(f"  dissolve: f{a}-{b} -> " +
                  ", ".join(f"f{a+r0}-{a+r1}" for r0, r1 in runs))
    print(f"  dissolve pass: {len(sh)} shots in, {len(out)} out, {nsplit} split or dropped")
    return out


def contact_sheet(seq, sh, out, cols=6, cw=300):
    d = VHAP / "data/monocular" / seq / "images"
    ims = []
    for k, s in enumerate(sh):
        m = (s["start"] + s["end"]) // 2
        im = Image.open(d / f"{m:06d}.jpg").convert("RGB")
        r = cw / im.width
        im = im.resize((cw, int(im.height * r)))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, cw - 1, 18], fill=(0, 0, 0))
        dr.text((4, 4), f"{k}: {NAMES[s['layout']]} f{s['start']}-{s['end']} {s['frames']/30:.0f}s",
                fill=(255, 255, 0))
        ims.append(im)
    if not ims:
        return
    ch = ims[0].height
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cw * cols, ch * rows), (20, 20, 20))
    for k, im in enumerate(ims):
        sheet.paste(im, ((k % cols) * cw, (k // cols) * ch))
    sheet.save(out)
    print("contact sheet ->", out, f"({len(ims)} shots)")


def crop_box(bb, s, W, H, k=3.2):
    """One FIXED box per shot, from the median face over that shot -- never per-frame.

    A per-frame box would track the face and so introduce apparent camera motion, which
    VHAP has no camera model for and would absorb into head pose: exactly the disease we
    are curing. A fixed box is a static virtual camera, which is what VHAP assumes.

    Sized as k x face width for every shot regardless of layout, so a panel shot and a
    full-frame shot end up with the face at the SAME apparent scale, centred. That makes
    every shot the same virtual camera, which is what lets one focal_length and one
    identity serve all of them.
    """
    f = bb[s["start"]:s["end"] + 1]
    fw = np.median(f[:, 2] - f[:, 0]) * W
    cx = np.median((f[:, 0] + f[:, 2]) / 2) * W
    cy = np.median((f[:, 1] + f[:, 3]) / 2) * H
    half = k * fw / 2
    # Symmetric about the face -- an asymmetric box would re-introduce the very off-axis
    # offset we are cropping to remove. It is NOT shrunk to fit: a panel face sits at
    # x=1088 of 1280, so fitting would clamp half to 192 instead of the ~298 that k*fw
    # asks for, and panel clips would come out visibly tighter than centred ones. Unequal
    # face scale across shots means unequal apparent depth, which is what we are trying to
    # stop VHAP from inventing. So the box is allowed to run off the frame and the CALLER
    # pads. Returns the box in (possibly negative) source coordinates.
    return int(cx - half), int(cy - half), int(2 * half), int(2 * half), fw


def load_padded(d, i, box, size):
    """One cropped frame, edge-padded where the box runs off the source."""
    x, y, w, h = box
    im = Image.open(d / f"{i:06d}.jpg").convert("RGB")
    pl, pt = max(0, -x), max(0, -y)
    pr, pb = max(0, x + w - im.width), max(0, y + h - im.height)
    if pl or pt or pr or pb:
        im = Image.fromarray(np.pad(np.asarray(im), ((pt, pb), (pl, pr), (0, 0)), mode="edge"))
    return im.crop((x + pl, y + pt, x + pl + w, y + pt + h)).resize((size, size), Image.LANCZOS)


def strips(seq, sh, outdir, per_shot=6, cell=210, shots_per_sheet=7):
    """A filmstrip per shot, cropped exactly as the clip will be, tiled for REVIEW.

    The heuristics propose; this is what lets a reader (human or model) actually look. One
    frame per shot -- the contact sheet -- shows the layout but cannot show a cut hiding
    INSIDE a shot, the face leaving the crop, or a stretch where he is turned away. Six
    frames spread across the shot show all three.
    """
    d = VHAP / "data/monocular" / seq / "images"
    lm = np.load(VHAP / "data/monocular" / seq / "landmark2d/STAR.npz", allow_pickle=True)
    bb = lm["bounding_box"]
    im0 = Image.open(d / "000000.jpg"); W, H = im0.size
    sd = outdir / "strips"; sd.mkdir(exist_ok=True)
    out = []
    for page in range(0, len(sh), shots_per_sheet):
        block = sh[page:page + shots_per_sheet]
        sheet = Image.new("RGB", (cell * per_shot, cell * len(block) + 18 * len(block)), (18, 18, 18))
        yy = 0
        for k, s in enumerate(block):
            x, y, w, h, fw = crop_box(bb, s, W, H)
            dr = ImageDraw.Draw(sheet)
            dr.text((4, yy + 4),
                    f"shot {page+k}  {NAMES[s['layout']]}  f{s['start']}-{s['end']}  "
                    f"{s['frames']/30:.0f}s  face {fw:.0f}px ({fw/w*100:.0f}% of crop)",
                    fill=(255, 220, 80))
            yy += 18
            for j, i in enumerate(np.linspace(s["start"], s["end"], per_shot).astype(int)):
                sheet.paste(load_padded(d, int(i), (x, y, w, h), cell), (j * cell, yy))
            yy += cell
        f = sd / f"strip_{page:02d}.jpg"
        sheet.save(f, quality=88)
        out.append(f)
        print("  ", f)
    return out


def emit(seq, sh, outdir, size, fps):
    d = VHAP / "data/monocular" / seq / "images"
    im0 = Image.open(d / "000000.jpg")
    W, H = im0.size
    lm = np.load(VHAP / "data/monocular" / seq / "landmark2d/STAR.npz", allow_pickle=True)
    bb = lm["bounding_box"]
    vdir = outdir / "clips"
    vdir.mkdir(exist_ok=True)
    rows = []
    for k, s in enumerate(sh):
        x, y, w, h, fw = crop_box(bb, s, W, H)
        name = f"{k:02d}_{NAMES[s['layout']]}_f{s['start']}-{s['end']}_{s['frames']//30}s"
        tmp = outdir / "_frames"
        if tmp.exists():
            subprocess.run(["rm", "-rf", str(tmp)])
        tmp.mkdir()
        for j, i in enumerate(range(s["start"], s["end"] + 1)):
            # PIL .crop() past an edge fills BLACK, which the matting model reads as
            # background and nvdiffrast's photometric term would try to explain. Replicate
            # the edge pixels instead: neutral, and outside the matte either way.
            im = Image.open(d / f"{i:06d}.jpg").convert("RGB")
            pl, pt = max(0, -x), max(0, -y)
            pr, pb = max(0, x + w - im.width), max(0, y + h - im.height)
            if pl or pt or pr or pb:
                im = Image.fromarray(np.pad(np.asarray(im),
                                            ((pt, pb), (pl, pr), (0, 0)), mode="edge"))
            im.crop((x + pl, y + pt, x + pl + w, y + pt + h)).resize(
                (size, size), Image.LANCZOS).save(tmp / f"{j:06d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(tmp / "%06d.png"), "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", "-crf", "18", str(vdir / (name + ".mp4"))], check=True)
        subprocess.run(["rm", "-rf", str(tmp)])
        rows.append(dict(shot=k, name=name, crop=[x, y, w, h],
                         face_px=round(float(fw), 1),
                         face_frac=round(float(fw / w), 3), **s))
        print(f"  {name}  crop {w}x{h} at ({x},{y})  face {fw:.0f}px -> {fw/w*100:.0f}% of clip")
    json.dump(rows, open(outdir / "clips.json", "w"), indent=1)
    print("\nclips ->", vdir)


def main():
    ap = argparse.ArgumentParser()
    _profile_arg(ap)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--guard", type=int, default=5, help="frames dropped either side of a cut")
    ap.add_argument("--minlen", type=int, default=90, help="shortest shot kept, frames")
    ap.add_argument("--cut-z", type=float, default=8.0, help="robust z on frame difference")
    ap.add_argument("--alpha-graphic", type=float, default=0.15,
                    help="alpha coverage below which a frame holds no real person. "
                         "RE-DERIVE: read off the valley in the coverage histogram")
    ap.add_argument("--panel-cx", type=float, default=0.62,
                    help="face centre-x above which a frame is a side panel. "
                         "RE-DERIVE: take 2 has a mode at 0.15 that is not a layout at all")
    ap.add_argument("--dissolve-dev", type=float, default=25.0,
                    help="per-channel deviation counting as changed, 0-255")
    ap.add_argument("--grf-frac", type=float, default=0.15,
                    help="fraction of pixels a graphic must hold for the persist window")
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit", action="store_true", help="write one cropped mp4 per shot")
    ap.add_argument("--strips", action="store_true", help="write per-shot filmstrips for review")
    ap.add_argument("--only", type=int, nargs="+", help="restrict --emit to these shot ids")
    ap.add_argument("--review", help="review.json: LLM verdicts to apply over the proposal")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    a = _resolve_profile(a, _load_profile(a.profile), {'cut_z': 'CUT_Z', 'grf_frac': 'DISSOLVE_FRAC', 'guard': 'GUARD', 'minlen': 'MIN_SHOT', 'alpha_graphic': 'ALPHA_GRAPHIC', 'panel_cx': 'PANEL_CX'})

    root = pathlib.Path(__file__).resolve().parent
    outdir = pathlib.Path(a.out) if a.out else root / "chunks" / a.seq
    outdir.mkdir(parents=True, exist_ok=True)

    bb, l2, alpha, diff, thumb, athumb, n = load_signals(a.seq, outdir / "signals.npz")
    lab, cx = label(bb, l2, alpha, n, a.alpha_graphic, a.panel_cx)
    sh, nhard = shots(lab, diff, n, a.guard, a.minlen, a.cut_z)
    print()
    sh = graphic_transitions(sh, thumb, athumb, a.grf_frac,
                             dev_th=a.dissolve_dev, minlen=a.minlen)

    # --- apply the review -------------------------------------------------------------
    # The heuristics are a PROPOSAL. A look at the filmstrips catches what they cannot:
    # a graphic that dissolves in rather than cutting, a shot from a different camera, a
    # title card that looks alarming but never touches the face. Those verdicts live in a
    # file so the run stays reproducible and the reasoning stays attached to the frames.
    if a.review:
        rv = json.load(open(a.review))
        # Keyed by FRAME RANGE "start-end", never by shot index. The dissolve pass splits
        # and drops shots, so every index downstream of a split shifts -- a review written
        # against one run would silently drop the wrong shot on the next. A frame range is
        # stable under renumbering. A verdict matches a shot if their ranges overlap.
        def parse(k):
            lo, _, hi = k.partition("-")
            return int(lo), int(hi)

        drops = {parse(k): v for k, v in rv.get("drop", {}).items()}
        trims = {parse(k): v for k, v in rv.get("trim", {}).items()}
        hit = set()
        out = []
        for s in sh:
            killed = False
            for (lo, hi), why in drops.items():
                if s["start"] <= hi and s["end"] >= lo:
                    print(f"  review: DROP f{s['start']}-{s['end']} -- {why[:66]}")
                    hit.add((lo, hi)); killed = True; break
            if killed:
                continue
            for (lo, hi), (tlo, thi) in trims.items():
                if s["start"] <= hi and s["end"] >= lo:
                    if tlo is not None: s["start"] = max(s["start"], tlo)
                    if thi is not None: s["end"] = min(s["end"], thi)
                    s["frames"] = s["end"] - s["start"] + 1
                    print(f"  review: TRIM -> f{s['start']}-{s['end']}")
                    hit.add((lo, hi))
            if s["frames"] >= a.minlen:
                out.append(s)
        for k in set(drops) | set(trims):
            if k not in hit:
                print(f"  review: WARNING verdict f{k[0]}-{k[1]} matched no shot")
        sh = out

    kept = sum(s["frames"] for s in sh)
    print(f"\n{a.seq}: {n} frames")
    print(f"  graphic frames        {int((lab==GRAPHIC).sum()):6d}  {(lab==GRAPHIC).mean()*100:5.1f}%")
    print(f"  hard cuts detected    {nhard:6d}")
    print(f"  shots kept            {len(sh):6d}")
    print(f"  frames kept           {kept:6d}  {kept/n*100:5.1f}%")
    for L in (CENTRED, PANEL):
        f = sum(s["frames"] for s in sh if s["layout"] == L)
        print(f"    {NAMES[L]:8s}            {f:6d}  {f/n*100:5.1f}%")

    json.dump(dict(seq=a.seq, n=n, guard=a.guard, minlen=a.minlen,
                   cut_z=a.cut_z, shots=sh),
              open(outdir / "shots.json", "w"), indent=1)
    np.save(outdir / "labels.npy", lab)
    contact_sheet(a.seq, sh, outdir / "contact_sheet.jpg")
    print("wrote", outdir / "shots.json")
    if a.strips:
        print()
        strips(a.seq, sh, outdir)
    if a.emit:
        print()
        emit(a.seq, [sh[i] for i in a.only] if a.only else sh, outdir, a.size, a.fps)


if __name__ == "__main__":
    main()
