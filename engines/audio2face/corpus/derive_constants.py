#!/usr/bin/env python3
"""Print the distributions the profile's RE-DERIVE constants are read off.

    python derive_constants.py --seq <take>

The recipe says "read off the valley in the coverage histogram". This is that
histogram. Without it a worker writes the same ad-hoc numpy every time -- it was
written eight times by hand while building this pipeline.

It SUGGESTS values and does not set them. A suggestion from one take is not a
constant: take 2's centre-x had a clean second mode that was not a layout at all,
it was the detector on a bodybuilder. Look at the montages before believing a
mode means what it looks like.

ORDERING, which is not obvious. Two of these constants cannot be derived before
the scan that needs them:

    chunk_take.py --seq X            # with ANY thresholds; signals.npz does not
                                     # depend on them
    derive_constants.py --seq X      # read the distributions
    edit profiles/X.py               # set them
    chunk_take.py --seq X --profile X  (re-run; the scan is cached, this is fast)

So the first pass runs on placeholders deliberately. The profile's job is to
record what was chosen and why, not to be right before any data exists.
"""
import argparse, json, os, pathlib
import numpy as np

VHAP = pathlib.Path(os.environ.get("VHAP_ROOT",
    str(pathlib.Path(__file__).resolve().parents[1] / "vhap")))


def hist(v, lo, hi, bins=24, width=52):
    h, e = np.histogram(v, bins=bins, range=(lo, hi))
    mx = max(h.max(), 1)
    for c, b in zip(h, e[:-1]):
        if c:
            print(f"    {b:6.3f} {'#' * int(width * c / mx):<{width}} {c}")


def gap(v, lo, hi, bins=40):
    """Widest empty run between two populated bins -- the valley to sit a
    threshold in. Returns None when the distribution is unimodal, which is
    itself the answer: there is no threshold to find."""
    h, e = np.histogram(v, bins=bins, range=(lo, hi))
    nz = np.nonzero(h)[0]
    if len(nz) < 2:
        return None
    runs = [(b - a, a, b) for a, b in zip(nz[:-1], nz[1:]) if b - a > 1]
    if not runs:
        return None
    _, a, b = max(runs)
    return (e[a + 1] + e[b]) / 2, e[a + 1], e[b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    a = ap.parse_args()
    work = pathlib.Path(__file__).resolve().parent / "chunks" / a.seq
    sig = work / "signals.npz"
    if not sig.exists():
        raise SystemExit(f"no {sig} -- run chunk_take.py --seq {a.seq} first "
                         f"(with any thresholds; the scan does not depend on them)")

    z = np.load(sig)
    bb = np.load(VHAP / "data/monocular" / a.seq / "landmark2d/STAR.npz",
                 allow_pickle=True)["bounding_box"]
    n = len(bb)
    print(f"{a.seq}: {n} frames\n")

    # ---------------------------------------------------------- ALPHA_GRAPHIC
    al = z["alpha"]
    print("ALPHA_GRAPHIC  -- alpha coverage; low = no real person in frame")
    hist(al, 0, float(np.percentile(al, 99.5)))
    g = gap(al, 0, float(np.percentile(al, 99.5)))
    print(f"    suggest {g[0]:.3f}   (valley {g[1]:.3f}-{g[2]:.3f})" if g else
          "    NO VALLEY -- unimodal. Either no graphic frames, or they are not "
          "separable by coverage; check the montages")

    # ------------------------------------------------------------- PANEL_CX
    ok = bb[:, 4] > 0
    cx = (bb[ok, 0] + bb[ok, 2]) / 2
    print(f"\nPANEL_CX  -- face centre-x, {ok.sum()} frames with a face "
          f"({100 - ok.mean() * 100:.1f}% have none)")
    hist(cx, 0, 1)
    g = gap(cx, 0, 1)
    print(f"    suggest {g[0]:.3f}   (valley {g[1]:.3f}-{g[2]:.3f})" if g else
          "    NO VALLEY -- one layout only, or the modes touch")
    print("    CHECK THE MONTAGES: a second mode is not necessarily a layout. On "
          "take 2 the mode at 0.15 was the detector on a bodybuilder.")

    # ------------------------------------------------- MIN_FACE_PX / usability
    W = int(z["thumb"].shape[2] / 64 * 1280) if "thumb" in z else 1280
    fw = (bb[ok, 2] - bb[ok, 0]) * W
    print(f"\nMIN_FACE_PX  -- face width in source px (frame width {W})")
    hist(fw, 0, float(np.percentile(fw, 99.5)))
    print(f"    p05 {np.percentile(fw,5):.0f}  p50 {np.percentile(fw,50):.0f}  "
          f"p95 {np.percentile(fw,95):.0f}")
    print(f"    reference: 186-223px was usable, 76-90px was a webcam inset that "
          f"cost 36% of take 3. THIS IS A JUDGEMENT, no script applies it.")

    # ---------------------------------------------------------------- CUT_Z
    d = z["diff"]
    m, mad = np.median(d), np.median(np.abs(d - np.median(d))) * 1.4826 + 1e-6
    zs = (d - m) / mad
    print(f"\nCUT_Z  -- robust z on whole-frame difference")
    for t in (4, 6, 8, 12, 20):
        print(f"    z>{t:<3} {int((zs>t).sum()):6d} cuts   1 per {n/max((zs>t).sum(),1):.0f} frames")
    print("    a cut every <30 frames means the detector is firing on content "
          "beside him, not on cuts. Take 2 over-fired 2x for that reason.")

    # -------------------------------------------------------- MIN_SHOT context
    hard = np.where(zs > 8)[0]
    if len(hard) > 1:
        runs = np.diff(hard)
        print(f"\nMIN_SHOT  -- gaps between hard cuts, frames")
        print(f"    p10 {np.percentile(runs,10):.0f}  p50 {np.percentile(runs,50):.0f}  "
              f"p90 {np.percentile(runs,90):.0f}")
        print(f"    a floor above p50 discards more than half the shots. 90 (3s) "
              f"destroyed 44.6% of take 2.")

    # --------------------------------------------------- IDENTITY, if scanned
    fp = work / "faces.npz"
    if fp.exists():
        fz = np.load(fp)
        nf = fz["nfaces"]
        print(f"\nfaces per frame  -- from faces.npz")
        for k in (0, 1, 2, 3):
            sel = (nf == k) if k < 3 else (nf >= 3)
            print(f"    {'>=3' if k==3 else k} faces {int(sel.sum()):6d}  {sel.mean()*100:5.1f}%")
        print("    IDENTITY_MIN_SIM is not derived here: it needs a reference, "
              "which needs verdicts. 0.5 held on both takes (gap 0.84 vs 0.05).")
    else:
        print(f"\n(no faces.npz yet -- run face_scan.py for the multi-face picture)")

    print("\nNothing above is set. Put the chosen values in "
          f"recipes/face-clips/profiles/<creator>.py with the number you saw.")


if __name__ == "__main__":
    main()
