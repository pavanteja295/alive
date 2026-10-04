#!/usr/bin/env python3
"""Eye opening per frame, measured from the tracker's own landmarks.

    ~/miniconda3/envs/stavatar/bin/python pipeline/gauss/build_ear.py --subject drk

Writes cache/ear_<subject>.npz: one [T, 4] array per chunk --

    0  left eye aspect ratio     eye height / eye width, from the 68-point set
    1  right eye aspect ratio
    2  frontality                1.0 face-on, falling as he turns away
    3  confidence                the tracker's own, per frame

blink_ear.py turns columns 0-1 into the rig's blink control and gates on 2-3.

WHY A RATIO

Height over width does not care how large the face is in frame, so it needs no
per-clip calibration. On this subject it runs about 0.38 open and 0.04 fully shut,
and the two eyes agree at +0.978 -- which is what a real closure looks like, and
what the solved blink channel does NOT do (it manages 0.280).

WHAT IT CANNOT TELL YOU

A squint, a blink and looking down are the same to this measure. That is fine for
cooking, where the job is to make the mesh match the picture whatever the reason,
and it is NOT a blink detector.

FRONTALITY, AND A WARNING ABOUT THE FIRST CACHE

The ratio collapses on a profile whatever the lid is doing, so off-angle frames have
to be excluded rather than guessed at. Frontality here is how centred the nose is
between the two jaw sides -- 1.0 when they are equal, falling as one foreshortens.

  The original ear_drk.npz was built by an unrecorded script and its frontality
  column does not match this definition (mean difference 0.032). Everything else in
  it reproduces exactly. Rebuilding shifts which frames pass the gate slightly, so
  a corpus filtered with the old cache and one filtered with a rebuilt cache are not
  the same set of frames.

FOR A NEW SUBJECT

Nothing here is specific to anyone: it reads whatever chunks the tracker produced.
The subject needs VHAP landmarks (landmark2d/STAR.npz per chunk), which the
tracking stage already writes.
"""
import argparse, pathlib, sys
import numpy as np

DATA = (pathlib.Path(__file__).resolve().parents[1] / "vhap/data/monocular")
CACHE = pathlib.Path(__file__).resolve().parent / "cache"
L_EYE, R_EYE = list(range(36, 42)), list(range(42, 48))


def ear(P, ids):
    """Eye aspect ratio: the two vertical lid gaps over the corner-to-corner width."""
    e = P[ids]
    v1 = np.linalg.norm(e[1] - e[5])
    v2 = np.linalg.norm(e[2] - e[4])
    h = np.linalg.norm(e[0] - e[3])
    return (v1 + v2) / (2 * h) if h > 1e-6 else 0.0


def frontality(P):
    """How centred the nose is between the jaw sides. 1.0 face-on."""
    jl = np.linalg.norm(P[30] - P[2])
    jr = np.linalg.norm(P[30] - P[14])
    return float(min(jl, jr) / max(jl, jr)) if max(jl, jr) > 1e-6 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--match", default="", help="only chunks containing this string")
    ap.add_argument("--data", default=str(DATA))
    a = ap.parse_args()

    root = pathlib.Path(a.data)
    # this person's recordings only: data/monocular holds every creator's chunks, and
    # the open/shut levels below are percentiles, so pooling people moves them
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "recipes/mesh-to-render/tools"))
    from _profile import load
    recs = load(a.subject).RECORDINGS
    chunks = sorted(p for p in root.iterdir()
                    if (p / "landmark2d/STAR.npz").exists() and a.match in p.name
                    and "__c" in p.name and p.name.split("__c")[0] in recs)
    if not chunks:
        raise SystemExit(f"no chunks with landmarks under {root}")
    out, tot = {}, 0
    for c in chunks:
        z = np.load(c / "landmark2d/STAR.npz")
        lm = z["face_landmark_2d"]
        conf = lm[..., 2].mean(1) if lm.shape[-1] > 2 else np.ones(len(lm))
        rows = np.zeros((len(lm), 4), np.float32)
        for i, P in enumerate(lm[:, :, :2]):
            rows[i] = (ear(P, L_EYE), ear(P, R_EYE), frontality(P), conf[i])
        out[c.name] = rows
        tot += len(rows)
    CACHE.mkdir(exist_ok=True)
    dst = CACHE / f"ear_{a.subject}.npz"
    np.savez_compressed(dst, **out)
    E = np.concatenate([v[:, :2] for v in out.values()])
    F = np.concatenate([v[:, 2] for v in out.values()])
    r = np.corrcoef(E[:, 0], E[:, 1])[0, 1]
    print(f"{len(out)} chunks, {tot} frames -> {dst}")
    print(f"  eye aspect ratio: open (p85) {np.percentile(E, 85):.3f}  "
          f"shut (p02) {np.percentile(E, 2):.3f}")
    print(f"  the two eyes agree at {r:+.3f}   (a real closure moves both)")
    print(f"  frontal frames (>0.60): {100*(F > 0.60).mean():.1f}%")


if __name__ == "__main__":
    main()
