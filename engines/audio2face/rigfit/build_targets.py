#!/usr/bin/env python3
"""Step: the surface targets the animation model is trained against, at full frame rate.

    ~/miniconda3/envs/stavatar/bin/python rigfit/build_targets.py --subject drk

WHY SURFACE AND NOT CONTROL VALUES
    The obvious target is the control track the per-frame solve produces. It is the wrong
    one. That solve is an independent optimisation per frame of a problem with many equally
    good answers, so it wanders among them: the result is three times jerkier than a real
    performance, and a model trained on it would spend its capacity reproducing that
    wandering. Scoring the SURFACE instead leaves the model free to pick any control
    setting that makes the right face, which turns the degeneracy from a defect in the
    target into freedom the model may use.

WHAT IS WRITTEN, PER CLIP, ONE ROW PER VIDEO FRAME
    skin  [T, M, 3]     where each scored point of the face should be, relative to rest
    eyes  [T, 1540, 3]  the same for the two eyeballs, carrying gaze as geometry
    t     [T]           seconds into the source recording, so audio can be lined up

    Teeth and tongue get no target. Nothing observes them: the tracker's teeth are
    synthesised from the lip rings, and the tongue is inside the mouth. The teeth follow
    the jaw, and the jaw is constrained densely by the skin.
"""
import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PIPE / "offset"))
from target import Correspondence, similarity, VHAP_ASSET             # noqa: E402


def rot_between(A, B):
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--rig", default="rig_beltrami.npz")
    ap.add_argument("--index", default=None,
                    help="default: the subject's FLAME_FULL, from its audio-to-mesh profile")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE / "recipes/audio-to-mesh/tools"))
    from _profile import load
    a.index = a.index or str(load(a.subject).FLAME_FULL / "index.json")

    import torch
    from riglogic_torch import TorchRig

    index = json.load(open(a.index))
    align = json.load(open(HERE / "cache/align.json"))
    neu = {}
    for c in index:
        neu.setdefault(c["video"], np.load(c["file"])["neutral"])
    ref = neu[index[0]["video"]]
    corr = Correspondence(a.subject, a.rig, flame_neutral=ref)
    M = int(corr.facial.sum())
    mk = pickle.load(open(VHAP_ASSET, "rb"), encoding="latin1")
    ball = {"L": mk["left_eyeball"], "R": mk["right_eyeball"]}

    # the rotation carrying tracker axes into rig axes, read off the same similarity
    o = corr.to_dna(np.zeros((1, 3)))[0]
    A = np.stack([corr.to_dna(np.eye(3)[i:i + 1] * 0.01)[0] - o for i in range(3)])
    A = A / np.linalg.norm(A, axis=1).mean()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rig = TorchRig(str(PIPE / "identity/subjects" / a.subject / a.rig), device=dev)
    with torch.no_grad():
        d, bsw = rig.behaviour(torch.zeros(1, 263, device=dev))
        S = rig.skin_matrices(d)
        E0 = torch.cat([rig.deform(3, S, bsw), rig.deform(4, S, bsw)], 1)[0].cpu().numpy()
    P0 = np.concatenate([E0[:770] - E0[:770].mean(0), E0[770:] - E0[770:].mean(0)])

    outdir = pathlib.Path(a.out or HERE / f"cache/targets_{a.subject}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(index)} clips | {M} skin points + 1540 eyeball points")
    rows = []
    for n, it in enumerate(index):
        dst = outdir / f"{it['chunk']}.npz"
        rows.append({"chunk": it["chunk"], "video": it["video"]})
        if dst.exists():
            continue
        z = np.load(it["file"])
        V = z["verts"].astype(np.float64)
        skin = corr.target(V, neutral=neu[it["video"]])
        eyes = np.empty((len(V), 1540, 3), np.float32)
        Vn = neu[it["video"]].astype(np.float64)
        for side, sl in (("L", slice(0, 770)), ("R", slice(770, 1540))):
            b = ball[side]
            refb = Vn[b] - Vn[b].mean(0)
            q = P0[sl]
            for t in range(len(V)):
                cur = V[t, b] - V[t, b].mean(0)
                Rm = A.T @ rot_between(refb, cur) @ A
                eyes[t, sl] = (q @ Rm - q).astype(np.float32)
        al = align[it["video"]]
        src0 = al["src_range"][it["chunk"]][0]
        t = (src0 + np.asarray(z["frames"], np.float64)) / al["fps"]
        np.savez(dst, skin=skin.astype(np.float16), eyes=eyes.astype(np.float16),
                 t=t.astype(np.float32), video=it["video"])
        print(f"  [{n+1}/{len(index)}] {it['chunk'][:56]:<58} {len(V):>4}f  "
              f"skin {np.linalg.norm(skin,axis=-1).mean()*10:5.2f} mm  "
              f"eyes {np.linalg.norm(eyes,axis=-1).mean()*10:5.2f} mm", flush=True)
    (outdir / "index.json").write_text(json.dumps(rows, indent=1))
    np.savez_compressed(outdir / "_geometry.npz", mask=corr.mask, facial=corr.facial,
                        vt=corr.vt, neutral=ref, A=A, E0=E0)
    print("wrote", outdir)


if __name__ == "__main__":
    main()
