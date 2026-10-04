#!/usr/bin/env python3
"""Per-chunk, per-frame rigid head motion, measured from the exported FLAME fit.

    ~/miniconda3/envs/vhap/bin/python pipeline/gauss/head_rigid.py

Runs in the `vhap` env, the only one with the FLAME model. Writes one small file per
chunk to `pipeline/gauss/cache/rigid/<chunk>.npz`:

    R [T,3,3]  t [T,3]   with   neutral @ R + t = posed, in the EXPORT's world
    residual [T]         how far from rigid the anchor region actually moved, in mm

WHY A PROCRUSTES AND NOT FLAME'S OWN ROTATION AND TRANSLATION
    FLAME's `rotation` and `neck_pose` turn joints inside a skinned model, so the head is
    not a rigid transform of anything -- the neck bends and the jaw swings. What the
    MetaHuman head needs is the rigid motion of the SKULL. Measuring it by matching a
    region that does not deform, against that chunk's own neutral, gives exactly that,
    and the residual says how well the assumption held on each frame.

    The anchor is forehead + scalp + nose, copied from `headpose/transfer.py`, which
    records why the lowest-residual choice is not used: the nose alone fits better and
    estimates the rotation worse, because 379 near-coplanar points barely constrain it.

ONE NEUTRAL PER CHUNK, NOT ONE PER PERSON
    The three recordings were each solved against their own identity -- shape
    coefficients differ between them by up to 1.90 -- so the neutral is built from the
    chunk's own shape. Using a shared neutral would push that difference into the
    residual, where it would look like head motion.
"""
import json, os, pathlib, pickle, sys
import numpy as np
import torch

P = pathlib.Path(__file__).resolve().parent.parent
VHAP = (pathlib.Path(__file__).resolve().parents[1] / "vhap")
CORPUS = VHAP / "export/corpus"
OUT = P / "gauss/cache/rigid"
sys.path.insert(0, str(VHAP))
from vhap.model.flame import FlameHead                                  # noqa: E402

N_FLAME = 5023
ANCHOR = ("forehead", "scalp", "nose")


def procrustes(A, B):
    """Rotation and translation taking A onto B, row-vector convention: A @ R + t."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, cb - ca @ R


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # FlameHead opens asset/flame/*.pkl by a RELATIVE path, so it only loads from the
    # VHAP root; run from anywhere else it failed with FileNotFoundError.
    os.chdir(VHAP)
    mk = pickle.load(open(VHAP / "asset/flame/FLAME_masks.pkl", "rb"), encoding="latin1")
    sk = np.unique(np.concatenate([mk[k] for k in ANCHOR]))
    sk = sk[sk < N_FLAME]
    print(f"anchor: {len(sk)} vertices from {', '.join(ANCHOR)}")

    flame = FlameHead(300, 100, add_teeth=True).cuda()
    # A tracked chunk, not a merged dataset and not a cooked one: both of those also
    # carry a transforms.json, and the cooked ones carry no FLAME parameters at all,
    # which is how this crashed on an empty directory listing.
    chunks = sorted(d.name for d in CORPUS.iterdir()
                    if d.is_dir() and (d / "transforms.json").exists()
                    and any((d / "flame_param").glob("*.npz")))
    print(f"{len(chunks)} chunks")

    worst = []
    for n, c in enumerate(chunks):
        dst = OUT / f"{c}.npz"
        if dst.exists():
            continue
        fp = sorted((CORPUS / c / "flame_param").glob("*.npz"))
        T = len(fp)
        first = np.load(fp[0])
        shape = torch.tensor(first["shape"])[None].cuda()
        static = torch.tensor(first["static_offset"]).cuda() if "static_offset" in first else None

        def run(expr, rot, neck, jaw, eyes, tr):
            with torch.no_grad():
                return flame(shape.expand(len(expr), -1), expr, rot, neck, jaw, eyes, tr,
                             return_verts_cano=False, static_offset=static)[0].cpu().numpy()

        z = torch.zeros(1, 3).cuda()
        Vn = run(torch.zeros(1, 100).cuda(), z, z, z, torch.zeros(1, 6).cuda(), z)[0]

        R = np.zeros((T, 3, 3)); t = np.zeros((T, 3)); res = np.zeros(T)
        B = 64
        for s in range(0, T, B):
            e = min(s + B, T)
            g = [np.load(p) for p in fp[s:e]]
            cat = lambda k: torch.tensor(np.concatenate([x[k] for x in g])).cuda()
            Vp = run(cat("expr"), cat("rotation"), cat("neck_pose"), cat("jaw_pose"),
                     cat("eyes_pose"), cat("translation"))
            for j in range(e - s):
                i = s + j
                R[i], t[i] = procrustes(Vn[sk].astype(np.float64), Vp[j][sk].astype(np.float64))
                res[i] = np.linalg.norm(Vn[sk] @ R[i] + t[i] - Vp[j][sk], axis=1).mean() * 1000
        np.savez_compressed(dst, R=R.astype(np.float32), t=t.astype(np.float32),
                            residual=res.astype(np.float32), anchor=sk.astype(np.int32),
                            neutral=Vn.astype(np.float32))
        worst.append((res.mean(), res.max(), c))
        if n % 20 == 0 or res.mean() > 3.0:
            print(f"  [{n:3d}] {c[-8:]}  {T:5d} frames  residual {res.mean():5.2f} mm mean, "
                  f"{res.max():5.2f} max")
    if worst:
        worst.sort()
        print(f"\nresidual across {len(worst)} chunks: "
              f"best {worst[0][0]:.2f} mm, median {worst[len(worst)//2][0]:.2f}, "
              f"worst {worst[-1][0]:.2f} ({worst[-1][2][-8:]})")


if __name__ == "__main__":
    main()
