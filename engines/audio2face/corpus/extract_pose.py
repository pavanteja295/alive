#!/usr/bin/env python3
"""Tracked FLAME params -> per-frame rigid head pose. The only thing the corpus run keeps.

Everything else VHAP writes is an intermediate: a 52 MB checkpoint is 50.33 MB of
`tex_extra`, a 2048^2 texture a landmark-only run never touches. What survives is R and t
per frame, ~350 KB per 3600 frames.

The pose is MEASURED, not read off a parameter, for the reason headpose/transfer.py
documents: VHAP runs `--global-mode none`, so the head is turned by `rotation` and
`neck_pose` together and FLAME has no head joint (K=4: neck, jaw, two eyes). Reading
`rotation` alone silently drops most of the motion. So: build the posed mesh, Procrustes
the skull anchor from this run's own neutral onto it.
"""
import argparse, os, pathlib, pickle, sys
import numpy as np, torch

N_FLAME = 5023
ANCHOR = ("forehead", "scalp", "nose")


def procrustes(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, cb - ca @ R


def read_topology_flags(cfg):
    want = {"add_teeth": False, "remove_lip_inside": False}
    for line in pathlib.Path(cfg).read_text().splitlines():
        k, _, v = line.partition(":")
        if k.strip() in want and v.strip() in ("true", "false"):
            want[k.strip()] = v.strip() == "true"
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--vhap-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = pathlib.Path(a.out).resolve()
    ckpt = pathlib.Path(a.ckpt).resolve()
    cfg = pathlib.Path(a.config).resolve()
    sys.path.insert(0, a.vhap_root)
    os.chdir(a.vhap_root)                       # FlameHead hardcodes asset/flame/*
    from vhap.model.flame import FlameHead

    flags = read_topology_flags(cfg)
    flame = FlameHead(300, 100, **flags)
    mk = pickle.load(open("asset/flame/FLAME_masks.pkl", "rb"), encoding="latin1")
    sk = np.unique(np.concatenate([mk[k] for k in ANCHOR]))
    sk = sk[sk < N_FLAME]

    d = dict(np.load(ckpt))
    n = len(d["rotation"])
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
    z3, z6, z100 = torch.zeros(1, 3), torch.zeros(1, 6), torch.zeros(1, 100)
    so = t(d["static_offset"])

    with torch.no_grad():
        neutral = flame(shape=t(d["shape"])[None], expr=z100, rotation=z3, neck=z3,
                        jaw=z3, eyes=z6, translation=z3, static_offset=so)[0][0].numpy()
    Vn = neutral[sk].astype(np.float64)

    R = np.zeros((n, 3, 3)); T = np.zeros((n, 3)); res = np.zeros(n)
    for i in range(n):
        with torch.no_grad():
            v = flame(shape=t(d["shape"])[None], expr=t(d["expr"][i])[None],
                      rotation=t(d["rotation"][i])[None], neck=t(d["neck_pose"][i])[None],
                      jaw=t(d["jaw_pose"][i])[None], eyes=t(d["eyes_pose"][i])[None],
                      translation=t(d["translation"][i])[None],
                      static_offset=so)[0][0].numpy()
        Vp = v[sk].astype(np.float64)
        R[i], T[i] = procrustes(Vn, Vp)
        res[i] = np.linalg.norm(Vn @ R[i] + T[i] - Vp, axis=1).mean()

    # NOT out.with_suffix(".npz.tmp"): np.savez_compressed APPENDS ".npz" when the name
    # does not already end in it, so that wrote "headpose.npz.tmp.npz" while the fsync
    # below opened "headpose.npz.tmp" and raised FileNotFoundError -- after the whole
    # sequence had been solved. The tmp name must end in ".npz" so numpy leaves it alone.
    # The writer owns its directory. Callers forgetting to mkdir has now cost this project
    # three separate incidents, and this one was the worst: stderr was swallowed by the
    # caller, so 38 chunks tracked fine and silently recorded no residual, surfacing five
    # hours later as "no donor candidate".
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, R=R, t=T, residual=res, neutral=neutral,
                        frames=np.arange(n), anchor_idx=sk,
                        shape=d["shape"], static_offset=d["static_offset"],
                        expr=d["expr"], jaw_pose=d["jaw_pose"], eyes_pose=d["eyes_pose"])
    # fsync before rename: rename is atomic, but only orders correctly against data that
    # has actually reached the disk. A power cut between write and fsync leaves an empty
    # file under the final name, which is exactly the case the .done marker cannot catch.
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, out)
    print(f"{n} frames, skull residual {res.mean()*1000:.2f} mm -> {out}")


main()
