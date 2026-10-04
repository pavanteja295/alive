#!/usr/bin/env python3
"""Step 2. For every chunk: the best the rig AS SHIPPED can do, and what it leaves over.

    ~/miniconda3/envs/stavatar/bin/python rigfit/solve.py --subject drk

For each frame this solves

    c*  =  argmin_c  || dV(c) - dF ||          c in [0, 1]^263

by gradient descent through the differentiable rig, where dV(c) is what the rig makes
when the sliders are set to c, and dF is what the tracker says the face did. c* is the
rig's ceiling: no slider setting does better. What is written out is the leftover,

    R  =  dF - dV(c*)

which is the thing the rest of the study is about. If R were noise, nothing could be
done. If R has structure, and that structure is predictable from the sliders, then it
is a piece of deformation the rig is missing and could be given.

The sliders are boxed to [0, 1] because that is the rig's actual domain. An unboxed
solve would report capacity the rig does not have at run time.

OUTPUT, one file per chunk
    c_fit  [T, 263]     f32   the ceiling slider track
    R      [T, M, 3]    f16   the leftover, in DNA centimetres
    dFn    [T, M]       f16   how far each point moved: the denominator for every ratio
    dVn    [T, M]       f16   how far the rig moved it
f16 holds about 3 decimal digits, and these are centimetre quantities of order 0.1, so
the storage error is ~1e-5 cm against a 1.4e-1 cm signal. Immaterial, and it keeps the
whole study in memory at once.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PIPE / "offset"))
from target import Correspondence                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--rig", default="rig_beltrami.npz")
    ap.add_argument("--w-dead", type=float, default=0.0, help=
                    "penalise each control IN INVERSE PROPORTION to how far it moves the "
                    "face. jawOpen travels 17.6 mm of lip and is barely touched; "
                    "neckThroatExhale travels 0.000 mm and is pinned. A uniform magnitude "
                    "prior cannot do this: it shrinks signal and degeneracy together, and "
                    "measurably makes the track jerkier because the spread falls faster "
                    "than the frame-to-frame steps. Needs cache/mouth_authority.json.")
    ap.add_argument("--w-mag", type=float, default=0.0, help=
                    "pull controls toward rest. Removes the freedom that let throat and "
                    "swallow controls absorb error they cannot possibly explain.")
    ap.add_argument("--w-tmp", type=float, default=0.0, help=
                    "penalise frame-to-frame change, so consecutive frames stop hopping "
                    "between control settings that make the same face.")
    ap.add_argument("--index", default=None,
                    help="default: the subject's FLAME_SHARED, from its audio-to-mesh profile")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--chunk-frames", type=int, default=48, help="frames per GPU batch")
    ap.add_argument("--limit", type=int, default=0, help="stop after N chunks (smoke test)")
    ap.add_argument("--selftest", action="store_true",
                    help="also refit a surface the rig provably can make, to measure "
                         "how much of the leftover is the optimiser rather than the rig")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE / "recipes/audio-to-mesh/tools"))
    from _profile import load
    a.index = a.index or str(load(a.subject).FLAME_SHARED / "index.json")

    import torch
    from riglogic_torch import TorchRig

    index = json.load(open(a.index))
    # Interleave the recordings. Solved in file order the first recording would finish
    # before the second was touched, so a run stopped early would carry no cross-recording
    # evidence at all -- which is the whole point of the study.
    byv = {}
    for it in index:
        byv.setdefault(it["video"], []).append(it)
    index = [it for grp in zip(*[v + [None] * (max(map(len, byv.values())) - len(v))
                                 for v in byv.values()]) for it in grp if it]
    if a.limit:
        index = index[:a.limit]
    # The tracker locks identity per recording. Verify that, and measure the drift
    # between recordings rather than assuming it away: it is the floor under every
    # claim about a fixed correction to the rest face.
    neu = {}
    for c in index:
        neu.setdefault(c["video"], []).append(np.load(c["file"])["neutral"])
    for v, ns in neu.items():
        assert all(np.array_equal(ns[0], n) for n in ns), f"{v}: rest face varies within a video"
        neu[v] = ns[0]
    ref = neu[index[0]["video"]]
    corr = Correspondence(a.subject, a.rig, flame_neutral=ref)
    if len(neu) > 1:
        # Reported two ways. Raw, it includes the overall size the tracker chose, which
        # a single camera cannot pin down and which is not an identity difference at all.
        # Aligned, a similarity is removed first, leaving only real shape disagreement.
        from target import similarity
        raw = [np.linalg.norm(n[:5023] - ref[:5023], axis=1).mean() for n in neu.values()]
        al = [np.linalg.norm(similarity(n[:5023], ref[:5023])(n[:5023]) - ref[:5023],
                             axis=1).mean() for n in neu.values()]
        sc = corr.to_dna(ref[:5023]).ptp(0).max() / ref[:5023].ptp(0).max()
        print(f"identity drift across {len(neu)} recordings, mean vertex displacement: "
              f"raw {np.mean(raw)*sc:.4f} cm, after removing scale and pose "
              f"{np.mean(al)*sc:.4f} cm  (max {np.max(al)*sc:.4f})")
    M = int(corr.facial.sum())
    print(f"{len(index)} chunks | {a.subject}/{a.rig} | {M} facial points "
          f"| median correspondence {np.median(corr.dist[corr.mask])*10:.2f} mm")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rig = TorchRig(str(PIPE / "identity/subjects" / a.subject / a.rig), device=dev)
    mi = torch.as_tensor(np.nonzero(corr.mask)[0], device=dev)
    fac = torch.as_tensor(np.nonzero(corr.facial)[0], device=dev)

    def head(c):
        d, bsw = rig.behaviour(c)
        return rig.deform(0, rig.skin_matrices(d), bsw)[:, mi][:, fac]

    with torch.no_grad():
        V0 = head(torch.zeros(1, 263, device=dev))[0]

    # the per-control penalty: large where the face cannot see the control at all
    PEN = None
    if a.w_dead:
        import json as _json
        _ja = HERE / "cache/mouth_authority.json"
        _d = _json.load(open(_ja))
        _auth = np.zeros(263, np.float32)
        _auth[np.array(_d["ctrl_idx"], np.int32)] = _d["authority_face_mm"]
        _ref = float(np.median(_auth[_auth > 1e-3]))
        _pen = 1.0 / (1.0 + _auth / max(_ref, 1e-6))
        PEN = torch.as_tensor(_pen, dtype=torch.float32, device=dev)
        print(f"  authority-weighted prior: penalty 1.00 on the "
              f"{int((_auth < 1e-3).sum())} controls the face cannot see, "
              f"{_pen[_auth > _ref].mean():.3f} on the ones above median authority")

    def solve(tgt):
        """tgt [T, M, 3] on device -> (residual [T,M,3], c [T,263]), both numpy.

        TWO PRIORS, AND WHY THEY ARE NOT OPTIONAL
            The tracker emits 100 FLAME expression coefficients plus a 3-axis jaw. We fit
            251 controls to that, so roughly 150 directions of control space are
            unconstrained: many settings make the same face and an unregularised fit picks
            among them arbitrarily. Measured on the unregularised solve, the controls it
            moved MOST were neckThroatExhale, neckThroatUp and neckSwallowPh4 -- throat
            muscles, on a mesh that barely contains a throat -- while jawOpen ranked 100th
            of 263.

            That costs nothing in reconstruction. It costs everything downstream: a ridge
            probe predicts xADA's own jawOpen from audio at R2 +0.66 and this solve's
            jawOpen at +0.075, with most other channels NEGATIVE. The curves are a valid
            solution and an unlearnable target, and the model's affine initialisation is
            fitted onto them.

            mag  pulls every control toward rest, so the unobservable directions stop
                 absorbing error and the observable ones carry the motion.
            tmp  penalises frame-to-frame change, so consecutive frames stop hopping
                 between equivalent settings. The unregularised jawOpen jumps 1.4x more
                 per frame than xADA's, relative to its own range.

            Both are weak by construction: with ~150 spare directions the fit gives them up
            almost for free, so the residual should barely move. If it moves a lot, the
            weight is too high and the prior is eating signal rather than degeneracy.
        """
        T = tgt.shape[0]
        R = np.empty((T, M, 3), np.float32)
        C = np.empty((T, 263), np.float32)
        for s in range(0, T, a.chunk_frames):
            e = min(s + a.chunk_frames, T)
            y = tgt[s:e]
            c = torch.zeros(e - s, 263, device=dev, requires_grad=True)
            opt = torch.optim.Adam([c], lr=a.lr)
            for _ in range(a.iters):
                opt.zero_grad(set_to_none=True)
                loss = ((head(c) - V0[None] - y) ** 2).sum(-1).mean()
                if a.w_mag:
                    loss = loss + a.w_mag * (c ** 2).mean()
                if a.w_dead:
                    loss = loss + a.w_dead * ((c ** 2) * PEN[None]).mean()
                if a.w_tmp and c.shape[0] > 1:
                    loss = loss + a.w_tmp * ((c[1:] - c[:-1]) ** 2).mean()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    c.clamp_(0.0, 1.0)
            with torch.no_grad():
                R[s:e] = (head(c) - V0[None] - y).cpu().numpy()
                C[s:e] = c.cpu().numpy()
        return R, C

    outdir = pathlib.Path(a.out or HERE / f"cache/solve_{a.subject}")
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "_correspondence.npz", mask=corr.mask,
                        facial=corr.facial, vt=corr.vt, dist=corr.dist,
                        neutral=ref,
                        **{f"region_{k}": v for k, v in corr.regions().items()})

    t0 = time.time()
    done = []
    for n, item in enumerate(index):
        dst = outdir / f"{item['chunk']}.npz"
        if dst.exists():
            done.append(item)
            continue
        z = np.load(item["file"])
        dF = corr.target(z["verts"], neutral=neu[item["video"]])
        tgt = torch.as_tensor(dF, device=dev)
        R, C = solve(tgt)
        dV = dF + R                                   # what the rig actually made
        np.savez(dst, c_fit=C, R=R.astype(np.float16),
                 dFn=np.linalg.norm(dF, axis=-1).astype(np.float16),
                 dVn=np.linalg.norm(dV, axis=-1).astype(np.float16),
                 video=item["video"], frames=z["frames"])
        got = 100 * (1 - np.linalg.norm(R, axis=-1).mean()
                     / np.linalg.norm(dF, axis=-1).mean())
        el = time.time() - t0
        print(f"  [{n+1}/{len(index)}] {item['chunk']:<62} {len(dF):>4}f "
              f"motion {np.linalg.norm(dF,axis=-1).mean():.4f} cm  ceiling {got:5.1f}%"
              f"   {el/60:.1f} min", flush=True)
        done.append(item)

    (outdir / "index.json").write_text(json.dumps(done, indent=1))
    print("wrote", outdir)


if __name__ == "__main__":
    main()
