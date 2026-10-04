#!/usr/bin/env python3
"""Step 4. Fit ONE corrective layer for the person, over every chunk at once, and write
it out as a reusable artefact.

    ~/miniconda3/envs/stavatar/bin/python rigfit/personalise.py --subject drk --k 16

This is the deliverable, not a measurement. capacity.py asks whether a layer generalises;
this builds the single layer that all the takes agree on and saves it so the rig can be
loaded with it. The held-out arm here is whole videos, because "good on all takes" is the
property being claimed.

WHAT COMES OUT
    corrective_<subject>_k<K>.npz
        m      [Nv, 3]     the fixed correction to the rest face
        B      [263, Nv*3] slider -> added displacement, already rank-reduced to K
        cbar   [263]       the slider values the layer is centred on
        basis  [K, Nv*3]   the K shapes on their own, for inspection and for plots
        cover  [Nv]        which vertices were actually observed, and how strongly

    The layer is one matrix multiply added after the existing rig:

        V(c)  ->  V(c) + m + (c - cbar) @ B

    It is linear, so it is differentiable in both c and its own parameters, which is
    what makes the next step -- refitting m and B against rendered pixels instead of
    against a tracked mesh -- a change of loss rather than a change of model.

THE HONEST EDGE
    Only the vertices the tracker actually constrains get a fitted value: the scored set
    is the part of the head that has a trustworthy FLAME counterpart. Everything else --
    the back of the skull, inside the mouth, the eyelids under the gate -- is filled by
    smoothing outward from the scored boundary so the surface has no crack, and is
    marked in `cover` as unobserved. Those vertices are carried, not measured, and any
    use of this layer should treat them as such.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def _index(d):
    """The chunks actually solved. Reads the run's index if it finished, otherwise the
    files on disk, so the study can be run against a solve still in progress."""
    f = d / "index.json"
    if f.exists():
        return json.load(open(f))
    out = []
    for p in sorted(d.glob("*.npz")):
        if p.name.startswith("_"):
            continue
        out.append({"chunk": p.stem, "video": str(np.load(p)["video"]),
                    "frames": int(np.load(p)["c_fit"].shape[0])})
    return out
PIPE = HERE.parent
sys.path.insert(0, str(HERE))


def stream(d, index, key="R"):
    for it in index:
        z = np.load(d / f"{it['chunk']}.npz")
        yield it, z


def fit(d, index, keep, ridge, K):
    """Streaming reduced-rank regression of the leftover on the sliders."""
    P = 263
    G = np.zeros((P, P))
    csum = np.zeros(P)
    H = None
    msum = None
    N = 0
    for it, z in stream(d, index):
        if it["chunk"] not in keep:
            continue
        C = z["c_fit"].astype(np.float64)
        R = z["R"].astype(np.float32).reshape(len(C), -1).astype(np.float64)
        if H is None:
            H = np.zeros((P, R.shape[1]))
            msum = np.zeros(R.shape[1])
        G += C.T @ C
        csum += C.sum(0)
        H += C.T @ R
        msum += R.sum(0)
        N += len(C)
    cbar = csum / N
    m = msum / N
    # centred normal equations, from the raw sums
    Gc = G - N * np.outer(cbar, cbar)
    Hc = H - N * np.outer(cbar, m)
    W = np.linalg.solve(Gc + ridge * np.eye(P), Hc)
    w, V = np.linalg.eigh(Gc + max(1e-12, 1e-9 * np.trace(Gc) / P) * np.eye(P))
    L = V * np.sqrt(np.clip(w, 0, None))                     # L L^T = Gc
    S = np.linalg.svd(L.T @ W, full_matrices=False)[2][:K]
    return m, cbar, W @ S.T @ S, S, N


def score(d, index, which, m, cbar, B, m_by_video=None):
    """mean residual length before and after the layer, per chunk."""
    out = {}
    for it, z in stream(d, index):
        if it["chunk"] not in which:
            continue
        C = z["c_fit"].astype(np.float64)
        R = z["R"].astype(np.float32).reshape(len(C), -1)
        mm = m if m_by_video is None else m_by_video[it["video"]]
        A = (R - (mm + (C - cbar) @ B)).reshape(len(C), -1, 3)
        out[it["chunk"]] = (float(z["dFn"].astype(np.float32).mean()),
                            float(np.linalg.norm(R.reshape(len(C), -1, 3), axis=-1).mean()),
                            float(np.linalg.norm(A, axis=-1).mean()), it["video"])
    return out


def rest_face(d, index, which, cbar, B):
    """The mean leftover of a set of clips, after the shape part is removed.

    Used to give a held-out recording its own rest-face correction while the shapes stay
    shared. It uses only that recording's own frames and no target from them beyond their
    average, which is the weakest thing that can be called a per-recording fit.
    """
    tot, n = None, 0
    for it, z in stream(d, index):
        if it["chunk"] not in which:
            continue
        C = z["c_fit"].astype(np.float64)
        R = z["R"].astype(np.float32).reshape(len(C), -1).astype(np.float64)
        r = (R - (C - cbar) @ B).sum(0)
        tot = r if tot is None else tot + r
        n += len(C)
    return tot / n


def spread_to_full(vals, scored_idx, faces, nv, iters=60):
    """Carry the fitted field off the scored patch so the mesh has no discontinuity.

    Umbrella smoothing with the scored vertices held fixed. It is a fill, not a fit:
    nothing outside the scored set was observed, and `cover` records that.
    """
    full = np.zeros((nv,) + vals.shape[1:], np.float64)
    full[scored_idx] = vals
    fixed = np.zeros(nv, bool)
    fixed[scored_idx] = True
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    order = np.argsort(e[:, 0], kind="stable")
    src, dst = e[order, 0], e[order, 1]
    cnt = np.bincount(src, minlength=nv).astype(np.float64)
    cnt[cnt == 0] = 1
    for _ in range(iters):
        acc = np.zeros_like(full)
        np.add.at(acc, src, full[dst])
        nxt = acc / cnt[:, None]
        full[~fixed] = nxt[~fixed]
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--holdout", default="videos", choices=("videos", "none"))
    ap.add_argument("--per-video-static", action="store_true",
                    help="give each recording its own rest-face correction and share only "
                         "the SHAPES. The tracker re-solves identity per recording and "
                         "those identities differ by more than the rig's own error, so a "
                         "single shared rest-face term is partly absorbing that drift. "
                         "This separates the two: it asks whether the shapes transfer, "
                         "given the rest face is allowed to be per-recording.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = HERE / f"cache/solve_{a.subject}"
    index = _index(d)
    corr = np.load(d / "_correspondence.npz")
    mask, facial = corr["mask"], corr["facial"]
    scored = np.nonzero(mask)[0][facial]                 # head vertex ids, [M]
    vids = sorted({it["video"] for it in index})
    print(f"{a.subject}: {len(index)} chunks, {len(vids)} videos, {len(scored)} scored "
          f"vertices, K={a.k}")

    # ---- leave-one-video-out, the claim being "one layer, good on every take" -------
    summary = {"subject": a.subject, "k": a.k, "chunks": len(index), "videos": vids,
               "leave_one_video_out": {}}
    if a.holdout == "videos" and len(vids) > 1:
        print(f"\n{'held-out video':<44}{'chunks':>7}{'before':>9}{'after':>9}{'gain':>8}")
        for v in vids:
            tr = {it["chunk"] for it in index if it["video"] != v}
            te = {it["chunk"] for it in index if it["video"] == v}
            m, cbar, B, _, n = fit(d, index, tr, a.ridge, a.k)
            mbv = None
            if a.per_video_static:
                # the held-out recording keeps its own rest-face term, measured on its
                # own frames; only the SHAPES come from the other recordings
                mbv = {v: m for v in vids}
                mbv[v] = rest_face(d, index, te, cbar, B)
            s = score(d, index, te, m, cbar, B, mbv)
            den = np.mean([x[0] for x in s.values()])
            b = np.mean([x[1] for x in s.values()])
            af = np.mean([x[2] for x in s.values()])
            print(f"{v[:42]:<44}{len(te):>7}{100*(1-b/den):>8.1f}%"
                  f"{100*(1-af/den):>8.1f}%{100*(b-af)/den:>7.1f}")
            summary["leave_one_video_out"][v] = {
                "chunks": len(te), "before_pct": 100 * (1 - b / den),
                "after_pct": 100 * (1 - af / den),
                "per_chunk_after": {k: 100 * (1 - x[2] / x[0]) for k, x in s.items()}}

    # ---- the layer everything agrees on --------------------------------------------
    allc = {it["chunk"] for it in index}
    m, cbar, B, S, N = fit(d, index, allc, a.ridge, a.k)
    s = score(d, index, allc, m, cbar, B)
    den = np.mean([x[0] for x in s.values()])
    b = np.mean([x[1] for x in s.values()])
    af = np.mean([x[2] for x in s.values()])
    print(f"\nfitted on all {N} frames: {100*(1-b/den):.1f}% -> {100*(1-af/den):.1f}% "
          f"explained (in-sample; the held-out numbers above are the claim)")

    # ---- carry it to the whole head and write it out --------------------------------
    H = np.load(PIPE / "head" / "head_assets.npz")
    faces = H["pos_idx"].astype(np.int64)[H["faces"].astype(np.int64)]
    nv = int(mask.shape[0])
    m_full = spread_to_full(m.reshape(-1, 3), scored, faces, nv)
    B_full = np.zeros((263, nv, 3))
    for k in range(263):
        B_full[k] = spread_to_full(B[k].reshape(-1, 3), scored, faces, nv, iters=40)
    cover = np.zeros(nv, np.float32)
    cover[scored] = 1.0

    out = pathlib.Path(a.out or HERE / f"cache/corrective_{a.subject}_k{a.k}.npz")
    np.savez_compressed(out, m=m_full.astype(np.float32),
                        B=B_full.reshape(263, -1).astype(np.float32),
                        cbar=cbar.astype(np.float32),
                        basis=S.astype(np.float32), cover=cover, scored=scored,
                        subject=a.subject, k=a.k,
                        per_chunk=json.dumps({k: v for k, v in s.items()}))
    summary["in_sample"] = {"before_pct": 100 * (1 - b / den), "after_pct": 100 * (1 - af / den)}
    summary["rest_face_mm"] = {
        "mean": float(np.linalg.norm(m.reshape(-1, 3), axis=1).mean() * 10),
        "max": float(np.linalg.norm(m.reshape(-1, 3), axis=1).max() * 10)}
    print("wrote", out)
    print(f"  correction to the rest face: mean {np.linalg.norm(m.reshape(-1,3),axis=1).mean()*10:.3f} mm"
          f"  max {np.linalg.norm(m.reshape(-1,3),axis=1).max()*10:.3f} mm")
    amp = np.linalg.norm(B.reshape(263, -1, 3), axis=-1).mean(1)
    top = np.argsort(-amp)[:8]
    names = np.load(PIPE / "offset/cache/rig_names.npz")
    rn = [str(x) for x in names["raw_names"]]
    print("  sliders that move the layer most:")
    summary["top_sliders"] = [{"name": rn[i], "mm_per_unit": float(amp[i] * 10)}
                              for i in top]
    for i in top:
        print(f"    {rn[i][:44]:<46}{amp[i]*10:.3f} mm per unit")
    js = out.with_suffix(".json")
    js.write_text(json.dumps(summary, indent=1))
    print("wrote", js)


if __name__ == "__main__":
    main()
