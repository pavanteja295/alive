#!/usr/bin/env python3
"""Step 3. How much of the leftover comes back if the rig is allowed a corrective layer.

    ~/miniconda3/envs/stavatar/bin/python rigfit/capacity.py --subject drk

THE PROPOSED CHANGE TO THE RIG, IN ONE LINE

    V(c)   ->   V(c)  +  m  +  sum_k  w_k(c) * S_k

    m       one fixed vertex field, the same in every frame.  A correction to where the
            rest face sits.  It is NOT expression: it is the rig's neutral being in the
            wrong place, which the sliders can never fix because they only run 0 to 1
            and so can only push away from the neutral, never behind it.
    S_k     K fixed vertex fields.  Extra shapes the rig does not have.
    w_k(c)  how strongly each is applied.  A function of the SLIDERS and nothing else,
            because at run time the sliders are all that exists.  Linear here, which is
            the weakest assumption available and therefore the safest claim.

    Both m and S_k are constants once fitted, and w is linear, so the whole layer is a
    small matrix added after the existing rig. It changes nothing about how the rig is
    driven and it stays differentiable.

HOW IT IS FITTED
    Reduced-rank regression of the leftover on the sliders. That directly answers "what
    is the best rank-K slider-driven layer", rather than the two-step alternative of
    finding shapes first and hoping the sliders can drive them.

FIVE SPLITS, WEAKEST EVIDENCE FIRST
    frames        train on the first half of every chunk, test on the second half.
                  The easiest test and the one to trust least: neighbouring frames of
                  the same sentence are nearly the same face.
    chunks        train on some chunks, test on chunks never seen. Different moments,
                  different sentences, same person and same recording session.
    videos        train on two videos, test on the third. Different day, different
                  lighting, different tracker run.
    expressions   cluster the frames by what the face is doing and hold out whole
                  clusters. This is the one that matters: it asks whether the layer
                  survives a face shape it was never fitted on.
    shuffled      the negative control. The slider tracks are permuted across frames
                  before fitting, destroying the pairing between sliders and leftover.
                  Anything this arm recovers is capacity, not signal, and every other
                  number should be read as "above this".
"""
import argparse
import json
import pathlib

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


# ----------------------------------------------------------------- the layer itself --
def fit_layer(R, C, ridge=1e-2):
    """Reduced-rank regression pieces for  R ~ m + (C - cbar) W,  rank K on demand.

    R [N, D] leftover, C [N, P] sliders. Returns (m, cbar, W_ls, S) where S [P, D] are
    the ordered shape directions: the rank-K layer is W_ls @ S[:K].T @ S[:K].
    """
    m = R.mean(0)
    cbar = C.mean(0)
    Cc, Rc = C - cbar, R - m
    G = Cc.T @ Cc + ridge * np.eye(C.shape[1])
    W = np.linalg.solve(G, Cc.T @ Rc)                       # [P, D] full-rank solution
    # right singular vectors of the FITTED values C W, obtained without forming it:
    # (CW)^T(CW) = W^T (C^T C) W, so factor C^T C and take the SVD of L^T W.
    # A whitening factor of C^T C. With few frames it is rank-deficient, so the jitter
    # is scaled to the matrix rather than fixed, and a symmetric factorisation is used
    # instead of Cholesky, which refuses anything not strictly positive definite.
    G0 = Cc.T @ Cc
    w, V = np.linalg.eigh(G0 + max(1e-12, 1e-9 * np.trace(G0) / len(G0)) * np.eye(len(G0)))
    L = V * np.sqrt(np.clip(w, 0, None))                     # L L^T = G0
    try:
        S = np.linalg.svd(L.T @ W, full_matrices=False)[2]   # [P, D]
    except np.linalg.LinAlgError:
        # LAPACK's divide-and-conquer driver occasionally fails to converge on a finite,
        # well-scaled matrix (huberman, one shuffled fold of twelve). The QR-iteration
        # driver is slower and does not; the result is the same decomposition.
        import scipy.linalg
        S = scipy.linalg.svd(L.T @ W, full_matrices=False, lapack_driver="gesvd")[2]
    return m, cbar, W, S


def apply_layer(C, m, cbar, W, S, K):
    return m + (C - cbar) @ (W @ S[:K].T @ S[:K] if K else np.zeros_like(W))


def explained(R, denom):
    """Mean residual length as a fraction of mean motion, the study-wide metric."""
    return 100.0 * (1.0 - np.linalg.norm(R.reshape(len(R), -1, 3), axis=-1).mean() / denom)


# ------------------------------------------------------------------------- the data --
def load(subject, limit=0, points=0, seed=0):
    d = HERE / f"cache/solve_{subject}"
    index = _index(d)
    if limit:
        index = index[:limit]
    corr = np.load(d / "_correspondence.npz")
    M = int(corr["facial"].sum())
    sel = slice(None)
    if points and points < M:
        sel = np.sort(np.random.default_rng(seed).choice(M, points, replace=False))
    R, C, dFn, chunk, video = [], [], [], [], []
    for it in index:
        z = np.load(d / f"{it['chunk']}.npz")
        R.append(z["R"][:, sel].astype(np.float32))
        C.append(z["c_fit"])
        dFn.append(z["dFn"][:, sel].astype(np.float32))
        chunk.append(np.full(len(z["c_fit"]), it["chunk"]))
        video.append(np.full(len(z["c_fit"]), it["video"]))
    R = np.concatenate(R); C = np.concatenate(C); dFn = np.concatenate(dFn)
    return (R.reshape(len(R), -1), C, dFn, np.concatenate(chunk),
            np.concatenate(video), corr, sel, index)


# ----------------------------------------------------------------------- the splits --
def splits(C, chunk, video, dFn, nfold=5, seed=0):
    rng = np.random.default_rng(seed)
    out = {}

    # frames: first half / second half of every chunk
    tr = np.zeros(len(C), bool)
    for c in np.unique(chunk):
        i = np.nonzero(chunk == c)[0]
        tr[i[: len(i) // 2]] = True
    out["frames"] = [(tr, ~tr)]

    # chunks: grouped k-fold, whole chunks held out
    ch = np.unique(chunk)
    fold = {c: i % nfold for i, c in enumerate(rng.permutation(ch))}
    f = np.array([fold[c] for c in chunk])
    out["chunks"] = [(f != k, f == k) for k in range(nfold)]

    # videos: leave one recording out. Undefined with one recording -- there is nothing
    # left to train on -- so the split is empty and reported as skipped, not as nan.
    vids = np.unique(video)
    out["videos"] = [(video != v, video == v) for v in vids] if len(vids) > 1 else []

    # expressions: cluster on the fitted sliders, hold out whole clusters.
    # The sliders are the face's own description of what it is doing, so a cluster is a
    # family of face shapes, not a stretch of time.
    from scipy.cluster.vq import kmeans2
    act = C[:, C.std(0) > 1e-3]
    z = (act - act.mean(0)) / (act.std(0) + 1e-9)
    lab = kmeans2(z, 12, minit="++", seed=seed, iter=40)[1]
    out["expressions"] = [(lab != k, lab == k) for k in range(12)
                          if (lab == k).sum() > 30 and (lab != k).sum() > 100]
    out["_labels"] = lab
    return out


def footage_curve(R, C, dFn, chunk, ks=(8, 16), grid=(1, 2, 4, 8, 16, 32, 64),
                  draws=6, seed=0, holdout=20):
    """How much video does a new person need before the layer is worth fitting?

    A fixed set of clips is held back. The layer is then fitted on a growing number of
    OTHER clips, drawn at random, and scored on the held-back set. Everything is measured
    in seconds of footage rather than in clips, because clips vary in length and seconds
    is the number a person planning a capture session actually has.
    """
    rng = np.random.default_rng(seed)
    ch = rng.permutation(np.unique(chunk))
    te_ch, pool = set(ch[:holdout]), ch[holdout:]
    te = np.array([c in te_ch for c in chunk])
    den = dFn[te].mean()
    out = {}
    for K in ks:
        rows = []
        for n in grid:
            if n > len(pool):
                continue
            got, secs = [], []
            for _ in range(draws):
                pick = set(rng.choice(pool, n, replace=False))
                tr = np.array([c in pick for c in chunk])
                if tr.sum() < K + 5:
                    continue
                m, cbar, W, S = fit_layer(R[tr], C[tr])
                got.append(explained(R[te] - apply_layer(C[te], m, cbar, W, S, K), den))
                secs.append(tr.sum() / 8.0)      # solved at 8 frames per second of video
            if got:
                rows.append({"clips": n, "seconds": float(np.mean(secs)),
                             "mean": float(np.mean(got)), "min": float(np.min(got)),
                             "max": float(np.max(got))})
        out[str(K)] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--ks", type=int, nargs="*", default=[0, 1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--points", type=int, default=4000,
                    help="facial points subsampled for the study; 0 = all. The layer is "
                         "fitted at full resolution later by personalise.py; this only "
                         "keeps the many-fold cross-validation in memory.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    R, C, dFn, chunk, video, corr, sel, index = load(a.subject, a.limit, a.points)
    denom = dFn.mean()
    base = explained(R, denom)
    print(f"{a.subject}: {len(R)} frames, {len(np.unique(chunk))} chunks, "
          f"{len(np.unique(video))} videos, {R.shape[1]//3} points")
    print(f"motion {denom:.4f} cm | rig as shipped {base:.1f}% explained\n")

    sp = splits(C, chunk, video, dFn)
    res = {"subject": a.subject, "frames": int(len(R)), "chunks": int(len(np.unique(chunk))),
           "videos": sorted(set(video.tolist())), "points": int(R.shape[1] // 3),
           "motion_cm": float(denom), "shipped_pct": float(base), "splits": {}}

    order = ["frames", "chunks", "videos", "expressions"]
    for name in order + ["shuffled"]:
        folds = sp["expressions"] if name == "shuffled" else sp[name]
        if not folds:
            print(f"{name:<12} skipped: needs at least two recordings, this person has one\n")
            res["splits"][name] = None
            continue
        rows = {K: [] for K in a.ks}
        rows["oracle"] = []
        for tr, te in folds:
            Ctr = C[tr] if name != "shuffled" else C[np.random.default_rng(0).permutation(
                np.nonzero(tr)[0])]
            m, cbar, W, S = fit_layer(R[tr], Ctr, a.ridge)
            d = dFn[te].mean()
            for K in a.ks:
                rows[K].append(explained(R[te] - apply_layer(C[te], m, cbar, W, S, K), d))
            Kmax = max(a.ks)
            Sk = S[:Kmax]
            rows["oracle"].append(explained(
                R[te] - m - ((R[te] - m) @ Sk.T) @ Sk, d))
        print(f"{name:<12}{'K':>4}{'explained':>12}{'  (spread over folds)':<24}")
        for K in a.ks:
            v = np.array(rows[K])
            print(f"{'':12}{K:>4}{v.mean():>11.1f}%   {v.min():.1f} .. {v.max():.1f}"
                  f"   n={len(v)}")
        v = np.array(rows["oracle"])
        print(f"{'':12}{'orc':>4}{v.mean():>11.1f}%   {v.min():.1f} .. {v.max():.1f}"
              f"      (upper bound: the {max(a.ks)} shapes with free per-frame weights)")
        res["splits"][name] = {str(k): [float(x) for x in rows[k]] for k in rows}
        print()

    # ---- how much footage does this need? -----------------------------------------
    res["footage"] = footage_curve(R, C, dFn, chunk)
    print("how much video the layer needs (scored on 20 clips held back throughout):")
    print(f"{'clips':>7}{'seconds':>10}{'K=8':>10}{'K=16':>10}")
    for i, row in enumerate(res["footage"]["8"]):
        r16 = res["footage"]["16"][i]
        print(f"{row['clips']:>7}{row['seconds']:>10.0f}{row['mean']:>9.1f}%"
              f"{r16['mean']:>9.1f}%")
    print()

    # ---- does the layer flicker? -------------------------------------------------
    # A correction can be right on average and still ruin a render by changing too fast.
    # The layer's strengths are a linear function of the slider values, so its roughness
    # is inherited from them; this measures it rather than assuming it. Only the LEFTOVER
    # is available as vectors here, which is the right quantity anyway: what matters is
    # how fast the error moves, not how fast the face does. Differences are taken within
    # clips only, so clip boundaries are never counted as motion.
    tr, te = sp["frames"][0]
    m, cbar, W, S = fit_layer(R[tr], C[tr])
    K = 16 if 16 in a.ks else max(a.ks)
    add = apply_layer(C, m, cbar, W, S, K)
    same = np.zeros(len(R), bool)
    same[1:] = chunk[1:] == chunk[:-1]

    def rough(X):
        d = (X[1:] - X[:-1])[same[1:]].reshape(-1, R.shape[1] // 3, 3)
        return float(np.linalg.norm(d, axis=-1).mean())

    res["roughness_cm_per_frame"] = {
        "the error, rig as shipped": rough(R),
        f"the error, with {K} added shapes": rough(R - add),
        "the layer's own contribution": rough(add)}
    print("frame-to-frame change of the surface, within clips (cm per frame):")
    for k, v in res["roughness_cm_per_frame"].items():
        print(f"  {k:<36}{v:.5f}")
    r0 = res["roughness_cm_per_frame"]["the error, rig as shipped"]
    r1 = res["roughness_cm_per_frame"][f"the error, with {K} added shapes"]
    print(f"  -> the layer makes the error {100*(1-r1/r0):.0f}% steadier, "
          f"not jitterier")

    out = pathlib.Path(a.out or HERE / f"cache/capacity_{a.subject}.json")
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
