#!/usr/bin/env python3
"""Cluster every frame, then look at one montage per cluster and label the clusters.

    python cluster_frames.py --seq <name> --k 28          # cluster + write montages
    python cluster_frames.py --seq <name> --apply verdicts.json

WHY CLUSTER INSTEAD OF THRESHOLD
    The threshold approach asks "what does a bad frame look like" and answers it with a
    number. That went wrong three times on this footage in ways that were only visible by
    looking: a mean-deviation score that was actually measuring his hands, a grayscale
    score blind to translucent colour overlays, a background mask that hid a card sitting
    on his chest. Each fix was found by rendering frames and looking at them.

    Clustering inverts the loop. Group frames that look alike, render one montage per
    group, and decide per GROUP. 22,635 frame decisions collapse to ~28 look-and-label
    decisions, which is a number a person or a model can actually do carefully. Nothing
    has to be right in one pass: a wrong label is one edit to a json file, not a threshold
    hunt.

FEATURES, and why these
    Deliberately cheap and already cached -- the point is not a clever embedding, it is a
    space where "looks alike" means "should be treated alike".

      colour thumbnail   a 12x16 RGB downsample. Carries layout, background, and any
                         graphic covering the frame. This is what separates ads, panels,
                         full-frame shots and a different shoot from each other.
      face geometry      bbox centre, size, and detector confidence, scaled up so a face
                         at x=0.85 cannot land in the same cluster as one at x=0.55 just
                         because the walls match.

    NOT a face-identity embedding, yet. ArcFace via insightface would add "is this him",
    which is the one question the pixels above cannot answer -- it is what the f10209
    different-shoot case needed. Worth adding; not needed to test whether the loop works.
"""
import argparse, json, pathlib
import os
import sys
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


def kmeans(X, k, iters=60, seed=0):
    """Plain k-means++ in numpy. sklearn is not installed in this env and one function is
    cheaper than a dependency."""
    rng = np.random.default_rng(seed)
    C = X[rng.integers(len(X))][None]
    for _ in range(k - 1):
        d = ((X[:, None] - C[None]) ** 2).sum(2).min(1) if len(C) < 8 else \
            np.min(((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]), axis=1)
        d = np.maximum(d, 0)
        p = d / (d.sum() + 1e-12)
        C = np.vstack([C, X[rng.choice(len(X), p=p)]])
    for _ in range(iters):
        a = np.argmin((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None], axis=1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    return a, C


def features(seq, cache, face_w=6.0):
    z = np.load(cache)
    th = z["thumb"]                                     # (n, 36, 64, 3) uint8
    lm = np.load(VHAP / "data/monocular" / seq / "landmark2d/STAR.npz", allow_pickle=True)
    bb = lm["bounding_box"]
    n = len(th)

    col = th[:, ::3, ::4, :].reshape(n, -1).astype(np.float32) / 255.0     # 12x16x3 = 576
    col -= col.mean(0)
    col /= col.std(0) + 1e-6
    col /= np.sqrt(col.shape[1])                        # so colour contributes ~unit norm

    cx = (bb[:, 0] + bb[:, 2]) / 2
    cy = (bb[:, 1] + bb[:, 3]) / 2
    w = bb[:, 2] - bb[:, 0]
    conf = bb[:, 4]
    geo = np.stack([cx, cy, w, conf], 1).astype(np.float32)
    geo = np.nan_to_num(geo)
    geo = (geo - geo.mean(0)) / (geo.std(0) + 1e-6)
    # Weighted up: layout is a small number of dimensions against 576 colour ones, and it
    # is the distinction we most need the clustering to respect.
    return np.hstack([col, geo * face_w]), bb, th


def montage(th, idx, cell=150, cols=8, rows=2):
    rng = np.random.default_rng(0)
    pick = rng.choice(idx, min(cols * rows, len(idx)), replace=False)
    pick = np.sort(pick)
    sheet = Image.new("RGB", (cell * cols, cell * rows), (16, 16, 16))
    for k, i in enumerate(pick):
        im = Image.fromarray(th[i]).resize((cell, cell))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, cell - 1, 12], fill=(0, 0, 0))
        d.text((3, 2), f"f{i}", fill=(180, 255, 180))
        sheet.paste(im, ((k % cols) * cell, (k // cols) * cell))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    _profile_arg(ap)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--k", type=int, default=28)
    ap.add_argument("--apply", help="verdicts.json mapping cluster id -> keep|drop")
    ap.add_argument("--subcluster", type=int, nargs="+",
                    help="re-cluster only these ids, in place, as <id>.0 .1 ...")
    ap.add_argument("--sub-k", type=int, default=4)
    a = ap.parse_args()
    a = _resolve_profile(a, _load_profile(a.profile), {'k': 'CLUSTER_K', 'sub_k': 'SUBCLUSTER_K'})

    root = pathlib.Path(__file__).resolve().parent
    outdir = root / "chunks" / a.seq
    cache = outdir / "signals.npz"
    assert cache.exists(), f"run chunk_take.py first to build {cache}"

    X, bb, th = features(a.seq, cache)
    k_requested = a.k          # a.k is mutated below; the CONFIG must record what was asked
    lab, _ = kmeans(X, a.k)
    if a.subcluster:
        # A mixed cluster is a clustering failure, not a labelling problem, and the fix is
        # NOT to raise k globally -- that reshuffles every id and invalidates every verdict
        # already written. Re-cluster only the offending ids, in place, appending new ids
        # at the end so existing labels keep meaning what they meant.
        nxt = a.k
        for j in a.subcluster:
            idx = np.where(lab == j)[0]
            if len(idx) < a.sub_k * 2:
                continue
            sub, _ = kmeans(X[idx], a.sub_k, seed=j)
            for t in range(a.sub_k):
                m = sub == t
                if not m.any():
                    continue
                lab[idx[m]] = j if t == 0 else nxt
                if t:
                    nxt += 1
            print(f"  subclustered c{j} ({len(idx)} frames) -> {a.sub_k} parts")
        a.k = nxt
    n = len(lab)
    np.save(outdir / "clusters.npy", lab)
    # Record how this clustering was produced. verdicts.json refers to cluster IDS, and a
    # re-run without the same --subcluster arguments renumbers them -- the verdicts then
    # silently point at different frames. Persisting the arguments is what makes the recipe
    # reproducible by someone who was not here when the decision was made.
    json.dump(dict(k=int(k_requested), subcluster=list(a.subcluster or []),
                   sub_k=int(a.sub_k), n_clusters=int(len(set(lab.tolist())))),
              open(outdir / "cluster_config.json", "w"), indent=1)

    cd = outdir / "clusters"
    cd.mkdir(exist_ok=True)
    rows = []
    print(f"{a.seq}: {n} frames -> {a.k} clusters\n")
    print(f"  {'id':>3} {'frames':>7} {'%':>6}  {'cx':>5} {'facew':>6} {'conf':>6}  spans")
    for j in range(a.k):
        idx = np.where(lab == j)[0]
        if not len(idx):
            continue
        cx = np.median((bb[idx, 0] + bb[idx, 2]) / 2)
        fw = np.median(bb[idx, 2] - bb[idx, 0])
        cf = np.median(bb[idx, 4])
        # contiguous spans, so a cluster that is one scene reads differently from one
        # scattered across the whole video
        brk = np.where(np.diff(idx) > 15)[0]
        spans = len(brk) + 1
        montage(th, idx).save(cd / f"c{j:02d}.jpg", quality=88)
        rows.append(dict(cluster=int(j), frames=int(len(idx)),
                         pct=round(len(idx) / n * 100, 2), cx=round(float(cx), 3),
                         face_w=round(float(fw), 3), conf=round(float(cf), 2),
                         spans=int(spans),
                         example=[int(x) for x in idx[::max(1, len(idx) // 4)][:4]]))
        print(f"  {j:>3} {len(idx):>7} {len(idx)/n*100:>5.1f}%  {cx:>5.2f} {fw:>6.3f} "
              f"{cf:>6.2f}  {spans}")
    json.dump(rows, open(outdir / "clusters.json", "w"), indent=1)
    print(f"\nmontages -> {cd}")
    print(f"summary  -> {outdir/'clusters.json'}")

    if a.apply:
        v = json.load(open(a.apply))
        keep = np.zeros(n, bool)
        for j, d in v.items():
            # verdict files carry "_" and other underscore keys as prose commentary; the
            # reasoning belongs with the labels, so skip them rather than banning them
            if not isinstance(d, dict) or not j.lstrip("-").isdigit():
                continue
            if d.get("verdict") == "keep":
                keep |= lab == int(j)
        np.save(outdir / "frame_keep.npy", keep)
        print(f"\napplied: keep {keep.sum()} / {n} frames ({keep.mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
