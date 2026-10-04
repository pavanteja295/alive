#!/usr/bin/env python3
"""Step 6. The demonstration: what the rig cannot do, and what the added shapes fix.

    ~/miniconda3/envs/stavatar/bin/python rigfit/demo_video.py --chunk <name> --k 16

FIVE PANELS, LEFT TO RIGHT
    1  the video frame                   what actually happened
    2  the tracked surface               what a photometric tracker fitted to it
    3  the rig, sliders solved           the best the rig as shipped can do against 2
    4  the rig + the corrective layer    the same sliders, richer rig
    5  what is still wrong               panel 4 against panel 2, painted on the face

Panels 3 and 4 differ ONLY in the rig. Same person, same identity, same slider values,
same frame. Any difference between them is the added shapes and nothing else.

The layer shown is fitted WITHOUT this chunk. What is on screen is generalisation, not
a reconstruction of frames the fit already saw.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PIPE / "offset"))
IMGROOT = (pathlib.Path(__file__).resolve().parents[1] / "vhap/data/monocular")
LABEL_H = 24


def normals(V, F):
    """Per-vertex normals, recomputed every frame so the shading follows the expression."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


def densify(V, N, F, col=None):
    """Vertices plus face centroids and edge midpoints.

    A splat of the 24k vertices alone leaves gaps at this zoom and the face comes out
    stippled. Sampling each triangle at its centre and edge midpoints as well quadruples
    the sample count for almost no cost and gives a solid surface.
    """
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    parts = [V, V[F].mean(1), V[e].mean(1)]
    nrm = [N, N[F].mean(1), N[e].mean(1)]
    out = [np.concatenate(parts), np.concatenate(nrm)]
    if col is not None:
        out.append(np.concatenate([col, col[F].mean(1), col[e].mean(1)]))
    return out


def raster(V, N, W, H, ctr, span, pt=5, colour=None, ss=3):
    """Shaded point splat with a depth buffer and back-face culling.

    Points are lit by their own normal, which is what makes a fold in the cheek or a
    tightening lip legible; a depth ramp alone washes the face out. Vertices facing away
    are dropped, otherwise the back of the skull draws over the face.
    """
    # supersample, then box down. The head has ~24k vertices against a few hundred
    # pixels, so splatting at output resolution beats a moire pattern into the surface;
    # rendering large and shrinking gives a solid, readable face.
    W, H = W * ss, H * ss
    front = N[:, 2] > -0.05
    V, N = V[front], N[front]
    col = colour[front] if colour is not None else None
    x = ((V[:, 0] - ctr[0]) / span + 0.5) * W
    y = (0.5 - (V[:, 1] - ctr[1]) / span) * H
    xi, yi = np.round(x).astype(int), np.round(y).astype(int)
    m = (xi >= -pt) & (xi < W + pt) & (yi >= -pt) & (yi < H + pt)
    xi, yi, z, N = xi[m], yi[m], V[m, 2], N[m]
    col = col[m] if col is not None else None
    lam = np.clip(0.30 + 0.70 * (0.55 * N[:, 2] + 0.55 * N[:, 1] + 0.35 * N[:, 0]), 0, 1)
    rgb = np.repeat(lam[:, None], 3, 1) if col is None else col * (0.55 + 0.45 * lam[:, None])
    img = np.zeros((H, W, 3), np.float32)
    zbuf = np.full((H, W), -1e9, np.float32)
    for dy in range(-pt, pt + 1):
        for dx in range(-pt, pt + 1):
            yy = np.clip(yi + dy, 0, H - 1); xx = np.clip(xi + dx, 0, W - 1)
            hit = z > zbuf[yy, xx]
            zbuf[yy[hit], xx[hit]] = z[hit]
            img[yy[hit], xx[hit]] = rgb[hit]
    img = img.reshape(H // ss, ss, W // ss, ss, 3).mean((1, 3))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def heat(v, vmax):
    """0 -> near black, vmax -> bright. Monotone in lightness so it survives greyscale."""
    t = np.clip(v / max(vmax, 1e-9), 0, 1)[:, None]
    stops = np.array([[0.04, 0.05, 0.14], [0.16, 0.30, 0.62],
                      [0.78, 0.28, 0.24], [1.00, 0.90, 0.45]])
    i = np.clip((t * 3).astype(int), 0, 2)
    f = (t * 3 - i)
    return stops[i[:, 0]] * (1 - f) + stops[i[:, 0] + 1] * f


def band(text, W, h=LABEL_H):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (W, h), (18, 21, 26))
    ImageDraw.Draw(im).text((7, 6), text, fill=(205, 212, 222))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="drk")
    ap.add_argument("--chunk", default=None, help="default: the chunk with the most frames")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--size", type=int, default=340)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--fps", type=float, default=8.0,
                    help="the solve is strided, so this is not the source frame rate")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from riglogic_torch import TorchRig
    from capacity import _index, fit_layer, apply_layer
    from personalise import spread_to_full

    d = HERE / f"cache/solve_{a.subject}"
    index = _index(d)
    if a.chunk is None:
        a.chunk = max(index, key=lambda it: it["frames"])["chunk"]
    item = next(it for it in index if it["chunk"] == a.chunk)
    print("chunk", a.chunk, "|", item["frames"], "frames |", item["video"][:40])

    # the layer, fitted on every chunk EXCEPT this one
    Rs, Cs = [], []
    for it in index:
        if it["chunk"] == a.chunk:
            continue
        z = np.load(d / f"{it['chunk']}.npz")
        Rs.append(z["R"].astype(np.float32).reshape(len(z["c_fit"]), -1))
        Cs.append(z["c_fit"])
    m, cbar, W, S = fit_layer(np.concatenate(Rs), np.concatenate(Cs))
    del Rs, Cs
    print(f"layer fitted on {len(index)-1} other chunks, K={a.k}")

    z = np.load(d / f"{a.chunk}.npz")
    corr = np.load(d / "_correspondence.npz")
    C = z["c_fit"][: a.max_frames]
    R = z["R"][: len(C)].astype(np.float32).reshape(len(C), -1)
    dFn = z["dFn"][: len(C)].astype(np.float32)
    add = apply_layer(C, m, cbar, W, S, a.k)                 # [T, M*3]
    after = (R - add).reshape(len(C), -1, 3)
    print(f"  this chunk: {100*(1-np.linalg.norm(R.reshape(len(C),-1,3),axis=-1).mean()/dFn.mean()):.1f}%"
          f" -> {100*(1-np.linalg.norm(after,axis=-1).mean()/dFn.mean()):.1f}% explained")

    # geometry: the rig with and without the layer, plus the tracked surface
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rig = TorchRig(str(PIPE / "identity/subjects" / a.subject / "rig_beltrami.npz"),
                   device=dev)
    scored = np.nonzero(corr["mask"])[0][corr["facial"]]

    def head(c):
        dj, bsw = rig.behaviour(c)
        return rig.deform(0, rig.skin_matrices(dj), bsw)

    with torch.no_grad():
        V0 = head(torch.zeros(1, 263, device=dev))[0].cpu().numpy()
        Vr = np.empty((len(C), V0.shape[0], 3), np.float32)
        for s in range(0, len(C), 64):
            e = min(s + 64, len(C))
            Vr[s:e] = head(torch.as_tensor(C[s:e], device=dev)).cpu().numpy()
    # Both displacement fields are defined only where the tracker constrains the surface.
    # Applied as-is they would tear the mesh at the edge of that region, so they are
    # carried outward by smoothing before being added. This is presentation only: every
    # number quoted is measured on the constrained region alone.
    Fa = np.load(PIPE / "head" / "head_assets.npz")
    fac = Fa["pos_idx"].astype(np.int64)[Fa["faces"].astype(np.int64)]
    nv = V0.shape[0]
    def carry(D, smooth=14):
        out = np.stack([spread_to_full(d.reshape(-1, 3), scored, fac, nv, iters=25)
                        for d in D])
        # the per-vertex field is noisy at the scale of one vertex; a few smoothing
        # passes make the surface readable without moving anything by more than the
        # measurement's own precision
        e = np.concatenate([fac[:, [0, 1]], fac[:, [1, 2]], fac[:, [2, 0]]])
        e = np.concatenate([e, e[:, ::-1]])
        cnt = np.maximum(np.bincount(e[:, 0], minlength=nv), 1)[:, None]
        for _ in range(smooth):
            acc = np.zeros_like(out)
            np.add.at(acc, np.s_[:, e[:, 0]], out[:, e[:, 1]])
            out = 0.4 * out + 0.6 * acc / cnt
        return out
    # Both are SUBTRACTED. R = rig - target, so the target surface is rig - R, and the
    # layer (which predicts R) corrects the rig by being taken off it.
    Vl = Vr - carry(add)                                     # the layer, on the mesh
    Vt = Vr - carry(R)                     # the tracked surface, on the same vertices

    # frame on the face itself: drop the neck and the back of the skull, then fit the
    # view to what is left with a small margin
    keep = (V0[:, 1] >= np.quantile(V0[:, 1], 0.30)) & (V0[:, 2] > np.median(V0[:, 2]))
    F = np.load(PIPE / "head" / "head_assets.npz")
    faces = F["pos_idx"].astype(np.int64)[F["faces"].astype(np.int64)]
    ctr = np.array([V0[keep, 0].mean(), V0[keep, 1].mean(), 0.0])
    span = max(np.ptp(V0[keep, 0]), np.ptp(V0[keep, 1])) * 1.12
    err_before = np.linalg.norm(R.reshape(len(C), -1, 3), axis=-1)
    err_after = np.linalg.norm(after, axis=-1)
    left = V0[scored, 0] < np.median(V0[scored, 0])   # split-face: before | after
    vmax = float(np.percentile(np.linalg.norm(R.reshape(len(C), -1, 3), axis=-1), 92))

    S_ = a.size
    imgdir = IMGROOT / a.chunk / "images"
    # the speaker is one part of a busy frame, so follow the tracked face rather than
    # showing the whole shot
    lmk = IMGROOT / a.chunk / "landmark2d/STAR.npz"
    crop = None
    if lmk.exists():
        L = np.load(lmk)
        pts = np.asarray(L["face_landmark_2d"])[:, :, :2]
        ok = np.asarray(L["bounding_box"])[:, 4] > 0.9
        for i in range(1, len(pts)):
            if not ok[i]:
                pts[i] = pts[i - 1]
        k = 9
        sm = lambda v: np.convolve(np.pad(v, (k, k), mode="edge"),
                                   np.ones(2 * k + 1) / (2 * k + 1), "valid")
        crop = (sm(pts[:, :, 0].mean(1)), sm(pts[:, :, 1].mean(1)),
                sm(2.3 * np.maximum(np.ptp(pts[:, :, 0], 1), np.ptp(pts[:, :, 1], 1))))
    frames = z["frames"][: len(C)]
    tmp = pathlib.Path(tempfile.mkdtemp())
    for i in range(len(C)):
        src = imgdir / f"{int(frames[i]):06d}.jpg"
        if src.exists():
            im = Image.open(src).convert("RGB")
            if crop is not None:
                j = min(int(frames[i]), len(crop[0]) - 1)
                h = float(np.clip(crop[2][j] * im.width, 60, min(im.size))) / 2
                cx = float(np.clip(crop[0][j] * im.width, h, im.width - h))
                cy = float(np.clip(crop[1][j] * im.height, h, im.height - h))
                box = (cx - h, cy - h, cx + h, cy + h)
            else:
                s = min(im.size)
                box = ((im.width - s) / 2, 0, (im.width + s) / 2, s)
            p1 = np.asarray(im.crop([int(v) for v in box]).resize((S_, S_)))
        else:
            p1 = np.zeros((S_, S_, 3), np.uint8)
        # one face, two halves: the same frame scored before and after the layer, on the
        # same colour scale. Anything the layer fixed shows as the right half going dark.
        ei = np.where(left, err_before[i], err_after[i])
        col = np.full((nv, 3), 0.04)
        col[scored] = heat(ei, vmax)
        dt = densify(Vt[i], normals(Vt[i], faces), faces)
        dr = densify(Vr[i], normals(Vr[i], faces), faces)
        dl = densify(Vl[i], normals(Vl[i], faces), faces, col)
        panels = [(p1, "video frame"),
                  (raster(dt[0], dt[1], S_, S_, ctr, span), "tracked surface"),
                  (raster(dr[0], dr[1], S_, S_, ctr, span), "rig as shipped"),
                  (raster(dl[0], dl[1], S_, S_, ctr, span), f"rig + {a.k} added shapes"),
                  (raster(dl[0], dl[1], S_, S_, ctr, span, colour=dl[2]),
                   "error:  left = shipped | right = fixed")]
        row = np.concatenate([np.concatenate([band(t, S_), p], 0) for p, t in panels], 1)
        Image.fromarray(row).save(tmp / f"f{i:05d}.png")
        if i % 50 == 0:
            print(f"  {i}/{len(C)}", flush=True)

    out = a.out or str(PIPE / f"viz/rigfit/{a.subject}_demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
                    "-i", str(tmp / "f%05d.png"), "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-crf", "20", out], check=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
