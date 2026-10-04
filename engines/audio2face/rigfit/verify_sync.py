#!/usr/bin/env python3
"""Show the whole synchronisation chain at once, with sound, so it can be judged by eye.

    ~/miniconda3/envs/stavatar/bin/python rigfit/verify_sync.py --n 6

Five joins have to be right and none of them is self-evidently right:

    audio  <->  source video      the container; assumed, and checked here by ear
    source video <-> chunk        sequences.json's src_range
    chunk  <->  FLAME target      the harvest's frame list
    FLAME target <-> controls     the per-frame solve
    audio  <->  controls          the measured driver lag, 110-115 ms

FOUR PANELS, THE SAME INSTANT IN EACH

    1  the video frame                what happened
    2  the tracker's FLAME mesh       what the tracker fitted to it, in the RIG's space
    3  rig + layer, solved controls   the target the model is trained to predict
    4  rig, driven by the audio       what the shipped driver does at that sound

Panels 1-3 must agree instant for instant. If they do not, the fault is upstream of any
model. Panel 4 against 1 is the audio join: it may be WRONG in content -- that is the
whole problem being worked on -- but it must be wrong at the RIGHT TIME. A mouth that
opens half a second late is a sync bug; a mouth that opens too little is not.

Everything is indexed exactly as rigfit/train_offset2.py indexes it.
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
IMG = (pathlib.Path(__file__).resolve().parents[1] / "vhap/data/monocular")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="how many clips")
    ap.add_argument("--secs", type=float, default=8.0, help="seconds from each clip")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--sheet", type=int, default=0, help=
                    "also write a contact sheet of this many instants per clip. The "
                    "video is for a person's EARS -- the audio join has to be heard. The "
                    "sheet is for a reader that cannot hear, and carries the three joins "
                    "that are spatial: panels 1-3 must agree instant for instant.")
    ap.add_argument("--profile", required=True,
                    help="recipes/audio-to-mesh/profiles/<name>.py: whose clips, rig and layer")
    ap.add_argument("--outdir", default=None, help="default viz/rigfit/sync_<subject>")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE / "recipes/audio-to-mesh/tools"))
    from _profile import load
    p = load(a.profile)
    SUBJ, IDS = p.SUBJECT, p.RECORDINGS
    a.outdir = a.outdir or str(PIPE / f"viz/rigfit/sync_{SUBJ}")

    import torch
    from PIL import Image
    from riglogic_torch import TorchRig
    from target import Correspondence, similarity
    from demo_video import normals, raster, densify, band
    from scipy.ndimage import gaussian_filter1d

    split = json.load(open(p.SPLIT))
    align = json.load(open(HERE / "cache/align.json"))
    fl = json.load(open(p.FLAME_FULL / "index.json"))
    neu = {}
    for c in fl:
        neu.setdefault(c["video"], np.load(c["file"])["neutral"])
    ref = neu[fl[0]["video"]]
    corr = Correspondence(SUBJ, "rig_beltrami.npz", flame_neutral=ref)

    dev = "cuda"
    rig = TorchRig(str(p.RIG), device=dev)
    lz = np.load(HERE / f"cache/corrective_{SUBJ}_k{p.CORRECTIVE_K}.npz")
    sc = torch.as_tensor(lz["scored"], dtype=torch.long, device=dev)
    L_m = torch.as_tensor(lz["m"], device=dev)[sc]
    L_B = torch.as_tensor(lz["B"], device=dev).view(263, -1, 3)[:, sc]
    L_cb = torch.as_tensor(lz["cbar"], device=dev)
    scored = lz["scored"]
    H = np.load(PIPE / "head/head_assets.npz")
    faces = H["pos_idx"].astype(np.int64)[H["faces"].astype(np.int64)]

    def head(c, layer):
        d, bsw = rig.behaviour(c)
        V = rig.deform(0, rig.skin_matrices(d), bsw)
        if layer:
            V = V.clone()
            V[:, sc] = V[:, sc] - L_m[None] - torch.einsum("bc,cmd->bmd", c - L_cb, L_B)
        return V

    with torch.no_grad():
        V0 = head(torch.zeros(1, 263, device=dev), False)[0].cpu().numpy()
    keep = (V0[:, 1] >= np.quantile(V0[:, 1], 0.30)) & (V0[:, 2] > np.median(V0[:, 2]))
    ctr = np.array([V0[keep, 0].mean(), V0[keep, 1].mean(), 0.0])
    span = max(np.ptp(V0[keep, 0]), np.ptp(V0[keep, 1])) * 1.12

    out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    xada = {v: np.load(HERE / f"cache/xada_{i}.npz") for v, i in IDS.items()}
    picked = [c for c in split["test"] if (HERE / f"cache/solve_{SUBJ}/{c}.npz").exists()][:a.n]

    for clip in picked:
        z = np.load(HERE / f"cache/solve_{SUBJ}/{clip}.npz")
        tz = np.load(HERE / f"cache/targets_{SUBJ}/{clip}.npz")
        fz = np.load(p.FLAME_FULL / f"{clip}.npz")
        vid = str(tz["video"])
        al = align[vid]
        fps = al["fps"]
        t_all = tz["t"].astype(np.float64)                     # seconds into the source
        n = min(int(a.secs * fps), len(t_all))
        # controls exist on the solve's strided frames; carry them to every frame
        k = np.asarray(z["frames"]); k = k[k < len(t_all)]
        C = np.stack([np.interp(np.arange(len(t_all)), k, z["c_fit"][:len(k), j])
                      for j in range(263)], 1)[:n].astype(np.float32)
        # the driver, at the same instants, with the measured lag
        x = xada[vid]
        tt, raw = np.asarray(x["t"]), np.asarray(x["raw"])
        ta = t_all[:n] + al.get("sample_offset_ms", -al["lag_ms"]) / 1000.0
        A = np.stack([np.interp(ta, tt, raw[:, j]) for j in range(263)], 1).astype(np.float32)
        # FLAME, in the rig's space, treated exactly as the target builder treats it
        Vn = neu[vid].astype(np.float64)[:5023]
        sim = similarity(Vn, corr.own)
        Vn_a = sim(Vn)
        from target import procrustes_rigid
        FL = np.empty((n, 5023, 3))
        rows_for_sheet = []
        for i in range(n):
            v = sim(fz["verts"][i].astype(np.float64)[:5023])
            R, tt2 = procrustes_rigid(v[corr.skull], Vn_a[corr.skull])
            FL[i] = corr.to_dna(v @ R + tt2)
        # FLAME is already in the RIG's own space, so panels 2, 3 and 4 share one
        # camera. That is the point: a sync error shows up as the same face at different
        # moments, side by side, on the same framing.
        fctr, fspan = ctr, span
        fF = corr.Ft            # FLAME triangles, minus eyeballs and scalp

        with torch.no_grad():
            Vc = np.concatenate([head(torch.as_tensor(C[s:s+64], device=dev), True)
                                 .cpu().numpy() for s in range(0, n, 64)])
            Va = np.concatenate([head(torch.as_tensor(A[s:s+64], device=dev), False)
                                 .cpu().numpy() for s in range(0, n, 64)])

        # follow the face in the video: he is a small inset whenever slides are up
        S = a.size
        lmk = IMG / clip / "landmark2d/STAR.npz"
        crop = None
        if lmk.exists():
            L = np.load(lmk)
            pts = np.asarray(L["face_landmark_2d"])[:, :, :2]
            ok = np.asarray(L["bounding_box"])[:, 4] > 0.9
            for i in range(1, len(pts)):
                if not ok[i]:
                    pts[i] = pts[i - 1]
            kk = 9
            sm = lambda v: np.convolve(np.pad(v, (kk, kk), mode="edge"),
                                       np.ones(2 * kk + 1) / (2 * kk + 1), "valid")
            crop = (sm(pts[:, :, 0].mean(1)), sm(pts[:, :, 1].mean(1)),
                    sm(2.3 * np.maximum(np.ptp(pts[:, :, 0], 1), np.ptp(pts[:, :, 1], 1))))
        tmp = pathlib.Path(tempfile.mkdtemp())
        for i in range(n):
            src = IMG / clip / "images" / f"{int(round(t_all[i]*fps)) - al['src_range'][clip][0]:06d}.jpg"
            if src.exists():
                im = Image.open(src).convert("RGB")
                ci_ = int(round(t_all[i] * fps)) - al["src_range"][clip][0]
                if crop is not None and 0 <= ci_ < len(crop[0]):
                    h = float(np.clip(crop[2][ci_] * im.width, 60, min(im.size))) / 2
                    cx = float(np.clip(crop[0][ci_] * im.width, h, im.width - h))
                    cy = float(np.clip(crop[1][ci_] * im.height, h, im.height - h))
                    box = (cx - h, cy - h, cx + h, cy + h)
                else:
                    s_ = min(im.size)
                    box = ((im.width - s_) / 2, 0, (im.width + s_) / 2, s_)
                p1 = np.asarray(im.crop([int(v) for v in box]).resize((S, S)))
            else:
                p1 = np.zeros((S, S, 3), np.uint8)
            # FLAME has a fifth of the rig's vertices, so it needs one extra round of
            # subdivision to read as a surface rather than as a dot screen
            nf = normals(FL[i], fF)
            e = np.concatenate([fF[:, [0, 1]], fF[:, [1, 2]], fF[:, [2, 0]]])
            fv = np.concatenate([FL[i], FL[i][fF].mean(1), FL[i][e].mean(1)])
            fn = np.concatenate([nf, nf[fF].mean(1), nf[e].mean(1)])
            mid = (fv[None] if False else None)
            df = (np.concatenate([fv, (FL[i][fF[:, [0, 1]]].mean(1) + FL[i][fF].mean(1)) / 2,
                                  (FL[i][fF[:, [1, 2]]].mean(1) + FL[i][fF].mean(1)) / 2,
                                  (FL[i][fF[:, [2, 0]]].mean(1) + FL[i][fF].mean(1)) / 2]),
                  np.concatenate([fn, nf[fF].mean(1), nf[fF].mean(1), nf[fF].mean(1)]))
            dc = densify(Vc[i], normals(Vc[i], faces), faces)
            da = densify(Va[i], normals(Va[i], faces), faces)
            panels = [(p1, "1  video frame"),
                      (raster(df[0], df[1], S, S, fctr, fspan), "2  tracker (FLAME)"),
                      (raster(dc[0], dc[1], S, S, ctr, span), "3  rig, controls from FLAME"),
                      (raster(da[0], da[1], S, S, ctr, span), "4  rig, driven by audio")]
            row = np.concatenate([np.concatenate([band(t, S), p], 0) for p, t in panels], 1)
            Image.fromarray(row).save(tmp / f"f{i:05d}.png")
            rows_for_sheet.append(row)
        if a.sheet:
            # Evenly spaced instants, stacked. Panels 1-3 must agree in each row;
            # panel 4 may be wrong in content but must be wrong at the right time,
            # and THAT part cannot be judged here -- it needs the video and ears,
            # or the measured correlation peak in align.json.
            pick = np.linspace(0, len(rows_for_sheet) - 1, a.sheet).round().astype(int)
            sheet = np.concatenate([rows_for_sheet[k] for k in pick], 0)
            sp = out / f"{clip}.sheet.png"
            Image.fromarray(sheet).save(sp)
            print(f"  {sp.name}  ({a.sheet} instants)")
        wav = HERE / f"cache/audio_local/{IDS[vid]}.wav"
        dst = out / f"{clip}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-framerate", f"{fps:.4f}", "-i", str(tmp / "f%05d.png"),
                        "-ss", f"{t_all[0]:.3f}", "-t", f"{n/fps:.3f}", "-i", str(wav),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-c:a", "aac", "-shortest", str(dst)], check=True)
        s0 = al["src_range"][clip][0]
        print(f"  {dst.name}")
        print(f"     {n} frames at {fps:.3f} fps | chunk starts at source frame {s0} "
              f"= {s0/fps:.2f} s | driver lag {al['lag_ms']:+.0f} ms")
        for i in (0, n // 2, n - 1):
            print(f"       panel frame {i:>4}  chunk idx {int(round(t_all[i]*fps))-s0:>4}"
                  f"  source frame {int(round(t_all[i]*fps)):>6}"
                  f"  video t {t_all[i]:8.3f}s  audio sampled at "
                  f"{t_all[i]+al['lag_ms']/1000:8.3f}s")
    print("wrote", out)


if __name__ == "__main__":
    main()
