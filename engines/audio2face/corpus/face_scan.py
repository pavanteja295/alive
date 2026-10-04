#!/usr/bin/env python3
"""Detect EVERY face per frame and embed it. Feeds two filters and one accumulator.

    ~/faceid_env/bin/python face_scan.py --seq <name>          # scan, write faces.npz
    ~/faceid_env/bin/python face_scan.py --seq <name> --report # what it found

Runs in ~/faceid_env (insightface + onnxruntime-gpu), NOT the vhap env -- that env has a
hand-built nvdiffrast and pytorch3d and is not worth risking for a dependency.

WHY THIS EXISTS
    VHAP's landmarks come from STAR, which returns ONE face per frame and does not check
    which. On take 2 that is wrong for 2,602 frames (7.0%): the box lands on a bodybuilder
    in a screenshot, or a synthetic face in a research figure, while he sits beside it at
    full size. Measured on f16527, STAR took a 76px face; the frame actually holds three,
    and he is the 213px one.

    Those frames are worse than the no-face frames. A no-face frame at least announces
    itself with confidence -1. These come back with confidence 1.0, and VHAP would fit
    FLAME to a stranger and fold it into the shared identity.

WHAT IT ENABLES, in increasing order of ambition
    1. MULTI-FACE FILTER. More than one face and no way to say which is him -> drop the
       frame. Needs no identity model, and on take 2 it alone catches every one of the
       2,602 wrong-face frames, since all of them hold 2-12 faces. Cheap and conservative:
       a discarded good frame costs a fraction of a percent, a kept bad one corrupts the
       identity that every other frame depends on.
    2. IDENTITY ACCUMULATION. Embed the faces in the frames already known clean, average
       into a reference, and "which of these three faces is him" becomes a lookup. The
       2,602 frames stop being dropped and start being corrected.
    3. CORPUS ANCHOR. The reference is not per-video. The face recurring across all 17
       takes is the subject; a bodybuilder appears in one video's screenshots, a guest in
       one episode. Carry the anchor forward and refine it per take.

    ORDER MATTERS for (2): build the reference only from frames already labelled clean. An
    embedding averaged over frames containing strangers poisons itself, and every later
    match gets worse.
"""
import argparse, json, pathlib, time
import os
import numpy as np

VHAP = pathlib.Path(os.environ.get("VHAP_ROOT",
    str(pathlib.Path(__file__).resolve().parents[1] / "vhap")))
MAXF = 12                    # faces stored per frame; the face-grid figures hit 12


def scan(seq, out, det_size=640, batch_report=2000):
    import cv2
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(det_size, det_size))

    imgdir = VHAP / "data/monocular" / seq / "images"
    files = sorted(imgdir.glob("*.jpg"))
    n = len(files)
    nfaces = np.zeros(n, np.int16)
    boxes = np.zeros((n, MAXF, 4), np.float32)
    dets = np.zeros((n, MAXF), np.float32)
    embs = np.zeros((n, MAXF, 512), np.float16)      # half: 657k x 12 x 512 fp32 is 16 GB
    t0 = time.time()
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        H, W = img.shape[:2]
        fs = app.get(img)
        # biggest first, so index 0 is the most likely subject and a truncation at MAXF
        # drops the least plausible candidates rather than an arbitrary set
        fs = sorted(fs, key=lambda x: -(x.bbox[2] - x.bbox[0]))[:MAXF]
        nfaces[i] = len(fs)
        for k, fc in enumerate(fs):
            boxes[i, k] = [fc.bbox[0] / W, fc.bbox[1] / H, fc.bbox[2] / W, fc.bbox[3] / H]
            dets[i, k] = fc.det_score
            e = fc.normed_embedding
            embs[i, k] = e.astype(np.float16)
        if i % batch_report == 0:
            el = time.time() - t0
            print(f"  {i}/{n}  {el:.0f}s  eta {(n-i)*el/max(i,1)/60:.1f}min", flush=True)
    np.savez_compressed(out, nfaces=nfaces, boxes=boxes, det=dets, emb=embs)
    print(f"wrote {out}  ({time.time()-t0:.0f}s)")


def report(seq, out):
    z = np.load(out)
    nf = z["nfaces"]
    bx = z["boxes"]
    n = len(nf)
    w = np.where(nf > 0, (bx[:, 0, 2] - bx[:, 0, 0]), 0)
    print(f"{seq}: {n} frames")
    print(f"  0 faces        {int((nf==0).sum()):6d}  {(nf==0).mean()*100:5.1f}%")
    print(f"  exactly 1      {int((nf==1).sum()):6d}  {(nf==1).mean()*100:5.1f}%")
    print(f"  2 or more      {int((nf>1).sum()):6d}  {(nf>1).mean()*100:5.1f}%   <- multi-face filter drops these")
    print(f"  max in a frame {int(nf.max()):6d}")
    print(f"  largest-face width (frames with a face): p05 {np.percentile(w[nf>0],5):.3f} "
          f"p50 {np.percentile(w[nf>0],50):.3f} p95 {np.percentile(w[nf>0],95):.3f}")

    lab_p = pathlib.Path(__file__).resolve().parent / "chunks" / seq / "clusters.npy"
    ver_p = lab_p.parent / "verdicts.json"
    if lab_p.exists() and ver_p.exists():
        lab = np.load(lab_p)
        v = json.load(open(ver_p))
        wrong = np.isin(lab, [int(j) for j, d in v.items()
                              if j.lstrip("-").isdigit() and isinstance(d, dict)
                              and d["kind"] == "wrong_face"])
        if wrong.any():
            print(f"\n  cross-check against the 'wrong_face' clusters ({int(wrong.sum())} frames):")
            print(f"    of those, multi-face catches {int((wrong & (nf>1)).sum())} "
                  f"({(wrong & (nf>1)).sum()/wrong.sum()*100:.1f}%)")
        keep = np.isin(lab, [int(j) for j, d in v.items()
                             if j.lstrip("-").isdigit() and isinstance(d, dict)
                             and d["verdict"] == "keep"])
        if keep.any():
            print(f"  of the KEPT frames ({int(keep.sum())}), multi-face would additionally "
                  f"drop {int((keep & (nf>1)).sum())} ({(keep & (nf>1)).sum()/keep.sum()*100:.1f}%)")


def main():
    global MAXF
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--max-faces", type=int, default=MAXF,
                    help="faces stored per frame; a face-grid figure hits 12")
    a = ap.parse_args()
    outdir = pathlib.Path(__file__).resolve().parent / "chunks" / a.seq
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "faces.npz"
    MAXF = a.max_faces
    if a.report:
        report(a.seq, out)
    else:
        scan(a.seq, out, a.det_size)


main()
