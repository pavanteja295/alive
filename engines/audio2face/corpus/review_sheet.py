#!/usr/bin/env python3
"""Render clusters for review. This is the step where a human or a model LOOKS.

    python review_sheet.py --seq <seq>              # every cluster, one montage each
    python review_sheet.py --seq <seq> --kept       # only the clusters labelled keep

WHY BOTH MODES
    The all-clusters sheet is for writing verdicts.json in the first place.

    The --kept sheet is for the check that matters more, and that is easy to skip: FALSE
    POSITIVES. A wrongly dropped cluster costs a fraction of a percent of the corpus; a
    wrongly KEPT one is tracked, fitted into the shared identity, and corrupts frames that
    were fine. On take 2 this pass caught a Memberships promo card sitting inside a
    keep_centred cluster, and a 17-frame cluster where he was clipped by the frame edge.
    Neither is visible in the all-clusters sheet, because both look plausible next to the
    obvious graphics.

WHAT IS DRAWN
    Real source frames, not the cached thumbnails -- the thumbnails are 64x36 and too small
    to judge. The detector's box is drawn on each frame: that is what makes a wrong-face
    cluster obvious at a glance, because the box is visibly on a bodybuilder while he sits
    beside it at full size.
"""
import argparse, json, pathlib
import os
import numpy as np
from PIL import Image, ImageDraw

VHAP = pathlib.Path(os.environ.get("VHAP_ROOT",
    str(pathlib.Path(__file__).resolve().parents[1] / "vhap")))


def load(seq):
    outdir = pathlib.Path(__file__).resolve().parent / "chunks" / seq
    lab = np.load(outdir / "clusters.npy")
    rows = {r["cluster"]: r for r in json.load(open(outdir / "clusters.json"))}
    bb = np.load(VHAP / "data/monocular" / seq / "landmark2d/STAR.npz",
                 allow_pickle=True)["bounding_box"]
    vp = outdir / "verdicts.json"
    v = {}
    if vp.exists():
        v = {int(j): d for j, d in json.load(open(vp)).items()
             if j.lstrip("-").isdigit() and isinstance(d, dict)}
    return outdir, lab, rows, bb, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--kept", action="store_true", help="only clusters labelled keep")
    ap.add_argument("--per", type=int, default=8, help="frames sampled per cluster")
    ap.add_argument("--cell", type=int, default=400)
    a = ap.parse_args()

    outdir, lab, rows, bb, v = load(a.seq)
    imgdir = VHAP / "data/monocular" / a.seq / "images"
    ids = sorted(set(lab.tolist()))
    if a.kept:
        if not v:
            raise SystemExit("--kept needs verdicts.json")
        ids = [j for j in ids if v.get(j, {}).get("verdict") == "keep"]
        ids.sort(key=lambda j: -rows.get(j, {}).get("frames", 0))

    dest = outdir / ("clusters_kept" if a.kept else "clusters_full")
    dest.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    cw, ch = a.cell, int(a.cell * 9 / 16)
    for j in ids:
        idx = np.where(lab == j)[0]
        if not len(idx):
            continue
        pick = np.sort(rng.choice(idx, min(a.per, len(idx)), replace=False))
        cols = min(4, len(pick))
        rowsn = (len(pick) + cols - 1) // cols
        sheet = Image.new("RGB", (cw * cols, ch * rowsn + 34), (12, 12, 12))
        for k, i in enumerate(pick):
            im = Image.open(imgdir / f"{i:06d}.jpg").convert("RGB").resize((cw, ch))
            d = ImageDraw.Draw(im)
            if bb[i, 4] > 0:
                d.rectangle([bb[i, 0] * cw, bb[i, 1] * ch, bb[i, 2] * cw, bb[i, 3] * ch],
                            outline=(255, 60, 60), width=2)
            d.rectangle([0, 0, cw, 14], fill=(0, 0, 0))
            d.text((3, 2), f"f{i}", fill=(160, 255, 160))
            sheet.paste(im, ((k % cols) * cw, 34 + (k // cols) * ch))
        info, ver = rows.get(j, {}), v.get(j, {})
        d = ImageDraw.Draw(sheet)
        col = (120, 255, 120) if ver.get("verdict") == "keep" else \
              (255, 110, 110) if ver.get("verdict") == "drop" else (255, 230, 90)
        d.text((8, 6), "c%02d  %s / %s  %d fr (%.1f%%)  cx %.2f  face_w %.3f  %d spans  | %s"
               % (j, ver.get("verdict", "UNLABELLED").upper(), ver.get("kind", "?"),
                  info.get("frames", len(idx)), info.get("pct", 0), info.get("cx", 0),
                  info.get("face_w", 0), info.get("spans", 0), ver.get("note", "")[:70]),
               fill=col)
        tag = f"{ver.get('verdict','na')}_{ver.get('kind','na')}" if ver else \
              f"w{info.get('face_w',0):.3f}_cx{info.get('cx',0):.2f}"
        sheet.save(dest / f"c{j:02d}_{tag}.jpg", quality=90)

    unl = [j for j in sorted(set(lab.tolist())) if j not in v] if v else []
    print(f"{a.seq}: wrote {len(ids)} montages -> {dest}")
    if unl:
        print(f"  UNLABELLED clusters: {unl}  ({sum((lab==j).sum() for j in unl)} frames)")
    if v and not a.kept:
        keep = np.isin(lab, [j for j, d in v.items() if d["verdict"] == "keep"])
        print(f"  verdicts cover {len(v)} clusters, keep {keep.sum()}/{len(lab)} "
              f"({keep.mean()*100:.1f}%)")
        print(f"  now run again with --kept to check for FALSE POSITIVES")


main()
