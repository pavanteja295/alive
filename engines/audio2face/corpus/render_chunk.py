#!/usr/bin/env python3
"""Sanity check a tracked chunk by drawing the solved FLAME mesh back onto its frames.

    python render_chunk.py --chunk <name>              # contact sheet
    python render_chunk.py --chunk <name> --video      # + mp4
    python render_chunk.py --seq <seq> --all           # one sheet per tracked chunk

A residual number can look fine while the fit is wrong -- a mesh that has drifted onto the
shoulder still Procrustes cleanly against its own neutral, because the residual measures
neutral-to-posed consistency, not agreement with the image. Only the overlay tests that.

CAMERA. VHAP is uncalibrated, so tracker.py:150-156 builds intrinsics itself:
    f = focal_length * max(h, w);   cx, cy = w/2, h/2      <- principal point HARDCODED
and the extrinsic is a pure translation (tracker.py:1344-45):
    RT = [I | (0,0,-1)]                                    <- camera at +1z looking down -z
so camera-space z is (z_world - 1) and the projection is a plain pinhole about the image
centre. That hardcoded principal point is the whole reason the frames are cropped: an
off-centre face is inexplicable to this model and gets absorbed into head pose instead.
"""
import argparse, json, os, pathlib, re, subprocess, sys, tempfile, shutil
import numpy as np

VHAP = pathlib.Path(os.environ.get("VHAP_ROOT",
    str(pathlib.Path(__file__).resolve().parents[1] / "vhap")))


def flags_from_config(cfg):
    want = {"add_teeth": False, "remove_lip_inside": False}
    for line in pathlib.Path(cfg).read_text().splitlines():
        k, _, v = line.partition(":")
        if k.strip() in want and v.strip() in ("true", "false"):
            want[k.strip()] = v.strip() == "true"
    return want


# which tracking pass to visualise: output/chunks (independent identity) or output/shared
# (one identity for the whole take). Without this the shared pass cannot be checked by eye
# at all, only by residual -- and the residual is exactly what cannot detect a consistently
# wrong mesh.
RUN_DIR = "output/chunks"


def find_run(chunk):
    runs = sorted((VHAP / RUN_DIR / chunk).glob("*/tracked_flame_params_30.npz"))
    return (runs[-1], runs[-1].parent / "config.yml") if runs else (None, None)


def project(v, focal, W, H):
    f = float(focal) * max(H, W)
    z = v[:, 2] - 1.0
    return f * v[:, 0] / -z + W / 2, -f * v[:, 1] / -z + H / 2


def worst_frames(chunk, seq_dir, per, min_gap_frac=0.05):
    """The frames with the HIGHEST per-frame skull residual, spread apart.

    Evenly spaced samples answer "does it look right in general". They cannot answer "does it
    ever go wrong", and that is the question -- a fit that fails for 2 seconds in a 40 s chunk
    is invisible at 15/50/85% and is exactly what would poison the identity. So sample where
    the fit is worst, which is where a failure must show if there is one.

    Spread enforced at `min_gap_frac` of the chunk: the top-3 residual frames are usually
    three consecutive frames of one bad moment, which shows the same failure three times and
    hides everything else.
    """
    pose = seq_dir / ("pose_shared" if RUN_DIR.endswith("shared") else "pose") / f"{chunk}.npz"
    if not pose.exists():
        return None
    res = np.load(pose)["residual"] * 1000
    n = len(res)
    gap = max(1, int(n * min_gap_frac))
    picks, order = [], np.argsort(-res)
    for i in order:
        if all(abs(int(i) - p) >= gap for p in picks):
            picks.append(int(i))
        if len(picks) == per:
            break
    return sorted(picks), res


def chunk_frames(chunk, per, cell, seq_dir=None):
    """`per` overlay frames -- worst-residual where available, else evenly spaced."""
    from PIL import Image, ImageDraw
    fn = _frame_fn(chunk)
    n = fn.n
    res = None
    if seq_dir is not None:
        w = worst_frames(chunk, seq_dir, per)
        if w is not None:
            picks, res = w
            picks = [min(p, n - 1) for p in picks]
    if res is None:
        picks = [int(n * q) for q in np.linspace(0.15, 0.85, per)]
    out = []
    for i in picks:
        im = fn(i).resize((cell, cell))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, cell, 13], fill=(0, 0, 0))
        lbl = f"f{i}  {res[i]:.2f}mm" if res is not None else f"f{i}"
        d.text((3, 1), lbl, fill=(255, 170, 120) if res is not None else (160, 255, 160))
        out.append(im)
    return out


def _frame_fn(chunk):
    """Build a cached per-frame overlay renderer for one chunk."""
    from PIL import Image, ImageDraw
    import torch
    ck, cfg = find_run(chunk)
    sys.path.insert(0, str(VHAP)); os.chdir(VHAP)
    from vhap.model.flame import FlameHead
    d = dict(np.load(ck))
    flame = FlameHead(300, 100, **flags_from_config(cfg))
    files = sorted((VHAP / "data/monocular" / chunk / "images").glob("*.jpg"))
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)

    def fn(i):
        im = Image.open(files[i]).convert("RGB")
        W, H = im.size
        with torch.no_grad():
            v = flame(shape=t(d["shape"])[None], expr=t(d["expr"][i])[None],
                      rotation=t(d["rotation"][i])[None], neck=t(d["neck_pose"][i])[None],
                      jaw=t(d["jaw_pose"][i])[None], eyes=t(d["eyes_pose"][i])[None],
                      translation=t(d["translation"][i])[None],
                      static_offset=t(d["static_offset"]))[0][0].numpy()
        u, w = project(v, d["focal_length"][0], W, H)
        dr = ImageDraw.Draw(im)
        for x, y in zip(u, w):
            if 0 <= x < W and 0 <= y < H:
                dr.point((x, y), fill=(0, 255, 255))
        return im
    fn.n = min(len(files), len(d["rotation"]))
    return fn


def render(chunk, out_dir, stride, video, cols=5, cell=300):
    from PIL import Image, ImageDraw
    import torch
    ck, cfg = find_run(chunk)
    if ck is None:
        print(f"  {chunk}: not tracked, skipping")
        return None
    sys.path.insert(0, str(VHAP)); os.chdir(VHAP)
    from vhap.model.flame import FlameHead
    d = dict(np.load(ck))
    flame = FlameHead(300, 100, **flags_from_config(cfg))
    imgdir = VHAP / "data/monocular" / chunk / "images"
    files = sorted(imgdir.glob("*.jpg"))
    n = min(len(files), len(d["rotation"]))
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)

    def frame(i):
        im = Image.open(files[i]).convert("RGB")
        W, H = im.size
        with torch.no_grad():
            v = flame(shape=t(d["shape"])[None], expr=t(d["expr"][i])[None],
                      rotation=t(d["rotation"][i])[None], neck=t(d["neck_pose"][i])[None],
                      jaw=t(d["jaw_pose"][i])[None], eyes=t(d["eyes_pose"][i])[None],
                      translation=t(d["translation"][i])[None],
                      static_offset=t(d["static_offset"]))[0][0].numpy()
        u, w = project(v, d["focal_length"][0], W, H)
        dr = ImageDraw.Draw(im)
        for x, y in zip(u, w):
            if 0 <= x < W and 0 <= y < H:
                dr.point((x, y), fill=(0, 255, 255))
        return im

    picks = list(range(0, n, max(1, n // (cols * 2)))) [:cols * 2]
    ims = []
    for i in picks:
        im = frame(i).resize((cell, cell))
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, cell, 14], fill=(0, 0, 0))
        dr.text((3, 2), f"f{i}", fill=(160, 255, 160))
        ims.append(im)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cell * cols, cell * rows + 20), (12, 12, 12))
    for k, im in enumerate(ims):
        sheet.paste(im, ((k % cols) * cell, 20 + (k // cols) * cell))
    ImageDraw.Draw(sheet).text((6, 5), f"{chunk}   {n} frames   focal {float(d['focal_length'][0]):.3f}",
                               fill=(255, 220, 90))
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{chunk}.jpg"
    sheet.save(p, quality=90)
    print(f"  {chunk}: sheet -> {p.name}")

    if video:
        tmp = pathlib.Path(tempfile.mkdtemp())
        for j, i in enumerate(range(0, n, stride)):
            frame(i).save(tmp / f"{j:06d}.png")
        mp4 = out_dir / f"{chunk}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(30 // stride),
                        "-i", str(tmp / "%06d.png"), "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", "-crf", "18", str(mp4)], check=True)
        shutil.rmtree(tmp)
        print(f"  {chunk}: video -> {mp4.name}")
    return p


def summary(seq, root, per=3, cell=250, sample=None, seed=0):
    """One sheet, the WORST frames from EVERY tracked chunk. The scan-everything view.

    Per-chunk sheets are for digging into one suspect chunk; this is for noticing that
    chunk 34 is the only one where the mesh has slid onto the shoulder, without opening 59
    files. Frames are the highest-residual ones in each chunk, spread apart -- if the worst
    frame of a chunk looks right, the chunk is right, and that is a far stronger statement
    than a sample at the middle.
    """
    from PIL import Image, ImageDraw
    rows = json.load(open(root / "chunks" / seq / "sequences.json"))
    tracked = [r for r in rows if find_run(r["name"])[0] is not None]
    if sample and len(tracked) > sample:
        rng = np.random.default_rng(seed)
        tracked = [tracked[i] for i in sorted(rng.choice(len(tracked), sample, replace=False))]
    if not tracked:
        print("no tracked chunks yet")
        return
    led = root / "chunks" / seq / ("shared_ledger.tsv" if RUN_DIR.endswith("shared") else "track_ledger.tsv")
    res = {}
    if led.exists():
        for ln in led.read_text().splitlines()[1:]:
            f = ln.split("\t")
            if len(f) >= 6 and f[3] == "ok":
                res[f[0]] = f[5]
    sheet = Image.new("RGB", (cell * per + 190, (cell + 4) * len(tracked)), (12, 12, 12))
    for r, rec in enumerate(tracked):
        ims = chunk_frames(rec["name"], per, cell, seq_dir=root / "chunks" / seq)
        y = r * (cell + 4)
        d = ImageDraw.Draw(sheet)
        d.text((6, y + cell // 2 - 22), f"c{rec['chunk']:03d} {rec['kind']}", fill=(255, 220, 90))
        d.text((6, y + cell // 2 - 6), f"{rec['frames']} fr", fill=(210, 210, 210))
        rr = res.get(rec["name"])
        d.text((6, y + cell // 2 + 10), f"resid {rr} mm" if rr else "resid ?",
               fill=(140, 255, 160) if rr and float(rr) < 3 else (255, 160, 160))
        for k, im in enumerate(ims):
            sheet.paste(im, (190 + k * cell, y + 2))
    out = root / "chunks" / seq / "overlay_summary.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    print(f"summary -> {out}  ({len(tracked)} chunks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk")
    ap.add_argument("--seq")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--summary", action="store_true", help="a few frames from every tracked chunk, one sheet")
    ap.add_argument("--sample", type=int, help="with --summary: only this many random chunks")
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--shared", action="store_true",
                    help="visualise the shared-identity pass (output/shared) instead")
    a = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parent
    global RUN_DIR
    if a.shared:
        RUN_DIR = "output/shared"
    if a.summary and a.seq:
        summary(a.seq, root, per=a.per, sample=a.sample)
    elif a.chunk:
        # rsplit, not split: a creator's sequence name could itself contain "__c"
        seq = re.sub(r"__c\d+$", "", a.chunk)
        render(a.chunk, root / "chunks" / seq / "overlay", a.stride, a.video)
    elif a.seq and a.all:
        rows = json.load(open(root / "chunks" / a.seq / "sequences.json"))
        od = root / "chunks" / a.seq / "overlay"
        for r in rows:
            render(r["name"], od, a.stride, False)
    else:
        ap.error("give --chunk NAME or --seq SEQ --all")


main()
