#!/usr/bin/env python3
"""Render the three identities side by side, driven by one set of control curves.

Called by build_identity.py --compare. All rows share the rig and the curves; only the
bind pose differs, so anything that differs on screen IS the identity.
"""
import pathlib, subprocess, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path.insert(0, str(PIPE / "offset"))
HEAD = 24049
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

ORDER = [("rig_archetype.npz", "ARCHETYPE  (no identity work)"),
         ("rig_conform.npz",   "MetaHuman Identity Solve"),
         ("rig_beltrami.npz",  "FLAME Beltrami wrap")]


def shade(V, F, S, span, yoff, torch):
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh.rasterizer import rasterize_meshes
    Vn = V.clone()
    # pytorch3d NDC is +X left, +Y up, +Z into the screen; the DNA is Y up with +Z toward
    # the viewer, so X and Z negate or the face renders from behind.
    Vn[..., 0] *= -1.0; Vn[..., 2] *= -1.0
    Vn[..., 1] -= yoff
    Vn = Vn / span
    Vn[..., 2] += 5.0
    m = Meshes(verts=[Vn[i] for i in range(V.shape[0])], faces=[F] * V.shape[0])
    pix, _, _, _ = rasterize_meshes(m, image_size=(S, S), blur_radius=0.0,
                                    faces_per_pixel=1, bin_size=None)
    pix = pix[..., 0]
    hit = pix >= 0
    n = m.faces_normals_packed()[torch.clamp(pix, min=0)]
    L = torch.tensor([0.32, 0.42, 0.85], device=V.device); L = L / L.norm()
    lit = (n @ L).abs().clamp(0, 1) * 0.78 + 0.20
    return ((torch.where(hit, lit, torch.ones_like(lit))).clamp(0, 1) * 255) \
        .to(torch.uint8).unsqueeze(-1).repeat(1, 1, 1, 3)


def run_compare(sub, curves, video, audio, start, dur, curves_offset,
                head_pose="rotation", fps=30, S=520, span=11.0, yoff=-1.2, chunk=48):
    """head_pose: none | rotation | full.

    TorchRig emits CANONICAL vertices and never applies head pose -- deliberately, because
    the six MetaHuman head curves are not a rotation about the head joint. They are a root
    delta conjugated into the ARCHETYPE head-bone frame,
    HeadPose = B * Root_relative * B^-1 (MetaHumanPerformanceExportUtils.cpp:2247-2302).
    Treating them as a pivot rotation about DNA joint 84 displaces vertices by 5.7 cm mean
    -- more than the whole take's head motion. head_pose.py inverts the conjugation
    properly; this just calls it.

    'rotation' drops the translation. Use it whenever the comparison includes a FLAME row:
    FLAME's translation is camera-space placement and MetaHuman's is a delta about the head
    joint referenced to a per-take frame, so the two are not the same gauge and only the
    rotation is comparable. 'full' is right when every row is MetaHuman.
    """
    import torch
    from riglogic_torch import TorchRig
    from utils.rig_utils import load_depth_curves, curves_to_raw
    from PIL import Image, ImageDraw, ImageFont
    import head_pose as hp

    if not curves:
        sys.exit("--compare needs --curves (a MetaHuman animation-sequence JSON). The "
                 "performance solve lives in Unreal; this tool does not produce curves.")
    rows = [(sub / f, lab) for f, lab in ORDER if (sub / f).exists()]
    if len(rows) < 2:
        sys.exit("need at least two rigs in {} -- run the build first".format(sub))
    print("rows: " + " | ".join(lab for _, lab in rows))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    raw = np.load(PIPE / "offset/cache/rig_names.npz")["raw_names"]
    keys, t, _ = load_depth_curves(str(curves), fps=60.0)
    c, _, _ = curves_to_raw(keys, t, raw)
    c, t = np.asarray(c), np.asarray(t) + curves_offset
    grid = np.arange(start, min(start + dur, t[-1]), 1.0 / fps)
    if len(grid) < 2:
        sys.exit(f"no overlap: curves span {t[0]:.1f}-{t[-1]:.1f}s, asked for "
                 f"{start:.1f}-{start+dur:.1f}s. Use --curves-offset if the file's clock "
                 f"starts at a non-zero take time.")
    C = np.stack([np.interp(grid, t, c[:, i]) for i in range(c.shape[1])], 1)
    print(f"{len(grid)} frames  {grid[0]:.1f}-{grid[-1]:.1f}s at {fps} fps")

    H6 = ["HeadPitch", "HeadYaw", "HeadRoll",
          "HeadTranslationX", "HeadTranslationY", "HeadTranslationZ"]
    POSE = None
    if head_pose != "none":
        missing = [k for k in H6 if k not in keys]
        if missing:
            print(f"  head pose: DISABLED, curve file lacks {missing}")
        else:
            POSE = np.stack([np.interp(grid,
                                       np.asarray([q[0] for q in keys[k]]) + curves_offset,
                                       np.asarray([q[1] for q in keys[k]])) for k in H6], 1)
            print(f"  head pose: {head_pose}"
                  f"   yaw range {POSE[:,1].ptp():.1f} deg, pitch {POSE[:,0].ptp():.1f} deg")

    H = np.load(PIPE / "head/head_assets.npz")
    F = torch.as_tensor(H["pos_idx"].astype(np.int64)[H["faces"].astype(np.int64)], device=dev)
    rigs, centres = [], []
    for p, _ in rows:
        r = TorchRig(str(p), device=dev)
        with torch.no_grad():
            v0 = r(torch.zeros(1, 263, device=dev))[0, :HEAD]
        # centre on the HEAD, not the centroid: MetaHuman carries a bust and most of its
        # vertices sit below the jaw, which drags a plain centroid into the shoulders.
        lo, hi = torch.quantile(v0[:, 1], torch.tensor([0.40, 0.99], device=dev))
        rigs.append(r); centres.append(v0[(v0[:, 1] > lo) & (v0[:, 1] < hi)].mean(0))

    try:
        font = ImageFont.truetype(FONT, 16)
    except Exception:
        font = ImageFont.load_default()
    tmp = pathlib.Path("/tmp/id_compare_" + sub.name); tmp.mkdir(exist_ok=True)
    for f in tmp.glob("*.png"):
        f.unlink()
    BAR = 32
    for s in range(0, len(grid), chunk):
        ix = np.arange(s, min(s + chunk, len(grid)))
        cols = []
        x = torch.as_tensor(C[ix], dtype=torch.float32, device=dev)
        for r, c0 in zip(rigs, centres):
            with torch.no_grad():
                V = r(x)[:, :HEAD]
            if POSE is not None:
                Vn = V.cpu().numpy().astype(np.float64)
                for k, fi in enumerate(ix):
                    M = hp.root_delta_dna(POSE[fi]).copy()
                    if head_pose == "rotation":
                        M[:3, 3] = 0.0
                    Vn[k] = hp.apply(Vn[k], M)
                V = torch.as_tensor(Vn, dtype=torch.float32, device=dev)
            cols.append(shade(V - c0, F, S, span, yoff, torch).cpu().numpy())
        for k, fi in enumerate(ix):
            strip = np.concatenate([cc[k] for cc in cols], 1)
            im = Image.new("RGB", (strip.shape[1], strip.shape[0] + BAR), (255, 255, 255))
            im.paste(Image.fromarray(strip), (0, BAR))
            d = ImageDraw.Draw(im)
            for j, (_, lab) in enumerate(rows):
                d.text((j * S + 12, 7), lab, fill=(20, 20, 24), font=font)
                if j:
                    d.line([(j * S, BAR), (j * S, im.height)], fill=(220, 220, 226), width=1)
            d.text((im.width - 104, im.height - 24), f"{grid[fi]:6.2f} s",
                   fill=(150, 150, 158), font=font)
            im.save(tmp / f"f{fi:05d}.png")
        if s % (chunk * 6) == 0:
            print(f"  {s}/{len(grid)}")

    out = sub / "compare.mp4"
    if video:
        fc = (f"[0:v]fps={fps},crop=in_w:in_w:0:(in_h*0.14),scale={S}:{S},"
              f"pad={S}:{S+BAR}:0:{BAR}:color=white,"
              f"drawtext=fontfile={FONT}:text='SOURCE':x=12:y=7:fontsize=16:"
              f"fontcolor=0x141418[src];[src][1:v]hstack=inputs=2[v]")
        cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", video,
               "-framerate", str(fps), "-start_number", "0", "-i", str(tmp / "f%05d.png"),
               "-filter_complex", fc, "-map", "[v]", "-map", "0:a?", "-t", str(dur),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               "-c:a", "aac", "-b:a", "128k", str(out)]
    else:
        cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(tmp / "f%05d.png")]
        if audio:
            cmd += ["-ss", str(start), "-i", audio, "-c:a", "aac", "-shortest"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit("ffmpeg failed:\n" + r.stderr[-1500:])
    print("wrote", out)
