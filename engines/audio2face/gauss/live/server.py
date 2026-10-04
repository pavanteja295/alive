#!/usr/bin/env python3
"""The serving path held open, streaming frames the instant they leave the card.

    python pipeline/gauss/live/server.py --profile drk --run G4_teeth --compile
    then open http://127.0.0.1:8730

WHAT THIS IS FOR

infer.py answers "does the chain work". It cannot answer "how fast is it", because
almost everything it costs is the command-line shape of it: two interpreters started
per request, two models loaded per request, 300 PNG files written so that ffmpeg can
read them back. Measured, that is 53 ms a frame, of which 31 ms is compression and
5.4 s is startup.

None of that is in the deployed shape. Here the models are resident, the geometry
crosses between the two processes through shared memory, and a rendered frame goes
straight from the card into the HTTP response. What is left is the pipeline.

    click a clip  ->  its audio to the resident phase-one worker
                  ->  geometry in shared memory
                  ->  frames rendered and pushed as they are produced
                  ->  a counter that is measuring, not estimating

NO FILE OF RENDERED FRAMES EXISTS. There is nothing to play back afterwards, on
purpose: a demo that renders to a video and then plays it is measuring the video
player. What the right-hand panel shows is the renderer's own output rate.

THE THREE THINGS THAT MAKE IT FAST, all measured in pipeline/gauss/serve_cost.py

  resident models       5.4 s per request -> 0
  the trunk cached      the encoder over the reference photograph and the convolution
                        over the resting-shape map read buffers that are never written
                        to, and were being recomputed 30 times a second. Bit-identical,
                        checked on every boot: --verify prints the difference.
  nothing to disk       no PNG, no npz. 31 ms a frame of compression, gone.

and one that is a real trade, off by default: --fp16.
"""
import argparse
import base64
import http.server
import io
import json
import math
import os
import pathlib
import queue
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent.parent
M = PIPE.parent
CLIPS = HERE / "clips"
FPS_OUT = 30.0

STATE = {}
LOCK = threading.Lock()
INBOX = HERE / "inbox"
SPOKEN = HERE / "spoken"


def wav_seconds(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(p)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def to_16k_mono(src, dst):
    """Whatever arrived, as the 16 kHz mono the audio model asserts on.

    Anything that can speak -- a text-to-speech reply, a microphone, a file someone
    dropped in -- converges here, so no caller has to know what the model wants and
    no caller can hand it something subtly wrong.
    """
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-vn", "-ac", "1", "-ar", "16000", str(dst)], check=True)
    return dst


# ==========================================================  THE RENDER HALF  ===
class Renderer:
    def __init__(self, profile, run, iteration, render_scale, compile_mode,
                 boot_verts, boot_ctrl):
        sys.path.insert(0, str(PIPE / "gauss/recipes/mesh-to-render/tools"))
        from _profile import load, run_assets
        self.p = p = load(profile)

        rundir = p.RELEASE / run                 # the released renderer, checkpoints/
        if not rundir.exists():
            rundir = p.RUNS / run                # a training run not released yet
        if not rundir.exists():
            raise SystemExit(f"no run or release called {run}")

        sys.path.insert(0, str(p.STAVATAR)); os.chdir(p.STAVATAR)
        sys.path.insert(0, str(PIPE / "gauss"))
        import torch
        import yaml
        from argparse import Namespace
        from scene.mesh_gaussian_model import MeshGaussianModel
        from gaussian_renderer import render
        from networks.dual_branch import DualBranchUNet
        from utils.graphics_utils import getProjectionMatrix

        self.torch, self.render = torch, render
        torch.backends.cudnn.benchmark = True

        cfg = yaml.safe_load((rundir / "config.yml").read_text()) or {}
        flat = {}
        for v in cfg.values():
            if isinstance(v, dict):
                flat.update(v)
        flat.update({k: v for k, v in cfg.items() if not isinstance(v, dict)})
        self.flat = flat

        pcs = sorted((rundir / "point_cloud").glob("iteration_*"),
                     key=lambda x: int(x.name.split("_")[1]))
        it = iteration if iteration > 0 else int(pcs[-1].name.split("_")[1])
        self.run, self.it = run, it

        assets = run_assets(p, flat.get("mesh_assets"))
        self.g = g = MeshGaussianModel(flat.get("sh_degree", 3), assets,
                                       uv_size=flat.get("uv_size", 256),
                                       pose_mode=flat.get("pose_mode", "posed"),
                                       source_path=None)
        g.mesh_paths, g.rigid = None, None
        g.metric_xyz = bool(flat.get("metric_xyz", p.METRIC_XYZ))
        g.load_ply(str(rundir / f"point_cloud/iteration_{it}/point_cloud.ply"))

        # A MESH HAS TO BE SELECTED BEFORE THE POSITION MAP EXISTS, AND THE NETWORK IS
        # BUILT AROUND THAT MAP -- it goes in as a register_buffer and is never written
        # again. train.py gets a real one from the Scene it constructs. A zero mesh of
        # the right shape is NOT a substitute: every triangle is degenerate, the map
        # comes out NaN, and so does every frame. Nothing raises; the page just shows
        # nothing. So the caller hands one real posed head in, from the phase-one
        # worker, before this is constructed.
        assert boot_verts is not None and np.isfinite(boot_verts).all(), (
            "the renderer needs one real posed mesh to build the position map around")
        bv = torch.as_tensor(np.asarray(boot_verts, np.float32))
        g.verts_seq = g.verts_cano_seq = bv[None] if bv.dim() == 2 else bv
        g.num_timesteps = len(g.verts_seq)
        g.ctrl = torch.as_tensor(np.asarray(boot_ctrl, np.float32), device="cuda")
        if g.ctrl.dim() == 1:
            g.ctrl = g.ctrl[None]
        g.select_mesh_by_timestep(0)

        uv_coords = torch.load(rundir / f"param/iteration_{it}/uv_coords.pt",
                               map_location="cuda")
        self.net = net = DualBranchUNet(
            device="cuda", uv_sample_coords=uv_coords, uv_mask=g.get_uv_mask(),
            position_map=g.get_position_map(), uv_size=int(flat.get("uv_size", 256)),
            condition_dim=g.condition.shape[1]).to("cuda")
        net.load_state_dict(torch.load(rundir / f"param/iteration_{it}/dual_branch.pth",
                                       map_location="cuda"))
        net.eval()

        # A RUN TRAINED WITH per_take_appearance CARRIES ITS OWN CONSTANT AND IT IS NOT
        # OPTIONAL. In that mode training replaces the network's channels 10:14 with a
        # learned constant per recording, so those four outputs are never trained on
        # anything. Rendering without it feeds untrained values into colour and opacity
        # -- which still looks like a face, just flat, and nothing reports it.
        self.appear = None
        ap = rundir / f"point_cloud/iteration_{it}/appearance.pt"
        if ap.exists():
            b = torch.load(ap, map_location="cuda")
            groups = list(b["groups"])
            want = self.p.DERIVED_ALL["deploy_camera"]["_from_chunk"].rsplit("__c", 1)[0]
            gi = next((k for k, x in enumerate(groups) if x == want), None)
            if gi is None:
                gi = next((k for k, x in enumerate(groups) if x.startswith(want[:24])), 0)
            self.appear = b["appearance"][:, gi * 4:(gi + 1) * 4].cuda()
            self.look = groups[gi]

        # ---- the camera deployment does not have -----------------------------
        cam = p.DERIVED_ALL["deploy_camera"]
        self._getProjectionMatrix, self.render_scale = getProjectionMatrix, render_scale
        self.cam, self.W, self.H = self.cam_for(cam)
        self.cam_from = cam["_from_chunk"]
        self.pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                              debug=False)
        self.bg = torch.tensor([1.0, 1.0, 1.0] if flat.get("white_background", True)
                               else [0.0, 0.0, 0.0], device="cuda")

        # ---- the two branches that never change ------------------------------
        with torch.no_grad():
            x1 = net.inc(net.reference_image); x2 = net.down1(x1)
            x3 = net.down2(x2); x4 = net.down3(x3)
            self.d3 = net.up3(net.up2(net.up1(x4, x3), x2), x1)
            self.uvf = net.uv_conv(net.fourier(net.position_map.float()))

        self.compiled = {}
        self.compile_mode = compile_mode
        self.blobs = int(g._xyz.shape[0])

    def cam_for(self, cam):
        """A renderer camera from a camera record (fl_x, fl_y, w, h, transform_matrix):
        the deploy camera, or a body clip's own tracked one."""
        torch = self.torch
        s = max(self.render_scale, 1e-6)
        W, H = int(round(cam["w"] * s)), int(round(cam["h"] * s))
        fovx = 2 * math.atan(W / (2 * cam["fl_x"] * s))
        fovy = 2 * math.atan(H / (2 * cam["fl_y"] * s))
        c2w = np.array(cam["transform_matrix"], np.float64); c2w[:3, 1:3] *= -1
        wvt = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32).transpose(0, 1).cuda()
        proj = self._getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx,
                                         fovY=fovy).transpose(0, 1).cuda()

        class Cam:
            image_width, image_height = W, H
            FoVx, FoVy = fovx, fovy
            world_view_transform = wvt
            full_proj_transform = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
            camera_center = torch.inverse(wvt)[3, :3]
        return Cam(), W, H

    # ---------------------------------------------------------------- real bodies
    def load_bodies(self, min_s=5.0):
        """Every centred clip of this creator that can carry a pasted head: it has its
        own tracked camera (the export) and its own head track (cache/rigid). Frames are
        read from disk per job, so nothing here costs card memory."""
        import json as _json
        p, torch = self.p, self.torch
        self.bodies = {}
        for rec in p.RECORDINGS:
            sq = PIPE / "corpus/chunks" / rec / "sequences.json"
            if not sq.exists():
                continue
            for e in _json.load(open(sq)):
                name, nfr = e["name"], int(e["frames"])
                tf = p.CORPUS / name / "transforms.json"
                rg = PIPE / f"gauss/cache/rigid/{name}.npz"
                if e.get("kind") != "centred" or nfr < min_s * FPS_OUT \
                        or not tf.exists() or not rg.exists():
                    continue
                fr0 = _json.load(open(tf))["frames"][0]
                cam, _, _ = self.cam_for(fr0)
                self.bodies[name] = dict(cam=cam, crop=[int(v) for v in e["crop"]],
                                         src0=int(e["src_range"][0]), frames=nfr,
                                         images=p.VHAP / "data/monocular" / rec / "images")
        # The head, and only the head: blobs on triangles that move with the skull, plus
        # the rig-only meshes (teeth, eyeballs) which sit inside it. Everything else --
        # shirt, shoulders -- is hidden for a body render, because the real frame has it.
        sys.path.insert(0, str(PIPE / "gauss"))
        from head_weight import HeadWeight
        hw = HeadWeight(p.SUBJECT, rig=str(p.RIG_DIR /
                                           "rig_beltrami.npz")).w
        A = np.load(self.flat.get("mesh_assets") or str(p.HEAD_ASSETS), allow_pickle=True)
        F = A["pos_idx"].astype(np.int64)[A["faces"].astype(np.int64)]
        nh = int(self.g.n_head_faces) if getattr(self.g, "n_head_faces", None) else len(F)
        head_tri = np.ones(len(F), bool)
        # 0.9, not 0.5: neck triangles that only partly follow the skull drew a faint
        # collar over the real shirt and microphone
        head_tri[:nh] = (hw[np.clip(F[:nh], 0, len(hw) - 1)] > 0.9).all(1)
        b = self.g.binding.long().view(-1)
        self.nonhead = ~torch.as_tensor(head_tri, device="cuda")[b]
        # HOW FRONTAL each body frame is, from the blink step's measurement (ear cache,
        # column 2). The renderer was trained on frames above FRONTAL_MIN only
        # (filter_frontal.py), so a replayed head turned past that is drawn from nothing
        # it has seen: speckled forehead, a smeared face. drk's first resting pick did it.
        self.frontal = {}
        ef = PIPE / "gauss/cache" / f"ear_{p.SUBJECT}.npz"
        if ef.exists():
            z = np.load(ef)
            self.frontal = {k: np.asarray(z[k][:, 2], np.float32) for k in z.files
                            if k in self.bodies}
        return len(self.bodies)

    # ---------------------------------------------------------------- the network
    def _per_frame(self, expr, dm):
        """Everything the frame actually changes. The trunk is not in here."""
        torch, net, d3, uvf = self.torch, self.net, self.d3, self.uvf
        ce = net.ctrl_embed(expr)
        cf = ce.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, d3.size(2), d3.size(3))
        og = net.global_decoder(torch.cat([d3, uvf, cf], 1))
        dsp = net.disp_conv(net.fourier(dm.float()))
        loc = [net.local_decoder(d3, dsp, expr, m, net.crop_boxes[k])
               for k, m in (("eye", net.eye_mask), ("nose", net.nose_mask),
                            ("lips", net.lips_mask), ("forehead", net.forehead_mask))]
        return net._offset_attr_process(net.final_fuse(torch.cat([og] + loc, 1)),
                                        net.uv_sample_coords)

    def forward_fn(self, fp16):
        key = bool(fp16)
        if key in self.compiled:
            return self.compiled[key]
        fn = self._per_frame
        if self.compile_mode:
            fn = self.torch.compile(self._per_frame, mode=self.compile_mode)
        self.compiled[key] = fn
        return fn

    def warm(self, fp16, n=12):
        """A whole frame, end to end, the number of times it takes to stop being slow.

        WARMING THE NETWORK ALONE IS NOT ENOUGH, and the first version of this did
        exactly that. The rasteriser is a CUDA extension that initialises on its first
        call, the jpeg encoder probes for a GPU path on its first call, and a compiled
        graph is compiled on its first call. Together they were 5.2 seconds, and they
        were being paid by whoever clicked first -- which made the demo's headline
        number, time to first frame, a measurement of the warm-up rather than of the
        pipeline. Everything the frame loop touches is touched here instead.
        """
        torch = self.torch
        fn = self.forward_fn(fp16)
        with torch.no_grad():
            for _ in range(n):
                self.g.select_mesh_by_timestep(0)
                dm = self.g.get_vertex_displace_map()
                expr = self.g.ctrl[0].unsqueeze(0)
                if fp16:
                    with torch.autocast("cuda", dtype=torch.float16):
                        off = fn(expr, dm)
                else:
                    off = fn(expr, dm)
                off = self.g.freeze_rig_only(off)   # teeth/eyes take no network offset, as in training
                if self.appear is not None:
                    off = off.float()
                    off[..., 10:14] = self.appear
                img = self.render(self.cam, self.g, self.pipe, self.bg, offset=off)["render"]
                encode_jpeg((img.clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8))
        torch.cuda.synchronize()

    # -------------------------------------------------------- is it still the same?
    def verify_trunk(self):
        """The cached branches against the network's own forward, on a real mesh.

        Run at boot. The claim is bit-identical, not approximately equal, and a claim
        like that is worth nothing unless something checks it where it is used.
        """
        torch = self.torch
        with torch.no_grad():
            dm = self.g.get_vertex_displace_map()
            expr = self.g.condition[0].unsqueeze(0)
            ref = self.net(self.g.condition, 0, dm)
            got = self._per_frame(expr, dm)
            return float((ref - got).abs().max())


# ================================================================  A JOB  =======
# The model's own lookahead, in seconds. drive.py pads every window by this much on
# BOTH sides before asking the encoder, because the encoder's answer at a moment
# depends on what comes after it. It is the floor under any lag this pipeline can
# have, and no amount of hardware touches it.
LOOKAHEAD_S = 3.0


class Job:
    def __init__(self, jid, clip, fp16, pace, stream=False, chunk_s=2.0,
                 head="rest", wav=None, seconds=0.0, label="", body="", body_start=-1,
                 body_hold=False):
        # THE AUDIO IS A PATH, NOT A SHELF NAME. The demo began with a fixed list of
        # clips and the job looked one up by name; the moment anything else wants to
        # speak -- a wav posted in, a file dropped in the inbox, a microphone -- that
        # lookup is the only thing in the way. A job now just carries the wav it is
        # for, and where it came from is the caller's business.
        self.id, self.clip, self.fp16, self.pace = jid, clip, bool(fp16), bool(pace)
        self.wav = pathlib.Path(wav) if wav else (CLIPS / f"{clip}.wav")
        self.seconds, self.label = float(seconds), label or clip
        self.stream, self.chunk_s = bool(stream), float(chunk_s)
        self.head = head
        self.body, self.body_start = body, int(body_start)   # "" = no body; "random" or a chunk
        # hold: ONE real frame, and the head at that frame's own placement, for the whole
        # job. For listening: nothing moves but the face itself -- a still body under a
        # moving head would be a head sliding off its neck, so both stop together.
        self.body_hold = bool(body_hold)
        # TWO CONSUMERS, TWO DIFFERENT NEEDS.
        #   q       the live mjpeg view. Bounded and drop-oldest, because that view is
        #           about watching frames appear and a slow browser must never become
        #           the thing being measured.
        #   ndjson  the aligned view. Every frame, in order, WITH ITS INDEX, so the page
        #           can show frame n at second n/30 of the sound it is hearing. No
        #           frame may be dropped here: a missing index is a held still.
        self.q = queue.Queue(maxsize=4)
        self.nd = queue.Queue()
        self.dropped = 0
        self.stats = dict(state="starting", clip=clip, frames=0, total=0,
                          fp16=self.fp16, paced=self.pace, stream=self.stream,
                          chunk_s=self.chunk_s, lookahead_s=LOOKAHEAD_S,
                          head=self.head, pose_why=None,
                          label=self.label, seconds=self.seconds,
                          ttff_ms=None, ms_per_frame=None, fps=None,
                          audio=None, stages=None, note=None,
                          lag_ms=None, lag_min_ms=None, lag_max_ms=None,
                          audio_pos_s=0.0, video_pos_s=0.0, chunks=[])
        self.t_click = time.time()
        self.cancelled = False

    def emit(self, i, jpg):
        self.nd.put((i, jpg))
        self._offer(jpg)

    def close(self):
        """Both consumers told the job is over, WITHOUT BLOCKING.

        This used to be a plain `q.put(None)`, and `q` is bounded. When nobody was
        attached to the live view -- which is now the normal case, since the page reads
        the indexed stream instead -- the queue stayed full and the sentinel blocked
        FOREVER, on the one thread that owns the card. Every later clip then sat in the
        queue untouched while the page showed "listening" and no error anywhere. A
        bounded queue and a blocking put on a shutdown path is the whole bug.
        """
        self.nd.put(None)
        self._offer(None)

    def _offer(self, item):
        while True:
            try:
                self.q.put_nowait(item); return
            except queue.Full:
                try:
                    self.q.get_nowait(); self.dropped += 1
                except queue.Empty:
                    return


FRONTAL_MIN = 0.60      # filter_frontal.py's cut: the renderer saw nothing turned further


def frontal_window(R, names, need, min_run=90):
    """Where an answer's body comes from -> (clip, start, span).

    The answer's head replays the clip's own head track, and the renderer only knows
    him from the front (FRONTAL_MIN), so every replayed frame must be frontal. Runs of
    frontal frames are found per clip; a run long enough for the whole answer is used
    straight (span 0), otherwise one of the longest runs is played back and forth
    inside itself (span = its length). Never the whole clip back and forth: that
    reached frames where he had turned away and the face smeared (drk, 67 s answer).
    No frontal measurement for this creator: any clip, any start, as before."""
    import random
    runs = []
    for n_ in names:
        fr = getattr(R, "frontal", {}).get(n_)
        if fr is None:
            continue
        ok = np.r_[False, fr[:R.bodies[n_]["frames"]] > FRONTAL_MIN, False]
        e = np.flatnonzero(np.diff(ok.astype(np.int8)))
        runs += [(int(b - a), n_, int(a)) for a, b in zip(e[::2], e[1::2]) if b - a >= min_run]
    if not runs:
        n_ = random.choice(names)
        return n_, random.randint(0, max(R.bodies[n_]["frames"] - need, 0)), 0
    whole = [r for r in runs if r[0] >= need]
    if whole:
        L, n_, a = random.choice(whole)
        return n_, a + random.randint(0, L - need), 0
    runs.sort(reverse=True)
    L, n_, a = random.choice(runs[:5])
    return n_, a, L


def calm_segment(R, names, need, step=6):
    """The stillest stretch; see calm_candidates."""
    c = calm_candidates(R, names, need, k=1, step=step)
    if not c:
        return max(names, key=lambda x: R.bodies[x]["frames"]), 0
    return c[0][1], c[0][2]


def calm_candidates(R, names, need, k=8, step=6):
    """The stillest stretch of real footage `need` frames long, for a body at rest
    between answers rather than one caught mid-gesture.

    Measured on the picture, because hands are what move and no track sees them: tiny
    greyscale thumbnails every `step` frames over ALL of this creator's footage, and a
    window scores its mean change between neighbours plus its single biggest one, so
    one gesture inside an otherwise quiet stretch still rules it out.

    KEEP `need` SHORT (a few seconds). A creator who talks with their hands may never
    hold them down for 20 s -- huberman does not, anywhere in 57 clips -- and a long
    window then picks the quietest gesturing, which still shows raised hands. Tried and
    failed: likeness to the clip's median frame (his median is mid-gesture) and a skin
    colour test (dark shirt, dark set and beard defeat it). ~10 s, once per process."""
    from PIL import Image
    if not hasattr(R, "_calm_scan"):
        R._calm_scan = {}
    cands = []
    for n_ in names:
        B = R.bodies[n_]
        if n_ not in R._calm_scan:
            th = []
            for j in range(0, B["frames"], step):
                im = Image.open(B["images"] / f"{B['src0'] + j:06d}.jpg"); im.draft("L", (80, 45))
                th.append(np.asarray(im.convert("L").resize((80, 45)), np.float32))
            th = np.stack(th)
            R._calm_scan[n_] = np.abs(np.diff(th, axis=0)).mean((1, 2))
        mv = R._calm_scan[n_]
        w = max(need // step - 1, 1)
        if len(mv) < w:
            continue
        fr = getattr(R, "frontal", {}).get(n_)
        for a in range(0, len(mv) - w + 1):
            st = a * step
            if fr is not None and fr[st:st + need].min(initial=1.0) <= FRONTAL_MIN:
                continue
            cands.append((float(mv[a:a + w].mean() + mv[a:a + w].max()), n_, st))
    # the k best windows that do not overlap one another, so a contact sheet of them
    # shows k different moments rather than one moment k times
    out = []
    for sc, n_, st in sorted(cands):
        if all(n_ != o[1] or abs(st - o[2]) >= need for o in out):
            out.append((sc, n_, st))
        if len(out) == k:
            break
    return out


def body_prefetch(B, n, workers=4, ahead=48):
    """Decode the body clip's frames for output frames 0..n-1 on CPU threads, ahead of
    the renderer, in the same ping-pong order the worker replays the head in. Returns a
    getter: frame i -> uint8 [3,H,W], pinned for a quick upload.

    A WINDOW, not the whole clip: a 75 s answer is 2250 frames, ~6 GB decoded. Only
    `ahead` frames past the one asked for are in flight or held."""
    import concurrent.futures as cf
    from torchvision.io import decode_jpeg, read_file
    T = B["frames"]
    pool = cf.ThreadPoolExecutor(max_workers=workers)
    futs, lock = {}, threading.Lock()

    def src(i):
        if B.get("hold"):
            return B["start"]
        if B.get("span"):                      # back and forth inside the chosen stretch
            L = B["span"]; k = i % max(2 * L - 2, 1)
            return B["start"] + (k if k < L else 2 * L - 2 - k)
        k = (B["start"] + i) % max(2 * T - 2, 1)
        return k if k < T else 2 * T - 2 - k

    def load(i):
        return decode_jpeg(read_file(str(B["images"] / f"{B['src0'] + src(i):06d}.jpg"))
                           ).pin_memory()

    def get(i):
        i = min(i, n - 1)
        with lock:
            for j in range(i, min(i + ahead, n)):
                if j not in futs:
                    futs[j] = pool.submit(load, j)
            for j in [j for j in futs if j < i]:
                del futs[j]
            fut = futs[i]
        return fut.result()
    get(0)
    return get


def paste_on_body(R, g, off, B, i):
    """The head, rendered through the body clip's own camera, pasted into that clip's
    real frame -- all on the card.

    Only head blobs are drawn (R.nonhead hidden), on white and on black: the black
    render IS the premultiplied colour and white minus black is exactly 1 - coverage,
    so there is no fringe. The layer goes back through the clip's crop box. The body
    frame index ping-pongs exactly as the worker's replay of that clip's head track did.
    """
    torch, F = R.torch, R.torch.nn.functional
    off = off.clone().float()
    off[..., R.nonhead, 13] = -2.0                        # opacity clamps to 0
    wht = R.render(B["cam"], g, R.pipe, torch.ones(3, device="cuda"), offset=off)["render"]
    blk = R.render(B["cam"], g, R.pipe, torch.zeros(3, device="cuda"), offset=off)["render"]
    cov = (1 - (wht - blk).mean(0, keepdim=True)).clamp(0, 1)
    lay = torch.cat([blk.clamp(0, 1), cov], 0)[None]                  # [1,4,h,w]
    x, y, w, h = B["crop"]
    lay = F.interpolate(lay, size=(h, w), mode="bilinear", align_corners=False)[0]
    # Isolated specks out: a stray light blob is invisible on white and glaring on his
    # real background. Coverage is kept only where its surroundings are covered too, so
    # the head survives whole and lone dots do not. Then a soft join.
    # premultiplied, so colour and coverage are masked TOGETHER -- masking coverage
    # alone leaves the speck's colour added on top
    near = F.avg_pool2d(lay[3:][None], 15, 1, 7)[0]
    keep = F.avg_pool2d((near > 0.35).float()[None], 5, 1, 2)[0]
    lay = lay * keep
    # PULL THE EDGE IN a few pixels. The outermost blobs carry the background the
    # renderer was trained against (drk: white studio), a light rim on a dark room.
    # The real hair is underneath in the same place, so it fills the boundary.
    core = 1 - F.max_pool2d(1 - lay[3:][None], 9, 1, 4)[0]
    lay = lay * F.avg_pool2d(core[None], 5, 1, 2)[0]
    # decoded on the CPU, ahead of time, by the job's prefetch thread: decoding on the
    # card from the render thread returned corrupted frames (coloured noise) at random
    real = B["frames_ahead"](i).cuda(non_blocking=True).float() / 255   # [3,H,W]
    Hf, Wf = real.shape[1:]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, Wf), min(y + h, Hf)
    L = lay[:, y0 - y:y1 - y, x0 - x:x1 - x]
    real[:, y0:y1, x0:x1] = L[:3] + (1 - L[3:]) * real[:, y0:y1, x0:x1]
    return (real.clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8)


def render_job(R, worker, job):
    """One clip, start to finish. Frames go into the job's queue as they are made."""
    torch = R.torch
    st = job.stats
    try:
        wav = job.wav
        st["state"] = "listening"

        t0 = time.time()
        B = None
        if job.body:
            # A REAL BODY, picked at random unless named: a clip of his, and a point in
            # it. The worker replays that clip's head from there; the paste below puts
            # the rendered head back onto those very frames.
            import random
            # "random:<recording>" / "calm:<recording>" keep to one recording. A creator
            # filmed in several sets would otherwise answer in one room and rest in another
            # (drk: fireplace, green room, and a set with a chat overlay burned in).
            mode, _, rec = job.body.partition(":")
            if mode in ("random", "calm"):
                names = [n_ for n_ in R.bodies if not rec or n_.startswith(rec + "__")]
                if not names:
                    st["state"] = "error"; st["note"] = f"no body clip in recording {rec!r}"
                    job.close(); return
            else:
                names = [job.body]
            if not names or names[0] not in R.bodies:
                st["state"] = "error"; st["note"] = f"no body clip {job.body!r}"
                job.close(); return
            need = int(round(wav_seconds(wav) * FPS_OUT)) + 1
            fits = [n_ for n_ in names if R.bodies[n_]["frames"] >= need]
            # a clip long enough for the whole reply, so the body never has to reverse;
            # if none is, the longest, and the replay ping-pongs
            bname = random.choice(fits) if fits else max(names, key=lambda n_: R.bodies[n_]["frames"])
            if mode == "calm":
                bname, start = calm_segment(R, fits or names, need)
            B = dict(R.bodies[bname], name=bname)
            room = max(B["frames"] - need, 0)
            if job.body_start >= 0:
                B["start"] = job.body_start
            elif mode == "calm":
                B["start"] = start
            else:
                bname, s0, span = frontal_window(R, names, need)
                B = dict(R.bodies[bname], name=bname, start=s0, span=span)
            B["hold"] = job.body_hold
            st["body"] = dict(clip=bname, start=B["start"], frames=B["frames"], hold=B["hold"],
                              span=int(B.get("span", 0)))
            B["frames_ahead"] = body_prefetch(B, need)
        req = dict(wav_path=str(wav), head=job.head)
        if B is not None:
            req.update(body=B["name"], body_start=B["start"], body_hold=B["hold"],
                       body_span=int(B.get("span", 0)))
        reply = worker.request(req)
        if not reply.get("ok"):
            st["state"] = "error"; st["note"] = reply.get("error", "phase one failed")
            job.close(); return
        st["pose_why"] = reply.get("pose_why")
        st["audio"] = dict(seconds=reply["seconds"], frames=reply["frames"],
                           ms=round((time.time() - t0) * 1000, 1),
                           stages={k: round(v * 1000, 2) for k, v in reply["timings"].items()})
        n, nv = reply["frames"], reply["vertices"]

        # the geometry, where phase one left it -- no copy, no decompression
        t_g = time.time()
        raw = np.fromfile(reply["blob"], dtype=np.float32)
        verts = torch.from_numpy(raw[: n * nv * 3].reshape(n, nv, 3).copy())
        ctrl = torch.from_numpy(raw[n * nv * 3:].reshape(n, 257).copy()).cuda()
        st["audio"]["read_back_ms"] = round((time.time() - t_g) * 1000, 1)

        g = R.g
        g.verts_seq = g.verts_cano_seq = verts
        g.num_timesteps = n
        g.ctrl = ctrl

        st["state"] = "rendering"; st["total"] = n
        fn = R.forward_fn(job.fp16)
        ac = torch.autocast("cuda", dtype=torch.float16) if job.fp16 else _Null()

        stage = dict(mesh=0.0, network=0.0, raster=0.0, encode=0.0)
        t_start = time.time()
        first = None
        n_timed = n
        with torch.no_grad():
            for i in range(n):
                if job.cancelled:
                    st["state"] = "error"; st["note"] = "superseded by a later clip"; return
                a = time.time()
                g.select_mesh_by_timestep(i)
                dm = g.get_vertex_displace_map()
                expr = ctrl[i].unsqueeze(0)
                torch.cuda.synchronize(); b = time.time()

                with ac:
                    off = fn(expr, dm)
                off = g.freeze_rig_only(off)        # teeth/eyes take no network offset, as in training
                if R.appear is not None:
                    off = off.float()
                    off[..., 10:14] = R.appear
                torch.cuda.synchronize(); c = time.time()

                if B is None:
                    img = R.render(R.cam, g, R.pipe, R.bg, offset=off)["render"]
                    buf = (img.clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8)
                else:
                    buf = paste_on_body(R, g, off, B, i)
                torch.cuda.synchronize(); d = time.time()

                jpg = encode_jpeg(buf)
                e = time.time()

                stage["mesh"] += b - a; stage["network"] += c - b
                stage["raster"] += d - c; stage["encode"] += e - d

                if first is None:
                    first = time.time() - job.t_click
                    st["ttff_ms"] = round(first * 1000, 1)
                    st["first_frame_ms"] = round((e - a) * 1000, 1)
                    st["first_frame_split"] = dict(
                        mesh=round((b - a) * 1000, 1), network=round((c - b) * 1000, 1),
                        raster=round((d - c) * 1000, 1), encode=round((e - d) * 1000, 1))
                    # the first frame through a compiled graph pays for the graph. It
                    # is a real cost once per process, and averaging it into the clip
                    # would misreport every clip after it, so it is held out.
                    for k in stage:
                        stage[k] = 0.0
                    t_start = time.time()
                    n_timed = n - 1

                if job.pace:
                    due = t_start + (i + 1) / FPS_OUT
                    if due > time.time():
                        time.sleep(due - time.time())
                job.emit(i, jpg)

                el = time.time() - t_start
                done_n = max(i + 1 - (n - n_timed), 1)
                busy = sum(stage.values())
                st["frames"] = i + 1
                st["dropped"] = job.dropped
                # TWO RATES, AND THEY ARE DIFFERENT QUESTIONS.
                #   render_fps  how fast frames are being made. The answer to "how fast
                #               is it", and the same in both stream modes.
                #   fps         how fast they are leaving this process, which in paced
                #               mode is 30 by construction and says nothing.
                st["render_ms_per_frame"] = round(busy / done_n * 1000, 2)
                st["render_fps"] = round(done_n / busy, 1) if busy else None
                st["ms_per_frame"] = round(el / (i + 1) * 1000, 2)
                st["fps"] = round((i + 1) / el, 1)

        el = time.time() - t_start
        nt = max(n_timed, 1)
        st["stages"] = {k: round(v / nt * 1000, 2) for k, v in stage.items()}
        st["render_ms_per_frame"] = round(sum(stage.values()) / nt * 1000, 2)
        st["render_fps"] = round(nt / sum(stage.values()), 1)
        st["wall_ms_per_frame"] = round(el / n * 1000, 2)
        st["realtime_x"] = round((n / FPS_OUT) / sum(stage.values()), 2)
        st["wall_realtime_x"] = round((n / FPS_OUT) / el, 2)
        st["state"] = "done"
    except Exception as e:
        import traceback
        st["state"] = "error"; st["note"] = f"{e}"
        sys.stderr.write(traceback.format_exc())
    finally:
        job.close()


def stream_job(R, worker, job):
    """The clip as if the sound were arriving now, and the gap that opens up.

    THE QUESTION THIS ANSWERS is not how fast frames are made -- render_job already
    answers that -- but how far behind the speaker the face is. Those are different
    numbers and the second one is the one a listener experiences.

    THE CLOCK. t_zero is the instant the first sample arrives. At wall time T, the
    speaker has said T seconds. A frame belonging to audio second `a` is late by
    (the wall time it is shown) - a. That is the lag, and everything below is an
    account of where it comes from.

    WHY IT CANNOT BE SMALL, AND IT IS NOT THE HARDWARE

      lookahead   drive.py pads every window by LOOKAHEAD_S seconds on both sides,
                  because what the encoder says about a moment depends on the sound
                  AFTER it. To answer for second a the pipeline must have heard
                  second a + 3. Structural. A faster card does not touch it.
      the window  frames are produced a window at a time, so the last frame of a
                  window waits for the whole window. Shrink the window and this
                  shrinks -- but see the next line.
      the encode  the encoder reads a FIXED 30-second block whatever you ask of it,
                  so a 2-second window and a 20-second window cost the same encode.
                  Halving the window does not halve the work, it doubles it.

    That trade is the whole reason the window size is a knob here rather than a
    constant: it is the one thing in this pipeline where latency and throughput pull
    against each other, and the page lets you watch them do it.
    """
    torch = R.torch
    st = job.stats
    try:
        wav = job.wav
        dur = job.seconds or wav_seconds(wav)
        total = int(round(dur * FPS_OUT))
        st.update(state="listening", total=total)

        t_zero = time.time()            # the first sample arrives
        g = R.g
        fn = R.forward_fn(job.fp16)
        ac = torch.autocast("cuda", dtype=torch.float16) if job.fp16 else _Null()

        stage = dict(mesh=0.0, network=0.0, raster=0.0, encode=0.0)
        emitted = 0
        t_first = None
        lags = []
        C = max(job.chunk_s, 0.5)
        a = 0.0
        with torch.no_grad():
            while a < dur - 1e-6:
                if job.cancelled:
                    st["state"] = "error"; st["note"] = "superseded by a later clip"; return
                b = min(a + C, dur)
                # THE WAIT. In a live setting the sound does not exist yet. The window
                # ends at b and the model needs LOOKAHEAD_S beyond it, so nothing about
                # this window can be computed before then. Enforced rather than assumed,
                # because skipping it is how a streaming demo reports a latency nobody
                # could actually have.
                need = b + LOOKAHEAD_S
                heard = time.time() - t_zero
                waited = max(need - heard, 0.0)
                if waited > 0:
                    st["state"] = "waiting for the sound"
                    time.sleep(waited)

                t_req = time.time()
                rep = worker.request(dict(wav_path=str(wav), t_from=a, t_to=b,
                                          frame0=int(round(a * FPS_OUT)),
                                          head=job.head))
                if not rep.get("ok"):
                    st.update(state="error", note=rep.get("error", "phase one failed"))
                    job.close(); return
                # the pose provenance travels with every window, not just the first.
                # A moving head that nothing says is GENERATED is the one thing this
                # demo must never imply it predicted.
                if st.get("pose_why") is None:
                    st["pose_why"] = rep.get("pose_why")
                enc = rep["timings"]["speech to controls"]
                stage["encode"] += enc
                n, nv = rep["frames"], rep["vertices"]
                raw = np.fromfile(rep["blob"], dtype=np.float32)
                verts = torch.from_numpy(raw[: n * nv * 3].reshape(n, nv, 3).copy())
                ctrl = torch.from_numpy(raw[n * nv * 3:].reshape(n, 257).copy()).cuda()
                g.verts_seq = g.verts_cano_seq = verts
                g.num_timesteps = n
                g.ctrl = ctrl
                t_ready = time.time()

                st["state"] = "rendering"
                for i in range(n):
                    if job.cancelled:
                        st["state"] = "error"
                        st["note"] = "superseded by a later clip"; return
                    p0 = time.time()
                    g.select_mesh_by_timestep(i)
                    dm = g.get_vertex_displace_map()
                    expr = ctrl[i].unsqueeze(0)
                    torch.cuda.synchronize(); p1 = time.time()
                    with ac:
                        off = fn(expr, dm)
                    off = g.freeze_rig_only(off)    # teeth/eyes take no network offset, as in training
                    if R.appear is not None:
                        off = off.float(); off[..., 10:14] = R.appear
                    torch.cuda.synchronize(); p2 = time.time()
                    img = R.render(R.cam, g, R.pipe, R.bg, offset=off)["render"]
                    buf = (img.clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8)
                    torch.cuda.synchronize(); p3 = time.time()
                    jpg = encode_jpeg(buf)
                    p4 = time.time()
                    stage["mesh"] += p1 - p0; stage["network"] += p2 - p1
                    stage["raster"] += p3 - p2

                    # THE PLAYOUT CLOCK. The first finished frame sets the lag; after
                    # that the face plays at 30 fps like anything else, and the lag only
                    # changes when the pipeline fails to keep a frame ready in time.
                    audio_t = (emitted) / FPS_OUT
                    if t_first is None:
                        t_first = time.time()
                        st["ttff_ms"] = round((t_first - t_zero) * 1000, 1)
                    due = t_first + audio_t
                    now = time.time()
                    if now < due:
                        time.sleep(due - now)
                    job.emit(emitted, jpg)
                    emitted += 1
                    shown = time.time() - t_zero
                    lag = shown - audio_t
                    lags.append(lag)
                    st.update(frames=emitted, lag_ms=round(lag * 1000, 0),
                              lag_min_ms=round(min(lags) * 1000, 0),
                              lag_max_ms=round(max(lags) * 1000, 0),
                              audio_pos_s=round(min(shown, dur), 2),
                              video_pos_s=round(audio_t + 1 / FPS_OUT, 2),
                              dropped=job.dropped)
                    busy = stage["mesh"] + stage["network"] + stage["raster"]
                    st["render_ms_per_frame"] = round(busy / emitted * 1000, 2)
                    st["render_fps"] = round(emitted / busy, 1) if busy else None

                st["chunks"].append(dict(
                    window=[round(a, 2), round(b, 2)], frames=n,
                    waited_for_sound_ms=round(waited * 1000, 0),
                    encode_ms=round(enc * 1000, 0),
                    phase_one_ms=round((t_ready - t_req) * 1000, 0),
                    render_ms=round(n * st["render_ms_per_frame"], 0)))
                a = b

        st["stages"] = {k: round(v / max(emitted, 1) * 1000, 2) for k, v in stage.items()}
        st["stages"]["encode (per window, not per frame)"] = round(
            stage["encode"] / max(len(st["chunks"]), 1) * 1000, 0)
        st["lag_mean_ms"] = round(sum(lags) / len(lags) * 1000, 0) if lags else None
        st["realtime_x"] = round(
            (emitted / FPS_OUT) / (stage["mesh"] + stage["network"] + stage["raster"]), 2)
        st["encode_duty"] = round(stage["encode"] / max(dur, 1e-6), 3)
        st["state"] = "done"
    except Exception as e:
        import traceback
        st.update(state="error", note=f"{e}")
        sys.stderr.write(traceback.format_exc())
    finally:
        job.close()


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


_JPEG = {"mode": None}


def encode_jpeg(buf_hwc_uint8):
    """A frame as jpeg bytes, on the card if torchvision will do it there."""
    import torch
    if _JPEG["mode"] is None:
        try:
            from torchvision.io import encode_jpeg as tvenc
            tvenc(buf_hwc_uint8.permute(2, 0, 1).contiguous(), quality=88)
            _JPEG["mode"] = "torchvision"
        except Exception:
            _JPEG["mode"] = "pil"
    if _JPEG["mode"] == "torchvision":
        from torchvision.io import encode_jpeg as tvenc
        return bytes(tvenc(buf_hwc_uint8.permute(2, 0, 1).contiguous(), quality=88)
                     .cpu().numpy())
    from PIL import Image
    bio = io.BytesIO()
    Image.fromarray(buf_hwc_uint8.cpu().numpy()).save(bio, "JPEG", quality=88)
    return bio.getvalue()


# ============================================================  PHASE ONE  =======
class Worker:
    """The resident phase-one process, and the pipe to it."""

    def __init__(self, env_python, profile, run):
        self.lock = threading.Lock()
        self.proc = subprocess.Popen(
            [env_python, str(HERE / "audio_worker.py"), "--profile", profile,
             "--run", run],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
            cwd=str(M))
        line = self.proc.stdout.readline()
        self.boot = json.loads(line)

    def request(self, req):
        with self.lock:
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            return json.loads(self.proc.stdout.readline())


# ===============================================================  HTTP  =========
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        R, worker = STATE["renderer"], STATE["worker"]

        if u.path in ("/", "/index.html"):
            return self._send(200, "text/html; charset=utf-8",
                              (HERE / "index.html").read_bytes())

        if u.path == "/debug":
            return self._send(200, "application/json", json.dumps(dict(
                queued=JOBQ.qsize(),
                threads=sorted(x.name for x in threading.enumerate()),
                jobs={k: v.stats["state"] for k, v in STATE["jobs"].items()},
            )).encode())

        if u.path == "/shelf":
            shelf = json.loads((CLIPS / "shelf.json").read_text())
            _b = max(int((HERE / f).stat().st_mtime)
                     for f in ("server.py", "index.html", "audio_worker.py"))
            return self._send(200, "application/json", json.dumps(dict(
                clips=shelf, build=_b,
                model=dict(run=R.run, iteration=R.it, blobs=R.blobs,
                           size=f"{R.W}x{R.H}", uv_size=R.flat.get("uv_size", 256),
                           camera=R.cam_from, look=getattr(R, "look", None),
                           compiled=bool(R.compile_mode),
                           trunk_difference=STATE["trunk_diff"],
                           boot_s=round(STATE["boot_s"], 1),
                           worker_boot_s=round(worker.boot.get("boot_s", 0), 1)),
            )).encode())

        if u.path.startswith("/clip/"):
            f = CLIPS / pathlib.Path(u.path[len("/clip/"):]).name
            if not f.exists():
                return self._send(404, "text/plain", b"no such clip")
            # THE PAGE PLAYS THE WAV, NOT THE MP4, and that is the point: it is the
            # exact file the model was given, so frame n and second n/30 of what the
            # listener hears cannot drift apart. The mp4's audio track is a re-encode
            # of it and its container is a few milliseconds longer.
            ct = {".wav": "audio/wav", ".mp4": "video/mp4"}.get(f.suffix, "application/octet-stream")
            return self._send(200, ct, f.read_bytes(), {"Accept-Ranges": "none"})

        if u.path == "/run":
            clip = q.get("clip", [""])[0]
            if not (CLIPS / f"{clip}.wav").exists():
                return self._send(404, "application/json", b'{"error":"no such clip"}')
            meta = next((c for c in json.loads((CLIPS / "shelf.json").read_text())
                         if c["id"] == clip), None)
            return self._send(200, "application/json", json.dumps(dict(
                job=start_job(q, clip=clip, wav=CLIPS / f"{clip}.wav",
                              seconds=float(meta["seconds"]) if meta else 0.0,
                              label=clip))).encode())

        if u.path == "/latest":
            return self._send(200, "application/json",
                              json.dumps(dict(job=STATE.get("latest"))).encode())

        if u.path == "/audio":
            job = STATE["jobs"].get(q.get("job", [""])[0])
            if job is None or not job.wav.exists():
                return self._send(404, "text/plain", b"no audio for that job")
            # THE PAGE ALWAYS PLAYS THE JOB'S OWN WAV, whatever produced it. Frame n
            # belongs to second n/30 of exactly these samples, so nothing the caller
            # does upstream can put the sound and the face on different clocks.
            return self._send(200, "audio/wav", job.wav.read_bytes(),
                              {"Accept-Ranges": "none"})


        # THE RESTING STRETCH IS CHOSEN BY LOOKING. /calm proposes the stillest windows
        # of real footage; /calm_sheet draws one as a contact sheet; whoever runs the
        # live-app recipe (Claude) looks at the sheets and writes the pick into the
        # creator's file. Stillness is necessary, not sufficient: it cannot see a hand
        # held up and still, eyes shut, or a look off camera.
        if u.path == "/calm":
            R = STATE.get("R")
            if R is None or not getattr(R, "bodies", None):
                return self._send(404, "application/json", b'{"error":"no body clips"}')
            secs = float(q.get("seconds", ["4"])[0]); k = int(q.get("k", ["8"])[0])
            need = int(round(secs * FPS_OUT))
            rec = q.get("recording", [""])[0]      # keep to one set, as body=random:<rec>
            c = calm_candidates(R, [n_ for n_ in R.bodies
                                    if not rec or n_.startswith(rec + "__")], need, k=k)
            return self._send(200, "application/json", json.dumps(
                [dict(clip=n_, start=st, score=round(sc, 3), seconds=secs) for sc, n_, st in c]
            ).encode())
        if u.path == "/calm_sheet":
            R = STATE.get("R")
            n_ = q.get("clip", [""])[0]
            if R is None or n_ not in getattr(R, "bodies", {}):
                return self._send(404, "text/plain", b"no such clip")
            from PIL import Image
            B = R.bodies[n_]; st = int(q.get("start", ["0"])[0])
            need = int(round(float(q.get("seconds", ["4"])[0]) * FPS_OUT))
            js = np.linspace(st, min(st + need, B["frames"]) - 1, 8).astype(int)
            sheet = Image.new("RGB", (4 * 480, 2 * 270))
            for i_, j in enumerate(js):
                im = Image.open(B["images"] / f"{B['src0'] + j:06d}.jpg").convert("RGB")
                sheet.paste(im.resize((480, 270)), (480 * (i_ % 4), 270 * (i_ // 4)))
            buf = io.BytesIO(); sheet.save(buf, "JPEG", quality=85)
            return self._send(200, "image/jpeg", buf.getvalue())

        if u.path == "/stats":
            job = STATE["jobs"].get(q.get("job", [""])[0])
            if job is None:
                return self._send(404, "application/json", b'{"error":"no such job"}')
            return self._send(200, "application/json", json.dumps(job.stats).encode())

        if u.path == "/frames":
            job = STATE["jobs"].get(q.get("job", [""])[0])
            if job is None:
                return self._send(404, "text/plain", b"no such job")
            return self._ndjson(job)

        if u.path == "/stream":
            job = STATE["jobs"].get(q.get("job", [""])[0])
            if job is None:
                return self._send(404, "text/plain", b"no such job")
            return self._stream(job)

        return self._send(404, "text/plain", b"not found")

    def do_POST(self):
        """Audio in, from whatever made it. This is the entry point a text-to-speech
        reply, a microphone or another service uses; the shelf is just a convenience
        on top of the same machinery."""
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path != "/speak":
            return self._send(404, "text/plain", b"not found")
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 200 * 1024 * 1024:
            return self._send(400, "application/json", b'{"error":"no audio in the body"}')
        raw = self.rfile.read(n)
        SPOKEN.mkdir(parents=True, exist_ok=True)
        stamp = f"{int(time.time()*1000)%10**9}"
        src = SPOKEN / f"in_{stamp}.bin"
        src.write_bytes(raw)
        wav = SPOKEN / f"in_{stamp}.wav"
        try:
            to_16k_mono(src, wav)
        except subprocess.CalledProcessError:
            return self._send(400, "application/json",
                              b'{"error":"ffmpeg could not read that audio"}')
        finally:
            src.unlink(missing_ok=True)
        label = q.get("label", ["spoken"])[0]
        jid = start_job(q, clip=label, wav=wav, seconds=wav_seconds(wav), label=label)
        return self._send(200, "application/json", json.dumps(
            dict(job=jid, seconds=wav_seconds(wav), label=label)).encode())

    def _ndjson(self, job):
        """Every frame, in order, each carrying the index of the 1/30th of a second it
        belongs to. The page holds them and paints against the audio element's own
        clock, which is the only clock the listener actually experiences."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(b):
            self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n")

        try:
            while True:
                try:
                    item = job.nd.get(timeout=1.0)
                except queue.Empty:
                    if job.stats["state"] in ("done", "error"):
                        break
                    continue
                if item is None:
                    break
                i, jpg = item
                chunk((json.dumps({"i": i, "b64": base64.b64encode(jpg).decode()})
                       + "\n").encode())
                self.wfile.flush()
            chunk(json.dumps({"end": True, "state": job.stats["state"]}).encode() + b"\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream(self, job):
        """Frames as they are produced. Nothing is buffered to disk, ever."""
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        try:
            while True:
                try:
                    jpg = job.q.get(timeout=1.0)
                except queue.Empty:
                    if job.stats["state"] in ("done", "error"):
                        break
                    continue
                if jpg is None:
                    break
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpg)).encode() +
                                 b"\r\n\r\n" + jpg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


JOBQ = queue.Queue()


def start_job(q, clip, wav, seconds, label):
    """Queue one clip and abandon whatever was pending.

    ONE CARD, ONE QUESTION. A new request means "show me this now", not "queue it
    behind the last four" -- and a backlog of jobs nobody is watching used to hold
    the render thread for minutes.
    """
    jid = f"j{int(time.time()*1000)%10**9}"
    for old_job in STATE["jobs"].values():
        if old_job.stats["state"] not in ("done", "error"):
            old_job.cancelled = True
    while True:
        try:
            JOBQ.get_nowait()
        except queue.Empty:
            break
    job = Job(jid, clip, q.get("fp16", ["0"])[0] == "1",
              q.get("pace", ["1"])[0] == "1",
              stream=q.get("stream", ["0"])[0] == "1",
              chunk_s=float(q.get("chunk", ["2.0"])[0]),
              head=q.get("head", ["rest"])[0],
              body=q.get("body", [STATE.get("body_default", "")])[0],
              body_start=int(q.get("body_start", ["-1"])[0]),
              body_hold=q.get("body_hold", ["0"])[0] == "1",
              wav=wav, seconds=seconds, label=label)
    STATE["jobs"][jid] = job
    for k in list(STATE["jobs"])[:-6]:
        STATE["jobs"].pop(k, None)
    JOBQ.put(job)
    return jid


def render_thread(R, worker, ready):
    """ONE thread owns the card, for its whole life.

    Not for exclusion -- a lock would have done that. A compiled graph pays a large
    one-off cost the first time it is entered on a given thread, and a thread per
    request pays it per request: measured here at 5.24 seconds on the first frame of
    EVERY clip, all of it inside the network, while frames two onward ran at 6 ms.
    Warming on the main thread did not help, because the main thread is not the one
    that renders.

    So the thread is created once, warmed once inside itself, and then takes clips off
    a queue forever. Time to first frame drops from 6.1 s to what phase one costs.
    """
    for fp16 in (False, True):
        t0 = time.time()
        R.warm(fp16)
        print(f"  {'fp16' if fp16 else 'fp32'}: {time.time()-t0:.1f} s", flush=True)
    ready.set()
    while True:
        job = JOBQ.get()
        if job is None:
            return
        if job.cancelled:
            job.stats["state"] = "error"
            job.stats["note"] = "superseded by a later clip"
            job.close(); continue
        (stream_job if job.stream else render_job)(R, worker, job)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64


# ================================================================  BOOT  ========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--run", default="G4_teeth")
    ap.add_argument("--iteration", type=int, default=-1)
    ap.add_argument("--render-scale", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8730)
    ap.add_argument("--compile", dest="compile_", action="store_true",
                    help="run the per-frame network through torch.compile. Costs about "
                         "10 s at boot and measured 6.50 -> 5.91 ms in full precision, "
                         "6.50 -> 2.52 with --fp16 selectable in the page")
    ap.add_argument("--python", default=str(pathlib.Path.home() /
                                            "miniconda3/envs/stavatar/bin/python"))
    ap.add_argument("--body", default="",
                    help="the body every job is drawn on unless the request names one: "
                         "'' for the studio render, 'random' for a real clip of this creator, 'calm' for its stillest stretch, "
                         "or a chunk name. Inbox jobs carry no query, so this "
                         "is how a caller that hands audio over by file gets a body")
    a = ap.parse_args()
    STATE["body_default"] = a.body

    if not (CLIPS / "shelf.json").exists():
        raise SystemExit(f"no clips yet. Run:\n"
                         f"  python {HERE / 'build_clips.py'} --profile {a.profile}")

    t0 = time.time()
    # PHASE ONE FIRST, and not for tidiness. The renderer's per-frame network is built
    # around a position map rasterised from whatever mesh happens to be selected, and
    # that map is a buffer it never writes again. Built around a placeholder it is
    # degenerate, every frame comes out NaN, and nothing anywhere raises -- the page
    # just shows a blank square. So one real posed head is fetched first and the
    # renderer is constructed around it.
    print("starting the phase-one worker ...", flush=True)
    worker = Worker(a.python, a.profile, a.run)
    print(f"  ready in {worker.boot.get('boot_s', 0):.1f} s", flush=True)

    seed_clip = sorted(CLIPS.glob("*.wav"))[0]
    seed = worker.request(dict(wav_path=str(seed_clip), seconds=1.0))
    if not seed.get("ok"):
        raise SystemExit(f"phase one failed on {seed_clip.name}: {seed.get('error')}")
    raw = np.memmap(seed["blob"], dtype=np.float32, mode="r")
    n, nv = seed["frames"], seed["vertices"]
    seed_v = np.array(raw[: n * nv * 3]).reshape(n, nv, 3)[:1]
    seed_c = np.array(raw[n * nv * 3:]).reshape(n, 257)[:1]
    print(f"  one real head to build the position map around, from {seed_clip.stem}")

    print("loading the renderer ...", flush=True)
    R = Renderer(a.profile, a.run, a.iteration, a.render_scale,
                 "default" if a.compile_ else None, seed_v, seed_c)
    diff = R.verify_trunk()
    print(f"  {R.blobs:,} blobs, {R.W}x{R.H}, uv {R.flat.get('uv_size', 256)}")
    print(f"  cached trunk vs the network's own forward: max difference {diff:.3e}"
          f"  {'(bit-identical)' if diff == 0.0 else '(NOT identical -- look at this)'}")
    if not np.isfinite(diff):
        raise SystemExit("the cached branches are NaN. The position map is degenerate; "
                         "the mesh handed to the renderer was not a real head.")
    nb = R.load_bodies()
    STATE["R"] = R
    print(f"  {nb} real-body clips ready for body=random "
          f"({int((~R.nonhead).sum()):,} head blobs drawn on a body)")

    print("warming the whole frame path on the thread that will render, both "
          "precisions ...", flush=True)
    ready = threading.Event()
    threading.Thread(target=render_thread, args=(R, worker, ready), daemon=True).start()
    ready.wait()

    STATE.update(renderer=R, worker=worker, jobs={}, latest=None, trunk_diff=diff,
                 boot_s=time.time() - t0)

    # AUDIO THAT LANDS, RATHER THAN AUDIO THAT IS ASKED FOR. Anything that can write
    # a file can drive the face: drop a wav in inbox/ and it speaks. No client, no
    # protocol, nothing to integrate against -- which is the point, because the thing
    # that will eventually feed this is a text-to-speech process that already writes
    # wavs. POST /speak is the same path for callers that would rather not touch disk.
    INBOX.mkdir(parents=True, exist_ok=True)
    SPOKEN.mkdir(parents=True, exist_ok=True)

    def watch_inbox():
        seen = set()
        while True:
            try:
                for f in sorted(INBOX.iterdir()):
                    if f.name.startswith(".") or f in seen or not f.is_file():
                        continue
                    size = f.stat().st_size
                    time.sleep(0.25)                 # let the writer finish
                    if f.stat().st_size != size:
                        continue
                    seen.add(f)
                    wav = SPOKEN / f"inbox_{int(time.time()*1000)%10**9}.wav"
                    try:
                        to_16k_mono(f, wav)
                    except subprocess.CalledProcessError:
                        print(f"[inbox] ffmpeg could not read {f.name}", flush=True)
                        continue
                    jid = start_job({}, clip=f.stem, wav=wav,
                                    seconds=wav_seconds(wav), label=f.stem)
                    STATE["latest"] = jid
                    print(f"[inbox] {f.name} -> job {jid}", flush=True)
            except Exception as e:
                print(f"[inbox] {e}", flush=True)
            time.sleep(0.5)

    threading.Thread(target=watch_inbox, daemon=True).start()

    srv = Server(("127.0.0.1", a.port), Handler)
    print(f"\nready in {time.time()-t0:.1f} s   ->  http://127.0.0.1:{a.port}\n"
          f"every request from here costs nothing to start.\n"
          f"speak by dropping audio in  {INBOX}\n"
          f"or:  curl --data-binary @reply.wav "
          f"'http://127.0.0.1:{a.port}/speak?label=reply'\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            worker.proc.stdin.write('{"quit":true}\n'); worker.proc.stdin.flush()
        except Exception:
            worker.proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
