#!/usr/bin/env python3
"""Phase one, resident. Sound in, posed meshes out, for as many requests as you like.

WHY THIS EXISTS AS A PROCESS AND NOT A FUNCTION

Two reasons, and only the first is the one people expect.

  The audio model and the renderer both ship a top-level package called `utils`, so
  they cannot be imported into the same interpreter. infer.py solves that by running
  phase one as a CHILD PER INVOCATION, which is correct for a command and wrong for a
  server: it reloads the driver, the rig and the corrective layer on every request,
  and that is 2.2 seconds before any sound has been read.

  Held open, that cost is paid once at boot and never again.

WHAT IT DOES DIFFERENTLY FROM infer.py, AND WHY IT IS STILL THE SAME ANSWER

infer.py copies the rig's output off the graphics card and then runs three steps on
the processor, one frame at a time: the corrective layer, the fit into the export's
frame, and the placement. Measured, that is 5.5 ms a frame, and the largest part of
it is the layer reading a 152 MB table once per frame to produce one correction.

All three are matrix multiplies over the whole clip. Done where the geometry already
is, they are 0.11 ms a frame together. The arithmetic is identical -- the same
subtraction, the same contraction, the same blend by the rig's own skin weights --
in single precision rather than double, on a correction whose own magnitude is
0.47 mm. `--verify` checks a clip both ways and prints the largest disagreement in
millimetres, so that claim is measured rather than asserted.

PROTOCOL: one JSON request per line on stdin, one JSON reply per line on stdout.
The meshes do not travel through the pipe. They are written to a shared-memory file
and the reply names it, because 96 MB down a pipe is a copy nobody needs.
"""
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent.parent
SHM = pathlib.Path("/dev/shm")
FPS_OUT = 30.0

# THE TWO THINGS THAT HAVE TO BE TRUE ABOUT WHERE THE SOUND SITS.
#
# LEAD_S  The encoder reads a FIXED 30-second block and its answer depends on where
#         inside that block the sound sits. drive.py pads each window by 3 s on both
#         sides, then clamps the read to the start of the file -- so audio beginning
#         at t=0 gets NO lead-in at all, and the clamp silently swallows the lag with
#         it. Measured on held-out chunk c089 against his tracked mouth: at t=0 the
#         predicted aperture correlates -0.078 with what he actually did. Placed 5 s
#         in, the same audio through the same model gives +0.760. Same weights, same
#         samples, only the position in the block changed.
#           at t=0          58% amplitude   r = -0.078
#           2 s in          85%             r = +0.295
#           5 s in          87%             r = +0.760
#           10 s in         87%             r = +0.759
#         Five seconds is where it saturates, so five is what is used.
#
# LAG_S   Cooking pairs the face at video time t with audio at t + sample_offset_ms,
#         about -110 ms on all three of drk's recordings, +105 ms on huberman's one. The model therefore learned
#         that relation and needs it asked for at inference too. Same chunk, audio
#         positioned correctly, lag on vs off: +0.664 vs +0.091 at zero offset.
#         infer.py applies it on --clip and NOT on --audio, which is why novel-audio
#         renders have never been in sync.
LEAD_S = 5.0
SLICE = 300        # frames of geometry built at once on the card (10 s)


class Phase1:
    def __init__(self, profile, run):
        sys.path.insert(0, str(PIPE / "gauss"))
        sys.path.insert(0, str(PIPE / "rigfit"))
        sys.path.insert(0, str(PIPE / "offset"))
        sys.path.insert(0, str(PIPE / "gauss/recipes/mesh-to-render/tools"))
        import torch
        from _profile import load, run_assets
        from drive import Driver, SR
        from decoder import Decoder, resolve
        from head_weight import HeadWeight
        from riglogic_torch import TorchRig
        from utils.baseline import read_wav16k
        import infer as INFER

        self.torch, self.SR, self.read_wav16k = torch, SR, read_wav16k
        self.p = p = load(profile)
        self.INFER = INFER

        self.drv = Driver(p.AUDIO_MODEL_DIR, device="cuda")
        self.dec = Decoder(resolve(p.AUDIO_MODEL_DIR))

        sub = p.RIG_DIR                       # the released rig, checkpoints/<subject>/face/rig
        self.rig = TorchRig(str(sub / "rig_beltrami.npz"), device="cuda")
        V_dna = np.load(sub / "rig_beltrami.npz")["m0_V0"].astype(np.float64)
        Wr = np.load(sub / "beltrami_wrap.npz")["verts_wrapped"].astype(np.float64)
        s0, R0, t0 = INFER.similarity(V_dna, Wr)

        hw = HeadWeight(p.SUBJECT, rig=str(sub / "rig_beltrami.npz"))
        self.N_HEAD = len(hw.w)

        # WHICH MESHES THIS RUN'S RENDERER EXPECTS. Same resolution infer.py does:
        # a run trained on the four-part asset has blobs bound to triangles past the
        # head's last one, and handing it a head-only mesh indexes off the end.
        self.MESHES = [0]
        assets = None
        for c in (p.RUNS / run / "cfg_args", p.RELEASE / run / "cfg_args"):
            if c.exists() and "mesh_assets=" in c.read_text():
                assets = c.read_text().split("mesh_assets=")[1].split(",")[0].strip().strip("'\"")
        if assets:
            z = np.load(run_assets(p, assets))
            if "parts" in z.files:
                parts = json.loads(str(z["parts"]))
                if len(parts) > 1:
                    self.MESHES = [q["dna_mesh_index"] for q in parts]
                    extra = [HeadWeight(p.SUBJECT, rig=str(sub / "rig_beltrami.npz"),
                                        mesh=m).w for m in self.MESHES[1:]]
                    hw.w = np.concatenate([hw.w] + extra); hw.wc = hw.w[:, None]
        self.hw = hw

        # ---- everything the per-clip path needs, on the card, once ------------
        d = "cuda"
        self.R0 = torch.as_tensor(R0, dtype=torch.float32, device=d)
        self.t0 = torch.as_tensor(t0, dtype=torch.float32, device=d)
        self.s0 = float(s0)
        self.wc = torch.as_tensor(hw.wc, dtype=torch.float32, device=d)
        if self.dec.m is not None:
            self.lm = torch.as_tensor(self.dec.m, dtype=torch.float32, device=d)
            self.lB = torch.as_tensor(self.dec.B.reshape(self.dec.B.shape[0], -1),
                                      dtype=torch.float32, device=d)
            self.lc = torch.as_tensor(self.dec.cbar, dtype=torch.float32, device=d)
        else:
            self.lm = None

        # the resting placement the deploy camera is framed on -- constant, so it is
        # resolved once rather than per request
        Rb, tb, self.dep = INFER.deploy_reference(p)
        self.Rb = torch.as_tensor(Rb, dtype=torch.float32, device=d)
        self.tb = torch.as_tensor(tb, dtype=torch.float32, device=d)
        self.head6 = HeadWeight.channels(Rb[None], tb[None])[0].astype(np.float32)
        self.HeadWeight = HeadWeight
        self._pose_cache = {}

        al = json.load(open(PIPE / "rigfit/cache/align.json"))
        # this creator's recordings only: align.json holds every creator's, and their
        # lags need not agree (drk -110 ms, huberman +105 ms)
        self.lag_s = float(np.median([v.get("sample_offset_ms", -v["lag_ms"])
                                      for k, v in al.items() if k in p.RECORDINGS])) / 1000.0

    # ------------------------------------------------------------------ the work
    def pose(self, policy, n, frame0=0, ref=None, start=0, span=0):
        """The head's placement per frame, from infer.py's own policies.

        `rest` holds him at a placement he actually held -- correct, verifiable, and
        the reason every clip looks alike: his real head moves a great deal and this
        one does not move at all. `generate` samples motion with his statistics about
        that same placement. It is NOT his motion for this audio, and nothing predicts
        that beyond about a third of a second, so it is honest as movement and dishonest
        as a prediction. The video's sidecar says which was used; so does the page.
        """
        import numpy as _np
        if policy == "rest":
            return None, None
        key = (policy, ref, int(start), int(span))
        R, t, rb, tb, why = self._pose_cache.get(key, (None,) * 5)
        if R is None or len(R) < frame0 + n:
            need = max(frame0 + n, 4096)
            R, t, rb, tb, why = self.INFER.pose_track(policy, self.p, need, ref=ref,
                                                      start=start, span=span)
            self._pose_cache[key] = (R, t, rb, tb, why)
        self._pose_ref = (rb, tb)          # where the body sits, for this request
        return (_np.ascontiguousarray(R[frame0:frame0 + n]),
                _np.ascontiguousarray(t[frame0:frame0 + n])), why

    def run(self, wav_path, seconds=0.0, blink="generate", blink_rate=17.0, seed=0,
            t_from=0.0, t_to=0.0, frame0=0, head="rest", body=None, body_start=0,
            body_hold=False, body_span=0):
        """One clip, or one slice of one.

        `t_from`/`t_to` exist for streaming: the caller asks for the frames covering a
        window of the sound rather than the whole file. THE COST IS NOT PROPORTIONAL.
        The encoder reads a fixed 30-second block whatever you ask for, so a 2-second
        window costs the same encode as the whole clip -- which is the structural fact
        a streaming demo is there to show, not to hide.

        `frame0` keeps the blink generator on one timeline across windows; seeded from
        zero in each window it would blink at the same instant in every one.
        """
        torch = self.torch
        T = {}

        t = time.time()
        wav = self.read_wav16k(str(wav_path))
        dur = len(wav) / self.SR
        if seconds:
            dur = min(dur, seconds)
        lo, hi = float(t_from), (float(t_to) if t_to else dur)
        hi = min(hi, dur)
        ts = np.arange(round(lo * FPS_OUT), round(hi * FPS_OUT)) / FPS_OUT
        n = len(ts)
        if n == 0:
            return dict(ok=False, error=f"no frames in [{lo:.2f}, {hi:.2f})")
        T["read wav"] = time.time() - t

        t = time.time()
        # the sound is placed LEAD_S into what the encoder reads, and asked for at the
        # offset the model was trained through. See the note at the top of this file --
        # both were missing and together they cost the entire lip sync.
        lead = np.zeros(int(LEAD_S * self.SR), np.float32)
        c257 = self.drv.controls(np.concatenate([lead, wav]),
                                 ts + LEAD_S + self.lag_s)
        T["speech to controls"] = time.time() - t

        t = time.time()
        if blink != "none":
            from blink_gen import BlinkGen
            # sampled on the WHOLE clip's timeline and sliced, so a window boundary
            # is not a place where he blinks again
            bl = BlinkGen(rate_per_min=blink_rate).sample(frame0 + n, seed=seed,
                                                          fps=FPS_OUT)[frame0:frame0 + n]
            c257[:, 10:12] = bl[:, None]
        # A REAL BODY: the head replays that clip's own track from `body_start`, about
        # that clip's own resting placement, so it lands where it was filmed and the
        # renderer can paste it back onto those frames.
        if body:
            head = f"replay:{body}"
        self._pose_ref = None
        pose_seq, pose_why = self.pose(head, n, frame0, ref=body, start=body_start,
                                       span=body_span if body else 0)
        if body and body_hold and pose_seq is not None:
            # the placement of that one frame, held: the renderer holds the frame too
            pose_seq = tuple(np.ascontiguousarray(np.repeat(x[:1], n, 0)) for x in pose_seq)
            pose_why = f"HELD at frame {body_start} of {body}, still body under it"
        if pose_seq is None:
            pose_why = (f"REST -- still, at the resting placement of {self.dep[:44]}, "
                        f"which is a pose he actually held")
            c257[:, 251:257] = self.head6
        else:
            # the renderer is told the angle it is looking at, from the same
            # implementation cooking used. Telling it one angle while rendering
            # another is silent.
            c257[:, 251:257] = self.HeadWeight.channels(*pose_seq).astype(np.float32)
        c263 = self.drv.to_263(c257)
        T["blinks and head channels"] = time.time() - t

        # IN SLICES OF THE CLIP, NOT ALL OF IT AT ONCE. Every stage below holds a few
        # full copies of the geometry, ~0.9 GB each for an 80 s answer, and the body
        # path holds more of them. Whole-clip, an 80 s reply took this process to
        # 12 GB, and the voice on the same card died mid-sentence for want of 18 MB.
        # Per frame the arithmetic is unchanged, so the result is identical.
        T["the rig"] = T["layer, fit and placement"] = 0.0
        vb = None
        for a_ in range(0, n, SLICE):
            b_ = min(a_ + SLICE, n)
            v_ = self._geometry(c263[a_:b_], b_ - a_, body, T,
                                None if pose_seq is None else (pose_seq[0][a_:b_], pose_seq[1][a_:b_]))
            if vb is None:
                vb = np.empty((n, v_.shape[1], 3), np.float32)
            vb[a_:b_] = v_
        torch.cuda.empty_cache()

        t = time.time()
        blob = SHM / f"live_geom_{id(self) & 0xffff}_{int(lo * 1000)}.bin"
        with open(blob, "wb") as f:
            f.write(vb.tobytes())
            f.write(c257.astype(np.float32).tobytes())
        T["hand over"] = time.time() - t

        return dict(ok=True, frames=n, seconds=float(dur), vertices=int(vb.shape[1]),
                    t_from=lo, t_to=float(ts[-1] + 1.0 / FPS_OUT), frame0=int(frame0),
                    blob=str(blob), timings=T, pose=head, pose_why=pose_why,
                    lead_s=LEAD_S, lag_s=self.lag_s)

    def _geometry(self, c263, n, body, T, pose_seq):
        """Rig, corrective layer and placement for one slice of frames -> [n, V, 3]."""
        torch = self.torch
        t = time.time()
        with torch.no_grad():
            d_, bsw = self.rig.behaviour(c263)
            sk = self.rig.skin_matrices(d_)
            V = torch.cat([self.rig.deform(m, sk, bsw) for m in self.MESHES], dim=1)
        torch.cuda.synchronize(); T["the rig"] += time.time() - t

        t = time.time()
        with torch.no_grad():
            if self.lm is not None:
                # SUBTRACT: the layer says how far the rig overshoots. One matmul over
                # the whole clip, on the head's vertices only -- beyond them there was
                # no target and the regression's values are fitted from nothing.
                C = c263.float()
                corr = ((C - self.lc) @ self.lB).view(n, -1, 3) + self.lm
                V[:, :self.N_HEAD] = V[:, :self.N_HEAD] - corr
            # into the export's frame, then to the resting placement. At `rest` the
            # head and body transforms are the same, so the skin-weight blend is an
            # identity and the whole thing is one affine map.
            cano = self.s0 * (V @ self.R0) + self.t0
            if pose_seq is None:
                verts = (cano @ self.Rb + self.tb).contiguous()
            else:
                # NOT one rigid transform. The head mesh does not stop at the head --
                # the rig gives its lowest vertices weight 0 on the head subtree, and
                # moving them with the skull is the shoulders swinging with it. Two
                # influences blended by the rig's own weights, exactly as cooking does.
                Rs = torch.as_tensor(pose_seq[0], dtype=torch.float32, device="cuda")
                tsq = torch.as_tensor(pose_seq[1], dtype=torch.float32, device="cuda")
                headv = torch.einsum("tnd,tde->tne", cano, Rs) + tsq[:, None]
                if self._pose_ref is not None and body:
                    _rb = torch.as_tensor(self._pose_ref[0], dtype=torch.float32, device="cuda")
                    _tb = torch.as_tensor(self._pose_ref[1], dtype=torch.float32, device="cuda")
                    bodyv = cano @ _rb + _tb
                else:
                    bodyv = cano @ self.Rb + self.tb
                verts = (self.wc * headv + (1.0 - self.wc) * bodyv).contiguous()
        torch.cuda.synchronize(); T["layer, fit and placement"] += time.time() - t

        return verts.to(torch.float32).cpu().numpy()

    # --------------------------------------------------- is it the same answer?
    def verify(self, wav_path, seconds=4.0):
        """The card path against infer.py's own processor path, in millimetres.

        Not a unit test -- a measurement. The two differ in precision and in nothing
        else, and this says by how much on real geometry rather than in principle.
        """
        torch = self.torch
        wav = self.read_wav16k(str(wav_path))
        dur = min(len(wav) / self.SR, seconds)
        ts = np.arange(0, dur, 1.0 / FPS_OUT)
        n = len(ts)
        c257 = self.drv.controls(wav, ts)
        c257[:, 251:257] = self.head6
        c263 = self.drv.to_263(c257)
        with torch.no_grad():
            d_, bsw = self.rig.behaviour(c263)
            sk = self.rig.skin_matrices(d_)
            V = torch.cat([self.rig.deform(m, sk, bsw) for m in self.MESHES], dim=1)

            C = c263.float()
            corr = ((C - self.lc) @ self.lB).view(n, -1, 3) + self.lm
            Vg = V.clone(); Vg[:, :self.N_HEAD] = Vg[:, :self.N_HEAD] - corr
            fast = ((self.s0 * (Vg @ self.R0) + self.t0) @ self.Rb + self.tb).cpu().numpy()

        # infer.py's path, verbatim
        Vn = V.cpu().numpy().astype(np.float64)
        cn = c263.cpu().numpy()
        R0 = self.R0.cpu().numpy().astype(np.float64)
        t0 = self.t0.cpu().numpy().astype(np.float64)
        Rb = self.Rb.cpu().numpy().astype(np.float64)
        tb = self.tb.cpu().numpy().astype(np.float64)
        slow = np.empty_like(Vn)
        for i in range(n):
            x = Vn[i].copy()
            x[:self.N_HEAD] = self.dec.apply(Vn[i][:self.N_HEAD], cn[i])
            cano = self.s0 * (x @ R0) + t0
            slow[i] = self.hw.place(cano, Rb, tb, Rb, tb)

        d = np.linalg.norm(fast - slow, axis=-1)
        # the rig's units are centimetres; the corrective layer reports in millimetres
        # at 10 mm per unit, and this uses the same convention
        return dict(frames=n, max_mm=float(d.max() * 10), mean_mm=float(d.mean() * 10),
                    layer_magnitude_mm=float(
                        np.linalg.norm(self.dec.m[self.dec.scored], axis=1).mean() * 10))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--run", default="G4_teeth")
    ap.add_argument("--verify", default="", help="a wav; compare both paths and exit")
    a = ap.parse_args()

    t0 = time.time()
    w = Phase1(a.profile, a.run)
    boot = time.time() - t0

    if a.verify:
        r = w.verify(a.verify)
        print(f"\n{r['frames']} frames, card path against infer.py's processor path")
        print(f"  largest disagreement   {r['max_mm']:.5f} mm")
        print(f"  mean disagreement      {r['mean_mm']:.5f} mm")
        print(f"  the layer's own effect {r['layer_magnitude_mm']:.2f} mm")
        return 0

    print(json.dumps(dict(ready=True, boot_s=boot)), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("quit"):
                return 0
            print(json.dumps(w.run(**req)), flush=True)
        except Exception as e:
            import traceback
            print(json.dumps(dict(ok=False, error=f"{e}",
                                  trace=traceback.format_exc()[-800:])), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
