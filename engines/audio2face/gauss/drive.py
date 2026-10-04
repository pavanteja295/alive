#!/usr/bin/env python3
"""Sound in, control values out. The one implementation, shared by cooking and by
inference, so the two cannot drift.

WHY IT IS A MODULE AND NOT A FUNCTION IN EACH CALLER

This is the second time the same shape of bug has been designed out of this pipeline.
The first was the decoder: the map from controls to geometry lived in one place at
training time and a different place at render time, and the difference was silent. A
second copy of the AUDIO path would be the same mistake one stage earlier -- the
windowing here is not obvious, and a caller that reimplements it slightly differently
produces controls that look fine and are wrong.

THE WINDOWING, WHICH IS THE PART NOBODY GUESSES RIGHT

The encoder reads a FIXED 30-second block and its answer depends on where inside that
block the audio sits. So a clip longer than the block cannot simply be cut into pieces
and concatenated. Each window keeps `pad_s` seconds on both sides and only its middle
is used, and `pad_s` is longer than the model's own 2.4 s of context, so no frame is
ever read from a window that could not see its own context.

WHAT IT DOES NOT DECIDE

The head channels. The last six of the 257 are head pose, and this returns whatever
the driver predicted for them. Every caller replaces them:
  cooking a tracked clip   with the MEASURED pose, so the render and the conditioning
                           agree about where the head is
  inference                with nothing -- they are zeroed, because for novel audio
                           there is no pose source yet. A pose generator is being built
                           separately, and `head` is the seam it plugs into.
"""
import pathlib

import numpy as np

SR, FPS50 = 16000, 50.0
PIPE = pathlib.Path(__file__).resolve().parent.parent


class Driver:
    """The audio model as deployed: the shipped encoder, plus a face_vN correction."""

    def __init__(self, ckpt_dir, device="cuda"):
        import torch
        import sys
        sys.path.insert(0, str(PIPE / "rigfit"))
        sys.path.insert(0, str(PIPE / "offset"))
        from utils.rig_utils import GuiToRaw
        from utils.baseline import XAdaTeacher, head5_to_6
        from arguments import OffsetConfig
        from networks.offset_model import OffsetNet

        self.torch, self.dev = torch, device
        self.head5_to_6 = head5_to_6
        cfg = OffsetConfig()
        self.g2r = GuiToRaw(cfg.rig_names)
        quat = {i for i, n in enumerate(self.g2r.raw_names)
                if n.startswith(("neck_01.q", "neck_02.q", "head.q"))}
        self.ctrl = np.array([i for i in range(263) if i not in quat], np.int32)

        ck = torch.load(pathlib.Path(ckpt_dir) / "best.pt", map_location=device,
                        weights_only=False)
        sd = ck["net"]
        hid = sd["gru.weight_ih_l0"].shape[0] // 3
        lay = 1 + max((int(k.split("_l")[1].split("_")[0])
                       for k in sd if k.startswith("gru.weight_ih_l")), default=0)
        self.net = OffsetNet(d_z=cfg.d_z, d_zp=cfg.d_zp, d_cp=cfg.d_cp, d_pp=0,
                             hidden=hid, layers=lay, s_init=1.0).to(device)
        self.net.load_state_dict(sd); self.net.eval()
        self.ckpt, self.step, self.hidden, self.layers = ckpt_dir, ck.get("step"), hid, lay
        self.teacher = XAdaTeacher(PIPE / "onnx/audio_encoder.onnx",
                                   PIPE / "onnx/animation_decoder.onnx",
                                   self.g2r.blink_slots)

    def controls(self, wav, t, win_s=24.0, pad_s=3.0):
        """[len(t), 257] control values at each time in `t` (seconds into `wav`).

        `wav` is 16 kHz mono float. Windowed as described above; a frame is only ever
        taken from a window that saw its context.
        """
        torch, dev = self.torch, self.dev
        t = np.asarray(t, np.float64)
        out = np.zeros((len(t), 257), np.float32)
        done = np.zeros(len(t), bool)
        start = t[0]
        while not done.all():
            sel = (t >= start - 1e-6) & (t < start + win_s) & ~done
            if not sel.any():
                start += win_s
                if start > t[-1]:
                    break
                continue
            ts = t[sel]
            g50 = np.arange(ts[0] - pad_s, ts[-1] + pad_s, 1 / FPS50)
            lo = g50[0] - 1.0                          # the encoder's own 30 s block
            i0 = max(int(lo * SR), 0)
            seg = wav[i0: i0 + 30 * SR]
            blk = np.zeros(30 * SR, np.float32); blk[:len(seg)] = seg
            o = self.teacher.encode(blk)
            gui = np.zeros((len(o.gui81), len(self.g2r.gui_names)))
            gui[:, self.g2r.ada_gui_idx] = o.gui81
            raw50, head50 = self.g2r.apply(gui), self.head5_to_6(o.head5)
            tw = np.arange(len(raw50)) / FPS50 + lo
            base = np.empty((len(g50), 257), np.float32)
            base[:, :251] = np.stack([np.interp(g50, tw, raw50[:, j])
                                      for j in self.ctrl], 1)
            base[:, 251:] = np.stack([np.interp(g50, tw, head50[:, j])
                                      for j in range(6)], 1)
            zi = np.clip(np.round((g50 - lo) * FPS50).astype(int), 0, len(o.Z) - 1)
            Z = o.Z[zi].astype(np.float32)
            with torch.no_grad():
                B = torch.as_tensor(base, device=dev)[None]
                d = self.net(torch.as_tensor(Z, device=dev)[None], B)
                # the correction never touches the head channels
                d = torch.cat([d[..., :251], torch.zeros_like(d[..., 251:])], -1)
                c257 = (B + d)[0].cpu().numpy()
            w = np.interp(ts, g50, np.arange(len(g50)))
            j0 = np.clip(np.floor(w).astype(int), 0, len(c257) - 2)
            f = (w - j0)[:, None]
            out[sel] = c257[j0] * (1 - f) + c257[j0 + 1] * f
            done |= sel
            start += win_s
        assert done.all(), f"{(~done).sum()} frames never predicted"
        return out

    def to_263(self, c257):
        """The 251 driven controls placed back into the rig's 263 slots, clamped to the
        range the rig accepts. The head channels are NOT part of this: they place the
        mesh, they do not deform it."""
        torch = self.torch
        CI = torch.as_tensor(self.ctrl, dtype=torch.long, device=self.dev)
        C = torch.as_tensor(np.asarray(c257)[:, :251], device=self.dev).clamp(0, 1)
        return torch.zeros(len(C), 263, device=self.dev).index_copy(1, CI, C)
