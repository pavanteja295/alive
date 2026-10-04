#!/usr/bin/env python3
"""The speaking-style offset network. A residual on frozen xADA.

    ~/miniconda3/envs/stavatar/bin/python offset_model.py

Spec and justification: the original design note (not in this release), section 04.
The self-test below is what produced the parameter counts and timings quoted there.

Shape contract -- everything here runs at xADA's NATIVE 50 Hz
-------------------------------------------------------------
    Z50   [B, T, d_z]   xADA encoder features at 50 Hz, UNTOUCHED
    base  [B, T, 257]   the frozen baseline: 251 raw controls + 6 head channels
    ->    [B, T, 257]   the correction, added to base by the caller

T is cfg.window_in (107 for a 64-frame output window), not the output length. The
30 Hz resample happens OUTSIDE this module, after `base + delta`, in control space.
This module is rate-agnostic and must stay that way: it never sees `fps`, never
sees `shift_ms`, and never interpolates.

The latent is never resampled anywhere. An earlier design fed `resample_poly(Z, 3, 5)`
at 30 Hz; that discards 6% of Z's variance -- 7.2% of its energy sits above the
15 Hz Nyquist -- and moves the decoder's response by 19% of channel sd, so delta
would be corrected against a baseline its own input did not produce.

Two initialisation properties that the training plan depends on
--------------------------------------------------------------
  * The output head is ZERO-initialised, so at step 0 the recurrent branch
    contributes exactly nothing and the model equals its affine branch.
  * The affine branch is loaded from the closed-form weighted least-squares
    solution (`init_affine`), so step 0 sits on a known-optimal linear
    correction rather than at zero. Neither branch can make the system worse
    than the baseline it wraps, and `affine_share` then reads out directly
    whether the 577,410 recurrent parameters earned their place.

Head pose lives in the last 6 channels, not in the rig, ordered as
rig_utils.HEAD_CURVES: roll 251, pitch 252, yaw 253, tx 254, ty 255, tz 256.
ADA emits only five of them -- there is no HeadTranslationX -- so `base[:, :, 254]`
is always zero and delta learns that channel from scratch. (This said 251 until
2026-08-19; 251 is roll, which is very much not zero.)
"""

import time

import torch
import torch.nn as nn

N_CTRL, N_HEAD = 251, 6
N_OUT = N_CTRL + N_HEAD          # 257


N_PSD = 545              # this DNA's correctives. Ada.dna has 476 -- not portable.


class OffsetNet(nn.Module):
    """Delta = affine(base) + s * tanh(head(biGRU(audio, base [, psd]))).

    `d_pp > 0` enables the optional PSD input. See PSDFeatures in
    ../riglogic_torch.py for why a deterministic function of `base` is not a
    redundant input, and why it must be computed from the BASELINE controls.
    Off by default so the ablation is a one-flag change against a fixed baseline.
    """

    def __init__(self, d_z=512, d_zp=64, d_cp=32, d_pp=0, hidden=128, layers=2,
                 s_init=1.0, zero_head=True):
        super().__init__()
        self.zproj = nn.Linear(d_z, d_zp)
        self.cproj = nn.Linear(N_OUT, d_cp)
        self.d_pp = d_pp
        self.pproj = nn.Linear(N_PSD, d_pp) if d_pp else None
        self.gru = nn.GRU(d_zp + d_cp + d_pp, hidden, num_layers=layers,
                          bidirectional=True, batch_first=True)
        self.head = nn.Linear(2 * hidden, N_OUT)
        # Per-channel output bound. tanh keeps a bad step from launching the rig
        # somewhere absurd; s is learnable so the bound adapts.
        #
        # s_init = 1.0, NOT 0.1, and the difference is not cosmetic. The raw
        # controls live in [0,1], so a required correction spans the full range:
        # measured on take 4, |c_gt - base| has p90 0.149, p95 0.240, p99 0.866 and
        # max 1.000, and 118 of the 251 channels exceed 0.1 at their own p99. At
        # s_init = 0.1 the recurrent branch is bounded an order of magnitude below
        # its own target, and run A plateaus at 0.306 mm instead of reaching the
        # ~5e-4 mm fp32 floor. s = 1.0 still bounds delta to exactly the control
        # range, so it keeps the property the bound exists for.
        self.log_s = nn.Parameter(torch.full((N_OUT,), float(torch.log(torch.tensor(s_init)))))
        self.a = nn.Parameter(torch.zeros(N_OUT))      # affine gain, on top of identity
        self.b = nn.Parameter(torch.zeros(N_OUT))      # affine bias
        if zero_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, z, base, psd=None):
        """z [B,T,512], base [B,T,257], psd [B,T,545] or None -> [B,T,257]."""
        parts = [self.zproj(z), self.cproj(base)]
        if self.pproj is not None:
            if psd is None:
                raise ValueError("model built with d_pp>0 but psd was not supplied")
            parts.append(self.pproj(psd))
        elif psd is not None:
            raise ValueError("psd supplied but model was built with d_pp=0")
        h, _ = self.gru(torch.cat(parts, -1))
        return self.affine(base) + self.log_s.exp() * torch.tanh(self.head(h))

    def affine(self, base):
        return self.a * base + self.b

    @torch.no_grad()
    def init_affine(self, base, target, w=None):
        """Closed-form per-channel weighted least squares for `a` and `b`.

        Solves  min_a,b  sum_t w[t] (a*base[t,j] + b - target[t,j])^2  per channel j.
        `base` and `target` are [T, 257]; `target` is the offset to reproduce
        (r_c for the control block, H_gt - H_ada for the head block).

        Per channel, not a joint solve, because that is the null model the
        literature describes: a gain-and-bias error on each curve.
        """
        w = torch.ones(len(base), device=base.device) if w is None else w
        sw = w.sum()
        mx = (w[:, None] * base).sum(0) / sw
        my = (w[:, None] * target).sum(0) / sw
        vx = (w[:, None] * (base - mx) ** 2).sum(0) / sw
        cxy = (w[:, None] * (base - mx) * (target - my)).sum(0) / sw
        a = torch.where(vx > 1e-12, cxy / vx.clamp_min(1e-12), torch.zeros_like(vx))
        self.a.copy_(a)
        self.b.copy_(my - a * mx)
        return self.a.clone(), self.b.clone()

    def param_groups(self, lr_main=3e-4, lr_affine=1e-4):
        """Two groups. The affine block starts at its optimum and needs
        refinement, not search; one group at lr_main drags it off a solution
        that is already correct."""
        # log_s is deliberately NOT in the slow group. That group exists because a
        # and b are loaded from the closed-form weighted least squares and need
        # refinement rather than search -- one group at lr_main drags them off a
        # solution that is already correct. log_s starts at no optimum at all, only
        # at a scale prior, so the argument does not apply to it. Grouping it with a
        # and b was a real bug: at lr 1e-4 it needs ~23k steps of consistent
        # gradient to move an order of magnitude, so a bad s_init could not be
        # recovered from inside a 2k-step run.
        slow = {self.a, self.b}
        return [
            {"params": [p for p in self.parameters() if p not in slow],
             "lr": lr_main, "name": "main"},
            {"params": list(slow), "lr": lr_affine, "name": "affine"},
        ]

    def affine_share(self, z, base, psd=None):
        """||delta_affine|| / ||delta||. Near 1.0 means ship the affine."""
        with torch.no_grad():
            d, da = self(z, base, psd), self.affine(base)
            return float(da.norm() / d.norm().clamp_min(1e-12))


# ------------------------------------------------------------------ self-test
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}, torch {torch.__version__}")

    for d_pp in (0, 32):
        tag = "no psd " if d_pp == 0 else "psd->32"
        for hidden in (96, 128, 160):
            m = OffsetNet(hidden=hidden, d_pp=d_pp)
            print(f"{tag}  hidden={hidden:4d}  "
                  f"total={sum(p.numel() for p in m.parameters()):>9,}"
                  f"   affine={m.a.numel() + m.b.numel()}"
                  f"   gru={sum(p.numel() for p in m.gru.parameters()):>9,}")

    m = OffsetNet(hidden=128).to(dev)
    z = torch.randn(32, 90, 512, device=dev)
    base = torch.randn(32, 90, N_OUT, device=dev)
    with torch.no_grad():
        y = m(z, base)
    print(f"\nforward {tuple(z.shape)} + {tuple(base.shape)} -> {tuple(y.shape)}")
    print(f"zero-init check: max|delta| with a=b=0 is {float(y.abs().max()):.1e}")

    # ---- the optional PSD input, end to end against the real tables ---------
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    npz = Path(__file__).resolve().parent.parent / "rig_tables.npz"
    if npz.exists():
        from riglogic_torch import PSDFeatures, TorchRig, N_RAW
        pf = PSDFeatures(npz, device=dev)
        n_par = sum(b.numel() for b in pf.buffers())
        print(f"\nPSDFeatures: {pf.max_terms} max terms, "
              f"{n_par:,} buffer elements, 0 parameters")

        # exactness: PSDFeatures alone must equal the full rig's stage 2
        rig = TorchRig(npz, device=dev, dtype=torch.float64)
        pf64 = PSDFeatures(npz, device=dev, dtype=torch.float64)
        c = torch.zeros(4, N_RAW, dtype=torch.float64, device=dev)
        c[:, :251] = torch.rand(4, 251, dtype=torch.float64, device=dev)
        with torch.no_grad():
            _, _ = rig.behaviour(c)
            u_psd = torch.clamp(rig.psd_w * torch.cat(
                [c, c.new_zeros(4, 545), rig.rbf.expand(4, 18), c.new_ones(4, 1)], 1
            )[:, rig.psd_idx.reshape(-1)].reshape(4, 545, rig.max_terms).prod(2), 0, 1)
            err = (pf64(c) - u_psd).abs().max()
        print(f"PSDFeatures vs TorchRig stage 2: {float(err):.2e}  "
              f"-> {'PASS' if err < 1e-12 else 'FAIL'}")

        # shape plumbing on a windowed batch, and the cost of computing it live
        mp = OffsetNet(hidden=128, d_pp=32).to(dev)
        craw = torch.zeros(32, 90, N_RAW, device=dev)
        craw[..., :251] = torch.rand(32, 90, 251, device=dev)
        torch.cuda.synchronize() if dev == "cuda" else None
        t0 = time.time()
        for _ in range(20):
            psd = pf(craw)
        torch.cuda.synchronize() if dev == "cuda" else None
        print(f"psd for a 32x90 batch: {tuple(psd.shape)}, "
              f"{(time.time()-t0)/20*1e3:.2f} ms live  "
              f"(precompute it into the cache instead and this is 0)")
        with torch.no_grad():
            yp = mp(z, base, psd)
        print(f"forward with psd -> {tuple(yp.shape)}, "
              f"zero-init max|delta| {float(yp.abs().max()):.1e}")
        for name, args in (("without psd", (z, base)), ("with psd   ", (z, base, psd))):
            mm = m if "without" in name else mp
            for _ in range(3):
                mm(*args).square().mean().backward()
            torch.cuda.synchronize() if dev == "cuda" else None
            t0 = time.time()
            for _ in range(20):
                mm.zero_grad(set_to_none=True)
                mm(*args).square().mean().backward()
            torch.cuda.synchronize() if dev == "cuda" else None
            print(f"  fwd+bwd 32x90 {name}: {(time.time()-t0)/20*1e3:6.2f} ms")
    else:
        print(f"\n{npz} not found -- skipping the PSD checks")

    # closed-form affine, then confirm the model reproduces it exactly at step 0
    T = 3600
    bb = torch.rand(T, N_OUT, device=dev)
    tt = 0.3 * bb - 0.05 + 0.01 * torch.randn(T, N_OUT, device=dev)
    a, b = m.init_affine(bb, tt)
    with torch.no_grad():
        err = (m(z, base) - m.affine(base)).abs().max()
    print(f"affine init: a in [{a.min():.3f}, {a.max():.3f}] (truth 0.300), "
          f"b in [{b.min():.3f}, {b.max():.3f}] (truth -0.050); "
          f"head still contributes {float(err):.1e}")

    def bench(B, T, n=20):
        z = torch.randn(B, T, 512, device=dev)
        bs = torch.randn(B, T, N_OUT, device=dev)
        for _ in range(3):
            m(z, bs).square().mean().backward()
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            m.zero_grad(set_to_none=True)
            m(z, bs).square().mean().backward()
        if dev == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / n * 1e3

    print("\nfwd+bwd")
    for B, T in [(32, 90), (1, 90), (1, 900), (1, 3600)]:
        print(f"  B={B:3d} T={T:5d}  {bench(B, T):7.2f} ms")

    # effective context: does a window reproduce the full-sequence centre frame?
    m2 = OffsetNet(hidden=128, zero_head=False).to(dev).eval()
    T = 3600
    zf = torch.randn(1, T, 512, device=dev)
    bf = torch.randn(1, T, N_OUT, device=dev)
    with torch.no_grad():
        full = m2(zf, bf)[0]
        c = T // 2
        print("\ncentre-frame delta, window vs full sequence")
        print("  RANDOM INIT -- describes recurrent state decay, not the trained model.")
        print("  Re-run after training before trusting the 90-frame window in stage 3.")
        for w in (30, 60, 90, 150, 300, 600):
            h = w // 2
            win = m2(zf[:, c - h:c + h], bf[:, c - h:c + h])[0, h]
            rel = (win - full[c]).norm() / full[c].norm()
            print(f"  window {w:4d} frames ({w / 30:5.2f} s)   rel err {rel * 100:6.3f}%")


if __name__ == "__main__":
    main()
