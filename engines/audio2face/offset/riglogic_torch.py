#!/usr/bin/env python3
"""RigLogic controls -> vertices, in torch, differentiable end to end.

    ~/miniconda3/envs/stavatar/bin/python riglogic_torch.py

A direct transcription of `riglogic_numpy.py` with every loop replaced by a batched
op, so that d(vertices)/d(c_raw) is an exact autograd gradient rather than a
finite-difference Jacobian. Needs only `rig_tables.npz` -- no DNA file, no
OpenRigLogic, no Unreal. Verified against the numpy reference in `verify()`.

Three places where the transcription is deliberately NOT literal
---------------------------------------------------------------
  * `riglogic_skin.deform` skips channels whose weight is zero. That guard is a
    speed shortcut in the forward pass and a correctness bug in the backward one:
    dV/dbsw[ch] = D_ch is nonzero at w = 0, so skipping means the optimiser can
    never turn an inactive shape on. 68.4% of the 782 channels are inactive at a
    typical frame. Here the blendshape stage is one sparse matmul with no branch.
  * The PSD segment-product becomes a padded [545, 6] gather and a prod over the
    last axis. Padding points at a constant 1.0 slot, so degree-2 and degree-6
    PSDs cost the same.
  * The hierarchy walk becomes 13 out-of-place `index_copy` calls, one per depth
    level, instead of 870 sequential matmuls. Parents precede children in DNA
    joint order, which makes the level decomposition valid.

Non-smooth points, both harmless in practice, both worth knowing
---------------------------------------------------------------
  * `clamp(psd, 0, 1)` has a kink. Only 6 of 545 PSDs carry weight 4.0 and can
    reach the upper bound at all; the other 539 have weight 1.0 and inputs in
    [0, 1], so they never touch it. Subgradient is 0 outside the interval.
  * `safe_normalize` is not used here, but STAvatar's `compute_face_orientation`
    downstream of this module does normalise triangle edges. Zero-area triangles
    would be a problem there, not here. This mesh has none.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_RAW, N_PSD, N_RBF = 263, 545, 18
N_U = N_RAW + N_PSD + N_RBF          # 826
PAD_ONE, PAD_ZERO = N_U, N_U + 1     # two extra slots appended to u


def _psd_index_table(psd_row, psd_col):
    """-> [545, max_terms] gather table, padded with PAD_ONE (a constant-1 slot)."""
    row, col = np.asarray(psd_row, np.int64), np.asarray(psd_col, np.int64)
    deg = np.bincount(row, minlength=N_PSD)
    idx = np.full((N_PSD, int(deg.max())), PAD_ONE, np.int64)
    fill = np.zeros(N_PSD, np.int64)
    for r, c in zip(row, col):
        idx[r, fill[r]] = c
        fill[r] += 1
    return idx


class PSDFeatures(nn.Module):
    """Stage 2 alone: c_raw [B, 263] -> the 545 corrective activations [B, 545].

    Split out of TorchRig because the offset network wants these as an *input
    feature* and has no use for the 189 MB of geometry tables. This module holds
    only the PSD gather table and weights, about 30 KB.

    Why they are worth feeding to the offset network
    ------------------------------------------------
    They are a deterministic function of c_raw, so this looks redundant. It is
    not: each is a clamped monomial of degree 2 to 6 (measured, this DNA:
    2->301, 3->166, 4->62, 5->14, 6->2). A tanh GRU cannot cheaply represent a
    degree-6 product of its own inputs, so supplying them adds representational
    reach rather than duplicating information.

    They are also the one per-frame quantity that is specific to THIS identity:
    this DNA has 545 correctives, Ada.dna has 476, and the weights differ. A
    constant identity vector would be absorbed into the first-layer bias and
    carry no information; this varies frame to frame, so it does.

    Feed them from the ADA BASELINE, never from c_raw = base + delta. The latter
    is circular, and the baseline is available before delta runs -- which also
    means they can be precomputed into the cache and cost the training loop
    nothing at all.
    """

    def __init__(self, npz_path, device="cuda", dtype=torch.float32):
        super().__init__()
        z = np.load(str(npz_path))
        self.register_buffer("psd_idx",
                            torch.as_tensor(_psd_index_table(z["psd_row"], z["psd_col"]),
                                            dtype=torch.int64, device=device),
                            persistent=False)
        self.register_buffer("psd_w",
                            torch.as_tensor(np.asarray(z["psd_w"]), dtype=dtype, device=device),
                            persistent=False)
        self.max_terms = self.psd_idx.shape[1]
        self.rbf = None
        if "rbf" in z:
            self.register_buffer("rbf_const",
                                torch.as_tensor(np.asarray(z["rbf"]), dtype=dtype, device=device),
                                persistent=False)
            self.rbf = True

    def forward(self, c_raw):
        """c_raw [..., 263] -> psd [..., 545]. Batched over any leading dims."""
        lead = c_raw.shape[:-1]
        flat = c_raw.reshape(-1, N_RAW)
        B = flat.shape[0]
        rbf = (self.rbf_const.expand(B, N_RBF) if self.rbf
               else flat.new_zeros(B, N_RBF))
        # PSDs read raw and RBF slots only (0 chained PSDs in this DNA), so the
        # PSD block can be zero on this single pass and the result is still exact.
        # inputs clamped to [0,1] -- see riglogic_numpy.psd_values for why this
        # clamp exists here and must NOT be applied to the downstream stages.
        u = torch.cat([flat.clamp(0.0, 1.0), flat.new_zeros(B, N_PSD),
                       rbf, flat.new_ones(B, 1)], 1)
        p = u[:, self.psd_idx.reshape(-1)].reshape(B, N_PSD, self.max_terms).prod(2)
        return torch.clamp(self.psd_w * p, 0.0, 1.0).reshape(*lead, N_PSD)


class TorchRig(nn.Module):
    """c_raw [B, 263] -> vertices [B, 31701, 3] in centimetres, canonical space.

    Head pose is NOT applied. It is a rigid transform about joint 84 that belongs
    on the caller's side, because it comes from a different signal path.
    """

    def __init__(self, npz_path, device="cuda", dtype=torch.float32):
        super().__init__()
        z = np.load(str(npz_path))
        meta = z["meta"]
        n_raw, n_psd, n_rbf, self.n_joint_out, self.n_bs, n_jg = [int(x) for x in meta]
        assert (n_raw, n_psd, n_rbf) == (N_RAW, N_PSD, N_RBF), f"unexpected layout {meta}"
        self.n_joints = self.n_joint_out // 9
        self.dtype, self.device_ = dtype, device
        reg = lambda k, v: self.register_buffer(k, v, persistent=False)
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
        I = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.int64, device=device)

        # ---- stage 2: PSD, padded [545, max_terms] gather then prod ----------
        idx = _psd_index_table(z["psd_row"], z["psd_col"])
        self.max_terms = idx.shape[1]
        reg("psd_idx", I(idx))
        reg("psd_w", T(z["psd_w"]))

        # ---- stage 3: joint groups merged into one dense [7830, 826] ---------
        A = np.zeros((self.n_joint_out, N_U), np.float32)
        for g in range(n_jg):
            ii, oi, vv = z[f"jg{g}_in"], z[f"jg{g}_out"], z[f"jg{g}_v"]
            A[np.ix_(oi.astype(np.int64), ii.astype(np.int64))] += vv
        self.jg_nnz = int((A != 0).sum())
        reg("A", T(A))

        # ---- stage 4: blendshape routing as one gather -----------------------
        bs_map = np.full(self.n_bs, PAD_ZERO, np.int64)
        bs_map[z["bs_out"].astype(np.int64)] = z["bs_in"].astype(np.int64)
        reg("bs_map", I(bs_map))

        # ---- stages 5-6: skeleton -------------------------------------------
        reg("nT", T(z["nT"]))
        reg("nR", T(z["nR"]))
        reg("bind_world_inv", T(z["bind_world_inv"]))
        par = z["parents"].astype(np.int64)
        assert (par < np.arange(len(par))).all(), "parents must precede children"
        depth = np.zeros(len(par), np.int64)
        for j, p in enumerate(par):
            depth[j] = 0 if p < 0 else depth[p] + 1
        reg("parents", I(np.maximum(par, 0)))
        self.levels = [I(np.nonzero(depth == d)[0]) for d in range(int(depth.max()) + 1)]
        reg("rbf", T(z["rbf"]))

        # ---- stages 7-9: per-mesh geometry ----------------------------------
        self.n_meshes = int(z["n_meshes"][0])
        self.nv = 0
        for m in range(self.n_meshes):
            V0 = z[f"m{m}_V0"]
            reg(f"V0_{m}", T(V0))
            reg(f"JI_{m}", I(z[f"m{m}_JI"]))
            reg(f"JW_{m}", T(z[f"m{m}_JW"]))
            self.nv += len(V0)
            ch, off = z[f"m{m}_bs_ch"].astype(np.int64), z[f"m{m}_bs_off"].astype(np.int64)
            vi, dd = z[f"m{m}_bs_vi"].astype(np.int64), z[f"m{m}_bs_d"]
            if len(ch) == 0:
                setattr(self, f"B_{m}", None)
                continue
            # sparse [nv*3, 782]: column = channel, row = vertex*3 + axis
            chan = np.repeat(ch, np.diff(off))
            rows = (vi[:, None] * 3 + np.arange(3)[None, :]).ravel()
            cols = np.repeat(chan, 3)
            B = torch.sparse_coo_tensor(
                np.stack([rows, cols]), dd.ravel().astype(np.float32),
                (len(V0) * 3, self.n_bs), dtype=dtype, device=device).coalesce()
            setattr(self, f"B_{m}", B)
        self.bs_nnz = sum(getattr(self, f"B_{m}")._nnz()
                          for m in range(self.n_meshes) if getattr(self, f"B_{m}") is not None)

    # ------------------------------------------------------------------ stages
    def behaviour(self, c_raw):
        """[B, 263] -> joint deltas [B, 7830], blendshape weights [B, 782]."""
        B = c_raw.shape[0]
        rbf = self.rbf.expand(B, N_RBF)
        zeros_psd = c_raw.new_zeros(B, N_PSD)
        ones1 = c_raw.new_ones(B, 1)
        zero1 = c_raw.new_zeros(B, 1)

        # stage 2. PSDs read raw and RBF slots only (0 chained PSDs in this DNA),
        # so a single pass with the PSD block zeroed is exact.
        # c_raw is CLAMPED here and NOWHERE ELSE: the joint-group and blendshape
        # stages below read the same controls unclamped. Verified against
        # OpenRigLogic on out-of-range poses; see riglogic_numpy.psd_values.
        u0 = torch.cat([c_raw.clamp(0.0, 1.0), zeros_psd, rbf, ones1], 1)   # [B, 827]
        p = u0[:, self.psd_idx.reshape(-1)].reshape(B, N_PSD, self.max_terms).prod(2)
        psd = torch.clamp(self.psd_w * p, 0.0, 1.0)

        u = torch.cat([c_raw, psd, rbf], 1)                        # [B, 826], raw unclamped
        delta = u @ self.A.t()                                     # stage 3
        u_pad = torch.cat([u, ones1, zero1], 1)
        bsw = u_pad[:, self.bs_map]                                # stage 4
        return delta, bsw

    def skin_matrices(self, delta):
        """[B, 7830] -> [B, 870, 4, 4] skinning matrices. Stages 5-6."""
        B = delta.shape[0]
        jo = delta.reshape(B, self.n_joints, 9)
        t = self.nT + jo[..., 0:3]
        r = torch.deg2rad(self.nR + jo[..., 3:6])
        s = 1.0 + jo[..., 6:9]

        cx, cy, cz = torch.cos(r).unbind(-1)
        sx, sy, sz = torch.sin(r).unbind(-1)
        # ZYX, composed left to right: Rz @ Ry @ Rx
        R = torch.stack([
            cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
            sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
            -sy,     cy * sx,                cy * cx,
        ], -1).reshape(B, self.n_joints, 3, 3)
        R = R * s[..., None, :]                     # compose(): columns scaled

        local = torch.zeros(B, self.n_joints, 4, 4, dtype=delta.dtype, device=delta.device)
        local[..., :3, :3] = R
        local[..., :3, 3] = t
        local[..., 3, 3] = 1.0

        W = local
        for lv in self.levels[1:]:                  # 13 levels, out-of-place
            W = W.index_copy(1, lv, W[:, self.parents[lv]] @ local[:, lv])
        return W @ self.bind_world_inv

    def deform(self, m, skin, bsw):
        """One mesh. Stages 7-9: blendshapes, then linear blend skinning."""
        Bn = bsw.shape[0]
        V0 = getattr(self, f"V0_{m}")
        Bm = getattr(self, f"B_{m}")
        V = V0.expand(Bn, *V0.shape)
        if Bm is not None:
            V = V + torch.sparse.mm(Bm, bsw.t()).t().reshape(Bn, -1, 3)
        Vh = torch.cat([V, V.new_ones(*V.shape[:-1], 1)], -1)
        JI, JW = getattr(self, f"JI_{m}"), getattr(self, f"JW_{m}")
        out = 0.0
        for k in range(JI.shape[1]):                # <=12 influences, gathered per slot
            Mk = skin[:, JI[:, k]]                  # [B, nv, 4, 4]
            out = out + JW[:, k, None] * torch.einsum("bnij,bnj->bni", Mk, Vh)[..., :3]
        return out

    def forward(self, c_raw):
        delta, bsw = self.behaviour(c_raw)
        skin = self.skin_matrices(delta)
        return torch.cat([self.deform(m, skin, bsw) for m in range(self.n_meshes)], 1)


# Head pose is NOT here, deliberately
# -------------------------------------
# It lives in head_pose.py, and the reason is that the obvious implementation is
# wrong. The six MetaHuman Animator head curves are not a rotation about the head
# joint: MetaHumanPerformanceExportUtils.cpp:2247-2302 writes
#
#     HeadPose = B * Root_relative * B^-1
#
# a root delta CONJUGATED into the ARCHETYPE head-bone frame. Treating it as a
# pivot rotation about DNA joint 84 displaces vertices by 5.7 cm mean / 7.6 cm max
# -- larger than the take's entire head motion, so it does not degrade gracefully.
# There is also no pivot to recover: a general rigid motion is a screw, so R^T - I
# is rank 2 and any least-squares "rotation centre" diverges.
#
# Use head_pose.root_delta_dna(head6) -> 4x4, then head_pose.apply(V, M).
# check_head_pose.py is the regression test. An earlier apply_head/check_head_axes
# pair lived here implementing the pivot formulation; it was deleted rather than
# fixed, because a wrong head transform sitting next to a verified rig is exactly
# the kind of thing that gets called by accident.
#
# TorchRig returns CANONICAL vertices in CENTIMETRES. Nothing here applies head
# pose, and nothing here converts units.

# ------------------------------------------------------------------ verify ---
def verify():
    here = Path(__file__).resolve().parent
    npz = here / "rig_tables.npz"
    sys.path.insert(0, str(here))
    # riglogic_numpy imports riglogic_skin, which does `import dna` at module scope.
    # The OpenRigLogic bindings are built for the 3.13 venv, not this interpreter, and
    # nothing used below touches the DNA reader -- only the pure-numpy `behaviour`,
    # `geometry_from_tables`, `compose` and `world_matrices`. Stub the import so the
    # reference stays the real reference file rather than a copy of it.
    import types
    for name in ("dna", "riglogic"):
        sys.modules.setdefault(name, types.ModuleType(name))
    import riglogic_numpy as rn

    z = np.load(str(npz))
    tables = rn.RigTables()
    tables.psd_row, tables.psd_col, tables.psd_w = z["psd_row"], z["psd_col"], z["psd_w"]
    tables.bs_in, tables.bs_out = z["bs_in"], z["bs_out"]
    m = z["meta"]
    tables.n_raw, tables.n_psd, tables.n_rbf = int(m[0]), int(m[1]), int(m[2])
    tables.n_joint_out, tables.n_bs = int(m[3]), int(m[4])
    tables.jg = [(z[f"jg{g}_in"], z[f"jg{g}_out"], z[f"jg{g}_v"].astype(np.float64))
                 for g in range(int(m[5]))]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rig64 = TorchRig(npz, device=dev, dtype=torch.float64)
    print(f"TorchRig on {dev}: {rig64.nv:,} verts, {rig64.n_joints} joints, "
          f"jg nnz {rig64.jg_nnz:,} of {rig64.n_joint_out * N_U:,} "
          f"({100*rig64.jg_nnz/(rig64.n_joint_out*N_U):.1f}% dense), "
          f"blendshape nnz {rig64.bs_nnz:,}, PSD max degree {rig64.max_terms}")

    # ---- forward agreement with the numpy reference ----------------------
    rng = np.random.default_rng(7)
    worst = 0.0
    for i in range(8):
        c = np.zeros(N_RAW)
        n = int(rng.integers(20, 120))
        # Half out of [0,1] deliberately -- see riglogic_numpy.verify. For the
        # authoritative check against OpenRigLogic itself rather than against the
        # numpy port, run utils/check_torch_rig.py over the fixture.
        c[rng.choice(251, n, replace=False)] = (rng.random(n) if i % 2
                                                else rng.uniform(-0.6, 1.6, n))
        jo, bsw, _ = rn.behaviour(c, z["rbf"].astype(np.float64), tables)
        V_np = rn.geometry_from_tables(jo, bsw, z)
        with torch.no_grad():
            V_t = rig64(torch.as_tensor(c, dtype=torch.float64, device=dev)[None])[0]
        worst = max(worst, float((V_t.cpu().numpy() - V_np).__abs__().max()) * 10.0)
    print(f"forward vs riglogic_numpy, 8 random poses: worst {worst:.3e} mm  "
          f"-> {'PASS' if worst < 1e-4 else 'FAIL'}")

    # ---- gradcheck on the exact map, fp64, a few output coordinates -------
    c = np.zeros(N_RAW)
    c[rng.choice(251, 40, replace=False)] = rng.random(40)
    active = np.nonzero(c)[0][:12]
    probe = torch.as_tensor(rng.choice(rig64.nv, 64, replace=False), device=dev)

    def f(sub):
        full = torch.as_tensor(c, dtype=torch.float64, device=dev).clone()
        full = full.index_copy(0, torch.as_tensor(active, device=dev), sub)
        return rig64(full[None])[0, probe].reshape(-1)

    sub = torch.as_tensor(c[active], dtype=torch.float64, device=dev).requires_grad_(True)
    # nondet_tol is required, and the reason is worth stating: the sparse blendshape
    # matmul and the per-slot skinning gather both reduce with CUDA atomics, so the
    # backward is not bitwise reproducible. gradcheck treats that as a failure by
    # default even when the gradient itself is right. The spread is measured below.
    v = torch.randn_like(f(sub))          # the same shape of grad_output gradcheck uses
    g = []
    for _ in range(6):
        sub.grad = None
        f(sub).backward(v)
        g.append(sub.grad.clone())
    spread = float(torch.stack(g).std(0).max())
    ok = torch.autograd.gradcheck(f, (sub,), eps=1e-6, atol=1e-6, rtol=1e-4,
                                  fast_mode=True, nondet_tol=1e-8)
    print(f"gradcheck, 12 controls x 192 vertex coords, fp64: {'PASS' if ok else 'FAIL'}"
          f"   (backward nondeterminism {spread:.1e}, from atomic reductions)")

    # ---- timing on the fp32 path, the one training uses ------------------
    del rig64
    if dev == "cuda":
        torch.cuda.empty_cache()
    rig = TorchRig(npz, device=dev, dtype=torch.float32)
    base = torch.cuda.memory_allocated() / 1e6 if dev == "cuda" else 0.0
    print(f"  resident tables, fp32: {base:.0f} MB")
    for B in (1, 4, 16):
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        cb = torch.rand(B, N_RAW, device=dev) * 0.3
        cb.requires_grad_(True)
        for _ in range(3):
            rig(cb).square().mean().backward()
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            cb.grad = None
            rig(cb).square().mean().backward()
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / 10 * 1e3
        mem = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else 0.0
        print(f"  fwd+bwd B={B}: {dt:7.2f} ms   peak {mem:6.0f} MB")
    return worst < 1e-4 and ok


if __name__ == "__main__":
    sys.exit(0 if verify() else 1)
