#!/usr/bin/env python3
"""RigLogic's behaviour layer, reimplemented in numpy from DNA tables only.

    cd externals/OpenRigLogic/build/python
    LD_LIBRARY_PATH=. PYTHONPATH=./dna:./riglogic \
        ~/.venvs/mh-offset/bin/python \
        offset/riglogic_numpy.py

Why this exists
---------------
OpenRigLogic is C++ with no autograd tape, so the controls -> vertices map could not
be differentiated. The usual workaround is a finite-difference Jacobian, which is an
approximation. It turns out no approximation is needed: the DNA exposes every table
RigLogic evaluates, and the whole map is a short composition of monomials, sparse
matrix products, sines and cosines.

This module extracts those tables and reproduces `rl.calculate()` exactly. Once the
tables are extracted they are plain arrays, so the same arithmetic transcribes to
torch with autograd and no further reverse-engineering.

Verified: 25 random poses, worst vertex disagreement 3.9e-06 mm. See `verify()`.

What was read to build this
---------------------------
Behaviour tables, all from the DNA reader (see `RigTables.from_dna`):
    getPSDRowIndices / getPSDColumnIndices / getPSDValues
    getJointGroupInputIndices / OutputIndices / Values  (per group)
    getBlendShapeChannelInputIndices / OutputIndices
    getRBFSolver*  (parameters only; see the note on the RBF stage below)
Geometry stages are taken from ../riglogic_skin.py unchanged: `compose`,
`world_matrices`, `MetaHumanRig.deform`.

Conventions that were determined by testing candidates, not by reading docs
--------------------------------------------------------------------------
  * PSD output is CLAMPED to [0, 1].  Raw controls are NOT clamped by RigLogic
    (setting 1.7 stores 1.7).  Without the PSD clamp, 1 pose in 25 disagrees by 6.6 mm.
  * PSD weight is the PRODUCT of its per-term values, not the max or the first.
    Equivalent here because no PSD has more than one non-unit term, but stated
    explicitly so the port does not depend on that coincidence.
  * getJointGroupValues is ROW-MAJOR [n_out, n_in].  Column-major gives 37.4 error
    yet still produces a smooth plausible face, so it will not announce itself.
  * `rig.evaluate()` in riglogic_skin.py calls mapGUIToRawControls FIRST and therefore
    cannot be used to evaluate raw-driven controls.  Drive with setRawControl +
    rl.calculate, as three_way.py does.

Measured facts about this DNA (the base MetaHuman face rig, DNA 2.5)
------------------------------------------------------
  control vector u: 826 = 263 raw + 545 PSD + 18 RBF + 0 ML
  PSD terms 1430; terms per PSD 2:301 3:166 4:62 5:14 6:2
  PSD weights: 4.0 for 6 PSDs, 1.0 for 539.  0 chained.  0 repeated columns.
  joint groups 122 (121 non-empty), 1,000,930 matrix entries, ~4 MB fp32
  joint outputs 7830 = 870 joints x 9
  blendshape channels 782
  RBF: 1 solver, 38 poses, inputs = raw[251:263] (the neck/head quaternions),
       none of which has a depth-solve curve -> constant for our pipeline
"""

import sys
import time
from pathlib import Path

import numpy as np

PIPE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPE))
from riglogic_skin import MetaHumanRig, compose, world_matrices  # noqa: E402

N_RAW, N_PSD, N_RBF = 263, 545, 18
N_U = N_RAW + N_PSD + N_RBF          # 826


class RigTables:
    """RigLogic's behaviour layer as plain arrays. No DNA reader needed after this."""

    __slots__ = ("psd_row", "psd_col", "psd_w", "jg", "bs_in", "bs_out",
                 "n_joint_out", "n_bs", "n_raw", "n_psd", "n_rbf")

    @classmethod
    def from_dna(cls, reader):
        t = cls()
        t.n_raw = reader.getRawControlCount()
        t.n_psd = reader.getPSDCount()
        t.n_rbf = reader.getRBFPoseControlCount()
        assert (t.n_raw, t.n_psd, t.n_rbf) == (N_RAW, N_PSD, N_RBF), \
            f"unexpected layout {(t.n_raw, t.n_psd, t.n_rbf)}; this DNA differs"

        # --- PSD: psd_j = clip(w_j * prod_k u[col_k], 0, 1) -----------------
        t.psd_row = np.asarray(reader.getPSDRowIndices()).astype(np.int64) - t.n_raw
        t.psd_col = np.asarray(reader.getPSDColumnIndices()).astype(np.int64)
        vals = np.asarray(reader.getPSDValues()).astype(np.float64)
        t.psd_w = np.ones(t.n_psd)
        for k in range(len(t.psd_row)):          # weight is the PRODUCT of its terms
            t.psd_w[t.psd_row[k]] *= vals[k]

        # --- joint groups: delta[out_g] += V_g @ u[in_g] --------------------
        t.jg = []
        for g in range(reader.getJointGroupCount()):
            ii = np.asarray(reader.getJointGroupInputIndices(g)).astype(np.int64)
            if len(ii) == 0:
                continue
            oi = np.asarray(reader.getJointGroupOutputIndices(g)).astype(np.int64)
            vv = np.asarray(reader.getJointGroupValues(g)).astype(np.float64)
            assert len(vv) == len(oi) * len(ii)
            t.jg.append((ii, oi, vv.reshape(len(oi), len(ii))))     # ROW-major
        t.n_joint_out = reader.getJointCount() * 9

        # --- blendshapes: bsw[out_k] = u[in_k] -----------------------------
        t.bs_in = np.asarray(reader.getBlendShapeChannelInputIndices()).astype(np.int64)
        t.bs_out = np.asarray(reader.getBlendShapeChannelOutputIndices()).astype(np.int64)
        t.n_bs = reader.getBlendShapeChannelCount()
        return t

    def save(self, path, rig=None, meshes=None, rbf=None):
        """Write every constant the forward pass needs.

        With `rig`, `meshes` and `rbf` supplied the result is SELF-CONTAINED: the
        torch port can load this one file and never open the DNA or link
        OpenRigLogic. Behaviour layer alone without them.
        """
        d = {"psd_row": self.psd_row, "psd_col": self.psd_col, "psd_w": self.psd_w,
             "bs_in": self.bs_in, "bs_out": self.bs_out,
             "meta": np.array([self.n_raw, self.n_psd, self.n_rbf,
                               self.n_joint_out, self.n_bs, len(self.jg)])}
        for i, (ii, oi, vv) in enumerate(self.jg):
            d[f"jg{i}_in"], d[f"jg{i}_out"], d[f"jg{i}_v"] = ii, oi, vv.astype(np.float32)

        if rbf is not None:
            d["rbf"] = np.asarray(rbf, np.float32)
        if rig is not None:                                   # skeleton constants
            d["nT"] = rig.nT.astype(np.float32)
            d["nR"] = rig.nR.astype(np.float32)
            d["parents"] = np.asarray(rig.parents, np.int32)
            d["bind_world_inv"] = rig.bind_world_inv.astype(np.float32)
        if meshes is not None:                                # per-mesh geometry
            d["n_meshes"] = np.array([len(meshes)])
            for m, mesh in enumerate(meshes):
                d[f"m{m}_V0"] = mesh["V0"].astype(np.float32)
                d[f"m{m}_JI"] = mesh["JI"].astype(np.int32)
                d[f"m{m}_JW"] = mesh["JW"].astype(np.float32)
                # blendshape targets, flattened with a per-target offset table
                ch = np.array([t[0] for t in mesh["bs"]], np.int32)
                lens = np.array([len(t[1]) for t in mesh["bs"]], np.int64)
                d[f"m{m}_bs_ch"] = ch
                d[f"m{m}_bs_off"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
                d[f"m{m}_bs_vi"] = (np.concatenate([t[1] for t in mesh["bs"]]).astype(np.int32)
                                    if len(ch) else np.zeros(0, np.int32))
                d[f"m{m}_bs_d"] = (np.concatenate([t[2] for t in mesh["bs"]]).astype(np.float32)
                                   if len(ch) else np.zeros((0, 3), np.float32))
        np.savez_compressed(path, **d)

    @property
    def n_jg_entries(self):
        return sum(v.size for _, _, v in self.jg)


# ---------------------------------------------------------------- behaviour --
def psd_values(u, t):
    """Stage 2. u -> the 545 PSD activations. Monomials of degree 2..6, clamped twice.

    BOTH clamps are load-bearing and they are not the same clamp:

      * inputs  -- the PSD stage, and ONLY the PSD stage, reads controls clipped
        to [0,1]. The joint-group and blendshape stages downstream read the SAME
        controls UNCLAMPED. Measured against OpenRigLogic over 51 poses: clamping
        the inputs everywhere gives 40.6 joint error, clamping them nowhere gives
        0.345 on PSD outputs and 2.1 mm on vertices.
      * output -- 6 of the 545 carry weight 4.0 and would exceed 1 without it.

    This only diverges when a raw control leaves [0,1], which the depth solve
    never does (measured range on Take 4: 0.0000 .. 1.0000). It bites as soon as
    c_raw = base + delta is evaluated without clipping, which is exactly what
    photometric training through the rig does.
    """
    p = np.ones(t.n_psd)
    np.multiply.at(p, t.psd_row, np.clip(u[t.psd_col], 0.0, 1.0))   # inputs clamped
    return np.clip(t.psd_w * p, 0.0, 1.0)                           # output clamped


def joint_deltas(u, t):
    """Stage 3. u -> 7830 joint transform deltas. One sparse linear map."""
    d = np.zeros(t.n_joint_out)
    for ii, oi, vv in t.jg:
        d[oi] += vv @ u[ii]
    return d


def blendshape_weights(u, t):
    """Stage 4. u -> 782 blendshape weights. Pure routing, one gather."""
    w = np.zeros(t.n_bs)
    w[t.bs_out] = u[t.bs_in]
    return w


def behaviour(c_raw, rbf, t):
    """raw controls -> (joint deltas, blendshape weights). Stages 2-4.

    `rbf` is the 18 RBF pose controls. They are driven only by raw[251:263], the
    neck and head quaternions, none of which our pipeline touches -- so for us they
    are a constant, read once from the instance. Pass zeros only if you have
    confirmed the quaternions are zero too.
    """
    u = np.zeros(N_U)
    u[:N_RAW] = c_raw                      # RigLogic does NOT clamp these
    u[N_RAW + N_PSD:] = rbf
    u[N_RAW:N_RAW + N_PSD] = psd_values(u, t)
    return joint_deltas(u, t), blendshape_weights(u, t), u


# ----------------------------------------------------------------- geometry --
def geometry_from_tables(delta, bsw, z):
    """Stages 5-9 using ONLY arrays from rig_tables.npz.

    This is the runtime path. It needs no DNA file, no OpenRigLogic and no Unreal --
    `z` is the loaded npz and nothing else. Use this, not `geometry()`, unless you
    already have a live MetaHuman rig object for some other reason.

        z = np.load("rig_tables.npz")
        jo, bsw, _ = behaviour(c_raw, z["rbf"], tables)
        V = geometry_from_tables(jo, bsw, z)

    Verified against OpenRigLogic at 8.4e-05 mm; looser than the fp64 path below
    only because the npz stores fp32.
    """
    jo = delta.reshape(-1, 9)
    nT, nR = z["nT"], z["nR"]
    loc = np.stack([compose(nT[j] + jo[j, 0:3], nR[j] + jo[j, 3:6], 1.0 + jo[j, 6:9])
                    for j in range(len(jo))])
    skin = world_matrices(loc, z["parents"]) @ z["bind_world_inv"]

    parts = []
    for m in range(int(z["n_meshes"][0])):
        V = z[f"m{m}_V0"].astype(np.float64).copy()
        ch, off = z[f"m{m}_bs_ch"], z[f"m{m}_bs_off"]
        vi, dd = z[f"m{m}_bs_vi"], z[f"m{m}_bs_d"]
        # NO `if w != 0` GUARD, deliberately. riglogic_skin.deform has one as a speed
        # shortcut, and at real frames it skips 68.4% of channels (247 of 782 nonzero).
        # Transcribed literally into torch that silently zeroes the gradient for every
        # inactive shape -- but dV/dbsw[ch] = D_ch is NONZERO at w = 0, so the optimiser
        # could never turn a shape ON. This loop is the shape the torch port must copy:
        # unconditional, and in torch a single index_add / sparse matmul.
        for k in range(len(ch)):
            np.add.at(V, vi[off[k]:off[k + 1]], bsw[ch[k]] * dd[off[k]:off[k + 1]])
        Vh = np.concatenate([V, np.ones((len(V), 1))], 1)
        M = skin[z[f"m{m}_JI"]]
        o = np.einsum("nkij,nj->nki", M, Vh)[..., :3]
        parts.append((o * z[f"m{m}_JW"][..., None]).sum(1))
    return np.concatenate(parts, 0)


def geometry(delta, bsw, rig, meshes):
    """Stages 5-9 via a live MetaHuman rig object.

    NOTE this REQUIRES OpenRigLogic: `rig` comes from riglogic_skin.MetaHumanRig,
    whose module does `import dna; import riglogic` at import time. For a runtime
    path with no OpenRigLogic dependency use `geometry_from_tables` above.
    """
    jo = delta.reshape(-1, 9)
    loc = np.stack([compose(rig.nT[j] + jo[j, 0:3],
                            rig.nR[j] + jo[j, 3:6],
                            1.0 + jo[j, 6:9]) for j in range(len(jo))])
    skin = world_matrices(loc, rig.parents) @ rig.bind_world_inv
    return np.concatenate([rig.deform(m, skin, bsw) for m in meshes], 0)


# ------------------------------------------------------------------ verify ---
def verify(dna_path=None, n_trials=25, seed=3):
    dna_path = dna_path or str(PIPE / "assets" / "face.dna")
    rig = MetaHumanRig(dna_path)
    inst, rl, r = rig.inst, rig.rl, rig.r
    t = RigTables.from_dna(r)
    meshes = [rig.load_mesh(m) for m in range(r.getMeshCount())
              if r.getMeshName(m).endswith("lod0_mesh")
              and r.getMeshName(m) != "eyelashes_lod0_mesh"]

    print(f"tables: u={N_U} ({t.n_raw}+{t.n_psd}+{t.n_rbf})  "
          f"PSD terms={len(t.psd_row)}  joint groups={len(t.jg)}  "
          f"jg entries={t.n_jg_entries:,}  bs channels={t.n_bs}")
    print(f"meshes: {len(meshes)}  vertices={sum(len(m['V0']) for m in meshes):,}\n")

    rng = np.random.default_rng(seed)
    worst = dict(psd=0.0, joint=0.0, bs=0.0, vert=0.0)
    t0 = time.time()
    for i in range(n_trials):
        c = np.zeros(N_RAW)
        n = int(rng.integers(20, 120))
        sel = rng.choice(251, n, replace=False)
        # Half the trials leave [0,1] on purpose. Drawing only from rng.random()
        # is what let the missing PSD input clamp survive: every stage agreed
        # inside the nominal range and diverged by 2.1 mm outside it.
        c[sel] = rng.random(n) if i % 2 else rng.uniform(-0.6, 1.6, n)

        for k in range(N_RAW):
            inst.setRawControl(int(k), float(c[k]))
        rl.calculate(inst)                       # NOT rig.evaluate(): that remaps GUI
        psd_t = np.asarray(inst.getPSDControlValues())
        jo_t = np.asarray(inst.getJointOutputs())
        bs_t = np.asarray(inst.getBlendShapeOutputs())
        rbf = np.asarray(inst.getRBFControlValues())

        jo_m, bs_m, u = behaviour(c, rbf, t)
        worst["psd"] = max(worst["psd"], np.abs(u[N_RAW:N_RAW + N_PSD] - psd_t).max())
        worst["joint"] = max(worst["joint"], np.abs(jo_m - jo_t).max())
        # A joints-only DNA (e.g. an Identity auto-rigged as JointsOnly) has zero
        # blendshape channels, so this array is empty and .max() has no identity.
        if bs_m.size:
            worst["bs"] = max(worst["bs"], np.abs(bs_m - bs_t).max())
        v_m = geometry(jo_m, bs_m, rig, meshes)
        v_t = geometry(jo_t, bs_t, rig, meshes)
        worst["vert"] = max(worst["vert"], np.abs(v_m - v_t).max() * 10.0)  # cm -> mm

    print(f"{n_trials} random poses in {time.time()-t0:.0f}s   worst absolute error:")
    print(f"   PSD outputs         {worst['psd']:.3e}")
    print(f"   joint outputs       {worst['joint']:.3e}   (fp32 tables vs fp64 accum)")
    print(f"   blendshape weights  {worst['bs']:.3e}")
    print(f"   VERTICES            {worst['vert']:.3e} mm")
    ok = worst["vert"] < 1e-4
    print(f"\n{'PASS' if ok else 'FAIL'}  (threshold 1e-4 mm)")

    # --- write a self-contained artifact and prove it needs nothing else ------
    out = PIPE / "offset" / "rig_tables.npz"
    t.save(out, rig=rig, meshes=meshes, rbf=rbf)
    print(f"\nself-contained tables -> {out}  ({out.stat().st_size/1e6:.1f} MB)")

    z = np.load(out)
    c = np.zeros(N_RAW)
    rng2 = np.random.default_rng(99)
    n = 80
    c[rng2.choice(251, n, replace=False)] = rng2.random(n)

    for k in range(N_RAW):                         # ground truth via OpenRigLogic
        inst.setRawControl(int(k), float(c[k]))
    rl.calculate(inst)
    v_true = geometry(np.asarray(inst.getJointOutputs()),
                      np.asarray(inst.getBlendShapeOutputs()), rig, meshes)

    jo_z, bsw_z, _ = behaviour(c, z["rbf"], t)     # ...and via the npz-only path
    v_npz = geometry_from_tables(jo_z, bsw_z, z)

    e = np.abs(v_npz - v_true).max() * 10.0
    print(f"npz-only evaluation vs OpenRigLogic: {e:.3e} mm  "
          f"-> {'the artifact is self-contained' if e < 1e-4 else 'MISMATCH'}")
    return ok and e < 1e-4


if __name__ == "__main__":
    sys.exit(0 if verify() else 1)
