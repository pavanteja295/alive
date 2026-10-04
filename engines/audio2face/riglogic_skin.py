#!/usr/bin/env python3
"""
Controls -> vertices, entirely outside Unreal.

    gui[174] -> mapGUIToRawControls -> calculate
             -> jointOutputs (870 x 9 deltas) + blendShapeOutputs (782)
             -> blendshape deltas -> LBS -> vertices

Everything here was determined by probing the DNA, not from documentation:

  * getJointOutputs() is 7830 floats = 870 joints x 9, JOINT-MAJOR.
    joint i occupies [i*9 : i*9+9] as [tx ty tz rx ry rz sx sy sz].
  * The 9 values are DELTAS on the neutral local transform, not absolutes.
    At zero pose every output is 0.0 while neutral joint 0 translation is
    [0, 115.96, 3.85]. So local = neutral + delta.
  * There is no getNeutralJointScale accessor, so neutral scale is implicitly 1
    and absolute scale is 1 + delta. Reading it as 0 + delta collapses the mesh.
  * Rotations are Euler DEGREES (Configuration.rotationType defaults to
    RotationType_EulerAngles = 3; neutral rotations read like -90, 12.36, 90).
  * Skin weights are indexed by POSITION index (getSkinWeightsCount(0) == 24049
    for head_lod0), not by the 24408 split/layout index.

THE ACCEPTANCE TEST is at the bottom: with all controls at zero, skinned
vertices must equal the DNA's stored neutral positions. If that passes to
floating-point tolerance, then the hierarchy walk, the Euler order, the delta
semantics and the weights are all correct together. If it fails, exactly one of
those four is wrong and nothing downstream is trustworthy.

Run:
    cd externals/OpenRigLogic/build/python
    LD_LIBRARY_PATH=. PYTHONPATH=./dna:./riglogic \
      /tmp/ortbench/bin/python .../pipeline/riglogic_skin.py
"""

import os

import numpy as np
import dna
import riglogic

DNA_PATH = os.environ.get("FACE_DNA", "/tmp/pv/face.dna")

# ZYX, i.e. R = Rz @ Ry @ Rx. Established by geometry, not by documentation: with ZYX
# all 843 FACIAL_* joints land inside the head mesh bounding box and the `head` joint
# sits at Y=154.7 against a mesh spanning 134.7..171.8 (centre 153.2). With XYZ, zero
# facial joints land inside and `head` sits at Y=114.8, i.e. spine height, because the
# 39.4 cm neck chain points along -Z instead of +Y.
#
# The neutral-pose test CANNOT catch this: skin = pose_world @ inv(bind_world), and at
# neutral pose_world == bind_world, so the product is identity for any order. That test
# proves consistency, not correctness. joints_inside_head() below is the real check.
EULER_ORDER = "ZYX"


# ----------------------------------------------------------------------------
# transforms
# ----------------------------------------------------------------------------
def euler_deg_to_mat(rx, ry, rz, order=None):
    """Per-axis rotation matrices composed in `order`, left to right.

    `order` defaults to the module-level EULER_ORDER at CALL time. Do not turn this
    into `order=EULER_ORDER`: a default argument binds once at def time, so reassigning
    the module global would silently have no effect.
    """
    order = EULER_ORDER if order is None else order
    cx, sx = np.cos(np.radians(rx)), np.sin(np.radians(rx))
    cy, sy = np.cos(np.radians(ry)), np.sin(np.radians(ry))
    cz, sz = np.cos(np.radians(rz)), np.sin(np.radians(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    M = np.eye(3)
    for ax in order:
        M = M @ {"X": Rx, "Y": Ry, "Z": Rz}[ax]
    return M


def compose(t, r_deg, s):
    M = np.eye(4)
    M[:3, :3] = euler_deg_to_mat(*r_deg) * np.asarray(s)[None, :]
    M[:3, 3] = t
    return M


def world_matrices(local, parents):
    """Walk the hierarchy once. Parents are guaranteed to precede children in
    DNA joint order, which the assert enforces."""
    W = np.empty_like(local)
    for j, p in enumerate(parents):
        W[j] = local[j] if p < 0 else W[p] @ local[j]
    return W


# ----------------------------------------------------------------------------
# rig
# ----------------------------------------------------------------------------
class MetaHumanRig:
    def __init__(self, dna_path=DNA_PATH, lod=0):
        stream = dna.FileStream(
            dna_path, dna.FileStream.AccessMode_Read, dna.FileStream.OpenMode_Binary
        )
        self.r = dna.BinaryStreamReader(stream, dna.DataLayer_All, 0)
        self.r.read()
        r = self.r

        self.rl = riglogic.RigLogic.create(r)
        self.inst = riglogic.RigInstance(self.rl)
        self.inst.setLOD(lod)

        self.J = r.getJointCount()
        self.n_gui = r.getGUIControlCount()
        self.gui_names = [r.getGUIControlName(i) for i in range(self.n_gui)]

        self.parents = np.array(
            [r.getJointParentIndex(j) for j in range(self.J)], dtype=np.int64
        )
        # DNA stores the root with parent == its own index; normalise to -1.
        self.parents[self.parents == np.arange(self.J)] = -1
        assert all(
            p < j for j, p in enumerate(self.parents)
        ), "parents must precede children"

        self.nT = np.stack(
            [
                np.asarray(r.getNeutralJointTranslationXs()),
                np.asarray(r.getNeutralJointTranslationYs()),
                np.asarray(r.getNeutralJointTranslationZs()),
            ],
            axis=1,
        )
        self.nR = np.stack(
            [
                np.asarray(r.getNeutralJointRotationXs()),
                np.asarray(r.getNeutralJointRotationYs()),
                np.asarray(r.getNeutralJointRotationZs()),
            ],
            axis=1,
        )

        bind_local = np.stack(
            [compose(self.nT[j], self.nR[j], (1, 1, 1)) for j in range(self.J)]
        )
        self.bind_world = world_matrices(bind_local, self.parents)
        self.bind_world_inv = np.linalg.inv(self.bind_world)

    # -- per-mesh static data ------------------------------------------------
    def load_mesh(self, m):
        r = self.r
        n_v = r.getVertexPositionCount(m)
        V0 = np.stack(
            [
                np.asarray(r.getVertexPositionXs(m)),
                np.asarray(r.getVertexPositionYs(m)),
                np.asarray(r.getVertexPositionZs(m)),
            ],
            axis=1,
        )

        # skin weights, packed to a fixed max influence count
        idx_rows, w_rows = [], []
        for v in range(n_v):
            idx_rows.append(np.asarray(r.getSkinWeightsJointIndices(m, v), dtype=np.int64))
            w_rows.append(np.asarray(r.getSkinWeightsValues(m, v), dtype=np.float64))
        K = max(len(x) for x in idx_rows)
        JI = np.zeros((n_v, K), dtype=np.int64)
        JW = np.zeros((n_v, K), dtype=np.float64)
        for v, (ii, ww) in enumerate(zip(idx_rows, w_rows)):
            JI[v, : len(ii)] = ii
            JW[v, : len(ww)] = ww

        # blendshape targets: channel index -> (vertex indices, deltas)
        n_bs = r.getBlendShapeTargetCount(m)
        bs = []
        for t in range(n_bs):
            ch = r.getBlendShapeChannelIndex(m, t)
            vi = np.asarray(r.getBlendShapeTargetVertexIndices(m, t), dtype=np.int64)
            d = np.stack(
                [
                    np.asarray(r.getBlendShapeTargetDeltaXs(m, t)),
                    np.asarray(r.getBlendShapeTargetDeltaYs(m, t)),
                    np.asarray(r.getBlendShapeTargetDeltaZs(m, t)),
                ],
                axis=1,
            )
            bs.append((ch, vi, d))

        return dict(
            name=r.getMeshName(m), V0=V0, JI=JI, JW=JW, K=K, bs=bs,
            pos_idx=np.asarray(r.getVertexLayoutPositionIndices(m), dtype=np.int64),
        )

    # -- evaluation ----------------------------------------------------------
    def set_gui(self, values):
        """values: dict {name|index: float} or a full length-174 sequence."""
        if isinstance(values, dict):
            for i in range(self.n_gui):
                self.inst.setGUIControl(i, 0.0)
            for k, v in values.items():
                i = self.gui_names.index(k) if isinstance(k, str) else k
                self.inst.setGUIControl(i, float(v))
        else:
            assert len(values) == self.n_gui
            for i, v in enumerate(values):
                self.inst.setGUIControl(i, float(v))

    def evaluate(self):
        """-> (skin_mats [J,4,4], blendshape_weights [n_channels])"""
        self.rl.mapGUIToRawControls(self.inst)
        self.rl.calculate(self.inst)

        jo = np.asarray(self.inst.getJointOutputs()).reshape(self.J, 9)
        t = self.nT + jo[:, 0:3]          # delta on neutral
        rot = self.nR + jo[:, 3:6]        # delta on neutral, degrees
        s = 1.0 + jo[:, 6:9]              # <-- neutral scale is implicitly 1

        local = np.stack([compose(t[j], rot[j], s[j]) for j in range(self.J)])
        pose_world = world_matrices(local, self.parents)
        skin = pose_world @ self.bind_world_inv

        return skin, np.asarray(self.inst.getBlendShapeOutputs())

    def deform(self, mesh, skin, bs_w):
        V = mesh["V0"].copy()
        for ch, vi, d in mesh["bs"]:
            w = bs_w[ch]
            if w != 0.0:
                V[vi] += w * d

        Vh = np.concatenate([V, np.ones((len(V), 1))], axis=1)      # (n,4)
        M = skin[mesh["JI"]]                                       # (n,K,4,4)
        out = np.einsum("nkij,nj->nki", M, Vh)[..., :3]             # (n,K,3)
        return (out * mesh["JW"][..., None]).sum(axis=1)            # (n,3)


# ----------------------------------------------------------------------------
def main():
    global EULER_ORDER
    rig = MetaHumanRig()
    print(f"joints {rig.J}  guiControls {rig.n_gui}")
    print(f"root joints (parent<0): {int((rig.parents < 0).sum())}")

    head = rig.load_mesh(0)
    print(f"mesh {head['name']}: verts {head['V0'].shape}  maxInfluences {head['K']}  "
          f"bsTargets {len(head['bs'])}")
    w_sum = head["JW"].sum(axis=1)
    print(f"weight sums: min {w_sum.min():.6f}  max {w_sum.max():.6f}")

    # ------------- ACCEPTANCE TEST 1: geometry (catches wrong Euler order) -------
    jn = [rig.r.getJointName(j) for j in range(rig.J)]
    P = rig.bind_world[:, :3, 3]
    bb0, bb1 = head["V0"].min(0), head["V0"].max(0)
    fac = [j for j, n in enumerate(jn) if n.startswith("FACIAL_")]
    inside = int(((P[fac] >= bb0 - 2) & (P[fac] <= bb1 + 2)).all(1).sum())
    hj = jn.index("head")
    print()
    print(f"  GEOMETRY TEST  euler order {EULER_ORDER}")
    print(f"    head mesh Y span     {bb0[1]:.1f} .. {bb1[1]:.1f}  (centre {(bb0[1]+bb1[1])/2:.1f})")
    print(f"    'head' joint world   {np.round(P[hj], 1)}")
    print(f"    FACIAL joints inside {inside}/{len(fac)}")
    geom_ok = inside == len(fac)
    print(f"    {'PASS' if geom_ok else 'FAIL'}")

    # ------------- ACCEPTANCE TEST 2: neutral round-trip ------------------------
    rig.set_gui({})                       # every control at 0
    skin, bs_w = rig.evaluate()
    V = rig.deform(head, skin, bs_w)
    err = np.linalg.norm(V - head["V0"], axis=1)

    print()
    print(f"  NEUTRAL TEST  euler order {EULER_ORDER}")
    print(f"    max error  {err.max():.6e} cm")
    print(f"    mean error {err.mean():.6e} cm")
    print(f"    active blendshape weights at rest: {int((bs_w != 0).sum())}")
    ok = err.max() < 1e-4
    print(f"    {'PASS' if ok else 'FAIL'}")

    if not ok:
        print("\n    trying other Euler orders:")
        for order in ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]:
            EULER_ORDER = order
            r2 = MetaHumanRig()
            h2 = r2.load_mesh(0)
            r2.set_gui({})
            sk, bw = r2.evaluate()
            e = np.linalg.norm(r2.deform(h2, sk, bw) - h2["V0"], axis=1).max()
            print(f"      {order}: max err {e:.6e}")
        return

    # ---------------- does it actually move? ----------------
    rig.set_gui({"CTRL_C_jaw.ty": 1.0})
    skin, bs_w = rig.evaluate()
    Vj = rig.deform(head, skin, bs_w)
    disp = np.linalg.norm(Vj - head["V0"], axis=1)
    print()
    print("  JAW OPEN  CTRL_C_jaw.ty = 1.0")
    print(f"    verts moved >0.1cm : {int((disp > 0.1).sum())} / {len(disp)}")
    print(f"    max displacement   : {disp.max():.3f} cm")
    print(f"    active blendshapes  : {int((bs_w != 0).sum())}")


if __name__ == "__main__":
    main()
