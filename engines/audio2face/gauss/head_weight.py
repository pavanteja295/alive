#!/usr/bin/env python3
"""How much of the head's rigid motion each vertex of the head mesh is entitled to.

    python pipeline/gauss/head_weight.py drk [--run D1_pred_sel]

    from head_weight import HeadWeight
    hw = HeadWeight("drk")
    Rr, tr = hw.reference(R, t)              # the chunk's own resting placement
    posed  = hw.place(cano, R[i], t[i], Rr, tr)

THE DEFECT THIS EXISTS TO FIX

Head pose was applied as ONE rigid transform to every vertex of the head mesh:

    posed = cano @ R + t

The head mesh does not stop at the head. Its lowest vertices are the collar, and the rig
gives them weight 0.000 on the head -- they are skinned to the neck and spine joints and
are not supposed to move when the head turns. A seventh of the deployed run's Gaussians
sit on triangles with zero head weight and every one of them was being rotated with the
skull. That is the shoulders swinging with the head.

NO NUMBER IS WRITTEN DOWN HERE. `--run` counts them off the point cloud and `--verify`
measures the displacement on real poses, because a figure copied into a docstring is one
stale figure waiting to happen and this project has already lost a day to one.

WHY IT LOOKED CORRECT, AND WHY THAT ARGUMENT DOES NOT APPLY

`flame-vs-metahuman.html` proves that because the skin weights are a partition of unity
-- zero of 24,049 vertices off by more than 1e-6 -- transforming the finished mesh is the
same as pre-multiplying all 870 joint matrices, agreeing to 4.6e-04 mm. That is true, and
it is the wrong theorem for this purpose. It says a rigid transform of the WHOLE HEAD is
safe. It does not say the whole head is what should be transformed.

WHY THE BLEND IS NOT AN APPROXIMATION OF THE RIG

Linear blend skinning averages the joint MATRICES, and the weights sum to one, so

    v (w M_head + (1-w) M_body)  ==  w (v M_head) + (1-w) (v M_body)

Blending the two transformed positions IS two-influence skinning, exactly. The head is
one influence, the body held at its reference placement is the other, and w is read out
of the rig's own `m0_JW` rather than invented.

THE SUBTREE, WHICH IS THE PART THAT IS EASY TO GET WRONG

"What weight does this vertex have on the head joint" is the wrong question. Under a head
rotation every joint BELOW the head inherits that rotation and adds nothing of its own,
so the quantity that matters is the weight on the whole SUBTREE. Asking about the joint
alone concludes that a face is blended when it is rigid.

WHAT THIS DOES NOT DO

The rig's neck joints stay at rest, so the neck gets a smooth gradient of the head's
rotation rather than a real twist. That cannot be fixed through the control vector: raw
controls 251-262 (`neck_01.q*`, `neck_02.q*`, `head.q*`) drive no joint group and no
corrective. Checked on this subject's own rig by `_assert_inert`, not assumed.

And the body does not move at all. Real shoulders shift a little under a big turn. That
is a different wrong, and a smaller one.
"""
import argparse
import pathlib

import numpy as np

P = pathlib.Path(__file__).resolve().parent.parent

# The MetaHuman face chain is spine_05 -> neck_01 -> neck_02 -> head, and `head` is joint
# 84. NAMED, NOT DERIVED: every rule for finding it structurally that was tried lands one
# joint too deep, on the facial root at 147.14 cm, whose subtree is the head's minus the
# head bone itself -- which silently drops every vertex skinned to the head bone. So it is
# declared here and `_assert_structure` checks the weight field it produces has the shape
# a head joint must produce: a dead collar, a rigid skull, and a ramp in between.
HEAD_JOINT = 84


class HeadWeight:
    """The per-vertex share of the head's rigid motion, off the subject's own rig."""

    def __init__(self, subject, head_joint=HEAD_JOINT, mesh=0, rig=None):
        self.subject, self.head_joint, self.mesh = subject, head_joint, mesh
        self.path = pathlib.Path(rig) if rig else (
            P / "identity/subjects" / subject / "rig_beltrami.npz")
        z = np.load(self.path)
        par = z["parents"].astype(int)
        self.V0 = z[f"m{mesh}_V0"].astype(np.float64)
        JI, JW = z[f"m{mesh}_JI"], z[f"m{mesh}_JW"]

        # the head joint and everything hanging off it
        desc = np.zeros(len(par), bool)
        desc[head_joint] = True
        while True:
            grew = (par >= 0) & desc[np.maximum(par, 0)] & ~desc
            if not grew.any():
                break
            desc |= grew
        self.subtree = desc

        w = np.zeros(len(self.V0))
        for k in range(JI.shape[1]):
            w += np.where(desc[JI[:, k]], JW[:, k], 0.0)
        self.w = np.clip(w, 0.0, 1.0)
        self.wc = self.w[:, None]

        # the rig's own partition of unity, re-checked here rather than trusted. If the
        # weights do not sum to one the blend below is not skinning and the mesh shears.
        tot = JW.sum(1)
        assert np.abs(tot - 1.0).max() < 1e-5, (
            f"skin weights are not a partition of unity: worst {np.abs(tot - 1).max():.2e}")
        self._assert_structure()
        self._assert_inert(z)

    # ---- the checks, which are the reason this is a module and not three lines -------
    def _assert_structure(self):
        """A head joint produces a dead collar, a rigid skull, and a ramp between them.

        This is a RELATION, not a threshold fitted to this person: whatever the head joint
        is, the bottom of the head mesh must not follow it and the top must follow it
        completely. A joint chosen wrongly fails one end or the other.
        """
        if self.mesh != 0:
            # The relation below is about the head mesh, which spans skull to collar. An
            # auxiliary mesh -- an eyeball, the teeth -- lies wholly inside the skull, so
            # "the bottom must not follow the head joint" is not a property it has. The
            # meaningful invariant there is the opposite one: all of it rides the skull.
            assert self.w.min() > 0.99, (
                f"mesh {self.mesh} is not rigid with the skull (min weight "
                f"{self.w.min():.3f}) -- joint {self.head_joint} is not its parent")
            return
        y = self.V0[:, 1]
        lo = self.w[y <= np.percentile(y, 3)]
        hi = self.w[y >= np.percentile(y, 60)]
        assert lo.max() < 0.01, (
            f"joint {self.head_joint} moves the bottom of the mesh (max weight "
            f"{lo.max():.3f}) -- it is not the head")
        assert hi.min() > 0.99, (
            f"joint {self.head_joint} leaves the skull behind (min weight "
            f"{hi.min():.3f}) -- it is not the head")

    def _assert_inert(self, z):
        """The 12 neck/head quaternion controls drive nothing, so the neck cannot be
        articulated through the control vector and the blend here is the only route."""
        live = [k for k in z.files
                if k.startswith("jg") and k.endswith("_in")
                and ((z[k] >= 251) & (z[k] <= 262)).any()]
        assert not live, f"raw controls 251-262 now drive {live} -- drive the neck instead"

    # ---- the placement ---------------------------------------------------------------
    @staticmethod
    def reference(R, t, hz=0.0, fps=30.0):
        """Where the body sits: the chunk's own resting placement.

        `hz = 0` gives one constant placement for the whole chunk -- the proper mean of
        the rotations, not a mean of Euler angles, which is not a rotation. A nonzero `hz`
        instead lets the body follow posture drift below that frequency and hold still
        against everything faster. Zero is the baseline; the knob exists because about
        12 mm of his translation is slow drift that a hard mean charges to the head.
        """
        R = np.asarray(R, np.float64)
        t = np.asarray(t, np.float64)
        if hz <= 0:
            U, _, Vt = np.linalg.svd(R.mean(0))
            Rm = U @ Vt
            if np.linalg.det(Rm) < 0:            # reflection, not a rotation
                U[:, -1] *= -1
                Rm = U @ Vt
            return Rm, t.mean(0)
        from scipy.ndimage import gaussian_filter1d
        sigma = fps / (2 * np.pi * hz)
        Rs = gaussian_filter1d(R, sigma, axis=0, mode="nearest")
        U, _, Vt = np.linalg.svd(Rs)
        Rs = U @ Vt
        flip = np.linalg.det(Rs) < 0
        if flip.any():
            U[flip][:, :, -1] *= -1
            Rs[flip] = (U @ Vt)[flip]
        return Rs, gaussian_filter1d(t, sigma, axis=0, mode="nearest")

    @staticmethod
    def channels(R, t):
        """The six numbers the renderer is conditioned on, from a placement.

        Intrinsic XYZ off the row-vector rotation in degrees, translation in millimetres
        so the six sit in a comparable range. ONE implementation: cooking and inference
        both call this, because a renderer told the head is at one angle while the mesh
        is at another is the exact failure the conditioning exists to avoid, and it is
        silent.
        """
        R = np.asarray(R, np.float64).reshape(-1, 3, 3)
        t = np.asarray(t, np.float64).reshape(-1, 3)
        sy = np.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2)
        e = np.degrees(np.stack([np.arctan2(R[:, 2, 1], R[:, 2, 2]),
                                 np.arctan2(-R[:, 2, 0], sy),
                                 np.arctan2(R[:, 1, 0], R[:, 0, 0])], 1))
        return np.concatenate([e, t * 1000.0], 1)

    @staticmethod
    def relative(R, t, R_ref=None, t_ref=None):
        """A pose track re-expressed as motion about its own resting placement.

        A chunk's R and t are in that chunk's own gauge: its own neutral, and a camera
        the exporter invented around its own mean head position. Across the corpus the
        per-chunk MEAN yaw alone spans 61 degrees, because the source video cuts between
        camera angles. Replaying one chunk's raw pose somewhere else therefore teleports
        the head before it moves at all. This strips the gauge and leaves the motion.
        """
        R = np.asarray(R, np.float64)
        t = np.asarray(t, np.float64)
        if R_ref is None:
            R_ref, t_ref = HeadWeight.reference(R, t)
        A = np.einsum("ji,tjk->tik", R_ref, R)          # R_ref^T R_t, row-vector form
        b = t - np.einsum("j,tjk->tk", t_ref, A)
        return A, b

    def place(self, cano, R, t, R_ref, t_ref):
        """The posed mesh. `cano` is [nv,3] in the chunk's canonical frame.

        Two influences, blended by the rig's own weights: the head at this frame's pose,
        the body at the reference placement. Vertices at weight 1 are bit-for-bit what the
        old rigid step produced; vertices at weight 0 do not move at all.
        """
        head = cano @ R + t
        body = cano @ R_ref + t_ref
        return self.wc * head + (1.0 - self.wc) * body

    # ---- what a person looks at ------------------------------------------------------
    def report(self, run=None):
        y = self.V0[:, 1]
        print(f"{self.subject}: {self.path}")
        print(f"head joint {self.head_joint}, {int(self.subtree.sum()) - 1} joints below "
              f"it, {len(self.V0)} vertices in mesh {self.mesh}")
        print(f"  weight 1.000 (rigid with the skull) : {100 * (self.w > 0.999).mean():5.1f}%")
        print(f"  weight 0.000 (must not move)        : {100 * (self.w < 0.001).mean():5.1f}%  "
              f"{int((self.w < 0.001).sum())} vertices")
        print(f"  in between (the blend)              : "
              f"{100 * ((self.w >= 0.001) & (self.w <= 0.999)).mean():5.1f}%")
        print("\n  mesh height, cm     mean head weight   vertices")
        b = np.linspace(y.min(), y.max(), 12)
        for i in range(11):
            m = (y >= b[i]) & (y < b[i + 1] if i < 10 else y <= b[i + 1])
            if m.any():
                print(f"   {b[i]:6.1f} - {b[i + 1]:6.1f}      {self.w[m].mean():.3f}"
                      f"            {int(m.sum()):6d}")
        if run:
            self._report_run(run)

    def _report_run(self, run):
        """How many of a trained run's Gaussians were being moved that should not be.

        Reads the binding out of the point cloud rather than recomputing it, because the
        question is what THAT run did, not what a run would do.
        """
        from plyfile import PlyData
        run = pathlib.Path(run)
        if not run.is_absolute():
            run = P / "gauss/runs" / run
        pcs = sorted((run / "point_cloud").glob("iteration_*"),
                     key=lambda x: int(x.name.split("_")[1]))
        if not pcs:
            print(f"\n{run} has no point cloud"); return
        e = PlyData.read(str(pcs[-1] / "point_cloud.ply")).elements[0]
        names = [p.name for p in e.properties]
        if "binding_0" not in names:
            print(f"\n{pcs[-1].name}: no binding, so these Gaussians are not mesh-bound")
            return
        ha = np.load(P / "head/head_assets.npz")
        F = ha["pos_idx"][ha["faces"]]                 # UV-split faces -> position verts
        if F.max() >= len(self.V0):
            print(f"\n{run.name}: bound to a different topology, skipped"); return
        wf = self.w[F].mean(1)
        w = wf[np.asarray(e["binding_0"]).astype(int)]
        yf = self.V0[F][:, :, 1].mean(1)[np.asarray(e["binding_0"]).astype(int)]
        print(f"\n{run.name}, {len(w)} Gaussians, by what the rig says their triangle owes"
              f" the head:")
        for lo, hi, lab in [(-0.01, 0.001, "0.00  must not move"),
                            (0.001, 0.999, "      partial"),
                            (0.999, 1.01, "1.00  follows fully")]:
            m = (w > lo) & (w <= hi)
            if m.any():
                print(f"   {lab:22s} {int(m.sum()):7d}  {100 * m.mean():5.1f}%   "
                      f"triangle height {yf[m].mean():6.1f} cm")
        print(f"   the rigid step moved all {len(w)} of them at weight 1.")


CORPUS = pathlib.Path(__file__).resolve().parents[1] / "vhap/export/corpus"


def verify(hw, chunk, cooked="predicted", out=None, n=4):
    """The old rigid step against the new blend, on one real chunk, with a picture.

    The number and the picture are different objects, deliberately. The number says the
    face did not move and the collar did; the picture says whether the collar's new
    resting place is where his collar actually is, which no number here can answer.
    """
    import json
    from PIL import Image, ImageDraw
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from overlay_check import project

    r = np.load(P / f"gauss/cache/rigid/{chunk}.npz")
    R, t = r["R"].astype(np.float64), r["t"].astype(np.float64)
    Rr, tr = HeadWeight.reference(R, t)
    off = np.degrees(np.arccos(np.clip(
        (np.trace(np.einsum("tij,kj->tik", R, Rr), axis1=1, axis2=2) - 1) / 2, -1, 1)))

    meshes = CORPUS / f"{cooked}__{chunk}" / "meshes"
    have = sorted(meshes.glob("*.npz")) if meshes.exists() else []
    src = "the cooked meshes" if have else "the rig's neutral (nothing cooked yet)"
    print(f"{chunk}: {len(R)} frames, rotation off the chunk mean median "
          f"{np.median(off):.1f} deg, max {off.max():.1f}. Geometry from {src}.")

    exact = hw.w == 1.0
    ramp = (hw.w > 0.001) & (hw.w < 1.0)
    collar = hw.w <= 0.001
    sel = np.linspace(0, len(R) - 1, 40).astype(int)
    acc = {"skull": [], "ramp": [], "collar": []}
    same = True
    for i in sel:
        cano = (np.load(have[i])["verts_cano"].astype(np.float64) if have else hw.V0)
        old = cano @ R[i] + t[i]
        new = hw.place(cano, R[i], t[i], Rr, tr)
        same &= bool((new[exact] == old[exact]).all())
        e = np.linalg.norm(new - old, axis=1) * 1000.0
        acc["skull"].append(e[exact].max())
        acc["ramp"].append(e[ramp].mean())
        acc["collar"].append(e[collar].mean())
    print("\nhow far each group moves between the old rigid step and the new blend, mm:")
    print(f"  skull, weight exactly 1 ({int(exact.sum())} verts)   max "
          f"{max(acc['skull']):.6f}   bit-identical {same}")
    print(f"  the blend  ({int(ramp.sum()):5d} verts)   mean {np.mean(acc['ramp']):6.2f}"
          f"   max {np.max(acc['ramp']):6.2f}")
    print(f"  the collar ({int(collar.sum()):5d} verts)   mean "
          f"{np.mean(acc['collar']):6.2f}   max {np.max(acc['collar']):6.2f}")
    print("  the collar figure is how far the shoulders were being swung by a rotation "
          "the rig says they do not follow.")
    if not same:
        raise SystemExit("the face moved. The blend is not reproducing the rigid step "
                         "where the rig says the head is rigid, and nothing below this "
                         "line means anything.")
    if not out:
        return

    db = json.load(open(CORPUS / chunk / "transforms.json"))
    by_ts = {f["timestep_index"]: f for f in db["frames"]}
    show = np.argsort(-off)[:n]
    move = hw.w < 0.999                      # everything the head should not fully carry
    tiles = []
    for i in sorted(show):
        f = by_ts.get(int(i))
        if f is None:
            continue
        cano = (np.load(have[i])["verts_cano"].astype(np.float64) if have else hw.V0)
        im = Image.open(CORPUS / chunk / f["file_path"]).convert("RGB")
        d = ImageDraw.Draw(im)
        for V, col in ((cano @ R[i] + t[i], (255, 40, 40)),
                       (hw.place(cano, R[i], t[i], Rr, tr), (40, 255, 80))):
            uv, _ = project(V[move], f)
            for x, y in uv[::7]:
                d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=col)
        d.text((6, 6), f"frame {i}   {off[i]:.1f} deg off the chunk mean"
                       f"   red: rigid   green: blended", fill=(255, 255, 0))
        tiles.append(im)
    if not tiles:
        print("no frames to draw"); return
    w, h = tiles[0].size
    cols = min(2, len(tiles)); rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), "black")
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * w, (k // cols) * h))
    out = pathlib.Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\nwrote {out}: the {len(tiles)} most rotated frames, only the vertices the "
          f"rig says are not rigid with the skull. Red is where the old step put them, "
          f"green is where the blend does. Green should sit on his collar.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("--run", default="", help="a run under gauss/runs, to count the "
                                              "Gaussians that were being moved wrongly")
    ap.add_argument("--rig", default="", help="a rig_beltrami.npz, instead of the "
                                              "subject's own")
    ap.add_argument("--verify", default="", help="a chunk name: the old rigid step "
                                                 "against the new blend, on real poses")
    ap.add_argument("--cooked", default="predicted", help="which cooked dataset to take "
                                                          "the canonical meshes from")
    ap.add_argument("--out", default="", help="where to write the picture for --verify")
    a = ap.parse_args()
    hw = HeadWeight(a.subject, rig=a.rig or None)
    if a.verify:
        verify(hw, a.verify, cooked=a.cooked, out=a.out or None)
    else:
        hw.report(a.run or None)


if __name__ == "__main__":
    main()
