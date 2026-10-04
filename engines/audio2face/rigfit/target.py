#!/usr/bin/env python3
"""Shared geometry: where on the MetaHuman head each FLAME point lands, and how far
that point moves in a given frame.

Imported by solve.py, capacity.py and personalise.py so that every number in the study
is scored on exactly the same set of points, in exactly the same frame of reference.
Nothing here is specific to a chunk.

THE FOUR CHOICES THAT MAKE A NUMBER MEAN SOMETHING, unchanged from the original
ceiling experiment so the results stay comparable to it:

  DISPLACEMENT, NOT POSITION.  Both sides are measured as "this frame minus this rig's
      own rest face". The identity transfer closes 84-88% of the shape gap, not 100%,
      and fitting absolute position would charge that leftover identity error to the
      expression rig.

  PER-FRAME SKULL PROCRUSTES.  The tracker leaves up to ~10 degrees of rigid head
      rotation in the sequence even with global pose disabled. Un-removed, the fit
      would be scored on neck motion the face rig is not asked to produce. The skull
      (forehead, scalp, nose) is the part of a face that does not deform, so aligning
      on it removes rigid motion without removing expression.

  BARYCENTRIC CORRESPONDENCE, FIXED AT THE REST FACE.  24049 MetaHuman vertices against
      5023 FLAME ones is about 5 to 1, so nearest-vertex would quantise the target.
      Each MetaHuman vertex instead gets a (triangle, weights) address on the FLAME rest
      face and keeps it for every frame, which makes the target a real trajectory rather
      than a re-snapping to whatever is closest that frame.

  TWO REGIONS DROPPED.  FLAME's cut edge at the neck stump is a modelling artefact with
      no MetaHuman counterpart, and the neck is driven by neck joints rather than the
      face controls. Scoring on them charges the expression rig for motion outside its
      job.
"""
import pathlib
import pickle
import sys

import numpy as np

PIPE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPE / "vhap_metahuman"))
sys.path.insert(0, str(PIPE / "offset"))

from fit_metahuman_to_flame import (similarity, procrustes_rigid,          # noqa: E402
                                    closest_on_triangles, vert_normals,
                                    VHAP_ASSET, N_FLAME)

HEAD_NV = 24049


class Correspondence:
    """Built once per (subject, rig, FLAME rest face). Reused for every chunk."""

    def __init__(self, subject, rig="rig_beltrami.npz", corr_rig=None,
                 flame_neutral=None, gate_cm=0.40):
        from scipy.spatial import cKDTree
        sub = PIPE / "identity" / "subjects" / subject
        fx = np.load(sub / "flame_neutral.npz")
        wrap = np.load(sub / "beltrami_wrap.npz")
        H = np.load(PIPE / "head" / "head_assets.npz")
        Va = H["neutral_dna_cm"].astype(np.float64)
        Fm = H["pos_idx"].astype(np.int64)[H["faces"].astype(np.int64)]
        base = np.load(PIPE / "offset" / "rig_tables.npz")

        # FLAME metres -> DNA centimetres. A similarity only: rotation, translation and
        # one scalar. It cannot move a mouth corner relative to a nose, so no expression
        # passes through it.
        back = similarity(wrap["verts_archetype"], Va)
        S2 = similarity(back(wrap["verts_wrapped"]), base["m0_V0"].astype(np.float64))
        self.to_dna = lambda Z: S2(back(Z))

        # The rest face the correspondence is built on. Chunks solved against one locked
        # identity all share it; pass it in so the study uses the tracker's own rest face
        # rather than the one this subject's identity transfer happened to be built from.
        #
        # A monocular tracker cannot fix absolute scale, so two solves of the same person
        # come out at different sizes and in different places. The supplied rest face is
        # therefore brought onto the subject's own by a similarity first -- rotation,
        # translation, one scalar. Without this the size difference alone would push most
        # of the head outside the correspondence gate, and it would look like a
        # correspondence failure rather than the units mismatch it is.
        self.own = own = fx["verts"].astype(np.float64)[:N_FLAME]
        if flame_neutral is None:
            self.Vn_f = own
        else:
            self.Vn_f = similarity(np.asarray(flame_neutral, np.float64)[:N_FLAME],
                                   own)(np.asarray(flame_neutral, np.float64)[:N_FLAME])
            self.align = similarity(np.asarray(flame_neutral, np.float64)[:N_FLAME], own)
        self.Ft = fx["faces_target"].astype(np.int64)
        Vn_d = self.to_dna(self.Vn_f)

        Vm0 = np.load(sub / (corr_rig or rig))["m0_V0"].astype(np.float64)
        _, cand = cKDTree(Vn_d[self.Ft].mean(1)).query(Vm0, k=12)
        tri, bary, dist = closest_on_triangles(Vm0, Vn_d, self.Ft, cand)
        Nf = vert_normals(Vn_d, self.Ft)[self.Ft[tri]]
        agree = (vert_normals(Vm0, Fm) * (bary[..., None] * Nf).sum(1)).sum(1)
        self.mask = (dist < gate_cm) & (agree > 0.5)
        self.dist = dist
        self.align = getattr(self, 'align', None)

        self.T = self.Ft[tri][self.mask]                       # [M, 3] FLAME triangle
        self.W = bary[self.mask]                               # [M, 3] barycentric
        self.P0 = (self.W[..., None] * self.to_dna(self.Vn_f)[self.T]).sum(1)

        # The skull: the parts used to take rigid head motion out of every frame.
        #
        # The choice is not cosmetic. Measured with the EARS as an independent referee --
        # they are rigid bone and can be left out of the anchor -- the obvious anchor of
        # forehead + scalp + nose leaves them sliding 1.7-2.0 mm, against a face motion of
        # only 2.3-2.8 mm. That is residual head pose sitting inside the target at nearly
        # the size of the signal. The forehead is not rigid (brows move) and the scalp
        # carries tracker error (there is no hair mask).
        #
        # scalp + nose + ears leaves the ears at 0.67-0.77 mm, which is the tracker's own
        # noise floor on rigid parts. The nose alone is the most rigid region of all
        # (0.93 mm) but too compact to pin a rotation: on its own it leaves the ears at
        # 6.8 mm. Rigidity and spatial spread are both needed.
        #
        # Switching to this anchor changes the target by 0.71 mm, 26-31% of its own
        # motion, so anything measured against the old one is not comparable.
        self.mk = pickle.load(open(VHAP_ASSET, "rb"), encoding="latin1")
        sk = np.unique(np.concatenate([self.mk["scalp"], self.mk["nose"],
                                       self.mk["left_ear"], self.mk["right_ear"]]))
        self.skull = sk[sk < N_FLAME]

        vt = self.T[:, 0]                                      # a FLAME vertex per point
        drop = np.zeros(len(vt), bool)
        for name in ("boundary", "neck"):
            drop |= np.isin(vt, self.mk[name])
        self.facial = ~drop
        self.vt = vt

    def regions(self):
        """{name: boolean selector over the facial points}, from FLAME's own masks."""
        out = {}
        for name in ("lips", "nose", "forehead", "left_eye_region", "right_eye_region"):
            sel = np.isin(self.vt[self.facial], self.mk[name])
            if sel.sum() >= 50:
                out[name] = sel
        return out

    def sample(self, V):
        """Vertex positions [.., 5023, 3] -> the corresponded points, in DNA cm."""
        return (self.W[..., None] * self.to_dna(V)[self.T]).sum(1)

    def target(self, verts, neutral=None):
        """FLAME frames [T, 5023, 3] -> displacement at the facial points, in DNA cm.

        `neutral` is the rest face those frames were solved against. It is a separate
        argument because the tracker locks identity per recording, not across
        recordings: the same person comes out 0.2 mm different in a different video.
        Measuring each frame against its own recording's rest face keeps that drift out
        of the motion, where it does not belong.
        """
        V = np.asarray(verts, np.float64)[:, :N_FLAME]
        if neutral is None:
            Vn = self.Vn_f
        else:
            Vn = np.asarray(neutral, np.float64)[:N_FLAME]
            al = similarity(Vn, self.own)          # monocular scale is not fixed
            Vn, V = al(Vn), np.stack([al(v) for v in V])
        P0 = self.sample(Vn)
        out = np.empty((len(V), int(self.mask.sum()), 3), np.float32)
        for n in range(len(V)):
            R, t = procrustes_rigid(V[n, self.skull], Vn[self.skull])
            out[n] = (self.sample(V[n] @ R + t) - P0).astype(np.float32)
        return out[:, self.facial]
