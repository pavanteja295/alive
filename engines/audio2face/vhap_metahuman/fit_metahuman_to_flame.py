#!/usr/bin/env python3
"""Can MetaHuman's expression rig REACH the shapes VHAP/FLAME fits per frame?

    ~/miniconda3/envs/stavatar/bin/python vhap_metahuman/fit_metahuman_to_flame.py

WHY THIS EXPERIMENT EXISTS
    FLAME's meshes look more alive than MetaHuman's on the same clip. Two different
    things could explain that and they were being conflated:

        A. STATE SPACE   FLAME's 100 expression directions can make shapes that
                         MetaHuman's 263 controls cannot.
        B. FITTING       FLAME was optimised against this person's pixels, frame by
                         frame, hundreds of steps. MetaHuman's mono solve is one
                         forward pass of a network trained on other people.

    B alone would produce everything observed so far, so A is unproven. This isolates
    A by giving MetaHuman the SAME fitting treatment: gradient descent, per frame,
    straight onto FLAME's fitted surface, through the differentiable rig.

    No FLAME parameter is ever mapped to a MetaHuman control. The only thing crossing
    between them is geometry.

THREE ARMS, ONE METRIC
    neutral   c = 0                     how far the shapes move at all (the denominator)
    mono      the shipped mono solve    what the product actually delivers today
    fit       argmin_c ||dV(c) - dF||   the rig's CEILING

    fit >> mono  ->  the rig is fine, the solver is the bottleneck  (B)
    fit ~~ neutral -> the rig genuinely cannot make these shapes    (A)

FOUR DECISIONS THAT MAKE THE NUMBER MEAN SOMETHING
  * DISPLACEMENT, not position. The Beltrami wrap closes 88.3% of the identity gap,
    not 100%. Fitting absolute position would charge the leftover 11.7% of IDENTITY
    error to the EXPRESSION rig. dV = V(c) - V(0) on both sides cancels it exactly.
  * PER-FRAME SKULL PROCRUSTES. VHAP's `global_mode=none` zeroes `rotation` and
    `translation` but keeps `neck_pose`, which still leaves up to 10.2 deg of rigid
    head rotation in the sequence (measured below). Un-removed, the fit would be
    scored on neck motion the face rig is not asked to produce.
  * BARYCENTRIC CORRESPONDENCE fixed at the neutral. 24049 MetaHuman vertices vs 5023
    FLAME ones is ~5:1, so nearest-vertex quantises the target. Each MetaHuman vertex
    gets a (triangle, barycentric) address on the FLAME neutral and keeps it for the
    whole sequence, which is what makes the per-frame target a real trajectory rather
    than a re-snapping of the closest point.
  * CONTROLS BOXED TO [0, 1]. That is the rig's actual domain -- the shipped mono
    curves for this subject occupy exactly [0.0, 1.0]. An unboxed run is also reported,
    because if the box is what binds then the answer is "the range", not "the space".
"""
import argparse
import pathlib
import pickle
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent
sys.path.insert(0, str(PIPE / "offset"))
sys.path.insert(0, str(PIPE / "identity"))

VHAP_ASSET = (pathlib.Path(__file__).resolve().parents[1] / "vhap/asset/flame/FLAME_masks.pkl")
N_FLAME = 5023          # the base head; the sequence carries 5143 with synthesised teeth
HEAD_NV = 24049


# --------------------------------------------------------------------- geometry --
def similarity(P, Q):
    """The umbrella-free similarity used by build_identity.stage_rigs, verbatim.

    Returns a callable so it can be composed. Uniform scale + rotation + translation:
    a similarity commutes with barycentric interpolation, which is what lets the FLAME
    targets be transformed once and tracked as displacements.
    """
    ca, cb = P.mean(0), Q.mean(0)
    X, Y = P - ca, Q - cb
    U, S, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    s = S.sum() / (X ** 2).sum()
    return lambda Z: s * ((Z - ca) @ R) + cb


def procrustes_rigid(A, B):
    """Rotation+translation taking A onto B. No scale: a skull does not resize."""
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R, cb - ca @ R


def closest_on_triangles(P, V, F, cand):
    """Closest point on a candidate set of triangles, returned barycentrically.

    P [n,3] query points, cand [n,k] triangle indices to test. Vectorised Ericson
    (Real-Time Collision Detection 5.1.5): the seven Voronoi regions of a triangle,
    resolved by sign tests, no branching per point.
    """
    n, k = cand.shape
    tri = F[cand]                                   # [n,k,3]
    a, b, c = V[tri[..., 0]], V[tri[..., 1]], V[tri[..., 2]]
    ab, ac, ap = b - a, c - a, P[:, None, :] - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = P[:, None, :] - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = P[:, None, :] - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = np.where(np.abs(va + vb + vc) < 1e-20, 1e-20, va + vb + vc)
    # interior
    v = vb / denom
    w = vc / denom
    # the six degenerate regions, applied in Ericson's order
    reg_a = (d1 <= 0) & (d2 <= 0)
    reg_b = (d3 >= 0) & (d4 <= d3)
    reg_c = (d6 >= 0) & (d5 <= d6)
    reg_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    reg_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    reg_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t_ab = np.divide(d1, d1 - d3, out=np.zeros_like(d1), where=(d1 - d3) != 0)
    t_ac = np.divide(d2, d2 - d6, out=np.zeros_like(d2), where=(d2 - d6) != 0)
    t_bc = np.divide(d4 - d3, (d4 - d3) + (d5 - d6),
                     out=np.zeros_like(d4), where=((d4 - d3) + (d5 - d6)) != 0)
    for cond, vv, ww in ((reg_bc, 1.0 - t_bc, t_bc),
                         (reg_ac, np.zeros_like(t_ac), t_ac),
                         (reg_ab, t_ab, np.zeros_like(t_ab)),
                         (reg_c, np.zeros_like(v), np.ones_like(v)),
                         (reg_b, np.ones_like(v), np.zeros_like(v)),
                         (reg_a, np.zeros_like(v), np.zeros_like(v))):
        v = np.where(cond, vv, v)
        w = np.where(cond, ww, w)
    u = 1.0 - v - w
    Q = u[..., None] * a + v[..., None] * b + w[..., None] * c
    d = np.linalg.norm(Q - P[:, None, :], axis=-1)          # [n,k]
    j = d.argmin(1)
    r = np.arange(n)
    return cand[r, j], np.stack([u[r, j], v[r, j], w[r, j]], 1), d[r, j]


def vert_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="drk")
    ap.add_argument("--rig", default="rig_beltrami.npz",
                    help="which identity the expressions are driven on top of")
    ap.add_argument("--corr-rig", default="",
                    help="take the correspondence and the scored vertex set from THIS "
                         "rig instead of --rig. Comparing identities needs it: the gate "
                         "is a distance to the FLAME neutral, so letting each identity "
                         "pick its own scored points would compare different metrics and "
                         "reward whichever identity is nearest FLAME for free.")
    ap.add_argument("--flame-seq", default="/tmp/drk_flame_nopose.npz")
    ap.add_argument("--mono", default="/tmp/drk_mono_c.npy")
    ap.add_argument("--step", type=int, default=2, help="frame stride")
    ap.add_argument("--f0", type=int, default=0, help="first frame")
    ap.add_argument("--f1", type=int, default=0, help="last frame (0 = end)")
    ap.add_argument("--init", default="zero", choices=("zero", "mono", "rand", "half"),
                    help="where the per-frame fit starts. The ceiling is only a CEILING "
                         "if it is initialisation-independent: Adam from one point that "
                         "converges proves convergence, not global optimality.")
    ap.add_argument("--restarts", type=int, default=1,
                    help="with --init rand, keep the best of N independent starts")
    ap.add_argument("--no-control", action="store_true",
                    help="skip the selftest arm. ONLY for producing renders -- the "
                         "selftest is what makes the residual interpretable, so a run "
                         "with this flag set proves nothing on its own.")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--gate-cm", type=float, default=0.40,
                    help="max neutral correspondence distance for a vertex to be scored")
    ap.add_argument("--unboxed", action="store_true",
                    help="also run without the [0,1] control box")
    ap.add_argument("--tv", type=float, nargs="*", default=[],
                    help="temporal-smoothness weights to sweep. A per-frame fit is free "
                         "to jitter; the shipped solver is not. Penalising |c[t]-c[t-1]|^2 "
                         "turns an unreachable upper bound into an ACHIEVABLE one.")
    ap.add_argument("--smooth-target", type=float, default=0.0,
                    help="Gaussian sigma (frames) applied to the FLAME target first. "
                         "VHAP optimises each frame independently, so its output carries "
                         "per-frame noise. If the ceiling jumps when the target is "
                         "smoothed, the unreachable part was noise, not shape.")
    ap.add_argument("--out", default="/tmp/mh_flame_fit.npz")
    a = ap.parse_args()

    import torch
    from riglogic_torch import TorchRig
    from scipy.spatial import cKDTree

    sub = PIPE / "identity" / "subjects" / a.subject
    rig_path = sub / a.rig
    fx = np.load(sub / "flame_neutral.npz")
    wrap = np.load(sub / "beltrami_wrap.npz")
    H = np.load(PIPE / "head" / "head_assets.npz")
    Va = H["neutral_dna_cm"].astype(np.float64)
    Fm = H["pos_idx"].astype(np.int64)[H["faces"].astype(np.int64)]
    base_rig = np.load(PIPE / "offset" / "rig_tables.npz")

    # ---- 1. the FLAME -> DNA placement, rebuilt exactly as stage_rigs built it ----
    # A similarity only. It carries a rotation, a translation and one scalar; it cannot
    # move a mouth corner relative to a nose, so nothing about expression passes through
    # it. Composing the two stages here keeps the targets in the same frame as the rig.
    back = similarity(wrap["verts_archetype"], Va)
    Vw_dna = back(wrap["verts_wrapped"])
    S2 = similarity(Vw_dna, base_rig["m0_V0"].astype(np.float64))
    to_dna = lambda Z: S2(back(Z))

    Vn_f = fx["verts"].astype(np.float64)                   # FLAME neutral, metres
    Ft = fx["faces_target"].astype(np.int64)                # eyeballs and scalp removed
    Vn_d = to_dna(Vn_f)                                     # ... in DNA centimetres

    # ---- 2. sequence, canonicalised so only expression survives -------------------
    seq = np.load(a.flame_seq)
    Vs = seq["verts"].astype(np.float64)[:, :N_FLAME]       # drop synthesised teeth
    mk = pickle.load(open(VHAP_ASSET, "rb"), encoding="latin1")
    skull = np.unique(np.concatenate([mk["forehead"], mk["scalp"], mk["nose"]]))
    skull = skull[skull < N_FLAME]
    frames = np.arange(a.f0, (a.f1 or len(Vs)), a.step)
    ang = np.zeros(len(frames))
    Vc = np.zeros((len(frames), N_FLAME, 3))
    for n, i in enumerate(frames):
        R, t = procrustes_rigid(Vs[i, skull], Vn_f[skull])
        Vc[n] = Vs[i] @ R + t
        ang[n] = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"[1] {len(frames)} frames (stride {a.step}); rigid head rotation removed: "
          f"mean {ang.mean():.2f} deg  max {ang.max():.2f}")

    # ---- 3. correspondence, fixed at the neutral ---------------------------------
    Vm0 = np.load(sub / (a.corr_rig or a.rig))["m0_V0"].astype(np.float64)
    cent = Vn_d[Ft].mean(1)
    _, cand = cKDTree(cent).query(Vm0, k=12)
    tri, bary, dist = closest_on_triangles(Vm0, Vn_d, Ft, cand)
    Nm = vert_normals(Vm0, Fm)
    Nf = vert_normals(Vn_d, Ft)[Ft[tri]]
    nagree = (Nm * (bary[..., None] * Nf).sum(1)).sum(1)
    mask = (dist < a.gate_cm) & (nagree > 0.5)
    print(f"[2] scored on {int(mask.sum())}/{HEAD_NV} head vertices "
          f"({100*mask.mean():.1f}%), median correspondence {np.median(dist[mask])*10:.2f} mm")

    # per-vertex FLAME target displacement, in DNA centimetres
    T = Ft[tri][mask]                                       # [M,3]
    W = bary[mask]                                          # [M,3]
    P0 = (W[..., None] * to_dna(Vn_f)[T]).sum(1)
    dF = np.empty((len(frames), int(mask.sum()), 3), np.float32)
    for n in range(len(frames)):
        dF[n] = ((W[..., None] * to_dna(Vc[n])[T]).sum(1) - P0).astype(np.float32)
    if a.smooth_target > 0:
        from scipy.ndimage import gaussian_filter1d
        rough0 = float(np.abs(np.diff(dF, axis=0)).mean())
        dF = gaussian_filter1d(dF, a.smooth_target, axis=0, mode="nearest")
        print(f"    target smoothed, sigma {a.smooth_target} frames: per-frame roughness "
              f"{rough0:.5f} -> {float(np.abs(np.diff(dF,axis=0)).mean()):.5f} cm")
    mag = np.linalg.norm(dF, axis=2)
    print(f"[3] FLAME expression displacement: mean {mag.mean():.4f} cm  "
          f"p95 {np.percentile(mag,95):.4f}  max {mag.max():.4f}")

    # ---- 4. the three arms --------------------------------------------------------
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rig = TorchRig(rig_path, device=dev)
    mi = torch.as_tensor(np.nonzero(mask)[0], device=dev)
    tgt = torch.as_tensor(dF, device=dev)

    def head_verts(c):
        delta, bsw = rig.behaviour(c)
        return rig.deform(0, rig.skin_matrices(delta), bsw)

    with torch.no_grad():
        V0 = head_verts(torch.zeros(1, 263, device=dev))[0, mi]

    def residual(c_np, label, box=True, iters=0, tv=0.0):
        """Returns per-frame per-vertex residual [T,M] in cm. iters>0 -> optimise.

        tv > 0 adds mean((c[t]-c[t-1])^2) * tv to the loss. Chunk boundaries are not
        coupled, so with chunk=32 the reported roughness carries 1 uncoupled seam per
        32 frames -- it slightly OVERSTATES the roughness of the smoothed fits, which
        is the safe direction for the claim being made.
        """
        out = np.empty((len(frames), len(mi)), np.float32)
        got = np.empty((len(frames), 263), np.float32)
        for s in range(0, len(frames), a.chunk):
            e = min(s + a.chunk, len(frames))
            y = tgt[s:e]
            if iters == 0:
                c = torch.as_tensor(c_np[s:e], dtype=torch.float32, device=dev)
                with torch.no_grad():
                    r = head_verts(c)[:, mi] - V0[None] - y
            else:
                if a.init == "mono":
                    c0 = torch.as_tensor(mono_init[s:e], dtype=torch.float32, device=dev)
                elif a.init == "half":
                    c0 = torch.full((e - s, 263), 0.5, device=dev)
                elif a.init == "rand":
                    c0 = torch.rand(e - s, 263, device=dev)
                else:
                    c0 = torch.zeros(e - s, 263, device=dev)
                c = c0.clone().requires_grad_(True)
                opt = torch.optim.Adam([c], lr=a.lr)
                for it in range(iters):
                    opt.zero_grad(set_to_none=True)
                    d = head_verts(c)[:, mi] - V0[None] - y
                    loss = (d ** 2).sum(-1).mean()
                    if tv > 0 and c.shape[0] > 1:
                        loss = loss + tv * ((c[1:] - c[:-1]) ** 2).mean()
                    loss.backward()
                    opt.step()
                    if box:
                        with torch.no_grad():
                            c.clamp_(0.0, 1.0)
                with torch.no_grad():
                    r = head_verts(c)[:, mi] - V0[None] - y
            out[s:e] = r.norm(dim=-1).detach().cpu().numpy()
            got[s:e] = c.detach().cpu().numpy()
            print(f"    {label} {e}/{len(frames)}", end="\r", flush=True)
        print(" " * 40, end="\r")
        return out, got

    print("[4] arms")
    r_neu, _ = residual(np.zeros((len(frames), 263), np.float32), "neutral")
    mono = np.load(a.mono).astype(np.float32)
    mono = mono[np.minimum(frames, len(mono) - 1)]
    mono_init = mono
    r_mono, _ = residual(mono, "mono")
    r_fit, c_fit = residual(None, "fit", box=True, iters=a.iters)
    arms = {"neutral": r_neu, "mono": r_mono, "fit": r_fit}
    if a.unboxed:
        r_un, c_un = residual(None, "unboxed", box=False, iters=a.iters)
        arms["fit_unboxed"] = r_un
    cs = {"fit": c_fit}
    for w in a.tv:
        r_tv, c_tv = residual(None, f"tv{w}", box=True, iters=a.iters, tv=w)
        arms[f"fit_tv{w:g}"] = r_tv
        cs[f"fit_tv{w:g}"] = c_tv

    # ---- 4b. THE CONTROL ----------------------------------------------------------
    # Everything above assumes 400 Adam steps actually find the optimum of a 263-variable
    # non-convex problem. If they do not, "the rig cannot reach it" and "the optimiser did
    # not get there" are indistinguishable and the whole experiment says nothing.
    # So: replace the FLAME target with a shape the rig can provably hit -- its own output
    # under the mono controls -- and re-run the identical fit from c = 0. The residual is
    # then pure optimiser error, and it is the floor under every number above.
    if a.no_control:
        print("    selftest SKIPPED (--no-control)")
        arms["selftest"] = np.zeros_like(r_neu)
        r_self = arms["selftest"]
    else:
      with torch.no_grad():
        self_t = torch.empty_like(tgt)
        for s in range(0, len(frames), a.chunk):
            e = min(s + a.chunk, len(frames))
            cm = torch.as_tensor(mono[s:e], dtype=torch.float32, device=dev)
            self_t[s:e] = head_verts(cm)[:, mi] - V0[None]
      keep_tgt = tgt
      tgt = self_t
      r_self, _ = residual(None, "selftest", box=True, iters=a.iters)
      tgt = keep_tgt
      floor = r_self.mean()
      print(f"    optimiser floor (refit the rig's own mono output): {floor:.4f} cm "
            f"vs {r_mono.mean():.4f} cm of signal -- "
            f"{100*floor/r_neu.mean():.2f}% of the motion being measured")
      arms["selftest"] = r_self

    # ---- 5. read-out ---------------------------------------------------------------
    base = r_neu.mean()
    print(f"\n{'arm':<14}{'mean cm':>10}{'p95 cm':>10}{'motion explained':>20}")
    print("-" * 54)
    for k, r in arms.items():
        print(f"{k:<14}{r.mean():>10.4f}{np.percentile(r,95):>10.4f}"
              f"{100*(1-r.mean()/base):>19.1f}%")

    # per-region, using FLAME's own masks pushed through the correspondence
    vt = Ft[tri][mask][:, 0]                     # a representative FLAME vertex per point
    print(f"\n{'region':<16}{'n':>7}{'motion cm':>11}{'mono':>10}{'fit':>10}"
          f"{'  mono%':>9}{'  fit%':>8}")
    print("-" * 71)
    regions = {}
    for name in ("lips", "nose", "forehead", "left_eye_region", "right_eye_region",
                 "boundary", "neck", "face"):
        if name not in mk:
            continue
        sel = np.isin(vt, mk[name])
        if sel.sum() < 50:
            continue
        b = r_neu[:, sel].mean()
        m_, f_ = r_mono[:, sel].mean(), r_fit[:, sel].mean()
        regions[name] = (int(sel.sum()), float(b), float(m_), float(f_))
        print(f"{name:<16}{sel.sum():>7}{b:>11.4f}{m_:>10.4f}{f_:>10.4f}"
              f"{100*(1-m_/b):>8.1f}%{100*(1-f_/b):>7.1f}%")

    # FACIAL headline: drop the two regions the face rig is not asked to produce.
    # `boundary` is FLAME's cut edge at the neck stump -- a modelling artefact with no
    # MetaHuman counterpart. `neck` is driven by the neck joints, not the 251 face
    # controls. Leaving them in charges the expression rig for motion outside its job.
    drop = np.zeros(len(vt), bool)
    for name in ("boundary", "neck"):
        if name in mk:
            drop |= np.isin(vt, mk[name])
    fa = ~drop
    print(f"\nFACIAL ONLY  ({int(fa.sum())} points, neck and mesh boundary removed)")
    bf = r_neu[:, fa].mean()
    for k, r in arms.items():
        if k == "selftest":
            continue
        print(f"  {k:<14}{r[:, fa].mean():>9.4f} cm{100*(1-r[:, fa].mean()/bf):>9.1f}% explained")
    print(f"  {'(opt floor)':<14}{arms['selftest'][:, fa].mean():>9.4f} cm")

    # Is the ceiling REACHABLE by a real solver, or does it need frame-to-frame jumps?
    # A per-frame fit has no temporal prior, so it is free to jitter. If the fitted curves
    # are far rougher than the shipped ones, the ceiling is an upper bound no smooth
    # regressor could sit on, and the honest comparison is against a smoothed fit.
    jit = lambda C: float(np.abs(np.diff(C, axis=0)).mean())
    print(f"\ntemporal roughness of the controls (mean |c[t]-c[t-1]|)")
    for k, C in cs.items():
        print(f"  {k:<14}{jit(C):.5f}   ({jit(C)/max(jit(mono),1e-9):.1f}x mono)")
    print(f"  {'mono':<14}{jit(mono):.5f}   (1.0x)")

    np.savez_compressed(a.out, frames=frames, mask=mask, c_mono=mono,
                        facial=fa, vt=vt, **{f"c_{k}": v for k, v in cs.items()},
                        **{f"r_{k}": v for k, v in arms.items()})
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
