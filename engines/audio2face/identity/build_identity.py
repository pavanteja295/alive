#!/usr/bin/env python3
"""Build and compare three MetaHuman identities for any subject, from a VHAP fit.

    # everything that needs no Unreal:
    python build_identity.py --subject dr_k --vhap-export <VHAP>/export/monocular/<run>

    # then, once you have control curves for that subject:
    python build_identity.py --subject dr_k --compare --curves <AS_Mono_xxx.json> \
                             --video <source.mov> --start 60 --dur 30

THE THREE IDENTITIES
    archetype   Epic's generic head. The do-nothing baseline. Always available.
    beltrami    The archetype wrapped onto the VHAP/FLAME head, with the offset expressed
                in the low Laplace-Beltrami modes of the archetype's own surface. Offline,
                no Unreal, no cloud. Closes ~84% of the archetype->FLAME gap on the author's face.
    conform     MetaHuman's own Identity Solve on the same FLAME mesh. Needs Unreal and
                one manual editor step; this script prepares the input and tells you what
                to click. Closes ~33% on the author's face.

WHY THE RIG IS THE ARCHETYPE'S FOR ALL THREE
    A new subject has no autorigged DNA, and the autorig is a cloud service. But the rig
    is far more identity-agnostic than it looks: driving the archetype rig and a real
    depth-derived rig with identical curves gives deformations that differ by only 10.3%
    of the motion (verify/e2_mapping.py). So all three identities here share the archetype
    rig and differ ONLY in the bind pose, which is the variable under test. Motion is
    therefore approximate for all three EQUALLY, which keeps the comparison fair.

WHAT THIS CANNOT DO
    Produce control curves. Mono and depth performance solves live in Unreal; VHAP's own
    expression was measured to track a depth solve worse than the mono solve does
    (median r 0.749 vs 0.955 on gated channels, verify/e5). Bring curves from Unreal.
"""
import argparse, json, os, pathlib, shutil, subprocess, sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PIPE = HERE.parent
SUBJECTS = HERE / "subjects"
ARCHETYPE_RIG = PIPE / "offset" / "rig_tables.npz"
HEAD_ASSETS = PIPE / "head" / "head_assets.npz"
VHAP_ROOT = pathlib.Path(os.environ.get(
    "VHAP_ROOT", str(pathlib.Path(__file__).resolve().parents[1] / "vhap")))
PY_VHAP = os.environ.get("PY_VHAP", str(pathlib.Path.home() / "miniconda3/envs/vhap/bin/python"))


def die(msg, *extra):
    print("\n! " + msg)
    for e in extra:
        print("  " + e)
    sys.exit(1)


def need_vhap(export_dir):
    """A VHAP export is the one input this cannot synthesise. Fail with instructions."""
    p = pathlib.Path(export_dir) if export_dir else None
    if p and (p / "canonical_flame_param.npz").exists():
        n = len(list((p / "flame_param").glob("*.npz"))) if (p / "flame_param").exists() else 0
        print(f"  VHAP export OK: {p.name}  ({n} tracked frames)")
        return p
    die("no VHAP export found at: {}".format(export_dir),
        "VHAP fits a FLAME head to the video; without it there is no identity to transfer.",
        "Run it on the subject's video first, e.g.:",
        "  cd {}".format(VHAP_ROOT),
        "  python vhap/track.py --data.root_folder <frames> --exp.output_folder output/<name>",
        "  python vhap/export_as_nerf_dataset.py --src_folder output/<name> \\",
        "      --tgt_folder export/monocular/<name>_whiteBg_staticOffset --background-color white",
        "then re-run this with --vhap-export export/monocular/<name>_whiteBg_staticOffset")


# ----------------------------------------------------------------- stage 1 --
NEUTRAL_SNIPPET = r'''
import os, sys, pathlib, numpy as np, torch, pickle
ROOT = pathlib.Path(sys.argv[1]); EXP = pathlib.Path(sys.argv[2]); OUT = pathlib.Path(sys.argv[3])
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from vhap.model.flame import FlameHead
c = np.load(EXP / "canonical_flame_param.npz")
t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
# remove_lip_inside drops the interior lip FACES and touches no vertices, so shape and
# static_offset still broadcast. It matters because VHAP's canonical jaw is ~17 deg open
# and zeroing it for a Neutral presses the inner and outer lip sheets together; a fitter
# then cannot tell them apart.
fh = FlameHead(300, 100, add_teeth=True, remove_lip_inside=True)
nv = fh.v_template.shape[0]
with torch.no_grad():
    V = fh(shape=t(c["shape"])[None], expr=torch.zeros(1, 100),
           rotation=torch.zeros(1,3), neck=torch.zeros(1,3), jaw=torch.zeros(1,3),
           eyes=torch.zeros(1,6), translation=torch.zeros(1,3),
           static_offset=t(c["static_offset"][:, :nv]))[0][0].numpy().astype(np.float64)
F = fh.faces.cpu().numpy().astype(np.int64)
V, F = V[:5023], F[(F < 5023).all(1)]                       # teeth are synthesised: drop
mk = pickle.load(open(ROOT/"asset/flame/FLAME_masks.pkl","rb"), encoding="latin1")
# eyeballs have no MetaHuman counterpart; scalp is HAIR, not skull, on a haired subject
drop = np.unique(np.concatenate([mk["left_eyeball"], mk["right_eyeball"], mk["scalp"]]))
drop = drop[drop < 5023]
keep = np.ones(5023, bool); keep[drop] = False
Ftgt = F[keep[F].all(1)]
np.savez_compressed(OUT/"flame_neutral.npz", verts=V, faces_full=F, faces_target=Ftgt, dropped=drop)
# OBJ in UNREAL axes for the conform path: X=+Z_f, Y=-X_f, Z=+Y_f, det -1 so winding flips
M = np.array([[0.,-1.,0.],[0.,0.,1.],[1.,0.,0.]])
Vu = (V @ M) * 100.0; Vu -= Vu.mean(0); Fu = F[:, ::-1]
n = np.cross(Vu[Fu[:,1]]-Vu[Fu[:,0]], Vu[Fu[:,2]]-Vu[Fu[:,0]])
frac = float((np.einsum("ij,ij->i", n, Vu[Fu].mean(1)-Vu.mean(0)) > 0).mean())
assert frac > 0.6, "winding wrong after handedness flip ({:.3f})".format(frac)
with open(OUT/"flame_neutral.obj","w") as f:
    f.write("# FLAME neutral: jaw zeroed, no teeth, no inner lip, Unreal axes, cm\n")
    for v in Vu: f.write("v {:.6f} {:.6f} {:.6f}\n".format(*v))
    for tri in Fu + 1: f.write("f {} {} {}\n".format(*tri))
print("  neutral: {} verts, {} faces ({} target faces after masking)".format(len(V), len(F), len(Ftgt)))
print("  outward-normal fraction {:.3f}".format(frac))
'''


def stage_neutral(sub_dir, export):
    print("[1/4] FLAME neutral  (runs in the vhap env)")
    snip = sub_dir / "_neutral.py"
    snip.write_text(NEUTRAL_SNIPPET)
    r = subprocess.run([PY_VHAP, str(snip), str(VHAP_ROOT), str(export), str(sub_dir)],
                       capture_output=True, text=True)
    snip.unlink(missing_ok=True)
    if r.returncode:
        die("neutral export failed", *(r.stderr.strip().splitlines()[-6:]))
    print(r.stdout.rstrip())


# ----------------------------------------------------------------- stage 2 --
def stage_wrap(sub_dir, kmax=1200, dihedral_budget=1.10):
    """Non-rigid ICP archetype -> FLAME, offset expressed in Laplace-Beltrami modes."""
    import scipy.sparse as sp, scipy.sparse.linalg as spl
    from scipy.spatial import cKDTree
    print("[2/4] Beltrami wrap")
    from wrap_lib import (loop, umbrella, cot_laplacian, tri_normals, vert_normals,
                          dihedral, similarity_icp)
    H = np.load(HEAD_ASSETS)
    Vm = H["neutral_dna_cm"].astype(np.float64)
    Fm = H["pos_idx"].astype(np.int64)[H["faces"].astype(np.int64)]
    fx = np.load(sub_dir / "flame_neutral.npz")
    # Loop subdivision, twice. Midpoint subdivision adds vertices ON the existing flat
    # triangles and no smoothness at all; Loop converges to a smooth limit surface, which
    # is what stops the wrap copying FLAME's tessellation as if it were shape.
    Vt, Ft = loop(*loop(fx["verts"].astype(np.float64), fx["faces_target"].astype(np.int64)))
    tree = cKDTree(Vt); Nt = vert_normals(Vt, Ft)
    V0 = similarity_icp(Vm, Vt, tree)
    L = umbrella(V0, Fm); LtL = (L.T @ L).tocsc()
    N0 = vert_normals(V0, Fm); d0, i0 = tree.query(V0)
    M = (d0 < 1.5) & ((N0 * Nt[i0]).sum(1) > 0.5)
    gp = lambda V: np.abs(((V - Vt[tree.query(V)[1]]) * Nt[tree.query(V)[1]]).sum(1))[M].mean()
    base, arch = gp(V0), dihedral(V0, Fm)
    print(f"  scored on {int(M.sum())}/{len(V0)} vertices ({100*M.mean():.1f}%)")

    V = V0.copy()
    for lam in (30, 30, 10, 10, 3, 3, 1, 1, .3, .3):
        Nv = vert_normals(V, Fm); dd, ii = tree.query(V); c, n = Vt[ii], Nt[ii]
        # normal gate at 0.8 (37 deg), not 0.5: a looser gate lets outer-lip vertices grab
        # the surface behind them wherever two sheets are close.
        w = (((Nv * n).sum(1) > 0.8) & (dd < 1.0)).astype(float)
        Wd = sp.diags(w)
        fac = spl.factorized((Wd + lam * LtL + 1e-4 * sp.identity(len(V0), format="csc")).tocsc())
        B = np.asarray(Wd @ ((V - ((V - c) * n).sum(1)[:, None] * n) - V0))
        V = V0 + np.column_stack([fac(B[:, k]) for k in range(3)])
    d_free = V - V0
    print(f"  free wrap {100*(1-gp(V)/base):.1f}% gap, {dihedral(V,Fm)/arch:.2f}x archetype dihedral")

    Lc, Mm = cot_laplacian(V0, Fm)
    vals, U = spl.eigsh(Lc, k=kmax, M=Mm, sigma=-1e-8, which="LM")
    U = U / np.sqrt(np.einsum("ij,ij->j", U, Mm @ U))
    best = None
    for K in (300, 600, 900, kmax):
        Uk = U[:, :K]
        Vk = V0 + Uk @ (Uk.T @ (Mm @ d_free))
        g, dh = 100 * (1 - gp(Vk) / base), dihedral(Vk, Fm) / arch
        print(f"    K={K:5d}  {g:5.1f}% gap  {dh:.2f}x dihedral")
        if dh <= dihedral_budget and (best is None or g > best[1]):
            best = (Vk, g, K, dh)
    if best is None:
        die("no K met the dihedral budget -- the target is probably too rough")
    Vb, g, K, dh = best
    np.savez_compressed(sub_dir / "beltrami_wrap.npz", verts_wrapped=Vb, verts_archetype=V0,
                        faces=Fm, flame_verts=Vt, flame_faces=Ft, K=K, gap=g, dihedral=dh)
    print(f"  chose K={K}: {g:.1f}% gap closed at {dh:.2f}x archetype dihedral")
    return g, K


# ----------------------------------------------------------------- stage 3 --
def stage_rigs(sub_dir, base_rig):
    """One rig, three bind poses. Only the neutral differs, which is the variable."""
    print("[3/4] rig tables")
    H = np.load(HEAD_ASSETS)
    Va = H["neutral_dna_cm"].astype(np.float64)
    z0 = dict(np.load(base_rig))
    dt = z0["m0_V0"].dtype
    w = np.load(sub_dir / "beltrami_wrap.npz")

    def similarity(P, Q):
        ca, cb = P.mean(0), Q.mean(0); X, Y = P - ca, Q - cb
        U, S, Vt = np.linalg.svd(X.T @ Y); R = U @ Vt
        if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
        s = S.sum() / (X ** 2).sum(); return lambda Z: s * ((Z - ca) @ R) + cb

    out = {}
    z = dict(z0); z["m0_V0"] = Va.astype(dt)
    np.savez_compressed(sub_dir / "rig_archetype.npz", **z); out["archetype"] = "rig_archetype.npz"

    back = similarity(w["verts_archetype"], Va)          # FLAME frame -> DNA space
    Vb = similarity(back(w["verts_wrapped"]), z0["m0_V0"].astype(np.float64))(back(w["verts_wrapped"]))
    z = dict(z0); z["m0_V0"] = Vb.astype(dt)
    np.savez_compressed(sub_dir / "rig_beltrami.npz", **z); out["beltrami"] = "rig_beltrami.npz"

    cf = sub_dir / "conform_neutral.npy"
    if cf.exists():
        Vc = np.load(cf)
        z = dict(z0); z["m0_V0"] = Vc.astype(dt)
        np.savez_compressed(sub_dir / "rig_conform.npz", **z); out["conform"] = "rig_conform.npz"
        print("  archetype, beltrami, conform")
    else:
        print("  archetype, beltrami   (no conform_neutral.npy -- see UNREAL STEPS below)")
    return out


def unreal_steps(sub_dir):
    print("""
UNREAL STEPS for the third identity (MetaHuman's own Identity Solve)
  These cannot be scripted end to end: the solve needs a promoted frame with tracked
  markers, which is editor UI.

  1. import          {obj}
  2. point a MeshCaptureData at it, and an Identity's Neutral pose at that
                     (Scripts/flame_identity_solve.py --stage prepare --obj <path>)
  3. IN THE EDITOR   open the Identity, Promote Frame front-on, Track Markers,
                     tick Use To Solve and Is Front View, save
  4. solve + read    Scripts/resolve_flame_fixed.py, then
                     Scripts/export_conformed_mesh.py --out /tmp/conform_now.obj
  5. bring it back   python build_identity.py --subject {sub} --import-conform /tmp/conform_now.obj

  Do NOT use export_dna_data_to_files to read the conform: it returns the buffer written
  by the last AUTORIG, not the last conform, so it silently hands back a stale mesh.
  Read the template mesh instead -- and note it comes out MIRRORED, which
  --import-conform undoes.
""".format(obj=sub_dir / "flame_neutral.obj", sub=sub_dir.name))


def import_conform(sub_dir, obj_path):
    """Bring export_conformed_mesh.py's OBJ back into DNA vertex order.

    That export applies the component transform with bReverseOrientationIfNeeded, so the
    mesh arrives MIRRORED. Fitting with a reflection ALLOWED recovers it index-for-index
    (measured 0.09 cm median on the author's face); fitting without one gives 6.2 cm and looks like a
    correspondence failure, which is what it cost to learn.
    """
    H = np.load(HEAD_ASSETS); Va = H["neutral_dna_cm"].astype(np.float64)
    Vn = np.array([[float(x) for x in l.split()[1:4]]
                   for l in open(obj_path) if l.startswith("v ")])
    if len(Vn) != len(Va):
        die("conform OBJ has {} verts, expected {}".format(len(Vn), len(Va)))
    ca, cb = Vn.mean(0), Va.mean(0); X, Y = Vn - ca, Va - cb
    U, S, Vt = np.linalg.svd(X.T @ Y); R = U @ Vt                 # reflection ALLOWED
    F = (S.sum() / (X ** 2).sum()) * ((Vn - ca) @ R) + cb
    d = np.linalg.norm(F - Va, axis=1)
    print(f"  imported conform: index-wise vs archetype median {np.median(d):.4f} cm  "
          f"p95 {np.percentile(d,95):.4f} cm   det {np.linalg.det(R):+.3f}")
    np.save(sub_dir / "conform_neutral.npy", F)
    print("  wrote conform_neutral.npy -- re-run with --rigs to build rig_conform.npz")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--vhap-export")
    ap.add_argument("--base-rig", default=str(ARCHETYPE_RIG))
    ap.add_argument("--import-conform", metavar="OBJ")
    ap.add_argument("--rigs", action="store_true", help="rebuild rig tables only")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--curves"); ap.add_argument("--video"); ap.add_argument("--audio")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=30.0)
    ap.add_argument("--head-pose", choices=("none", "rotation", "full"), default="rotation",
                    help="TorchRig emits canonical vertices; this applies the six head "
                         "curves through head_pose.py. 'rotation' when a FLAME row is in "
                         "the comparison, 'full' when every row is MetaHuman.")
    ap.add_argument("--curves-offset", type=float, default=0.0,
                    help="take time the curve file's own t=0 corresponds to")
    ap.add_argument("--wrap-kmax", type=int, default=None,
                    help="override the profile's IDENTITY_WRAP['kmax']")
    ap.add_argument("--wrap-dihedral-budget", type=float, default=None,
                    help="override the profile's IDENTITY_WRAP['dihedral_budget']")
    a = ap.parse_args()

    sub = SUBJECTS / a.subject
    sub.mkdir(parents=True, exist_ok=True)
    print(f"subject: {a.subject}   ->  {sub}")

    if a.import_conform:
        import_conform(sub, a.import_conform); return
    if a.compare:
        from compare_identities import run_compare
        run_compare(sub, a.curves, a.video, a.audio, a.start, a.dur, a.curves_offset,
                    a.head_pose); return
    if a.rigs:
        stage_rigs(sub, a.base_rig); return

    # The wrap's two method constants live in the subject's audio-to-mesh profile,
    # the recipe that consumes this rig. A subject with no profile must pass both.
    prof = PIPE / "rigfit/recipes/audio-to-mesh/profiles" / f"{a.subject}.py"
    wrap = {}
    if prof.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"_prof_{a.subject}", prof)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        wrap = dict(getattr(m, "IDENTITY_WRAP", {}))
    if a.wrap_kmax is not None:
        wrap["kmax"] = a.wrap_kmax
    if a.wrap_dihedral_budget is not None:
        wrap["dihedral_budget"] = a.wrap_dihedral_budget
    if set(wrap) != {"kmax", "dihedral_budget"}:
        die(f"no wrap constants for '{a.subject}'",
            f"declare IDENTITY_WRAP = dict(kmax=..., dihedral_budget=...) in {prof}",
            "(copy it from profiles/_template.py), or pass --wrap-kmax and --wrap-dihedral-budget")
    print(f"  wrap constants: kmax={wrap['kmax']} dihedral_budget={wrap['dihedral_budget']}")

    export = need_vhap(a.vhap_export)
    stage_neutral(sub, export)
    gap, K = stage_wrap(sub, **wrap)
    rigs = stage_rigs(sub, a.base_rig)
    (sub / "report.json").write_text(json.dumps(
        {"subject": a.subject, "vhap_export": str(export), "base_rig": a.base_rig,
         "beltrami_gap_closed_pct": round(gap, 1), "beltrami_K": K, "wrap": wrap,
         "rigs": rigs}, indent=2))
    print(f"\n[4/4] wrote {sub/'report.json'}")
    unreal_steps(sub)
    print("Then compare:")
    print(f"  python build_identity.py --subject {a.subject} --compare \\")
    print(f"      --curves <AS_Mono_xxx.json> --video <source.mov> --start 60 --dur 30")


if __name__ == "__main__":
    main()
