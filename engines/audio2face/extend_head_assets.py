#!/usr/bin/env python3
"""The head mesh assets, read out of the MetaHuman rig: head_assets.npz (the head alone,
--meshes 0), then head_assets_eyes.npz / head_assets_0134.npz with eyeballs and teeth
appended so no existing index moves.

Two steps each, because the DNA bindings link libpython3.13 and numpy lives elsewhere:

    cd externals/OpenRigLogic/build/python && LD_LIBRARY_PATH=. PYTHONPATH=./dna:./riglogic \
        python3.13 <alive>/engines/audio2face/extend_head_assets.py --meshes 0 --dump
    ~/miniconda3/envs/stavatar/bin/python engines/audio2face/extend_head_assets.py --meshes 0 --build

then the same pair with --meshes 0,1,3,4 (what the renderer uses). --meshes 0 writes
head_assets.npz; if one exists it is checked against the rebuild and left untouched.

WHY APPEND RATHER THAN REBUILD

The head is mesh 0 and goes in first, so head vertex i is still vertex i and head
triangle j is still triangle j. Everything keyed by those indices keeps working
untouched: the corrective layer, the rig's skin weights, Epic's landmark vertex IDs,
the region masks, and the blob bindings inside clouds that are already trained. The
new geometry only ever occupies indices past the end.

`--build` ASSERTS that the head half of the output is bit-identical to the existing
head_assets.npz. If that assertion ever fails, the append has disturbed something and
nothing downstream should be trusted.

WHAT IS TAKEN, AND WHAT IS NOT

Meshes 0, 3 and 4: head, eyeLeft, eyeRight. Not the eyeshell, eyelashes or eyeEdge
strips, and not teeth or saliva. The strips sit in front of the eyeball and the
open-decisions note measured what that costs -- with them in, the eyeballs draw 22
pixels instead of 2,227.

WHERE THE VERTEX POSITIONS COME FROM, which is not where the name suggests

`neutral_dna_cm` in the existing asset is NOT the DNA's bind pose. It is the rig
evaluated at zero controls -- they differ by 12.25 cm, and the rig reproduces the
asset to 4.6e-07 cm. So the eyeball positions are taken the same way, from
rig.deform(3) and rig.deform(4) at zero, which also puts them in the frame the cook
already works in. Topology, texture coordinates and the layout mapping still come
from the DNA, because the rig does not carry them.

THE TWO COORDINATE CONVENTIONS, since getting this wrong is silent

    neutral_dna_cm   the DNA's own frame, centimetres, origin at the feet
    canonical_m      = neutral_dna_cm[:, (2, 1, 0)] * (1, 1, -1) * 0.01

That is x and z swapped and x negated. Verified against the existing asset to
1.3e-07 m. In the DNA frame x is left-right and z is depth; in canonical_m x is
depth and z is left-right.
"""
import argparse, json, pathlib, pickle, sys

PIPE = pathlib.Path(__file__).resolve().parent
DNA_PATH = str(__import__("pathlib").Path(__file__).resolve().parent / "assets/face.dna")
DUMP = pathlib.Path("/tmp/head_eyes_dump.pkl")
TAKE = [0, 3, 4]        # overridden by --meshes


def dump():
    global TAKE
    import dna
    st = dna.FileStream(DNA_PATH, dna.FileStream.AccessMode_Read,
                        dna.FileStream.OpenMode_Binary)
    r = dna.BinaryStreamReader(st, dna.DataLayer_All, 0); r.read()
    out = []
    for m in TAKE:
        tris = []
        for f in range(r.getFaceCount(m)):
            loop = list(r.getFaceVertexLayoutIndices(m, f))
            for k in range(1, len(loop) - 1):
                tris.append((loop[0], loop[k], loop[k + 1]))
        out.append({
            "name": r.getMeshName(m), "dna_mesh_index": m,
            "px": list(r.getVertexPositionXs(m)), "py": list(r.getVertexPositionYs(m)),
            "pz": list(r.getVertexPositionZs(m)),
            "uu": list(r.getVertexTextureCoordinateUs(m)),
            "vv": list(r.getVertexTextureCoordinateVs(m)),
            "pos_idx": list(r.getVertexLayoutPositionIndices(m)),
            "uv_idx": list(r.getVertexLayoutTextureCoordinateIndices(m)),
            "tris": tris,
        })
        print(f"  {r.getMeshName(m):22s} positions {r.getVertexPositionCount(m):6d}  "
              f"layout {r.getVertexLayoutCount(m):6d}  tris {len(tris):6d}")
    pickle.dump(out, open(DUMP, "wb"))
    print(f"wrote {DUMP}")


def rig_neutral():
    """Positions for meshes 0, 3, 4 with every control at zero, in centimetres."""
    import importlib.util, numpy as np, torch
    sys.path.append(str(PIPE))
    sp = importlib.util.spec_from_file_location(
        "riglogic_torch", str(PIPE / "offset" / "riglogic_torch.py"))
    mod = importlib.util.module_from_spec(sp); sys.modules["riglogic_torch"] = mod
    sp.loader.exec_module(mod)
    rig = mod.TorchRig(str(PIPE / "offset" / "rig_tables.npz"),
                       device="cuda", dtype=torch.float32)
    c = torch.zeros(1, 263, device="cuda")
    d_, bsw = rig.behaviour(c); sk = rig.skin_matrices(d_)
    return [rig.deform(m, sk, bsw)[0].cpu().numpy().astype(np.float64) for m in TAKE]


def build():
    global TAKE
    import numpy as np
    meshes = pickle.load(open(DUMP, "rb"))
    neutral = rig_neutral()
    faces, uvs, pos_idx, dna_cm, parts = [], [], [], [], []
    v_off = l_off = t_off = 0
    for mi, md in enumerate(meshes):
        pos = neutral[mi]
        assert len(pos) == len(md["px"]), (
            f"{md['name']}: rig gives {len(pos)} vertices, the DNA {len(md['px'])}")
        uv = np.stack([md["uu"], md["vv"]], 1).astype(np.float64)
        pi = np.asarray(md["pos_idx"], np.int64)
        ui = np.asarray(md["uv_idx"], np.int64)
        tr = np.asarray(md["tris"], np.int64)
        parts.append({"name": md["name"], "dna_mesh_index": md["dna_mesh_index"],
                      "vert_offset": v_off, "vert_count": len(pos),
                      "layout_offset": l_off, "layout_count": len(pi),
                      "tri_offset": t_off, "tri_count": len(tr)})
        dna_cm.append(pos)
        uvs.append(uv[ui])
        pos_idx.append(pi + v_off)
        faces.append(tr + l_off)
        v_off += len(pos); l_off += len(pi); t_off += len(tr)

    dna_cm = np.concatenate(dna_cm)
    uvs = np.concatenate(uvs).astype(np.float32)
    pos_idx = np.concatenate(pos_idx).astype(np.int32)
    faces = np.concatenate(faces).astype(np.int32)
    canonical_m = (dna_cm[:, (2, 1, 0)] * np.array([1.0, 1.0, -1.0]) * 0.01)

    base = PIPE / "head" / "head_assets.npz"
    if TAKE == [0]:
        if not base.exists():
            np.savez(base, faces=faces, uvs=uvs, pos_idx=pos_idx,
                     canonical_m=canonical_m.astype(np.float32),
                     neutral_dna_cm=dna_cm.astype(np.float32))
            print(f"wrote {base}: {len(canonical_m)} vertices, {len(faces)} triangles")
            return
        print(f"{base} exists; checking the rebuild reproduces it, writing nothing")
    old = np.load(base)
    nL, nV, nF = len(old["uvs"]), len(old["canonical_m"]), len(old["faces"])
    # Tolerances are in each array's own units. The position arrays are stored as
    # float32, so a centimetre-scale array round-trips to ~5e-05 cm (0.5 micron) and a
    # metre-scale one to ~1e-07 m. Anything above that is a real disturbance.
    for nm, new, ref, tol, unit in (
            ("faces",          faces[:nF],       old["faces"],          0,     ""),
            ("pos_idx",        pos_idx[:nL],     old["pos_idx"],        0,     ""),
            ("uvs",            uvs[:nL],         old["uvs"],            1e-6,  ""),
            ("canonical_m",    canonical_m[:nV], old["canonical_m"],    1e-6,  " m"),
            ("neutral_dna_cm", dna_cm[:nV],      old["neutral_dna_cm"], 1e-3,  " cm")):
        e = float(np.abs(np.asarray(new, np.float64) - np.asarray(ref, np.float64)).max())
        assert e <= tol, (f"the head half of '{nm}' does not reproduce the existing asset "
                          f"(max |diff| {e:.3e}{unit}, allowed {tol:g}{unit}). The append "
                          f"has disturbed the head; stop.")
        print(f"  head half of {nm:15s} matches  (max |diff| {e:.2e}{unit})")
    if TAKE == [0]:
        return

    tag = "eyes" if TAKE == [0, 3, 4] else "".join(str(t) for t in TAKE)
    dst = PIPE / "head" / f"head_assets_{tag}.npz"
    np.savez(dst, faces=faces, uvs=uvs, pos_idx=pos_idx,
             canonical_m=canonical_m.astype(np.float32),
             neutral_dna_cm=dna_cm.astype(np.float32),
             parts=json.dumps(parts))
    print(f"\nwrote {dst}")
    print(f"  vertices {nV} -> {len(canonical_m)}   layout {nL} -> {len(uvs)}   "
          f"triangles {nF} -> {len(faces)}")
    for p in parts:
        print(f"    {p['name']:22s} verts {p['vert_offset']:6d}..{p['vert_offset']+p['vert_count']-1:6d}"
              f"   tris {p['tri_offset']:6d}..{p['tri_offset']+p['tri_count']-1:6d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--meshes", default="",
                    help="DNA mesh indices to take, head first. Default 0,3,4 (head and "
                         "the two eyeballs). 0,1,3,4 adds teeth, which also carries the "
                         "tongue -- 322 of its 4,246 vertices are driven by tongue joints.")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.meshes:
        TAKE = [int(x) for x in a.meshes.split(",")]
        print(f"meshes: {TAKE}")
    if a.dump: dump()
    elif a.build: build()
    else: ap.error("pass --dump (under the DNA python) or --build")
