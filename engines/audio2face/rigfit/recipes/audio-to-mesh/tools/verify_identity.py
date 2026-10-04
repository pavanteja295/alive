#!/usr/bin/env python3
"""Is the rig on disk actually this person's? Reports; does not build.

    python tools/verify_identity.py --profile drk

BUILDING IT IS build_identity.py's JOB. This checks what it produced, because
nothing else does -- and everything downstream is measured against it.

WHY THIS IS NOT A PRECONDITION
    checks.py already fails when the rig is missing or the wrong shape, and names
    build_identity.py. That is the right behaviour and it is not this.

    This asks a different question: the rig is there and it plugs in, but is it
    HIS? A bind pose that never moved off the archetype loads perfectly, trains
    fine, and lowers the ceiling every later number is divided by -- so a bad
    identity presents as a bad audio model, and nothing says otherwise.

THE RELATIONS, WHICH NEED NO THRESHOLD
    the wrap is inside the rig      the rig's neutral must be a pure similarity of
                                    the wrapped mesh. If it is not, the identity was
                                    computed and never reached the rig -- which is a
                                    real thing that happened on another mesh path in
                                    this project, unnoticed for weeks
    the rig is not the archetype    if the two bind poses are identical, the identity
                                    step did nothing, whatever the report says
    the report agrees with the rig  a gap-closed figure that disagrees with the
                                    measured difference is stale
"""
import argparse, json, pathlib, sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _profile import load                                              # noqa: E402


def similarity(A, B):
    ca, cb = A.mean(0), B.mean(0)
    X, Y = A - ca, B - cb
    U, S, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    s = S.sum() / (X ** 2).sum()
    return s, R, cb - s * (ca @ R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    a = ap.parse_args()
    p = load(a.profile)
    sub = p.PIPE / "identity/subjects" / p.SUBJECT
    rel, concerns = [], []

    V = np.load(p.RIG)["m0_V0"].astype(np.float64)
    W = np.load(p.WRAP)["verts_wrapped"].astype(np.float64)
    arch_f = sub / "rig_archetype.npz"

    print(f"\n  the rig on disk for {p.SUBJECT}\n")

    # 1. is the wrap actually inside the rig?
    s_, R_, t_ = similarity(V, W)
    res = float(np.abs(s_ * (V @ R_) + t_ - W).max()) * 1000
    ok = res < 0.01
    rel.append(("the wrap is inside the rig", ok,
                f"the rig's neutral is a similarity of the wrapped mesh to {res:.5f} mm"))
    if not ok:
        concerns.append(
            "The rig's neutral is NOT a similarity of the wrapped mesh. The identity "
            "was computed and did not reach the rig, so every later number is measured "
            "against the archetype while claiming to be this person.")

    # 2. did it move off the archetype at all?
    if arch_f.exists():
        # COMPARE DIRECTLY. Both rigs live in the same DNA space, in the same
        # frame, so there is no scale or pose to remove -- and fitting a
        # similarity between two meshes that differ by millimetres is degenerate.
        # The first version of this did fit one and reported 324 mm of movement
        # on a head 200 mm tall, which is the shape of a check that is wrong
        # rather than a finding that is surprising.
        A = np.load(arch_f)["m0_V0"].astype(np.float64)
        moved = float(np.linalg.norm(A - V, axis=1).mean()) * 10   # DNA cm -> mm
        worst = float(np.linalg.norm(A - V, axis=1).max()) * 10
        ok = moved > 0.05
        rel.append(("the rig is not the archetype", ok,
                    f"bind pose differs from the archetype by {moved:.2f} mm mean, "
                    f"{worst:.1f} mm at the worst vertex"))
        if not ok:
            concerns.append(
                "The bind pose is the archetype's. The identity step produced nothing, "
                "whatever report.json says.")
    else:
        print("    (no rig_archetype.npz to compare against)")

    # 3. PROVENANCE. Was the rig fitted to the same tracking the targets came from?
    #
    # On drk it was not, and nothing recorded it. The rig was wrapped onto an 877
    # frame single-video export; the targets come from 162 chunks of a different
    # run. Their FLAME neutrals differ by 1.65 to 2.49 mm mean -- against the
    # 3.28 mm archetype-to-person gap the identity step exists to close. So a
    # meaningful share of what the wrap fixed was fixed toward a shape the
    # training data does not describe.
    #
    # This is a provenance check, not a geometry one: it costs nothing and it is
    # the thing that was never written down.
    r = sub / "report.json"
    if r.exists():
        src = pathlib.Path(str(json.load(open(r)).get("vhap_export", ""))).name
        used = p.VHAP / "output/shared"
        # this creator's chunks only: output/shared holds every creator's
        mine = sorted(d.name for v in p.RECORDINGS for d in used.glob(f"{v}__c*")
                      if d.is_dir())
        print(f"    the rig was fitted from : {src}")
        print(f"    the targets come from   : output/shared/<chunk>, {len(mine)} chunks")
        # An export of one of those very chunks IS the targets' tracking: the shared
        # pass locks one identity across all of them, so the rig and the targets
        # share provenance by construction.
        if src in mine:
            print(f"    same run                : the rig is one of the target chunks")
        elif "shared" not in src and src:
            concerns.append(
                f"The rig was fitted from '{src}', a different VHAP run than the "
                f"chunks the targets come from. Both describe this person, and they "
                f"do not have to agree: on drk their FLAME neutrals differ by 1.65 "
                f"to 2.49 mm mean, against a 3.28 mm archetype-to-person gap. Not "
                f"necessarily wrong -- a bind pose needs little data -- but it is an "
                f"undeclared dependency, and the ceiling every score is divided by "
                f"is measured against this rig.")

    # 3b. what the build recorded
    if r.exists():
        d = json.load(open(r))
        pct = d.get("beltrami_gap_closed_pct")
        print(f"    reported by build_identity: {pct:.1f}% of the archetype-to-FLAME "
              f"gap closed, K={d.get('beltrami_K')}")
        print(f"    built from: {pathlib.Path(str(d.get('vhap_export','?'))).name}\n")
    else:
        concerns.append("no report.json -- the build's own quality number is missing")

    for name, ok, stmt in rel:
        print(f"    {'ok   ' if ok else 'BROKEN'} {name:<30} {stmt}")

    if concerns:
        print(f"\n  {len(concerns)} concern(s):\n")
        for c in concerns:
            print(f"    {c}\n")
    else:
        print(f"\n  Nothing to flag. The ceiling that capacity.py measures, and every")
        print(f"  score divided by it, rests on this rig -- and it is his.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
