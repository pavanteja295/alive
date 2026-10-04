#!/usr/bin/env python3
"""Preconditions. This raises; it never builds.

    python tools/checks.py --profile drk

EVERY INPUT THIS RECIPE READS IS ANOTHER RECIPE'S OUTPUT.
    When one is missing the recipe FAILS and names the recipe that owns it. It
    does not run the upstream tool, even when that tool is one line away. An
    input this recipe creates is an input nothing can check, and a subtly wrong
    one is then rebuilt subtly wrong on every run.

    Three of the four worst incidents on this pipeline were an input that was
    silently partial -- a cook that produced 124 of 162 chunks and reported
    success, a dataset whose meshes were never loaded, a corpus missing its
    placements. None of them raised where the mistake was.
"""
import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _profile import load                                              # noqa: E402
from _corpus import shape, report                                      # noqa: E402


def fail(what, where, owner):
    raise SystemExit(
        f"\nPRECONDITION FAILED -- this recipe fails rather than builds.\n\n"
        f"  missing : {what}\n"
        f"  expected: {where}\n"
        f"  owned by: {owner}\n\n"
        f"  Produce it there, then re-run.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    a = ap.parse_args()
    p = load(a.profile)
    RIG = p.PIPE / "rigfit"
    ok = []

    # --- the audio model, and the split it was trained under --------------
    rel = RIG / "release" / p.AUDIO_MODEL
    for f, what in ((rel / "best.pt", f"the audio model {p.AUDIO_MODEL}"),
                    (rel / "eval.json", f"{p.AUDIO_MODEL}'s held-out score")):
        if not f.exists():
            fail(what, f, "recipes/audio-to-mesh")
    m = json.load(open(rel / "eval.json")).get("mean", {})
    ok.append(f"{p.AUDIO_MODEL}: {m.get('skin_pct', float('nan')):.2f}% transferred")

    f = p.SPLIT
    if not f.exists():
        fail("the canonical split", f, "recipes/audio-to-mesh")
    sp = json.load(open(f))
    clips = sp["train"] + sp["val"] + sp["test"]
    ok.append(f"the split: {len(clips)} clips, {len(sp['test'])} held out from BOTH models")

    f = RIG / "cache/align.json"
    if not f.exists():
        fail("the measured audio lag", f, "recipes/audio-to-mesh")
    ok.append("the measured lag")

    d = RIG / f"cache/targets_{p.SUBJECT}"
    absent = [c for c in clips if not (d / f"{c}.npz").exists()]
    if absent:
        fail(f"frame timings for {len(absent)} of {len(clips)} clips "
             f"(first: {absent[0][:40]})", f"{d}/<clip>.npz", "recipes/audio-to-mesh")
    ok.append(f"frame timings for all {len(clips)} clips")

    # --- the person's rig -------------------------------------------------
    for n, what in (("rig_beltrami.npz", "the person's rig"),
                    ("beltrami_wrap.npz", "the resting-face wrap")):
        f = p.PIPE / "identity/subjects" / p.SUBJECT / n
        if not f.exists():
            fail(what, f, "no recipe. The VHAP export is a documented MANUAL procedure --\n             pipeline/identity/README.md, Step 2 -- and it is NOT the same path\n             as recipes/face-clips: that tracks per CHUNK into output/shared/,\n             while the export comes from a single full-video run into\n             output/monocular/<subject>. Then:\n               python identity/build_identity.py --subject <name> \\\n                      --vhap-export <VHAP>/export/monocular/<run>")
    ok.append(f"rig and wrap for {p.SUBJECT}")

    # --- tracked chunks, with their pictures ------------------------------
    shared = p.VHAP / "output/shared"
    absent = [c for c in clips if not any((shared / c).glob("*/tracked_flame_params_30.npz"))]
    if absent:
        fail(f"tracked chunks for {len(absent)} of {len(clips)} clips in the split "
             f"(first: {absent[0][:40]})",
             f"{shared}/<clip>/<run>/tracked_flame_params_30.npz",
             "recipes/face-clips")
    data = p.VHAP / "data/monocular"
    absent = [c for c in clips if not (data / c / "images").is_dir()]
    if absent:
        fail(f"frames for {len(absent)} clips", f"{data}/<clip>/images/", "recipes/face-clips")
    ok.append(f"tracked chunks and frames for all {len(clips)} clips")

    # --- the renderer's own assets ----------------------------------------
    if not p.HEAD_ASSETS.exists():
        fail("the head topology", p.HEAD_ASSETS, "shipped with the project (pipeline/head/)")
    uv = p.TRAIN["uv_size"]
    masks = p.HEAD_ASSETS.parent / f"uv_region_masks_{uv}.pkl"
    plain = p.HEAD_ASSETS.parent / "uv_region_masks.pkl"
    if not masks.exists() and not plain.exists():
        fail(f"region masks at uv_size {uv}", masks, "shipped with the project")
    if not masks.exists():
        print(f"  note: no uv_region_masks_{uv}.pkl; the model will fall back to "
              f"{plain.name}, whose resolution then silently sets the training cost")
    if not (p.STAVATAR / "train.py").exists():
        fail("the renderer", p.STAVATAR / "train.py", "the STAvatar checkout")
    ok.append(f"head topology, region masks, and the renderer")

    # --- what the NATIVE variant needs, and the 512 one does not --------
    # Reported, never raised. The native corpus is one of two variants, so a missing
    # source video blocks that variant and nothing else -- failing the whole recipe over
    # it would stop the 512 path for no reason.
    #
    # It earns a line because it is the one input this recipe can LOSE. The 512 corpus is
    # built from a 1280-wide extraction and never needs the sources again, so they look
    # like 300 MB of dead weight each when disk runs short. Re-cutting at native reads
    # them directly, and once they are gone the native variant cannot be rebuilt from
    # anything still on disk.
    takes = sorted({c.rsplit("__c", 1)[0] for c in clips})
    srcs = {}
    for t in takes:
        srcs[t] = next((q for q in (p.PIPE / "takes").glob(f"*/{t}/source.*")
                        if q.suffix.lower() in (".mkv", ".mp4", ".webm", ".mov")), None)
    gone = [t for t, q in srcs.items() if q is None]
    if gone:
        print(f"  note: {len(gone)} of {len(takes)} source videos are missing, so the "
              f"native-resolution variant cannot be built.\n"
              f"        The 512 path is unaffected. Missing: "
              f"{', '.join(t[:34] for t in gone)}\n"
              f"        Expected under: {p.PIPE / 'takes'}/<channel>/<take>/source.*")
    else:
        gb = sum(q.stat().st_size for q in srcs.values()) / 1e9
        ok.append(f"all {len(takes)} source videos ({gb:.1f} GB) -- the native variant "
                  f"can be rebuilt")

    print(f"\n  preconditions hold for {p.SUBJECT}\n")
    for line in ok:
        print(f"    ok  {line}")
    print()
    report(shape(p.PIPE, p.VHAP, p.RECORDINGS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
