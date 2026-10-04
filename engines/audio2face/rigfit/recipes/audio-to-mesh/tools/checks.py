#!/usr/bin/env python3
"""Preconditions. This raises; it never builds.

    python tools/checks.py --profile drk

EVERY INPUT THIS RECIPE READS IS SOMEONE ELSE'S OUTPUT.
    When one is missing the recipe FAILS and names the recipe that owns it. It
    does not run the upstream tool, even when that tool is one line away. A
    recipe that builds its own inputs has no preconditions -- it has a longer
    body, and the first time an upstream artifact is subtly wrong it is rebuilt
    subtly wrong on every run with nothing left to compare against.

    So a failure here is a decision for the user, and it is one sentence to ask.

CHECKED AT THE TOP, ALL OF THEM, BEFORE ANY WORK STARTS.
    A run that discovers a missing input at minute 90 has burned the run and has
    usually half-written its outputs by then.
"""
import argparse, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _profile import load                                              # noqa: E402
from _corpus import shape, report                                      # noqa: E402
from _shape import clip, audio as audio_shape, rig as rig_shape        # noqa: E402


class Missing(SystemExit):
    pass


def fail(what, where, owner):
    raise Missing(
        f"\nPRECONDITION FAILED -- this recipe fails rather than builds.\n\n"
        f"  missing : {what}\n"
        f"  expected: {where}\n"
        f"  owned by: {owner}\n\n"
        f"  Produce it there, then re-run. Do not build it from inside this\n"
        f"  recipe: an input this recipe creates is an input nothing can check.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    a = ap.parse_args()
    p = load(a.profile)
    ok = []

    # --- the person's rig -------------------------------------------------
    for f, what, key in ((p.RIG, "the person's rig", "m0_V0"),
                         (p.WRAP, "the resting-face wrap", "verts_wrapped")):
        if not f.exists():
            fail(what, f, "no recipe. The VHAP export is a documented MANUAL procedure --\n             pipeline/identity/README.md, Step 2 -- and it is NOT the same path\n             as recipes/face-clips: that tracks per CHUNK into output/shared/,\n             while the export comes from a single full-video run into\n             output/monocular/<subject>. Then:\n               python identity/build_identity.py --subject <name> \\\n                      --vhap-export <VHAP>/export/monocular/<run>")
        good, det = rig_shape(f, key=key)
        if not good:
            fail(f"{what} is not the right shape to plug in -- {det}", f,
                 "no recipe. The VHAP export is a documented MANUAL procedure --\n             pipeline/identity/README.md, Step 2 -- and it is NOT the same path\n             as recipes/face-clips: that tracks per CHUNK into output/shared/,\n             while the export comes from a single full-video run into\n             output/monocular/<subject>. Then:\n               python identity/build_identity.py --subject <name> \\\n                      --vhap-export <VHAP>/export/monocular/<run>")
    ok.append(f"rig and wrap for {p.SUBJECT}, 24,049 vertices each")

    # --- the driver -------------------------------------------------------
    for n in ("audio_encoder.onnx", "animation_decoder.onnx"):
        f = p.PIPE / "onnx" / n
        if not f.exists():
            fail(f"the driver model {n}", f, "shipped with the project (pipeline/onnx/)")
    ok.append("the driver's encoder and decoder")

    # --- tracked chunks, which are the face-clips recipe's output ---------
    shared = p.VHAP / "output/shared"
    if not shared.is_dir():
        fail("tracked chunks", shared, "recipes/face-clips (pipeline/corpus/recipes/face-clips)")
    per_rec, total = {}, 0
    for vid in p.RECORDINGS:
        n = sum(1 for d in shared.glob(f"{vid}__c*")
                if any(d.glob("*/tracked_flame_params_30.npz")))
        per_rec[vid] = n
        total += n
    # DOES IT LOAD AND LINE UP? The one thing a precondition can settle for a
    # person nobody has seen, because it does not depend on the person: a field
    # is there or it is not, and T frames of pose either match T frames of
    # picture or they do not. Existence was never the failure mode -- a
    # truncated npz and a frame count off by a take's end trim both exist.
    data = p.VHAP / "data/monocular"
    malformed = []
    for d in sorted(shared.glob("*__c*")):
        f = sorted(d.glob("*/tracked_flame_params_30.npz"))
        if not f:
            continue
        good, det = clip(f[-1], data / d.name / "images", data / d.name / "alpha_maps")
        if not good:
            malformed.append((d.name, det))
    if malformed:
        lines = "".join(f"      {n[-12:]}  {d}\n" for n, d in malformed[:8])
        fail(f"{len(malformed)} clips are not the right shape to plug in:\n{lines}",
             f"{shared}/<clip>/<run>/tracked_flame_params_30.npz",
             "recipes/face-clips")

    empty = [v for v, n in per_rec.items() if n == 0]
    if empty:
        fail(f"tracked chunks for {len(empty)} of {len(p.RECORDINGS)} recordings "
             f"({', '.join(v[:28] for v in empty)})",
             f"{shared}/<recording>__cNNN/*/tracked_flame_params_30.npz",
             "recipes/face-clips (pipeline/corpus/recipes/face-clips)")
    ok.append(f"{total} tracked chunks across {len(p.RECORDINGS)} recordings, all conforming")

    # --- the audio, which nothing in this project produces ----------------
    ad = p.PIPE / "rigfit/cache/audio_local"
    absent = [i for i in p.RECORDINGS.values() if not (ad / f"{i}.wav").exists()]
    if absent:
        fail(f"audio for {len(absent)} recording(s): {', '.join(absent)}",
             f"{ad}/<id>.wav, 16 kHz mono",
             "no recipe -- it is downloaded alongside the video, by you")
    wrong = []
    for i in p.RECORDINGS.values():
        good, det = audio_shape(ad / f"{i}.wav")
        if not good:
            wrong.append(f"{i}: {det}")
    if wrong:
        fail("audio the driver cannot read -- " + "; ".join(wrong),
             f"{ad}/<id>.wav, 16 kHz mono",
             "no recipe -- it is downloaded alongside the video, by you")
    ok.append(f"{len(p.RECORDINGS)} audio files, 16 kHz mono")

    print(f"\n  preconditions hold for {p.SUBJECT}\n")
    for line in ok:
        print(f"    ok  {line}")
    print()
    # Existence is checked. Size and quality are REPORTED, and judged by whoever
    # reads this -- there is no number of clips that is enough, and a threshold
    # here would be fitted to the one creator this has run on.
    report(shape(p.PIPE, p.VHAP, p.RECORDINGS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
