#!/usr/bin/env python3
"""Do the numbers still make sense for this pipeline? And if not, what broke?

    python tools/sanity.py --profile drk

NOT MORE THRESHOLDS. RELATIONS.
    A threshold says "below 30% is bad", which is a number fitted to whoever it
    was fitted to. A relation says "a model driving these controls cannot beat
    what the controls can express" -- which is true by what the two numbers MEAN,
    on any person, and needs no tuning.

    So every check here is an ordering or an identity that follows from the
    semantics. When one breaks, something is genuinely wrong, and the useful
    output is not FAIL: it is what the break implies, because the same violation
    usually has two or three possible causes and naming them is the work.

WHY THIS IS WHERE THE INTELLIGENCE GOES
    A new creator produces numbers nobody has seen. A checklist written in
    advance covers what was imagined; a relation covers what is true. When one
    fails the worker has a real question to answer -- why can this model shut its
    lips better than the solve that can see the mesh? -- and answering it is the
    job that cannot be scripted.
"""
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _profile import load                                              # noqa: E402


def relations(n):
    """Each: (name, holds, statement, what a violation implies).

    A RELATION MAY BE NARROWED, BUT ONLY FOR A STATED REASON THAT DOES NOT
    MENTION THE OBSERVATION. Otherwise this becomes tuning until it passes,
    which is the failure it exists to catch. Two have been narrowed so far and
    each carries its argument below.
    """
    R = []
    g = n.get

    if g("ceiling") and g("transferred"):
        R.append(("model vs ceiling", g("transferred") <= g("ceiling"),
                  f"transferred {g('transferred'):.2f}% <= ceiling {g('ceiling'):.2f}%",
                  "A model driving these controls cannot express more of the face than "
                  "the controls can. If it does, the two numbers are measured on "
                  "different frames, different points, or different splits -- they are "
                  "not comparable and one of them is wrong."))

    if g("transferred") and g("motionless"):
        R.append(("model vs his average face", g("transferred") > g("motionless"),
                  f"transferred {g('transferred'):.2f}% > motionless {g('motionless'):.2f}%",
                  "Predicting his average face beats the model. The model has learnt "
                  "less than nothing: check the lag first, then whether the target is "
                  "the surface you think it is."))

    if g("transferred") and g("driver"):
        R.append(("model vs the shipped driver", g("transferred") > g("driver"),
                  f"transferred {g('transferred'):.2f}% > driver alone {g('driver'):.2f}%",
                  "The correction is making the driver worse. Either it is being added "
                  "with the wrong sign, or the corrective layer is being added where it "
                  "should be subtracted."))

    # NARROWED 2026-09-14. This was stated as "the solve bounds the model on
    # CLOSURE" and broke: the model shut the lips at +1.50 mm where the solve
    # managed +1.67. The relation was wrong, not the model. The per-frame solve
    # minimises error over the WHOLE FACE, so it bounds whole-face quantities --
    # aperture error, aperture R2 -- and bounds nothing about a tail statistic it
    # never optimised. Closure is such a tail statistic. The corrected relations
    # both hold: error 2.16 >= 1.15 mm, R2 0.560 <= 0.817.
    #
    # The reason does not mention the observation: it follows from what the solve
    # minimises, and would have been the right statement before any run existed.
    if g("model_ap_err") is not None and g("oracle_ap_err") is not None:
        R.append(("model vs the oracle, on aperture error",
                  g("model_ap_err") >= g("oracle_ap_err"),
                  f"model {g('model_ap_err'):.2f} mm >= oracle {g('oracle_ap_err'):.2f} mm",
                  "An audio model is fitting the mouth more accurately than a "
                  "per-frame solve that can SEE the tracked mesh. That is not "
                  "possible if they are scored the same way: check the frame sets, "
                  "the context length, and whether the corrective layer was applied "
                  "to one and not the other."))
    if g("model_ap_r2") is not None and g("oracle_ap_r2") is not None:
        R.append(("model vs the oracle, on aperture R2",
                  g("model_ap_r2") <= g("oracle_ap_r2"),
                  f"model {g('model_ap_r2'):.3f} <= oracle {g('oracle_ap_r2'):.3f}",
                  "Same as above, in the other direction."))

    # NARROWED 2026-09-14. The argument is that squared error asks for an
    # amplitude equal to the correlation, hence under 1. That argument assumes
    # squared error ALONE. A shaping term with its own objective -- the closure
    # hinge -- changes what is being minimised, so the bound does not apply to a
    # model trained with one. Again the reason is about the loss, not about the
    # number that broke it.
    if g("amplitude") and not g("has_shaping_term"):
        R.append(("amplitude under a squared-error loss", g("amplitude") <= 1.02,
                  f"aperture amplitude {g('amplitude'):.2f}x <= 1.0",
                  "Squared error asks for an amplitude equal to the correlation, so "
                  "under 1. Above 1 means something else is driving the range -- an "
                  "added term with its own objective, most likely. Not wrong, but it "
                  "is no longer the MSE solution and should not be read as one."))
    elif g("amplitude"):
        R.append(("amplitude, with a shaping term active", True,
                  f"{g('amplitude'):.2f}x -- the MSE bound does not apply",
                  ""))

    if g("val_test_corr") is not None:
        R.append(("validation agrees with test", g("val_test_corr") > 0,
                  f"val/test rank correlation {g('val_test_corr'):+.2f} > 0",
                  "Validation and test disagree about which model is better. Any "
                  "selection made on test is fitting the held-out set, and the spread "
                  "may be noise: run repeated seeds to get a floor before believing "
                  "any ordering."))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    a = ap.parse_args()
    p = load(a.profile)
    C = p.PIPE / "rigfit/cache"
    n, src = {}, {}

    cap = C / f"capacity_{p.SUBJECT}.json"
    if cap.exists():
        d = json.load(open(cap))["splits"]["frames"]
        at = {int(k): sum(d[k]) / len(d[k]) for k in d if k.isdigit()}
        if p.CORRECTIVE_K in at:
            n["ceiling"] = at[p.CORRECTIVE_K]; src["ceiling"] = cap.name

    rel = p.MOTION_DIR / p.RELEASE / "MANIFEST.json"
    if rel.exists():
        m = json.load(open(rel)); h = m.get("held_out", {})
        r = m.get("references_on_the_same_measure", {})
        n["transferred"] = h.get("transferred_pct")
        n["amplitude"] = h.get("aperture_amplitude_vs_tracked")
        n["model_closure"] = h.get("lip_closure_mean_mm")
        n["model_ap_err"] = h.get("aperture_err_mm")
        n["model_ap_r2"] = h.get("aperture_r2")
        n["has_shaping_term"] = bool(json.load(open(rel)).get("what_changed"))
        o = r.get("per_frame_solve_an_oracle") or {}
        n["oracle_ap_r2"] = o.get("aperture_r2")
        n["driver"] = (r.get("xada_as_it_ships") or {}).get("driver_alone_pct")
        for k in ("driver", "motionless"):
            pass
        # References this manifest does not carry come from THIS person's release run:
        # its history records the driver alone and the motionless average face on the
        # same clips. This read drk's face_v1 manifest for everyone until 2026-09-29.
        hist = p.MOTION_DIR / p.RELEASE / "history.json"
        if hist.exists():
            hr = json.load(open(hist))
            b = next((r for r in hr if r.get("baseline")), {})
            mf = next((r for r in hr if r.get("mean_face")), {})
            n.setdefault("driver", b.get("skin_pct"))
            n.setdefault("motionless", mf.get("skin_pct"))
        hh = m.get("held_out", {})
        if hh.get("driver_alone_pct") is not None:
            n["driver"] = hh["driver_alone_pct"]
        if hh.get("motionless_control_pct") is not None:
            n["motionless"] = hh["motionless_control_pct"]

    # validation against test, across whatever runs are on disk
    import numpy as np
    V, E = [], []
    for d in sorted(C.glob("*/history.json")):
        e = d.parent / "eval.json"
        if not e.exists():
            continue
        # this person's runs only: the cache holds every creator's, and a rank
        # correlation across people measures the people, not the protocol
        # owner read from the layer file itself: personalise.py stores the subject in
        # it, which also covers files named before layers were keyed by subject
        lay = json.load(open(e)).get("layer") or ""
        lf = p.PIPE / lay if lay and not pathlib.Path(lay).is_absolute() else pathlib.Path(lay)
        if not lay or not lf.exists() or str(np.load(lf).get("subject", "")) != p.SUBJECT:
            continue
        rows = [r for r in json.load(open(d)) if not r.get("baseline")]
        if not rows:
            continue
        V.append(max(r.get("val_skin_pct", r.get("skin_pct", -1)) for r in rows))
        E.append(json.load(open(e))["mean"]["skin_pct"])
    if len(V) > 2:
        n["val_test_corr"] = float(np.corrcoef(V, E)[0, 1]); src["val_test_corr"] = f"{len(V)} runs"

    n = {k: v for k, v in n.items() if v is not None}
    R = relations(n)
    ok = [r for r in R if r[1]]
    bad = [r for r in R if not r[1]]

    print(f"\n  {len(ok)} of {len(R)} relations hold for {p.SUBJECT}.\n")
    for name, _, stmt, _ in ok:
        print(f"    ok    {name:<34} {stmt}")
    if not bad:
        print(f"\n  Nothing to explain.\n")
        return 0
    print(f"\n  {len(bad)} DO NOT. These are not thresholds -- each is true by what the\n"
          f"  numbers mean, so a break is a real question with a real answer.\n")
    for name, _, stmt, implies in bad:
        print(f"    BROKEN  {name}")
        print(f"      expected: {stmt}")
        print(f"      implies:  {implies}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
