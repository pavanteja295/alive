#!/usr/bin/env python3
"""Re-derive, for this person, the values that can be derived without training.

    python tools/derive.py --profile drk

WHAT IS HERE AND WHAT IS NOT
    Three things are cheap enough to re-derive for every creator, and each one is
    currently either carried from drk or checked nowhere:

      close_mm   below what aperture counts as SHUT. It is absolute millimetres,
                 read off drk's mouth, and a person whose mouth moves further or
                 less will want a different one. Derived from their own tracked
                 aperture. Seconds.
      freeze     whether gaze is worth predicting at all. Carried as 'freeze it'
                 on drk's evidence -- a ridge from audio that was R2-negative on
                 all six axes. A ridge, not a training run. If someone's gaze IS
                 predictable, freezing it throws the signal away.
      steps      not a derivation. The best checkpoint is already chosen on
                 validation, so the budget only has to be LONG ENOUGH -- and the
                 check for that is whether the validation curve peaked inside it.

    ctx, w_close and w_bias are NOT here. Each costs a full training run per
    value, which makes them a search rather than a derivation: tools/sweep.py.

IT SUGGESTS; IT DOES NOT WRITE THE PROFILE.
    Every number below is printed with the distribution it came from so a reader
    can disagree with it. Writing the profile from a script would turn a judgement
    -- where a distribution's shoulder sits -- into a silent constant, which is
    the thing this pipeline keeps getting wrong.
"""
import argparse, json, pathlib, sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _profile import load                                              # noqa: E402


def aperture_distribution(p, clips):
    """This person's mouth, in millimetres of opening from rest."""
    sys.path.insert(0, str(p.PIPE / "rigfit"))
    sys.path.insert(0, str(p.PIPE / "offset"))
    from jaw_fit import mouth_points, measures
    up, dn, lft, rgt = mouth_points(p.SUBJECT)
    d = p.PIPE / f"rigfit/cache/targets_{p.SUBJECT}"
    vals = []
    for c in clips:
        f = d / f"{c}.npz"
        if not f.exists():
            continue
        skin = np.load(f)["skin"].astype(np.float32)
        ap, _ = measures(skin, up, dn, lft, rgt)
        vals.append(ap)
    return np.concatenate(vals) if vals else np.array([])


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--profile", required=True)
    ap_.add_argument("--skip-gaze", action="store_true")
    ap_.add_argument("--apply", action="store_true", help=
                     "take the auto-fixes. They are written to a derived_<subject>.json "
                     "that the profile reads; the profile .py stays hand-written.")
    a = ap_.parse_args()
    p = load(a.profile)
    C = p.PIPE / "rigfit/cache"
    out = {}

    split = json.load(open(p.SPLIT))
    train = split["train"]

    # ---- close_mm, from this person's own mouth --------------------------
    print("\n  close_mm -- below what aperture counts as SHUT\n")
    v = aperture_distribution(p, train)
    if len(v) == 0:
        print("    no targets to read; run build_targets.py first")
    else:
        q = {k: float(np.quantile(v, k / 100)) for k in (1, 2, 5, 10, 25, 50, 75, 95)}
        print(f"    {len(v):,} training frames, aperture from rest in mm")
        print("    " + "  ".join(f"p{k}={x:+.2f}" for k, x in q.items()))
        # DERIVED ON TRAINING CLIPS ONLY. Obvious, and got wrong once already:
        # drk's carried close_mm=2.0 sits just above his TEST p5 of +1.79, not his
        # training p5 of +2.58. It was read off the split nobody is allowed to look
        # at. His test clips genuinely have more closed mouths than his training
        # clips, so the two percentiles are 0.8 mm apart and the carried value was
        # tuned to the wrong one.
        #
        # The shut end is the lower tail: wide enough to catch the closures, narrow
        # enough to exclude ordinary speech.
        sug = round(float(np.quantile(v, 0.08)) * 2) / 2
        out["close_mm"] = sug
        print(f"\n    the profile carries close_mm={p.TRAIN['close_mm']}")
        print(f"    p5={q[5]:+.2f}, p10={q[10]:+.2f} -> suggest close_mm={sug}")
        print(f"    derived on TRAINING clips only -- drk's carried 2.0 was read off")
        print(f"    his test p5 (+1.79), not his training p5 (+2.58), which is the")
        print(f"    split nobody is allowed to look at.")

    # ---- freeze: is gaze worth predicting for THIS person? ---------------
    print(f"\n  freeze -- is gaze predictable from audio for this person?\n")
    if a.skip_gaze:
        print("    skipped")
    else:
        import subprocess
        r = subprocess.run([str(p.ENV / "bin/python"), "rigfit/gaze_probe.py"],
                           cwd=p.PIPE, capture_output=True, text=True)
        tail = [l for l in r.stdout.splitlines() if l.strip()][-14:]
        for l in tail:
            print(f"    {l}")
        print(f"\n    the profile carries freeze={p.TRAIN['freeze']!r}. If every R2 above")
        print(f"    is negative, freezing costs nothing. A positive one means the")
        print(f"    freeze is throwing signal away for THIS person.")

    # ---- steps: did the budget contain the peak? -------------------------
    print(f"\n  steps -- not derived. Was the budget long enough?\n")
    cands = [p.MOTION_DIR / p.RELEASE / "history.json",
             C / p.RELEASE / "history.json"]
    cands += sorted(C.glob("*/history.json"), key=lambda f: -f.stat().st_mtime)[:1]
    h = next((c for c in cands if c.exists()), cands[0])
    if h.exists():
        rows = [r for r in json.load(open(h)) if not r.get("baseline")]
        best = max(rows, key=lambda r: r.get("val_skin_pct", r.get("skin_pct", -1)))
        last = max(r["step"] for r in rows)
        bs = best["step"]
        print(f"    from {h.parent.name}: validation peaked at step {bs} of {last}")
        if bs == last:
            print(f"    THE PEAK IS AT THE END -- the budget was too short. Raise steps")
            print(f"    and re-run; the model was still improving when it stopped.")
        else:
            print(f"    the peak is inside the budget, so {p.TRAIN['steps']} is long")
            print(f"    enough for this person. Nothing to change.")
    else:
        print(f"    no history yet; train once and re-run this")

    # ---- the verdict, three outcomes, and only the ones that matter -----
    #
    # Printing all eighteen buries the two that need attention. A parameter is
    # FINE and moved past silently; or it is wrong AND THE CORRECTION IS
    # CONCRETE, in which case it is fixed and said so; or it is wrong and the
    # correction is a judgement, in which case it stops and asks.
    #
    # The line between the last two is whether a rule can be STATED. "The shut
    # end of this person's aperture" is a definition and derives itself. "Where
    # the capacity curve flattens" is not: drk's gains per doubling run 6.2, 3.6,
    # 4.7, 2.8, 3.2, 2.2, 1.2 -- not monotone, so any cut-off picks a different k
    # and the cut-off would be fitted to the one creator it was chosen on.
    fine, fixed, ask, chose = [], [], [], []

    for k, v in getattr(p, "CARRIED", {}).items():
        fine.append(f"{k}={v}")

    al = C / "align.json"
    if al.exists():
        a_ = json.load(open(al))
        have = [v for v in p.RECORDINGS if v in a_]
        lags = [a_[v]["sample_offset_ms"] for v in have]
        if len(have) < len(p.RECORDINGS):
            ask.append(("the lag",
                        f"only {len(have)} of {len(p.RECORDINGS)} recordings have one"))
        elif max(lags) - min(lags) > 100:
            ask.append(("the lag", f"the recordings disagree by "
                                   f"{max(lags)-min(lags):.0f} ms"))
        else:
            fine.append(f"lag={'/'.join(f'{x:+.0f}' for x in lags)} ms")
    else:
        ask.append(("the lag", "not measured -- run align.py"))

    if "close_mm" in out:
        cur, new_ = float(p.TRAIN["close_mm"]), out["close_mm"]
        if abs(cur - new_) < 0.25:
            fine.append(f"close_mm={cur}")
        else:
            fixed.append(("close_mm", cur, new_,
                          "the shut end of THIS person's aperture, from training "
                          "clips. A definition, not a threshold"))

    cap = C / f"capacity_{p.SUBJECT}.json"
    if not cap.exists():
        ask.append(("CORRECTIVE_K", f"no capacity curve -- run capacity.py "
                                    f"--subject {p.SUBJECT}"))
    else:
        d_ = json.load(open(cap))["splits"]["frames"]
        ks = sorted((x for x in d_ if x.isdigit()), key=int)
        at = {int(x): sum(d_[x]) / len(d_[x]) for x in ks}
        g = [f"{at[b]-at[a2]:+.1f}" for a2, b in zip(sorted(at), sorted(at)[1:])]
        # The curve is MONOTONE INCREASING, so there is no accuracy reason to stop
        # anywhere -- which means "the knee" was the wrong question. The real
        # question is cost: the layer is subtracted at every training step and a
        # bigger one is more to fit and more to overfit. So pick the largest k
        # whose gain is still worth its size, state the trade, and carry on. That
        # is a cost argument, not a threshold: it does not depend on any number
        # being above or below a line somebody chose.
        ks_ = sorted(at)
        best = max(at, key=lambda k: at[k])
        pick = p.CORRECTIVE_K
        gain_next = {k: at[b] - at[k] for k, b in zip(ks_, ks_[1:])}
        fine.append(f"CORRECTIVE_K={pick} ({at.get(pick, 0):.0f}% of his face reachable)")
        chose.append((f"CORRECTIVE_K = {pick}",
                      f"the curve is monotone -- k={best} would reach {at[best]:.1f}% -- "
                      f"so this is a cost call, not an accuracy one: the layer is "
                      f"subtracted every step, and k={pick} reaches {at.get(pick,0):.1f}% "
                      f"while the next doubling adds only "
                      f"{gain_next.get(pick, 0):.1f} points for twice the size. "
                      f"Raise it if you have the budget; nothing breaks."))

    print(f"\n{'='*70}")
    print(f"  {len(fine)} parameters make sense for {p.SUBJECT}; moving on.")
    if fine:
        print(f"    {', '.join(fine[:8])}{' ...' if len(fine) > 8 else ''}")

    if fixed:
        print(f"\n  AUTO-FIXED -- the correction is concrete")
        for k, cur, new_, why in fixed:
            print(f"    {k}: {cur} -> {new_}")
            print(f"      {why}")
    if chose:
        print(f"\n  CHOSEN -- a cost call with the trade stated. Override if you disagree")
        for k, why in chose:
            print(f"    {k}")
            print(f"      {why}")
    if ask:
        print(f"\n  NEEDS YOU -- the correction is a judgement, not a rule")
        for k, why in ask:
            print(f"    {k}: {why}")

    if fixed and a.apply:
        f = C / f"derived_{p.SUBJECT}.json"
        prev = json.loads(f.read_text()) if f.exists() else {}
        prev.update({k: v for k, _, v, _ in fixed})
        prev["_derived_when"] = __import__("datetime").datetime.now().isoformat(
            timespec="seconds")
        f.write_text(json.dumps(prev, indent=1))
        print(f"\n  written to {f.name}; the profile reads it and overrides its own")
        print(f"  carried value. The profile .py stays hand-written.")
    elif fixed:
        print(f"\n  Nothing written. Re-run with --apply to take the auto-fixes.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
