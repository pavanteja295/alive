#!/usr/bin/env python3
"""Promote a trained checkpoint to a release the next recipe can consume.

    python tools/promote.py --profile drk --run G5 --as face_v3
    python tools/promote.py --profile drk --run G5 --as face_v3 --wrong "what is still wrong"

IT REFUSES A CHECKPOINT THAT WAS SELECTED ON TEST.
    This pipeline released one. Six closure variants were compared on their test
    scores and the best was shipped; validation ranked the same six almost in
    reverse. The correction had to be hand-written into the manifest afterwards,
    which only happened because somebody went looking.

    So promotion checks the protocol, not just the files: if other runs exist
    with test scores and the one being promoted is not also the validation
    winner, it stops and says which run is.

THE MANIFEST IS ASSEMBLED, NOT WRITTEN.
    Everything in it comes off disk -- the command, the step chosen on
    validation, the held-out numbers, the closure numbers, this person's ceiling
    and the fraction of it reached. The one field that cannot be derived is what
    is still WRONG with the model, and promotion asks for it rather than leaving
    it blank, because a release with no known faults is a release nobody checked.
"""
import argparse, datetime, json, pathlib, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _profile import load                                              # noqa: E402


def val_best(d):
    h = json.load(open(d / "history.json"))
    rows = [r for r in h if not r.get("baseline")]
    if not rows:
        return None, None
    b = max(rows, key=lambda r: r.get("val_skin_pct", r.get("skin_pct", -1)))
    return b.get("val_skin_pct", b.get("skin_pct")), b.get("step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--run", required=True, help="a directory under rigfit/cache/")
    ap.add_argument("--as", dest="name", required=True)
    ap.add_argument("--wrong", default=None,
                    help="what is still wrong with it. Required: a release with no "
                         "known faults is a release nobody checked.")
    ap.add_argument("--anyway", action="store_true",
                    help="promote despite not being the validation winner. Records why.")
    a = ap.parse_args()
    p = load(a.profile)
    C = p.PIPE / "rigfit/cache"
    src = C / a.run
    if not (src / "best.pt").exists():
        raise SystemExit(f"no checkpoint at {src}/best.pt")

    v, step = val_best(src)

    # ---- the protocol check ----------------------------------------------
    # ONLY RUNS SCORED THE SAME WAY ARE RIVALS.
    #
    # The first version of this compared against everything on disk and refused a
    # perfectly good promotion because a jaw-only run scored 47.96% against a
    # whole-face run's 33.00%. Those yardsticks are not the same thing. Runs now
    # write a yardstick.json; a run without one cannot be shown to be comparable,
    # so it is left out rather than assumed.
    def yard(d):
        f = d / "yardstick.json"
        return json.load(open(f)) if f.exists() else None

    mine = yard(src)
    rivals, unknown = [], 0
    for d in sorted(C.glob("*/history.json")):
        if d.parent == src or not (d.parent / "eval.json").exists():
            continue
        y = yard(d.parent)
        if mine is None or y is None or y != mine:
            unknown += y is None
            continue
        rv, _ = val_best(d.parent)
        if rv is not None:
            rivals.append((d.parent.name, rv))
    better = [(n, rv) for n, rv in rivals if rv > (v or -1)]
    if better and not a.anyway:
        better.sort(key=lambda r: -r[1])
        raise SystemExit(
            f"\nNOT THE VALIDATION WINNER -- refusing to promote.\n\n"
            f"  {a.run} scores {v:.2f}% on validation.\n"
            f"  These score higher:\n"
            + "".join(f"    {n:<16}{rv:.2f}%\n" for n, rv in better[:5]) +
            f"\n  If you are promoting this one because its TEST score is better, that\n"
            f"  is fitting the held-out set: this pipeline shipped one model that way\n"
            f"  and the correction had to be written in afterwards.\n\n"
            f"  Promote the validation winner, or pass --anyway and say why in --wrong.\n")

    if a.wrong is None:
        raise SystemExit(
            f"\n--wrong is required. What is still wrong with this model?\n\n"
            f"  Every release here has real faults and the useful ones are recorded:\n"
            f"  face_v1 never closed the lips; face_v2 over-moves at 1.12x amplitude\n"
            f"  and stays open on half of his closed frames. A release with no known\n"
            f"  faults is a release nobody checked.\n")

    # ---- assemble, do not write by hand ----------------------------------
    dst = p.MOTION_DIR / a.name
    dst.mkdir(parents=True, exist_ok=True)
    # yardstick.json belongs in the release: it is the only file that records which
    # decoder the weights mean anything through. Dropping it is how stage B came to
    # render the bare rig for a model trained against rig-minus-layer.
    for f in ("best.pt", "eval.json", "history.json", "yardstick.json"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    # ...and the decoder it means anything through: the corrective layer goes beside the
    # weights, the rig into checkpoints/<SUBJECT>/face/rig. The live app reads both there.
    sys.path.insert(0, str(p.PIPE / "gauss"))
    from decoder import resolve
    dec = resolve(dst)
    if dec.get("layer"):
        lay = pathlib.Path(dec["layer"]); lay = lay if lay.is_absolute() else p.PIPE / lay
        shutil.copy(lay, dst / lay.name)
    rig = p.CHECKPOINTS / "face/rig"; rig.mkdir(parents=True, exist_ok=True)
    for f in ("rig_beltrami.npz", "beltrami_wrap.npz"):
        shutil.copy(p.PIPE / "identity/subjects" / p.SUBJECT / f, rig / f)

    ceiling = None
    cap = C / f"capacity_{p.SUBJECT}.json"
    if cap.exists():
        d_ = json.load(open(cap))["splits"]["frames"]
        at = {int(k): sum(d_[k]) / len(d_[k]) for k in d_ if k.isdigit()}
        ceiling = at.get(p.CORRECTIVE_K)
    got = json.load(open(dst / "eval.json"))["mean"].get("skin_pct") \
        if (dst / "eval.json").exists() else None

    m = {
        "name": a.name,
        "subject": p.SUBJECT,
        "from_run": a.run,
        "promoted": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True,
                                 cwd=p.PIPE).stdout.strip(),
        "selected_on": "validation",
        "validation_pct": v,
        "step_of_best_checkpoint": step,
        "rivals_on_validation": dict(sorted(rivals, key=lambda r: -r[1])[:6]),
        "held_out": {"transferred_pct": got},
        "this_persons_ceiling_pct": ceiling,
        "fraction_of_ceiling": (got / ceiling) if (got and ceiling) else None,
        "parameters": {**getattr(p, "CARRIED", {}), **getattr(p, "FITTED", {}),
                       **getattr(p, "DERIVED", {}), "CORRECTIVE_K": p.CORRECTIVE_K},
        "what_is_still_wrong": a.wrong,
    }
    if a.anyway and better:
        m["PROMOTED_DESPITE"] = {
            "higher_on_validation": dict(better[:5]),
            "why": a.wrong}
    json.dump(m, open(dst / "MANIFEST.json", "w"), indent=1)

    print(f"\n  promoted {a.run} -> {dst}\n")
    if unknown:
        print(f"    note: {unknown} runs on disk record no yardstick, so they could")
        print(f"    not be shown comparable and were not ranked against this one.")
    print(f"    selected on validation: {v:.2f}% at step {step}")
    if got is not None:
        print(f"    test, read once: {got:.2f}%"
              + (f" = {got/ceiling*100:.0f}% of this person's ceiling" if ceiling else ""))
    print(f"    still wrong: {a.wrong}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
