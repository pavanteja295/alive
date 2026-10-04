#!/usr/bin/env python3
"""Search the fitted parameters, and rank ONLY on validation.

    python tools/sweep.py --profile drk --vary w_close=0,0.01,0.02 --vary w_bias=0,1
    python tools/sweep.py --profile drk --seeds 2            # the noise floor first
    python tools/sweep.py --profile drk --report             # read what is on disk

WHY THIS TOOL EXISTS AND NOT A FOR-LOOP
    Because the protocol is the hard part, and it was got wrong by hand on this
    very pipeline. Six runs of the closure term were compared on TEST, and the
    best test score was released as face_v2. Validation ranked the same six runs
    almost in reverse -- correlation -0.44 -- and by validation the winner was the
    model with no closure term at all.

    A search that selects on test does not measure a model, it fits the held-out
    set. The more variants, the better the winner looks and the less it means.

SO:
    - ranking is on validation, read from history.json. Always.
    - test is read ONLY for the single config that already won on validation,
      and only once, and it is never used to choose.
    - the noise floor comes first. Two seeds of one config bound how big a
      difference has to be before it is a difference at all. Without it a 2-point
      spread is indistinguishable from six samples of the same model.
"""
import argparse, itertools, json, pathlib, subprocess, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _profile import load                                              # noqa: E402


def val_best(d):
    """The best VALIDATION score in a run, and where it was. Never test."""
    h = json.load(open(d / "history.json"))
    rows = [r for r in h if not r.get("baseline")]
    if not rows:
        return None, None
    b = max(rows, key=lambda r: r.get("val_skin_pct", r.get("skin_pct", -1)))
    return b.get("val_skin_pct", b.get("skin_pct")), b.get("step")


def test_of(d):
    f = d / "eval.json"
    return json.load(open(f))["mean"]["skin_pct"] if f.exists() else None


def cmd_for(p, tag, over):
    t = {**p.TRAIN, **over}
    c = p.PIPE / f"rigfit/cache/corrective_{p.SUBJECT}_k{p.CORRECTIVE_K}.npz"
    return [str(p.ENV / "bin/python"), "rigfit/train_offset2.py",
            "--out", f"rigfit/cache/{tag}", "--subject", p.SUBJECT,
            "--steps", str(t["steps"]), "--eval-every", str(t["eval_every"]),
            "--batch", str(t["batch"]), "--span", str(t["span"]),
            "--ctx", str(t["ctx"]), "--hidden", str(t["hidden"]),
            "--layers", str(t.get("layers", 2)),
            "--freeze", t["freeze"], "--live",
            "--per-window", t["per_window"], "--audio-ctx", t["audio_ctx"],
            "--workers", str(t["workers"]), "--seed", str(t["seed"]),
            "--scope", t["scope"], "--zero-base", t["zero_base"],
            "--layer", str(c.relative_to(p.PIPE)),
            "--w-close", str(t["w_close"]), "--close-mm", str(t["close_mm"]),
            "--w-bias", str(t["w_bias"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--vary", action="append", default=[],
                    help="name=v1,v2,v3 -- one of the FITTED parameters")
    ap.add_argument("--seeds", type=int, default=0,
                    help="run one config this many times to establish the noise floor")
    ap.add_argument("--tag", default="S")
    ap.add_argument("--report", action="store_true", help="read what is on disk, run nothing")
    ap.add_argument("--no-floor", action="store_true",
                    help="skip establishing the noise floor. The ranking is then not "
                         "distinguishable from repeated samples of one model.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    p = load(a.profile)
    cache = p.PIPE / "rigfit/cache"
    fitted = set(getattr(p, "FITTED", {}))

    # ---- build the variants ------------------------------------------------
    grid = {}
    for v in a.vary:
        k, vals = v.split("=", 1)
        if k not in fitted:
            raise SystemExit(
                f"'{k}' is not one of the FITTED parameters: {', '.join(sorted(fitted))}.\n"
                f"Carried values name their evidence and are not the search space; if "
                f"you mean to search one, move it to FITTED in the profile and say why.")
        grid[k] = [float(x) if "." in x or x.replace('-','').isdigit() else x
                   for x in vals.split(",")]
    # THE FLOOR COMES FIRST, WITHOUT BEING ASKED FOR.
    #
    # Ranking without one is the default failure: six closure variants on this
    # pipeline spanned two points, were ranked, and the ordering turned out to be
    # arbitrary. Someone running this should not have to know that in advance, so
    # if variants are wanted and no repeated-seed run exists, two are added ahead
    # of them. --no-floor opts out, deliberately and visibly.
    have_floor = len(sorted(cache.glob(f"{a.tag}_seed*/history.json"))) > 1
    if grid and not have_floor and not a.seeds and not a.no_floor:
        print(f"  no noise floor on disk -- adding 2 seed runs ahead of the sweep, so\n"
              f"  the ranking below can be told from run-to-run variance.\n")
        a.seeds = 2

    variants = []
    if a.seeds:
        for i in range(a.seeds):
            variants.append((f"{a.tag}_seed{i}", {"seed": p.TRAIN["seed"] + i}))
    for combo in itertools.product(*grid.values()) if grid else []:
        over = dict(zip(grid, combo))
        tag = a.tag + "_" + "_".join(f"{k}{v}".replace(".", "") for k, v in over.items())
        variants.append((tag, over))

    # ---- run what is missing ----------------------------------------------
    if not a.report:
        for tag, over in variants:
            d = cache / tag
            if (d / "history.json").exists():
                print(f"  have  {tag}")
                continue
            c = cmd_for(p, tag, over)
            print(f"  run   {tag}  {over}")
            if a.dry_run:
                print("        " + " ".join(c[1:]))
                continue
            t0 = time.time()
            r = subprocess.run(c, cwd=p.PIPE)
            print(f"        {'ok' if r.returncode == 0 else 'FAILED'} "
                  f"in {(time.time()-t0)/60:.1f} min")

    # ---- rank, on validation, and say what the spread means ---------------
    rows = []
    for tag, over in variants:
        d = cache / tag
        if not (d / "history.json").exists():
            continue
        v, step = val_best(d)
        rows.append((tag, over, v, step, d))
    if not rows:
        print("\n  nothing to rank yet\n")
        return 0
    rows.sort(key=lambda r: -(r[2] or -1))

    seeds = [r for r in rows if "seed" in r[0]]
    floor = None
    if len(seeds) > 1:
        vs = [r[2] for r in seeds]
        floor = max(vs) - min(vs)

    print(f"\n  ranked on VALIDATION. Test is not read for anything but the winner.\n")
    print(f"  {'run':22}{'validation':>12}{'step':>7}   what varied")
    for tag, over, v, step, _ in rows:
        print(f"  {tag:22}{v:12.2f}{step:7}   {over or '(baseline)'}")
    if floor is not None:
        print(f"\n  noise floor: {len(seeds)} seeds of one config spread {floor:.2f} points.")
        real = [r for r in rows if r[2] is not None and rows[0][2] - r[2] > floor]
        print(f"  A difference smaller than that is not a difference. "
              f"{len(rows) - len(real) - 1} of {len(rows)-1} rivals are inside it.")
    else:
        print(f"\n  NO NOISE FLOOR. Run --seeds 2 first: without it a spread of "
              f"{rows[0][2] - rows[-1][2]:.2f} points\n  cannot be told from repeated "
              f"samples of the same model.")

    w = rows[0]
    t = test_of(w[4])
    print(f"\n  winner on validation: {w[0]}  ({w[1] or 'baseline'})")
    if t is not None:
        print(f"  its test score, read once and used for nothing else: {t:.2f}%")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
