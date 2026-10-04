#!/usr/bin/env python3
"""Search the renderer's few searchable settings, ranked on VALIDATION only.

    python tools/sweep.py --profile drk --plan          # what to run, and why
    python tools/sweep.py --profile drk --rank          # rank what has finished

WHY THIS CANNOT RANK ON TEST, STRUCTURALLY RATHER THAN BY INTENTION

A renderer run is scored by train.py on whatever its dataset calls `transforms_test`.
So the dataset decides what "held out" means, and merge_corpus builds two:

    <name>_sel    train = the split's train clips, test = the split's VALIDATION clips
    <name>        train = the split's train clips, test = the split's TEST clips

A search arm is a run on the `_sel` dataset. This tool refuses to rank a run whose
dataset is not a `_sel` one, so choosing on test is not a thing you can do carelessly
here -- it would take deliberately renaming a dataset.

Note what the existing runs on drk did: they trained on train PLUS validation and held
out only test, so they have no validation number at all and cannot enter a ranking.
That is why the `_sel` datasets exist.

WHAT IT WILL AND WILL NOT VARY

It varies only what the profile lists as SEARCHED. It REFUSES to vary anything in
CARRIED or DECIDED, and says so, because those two categories are claims: carried means
"measured on one creator and expected to transfer", and a search that quietly moves one
has invalidated the claim without anybody noticing. If a carried value should be
searched, move it in the profile first -- that edit is the record.

THE NOISE FLOOR COMES FIRST

Two arms that differ only by seed are inserted ahead of everything else. Without them
no difference between two variants can be told from run-to-run variance, and on a
renderer at this scale nobody's intuition about that is worth anything. If the seed
spread is as wide as the best-to-worst spread, the honest answer is that the search
found nothing, and this tool will say that.
"""
import argparse
import itertools
import json
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _profile import load                                              # noqa: E402

# The metric the arms are ranked by. LPIPS, because the eventual judge of a talking
# head is a person, and of the three numbers train.py records it is the one that
# tracks perceived quality; PSNR and SSIM are printed beside it so a disagreement
# between them is visible rather than hidden. Lower is better.
RANK_BY, LOWER_IS_BETTER = "LPIPS", True


def cfg_of(run):
    f = run / "config.yml"
    if not f.exists():
        return {}
    y = yaml.safe_load(f.read_text()) or {}
    flat = {}
    for v in y.values():
        if isinstance(v, dict):
            flat.update(v)
    flat.update({k: v for k, v in y.items() if not isinstance(v, dict)})
    return flat


def arms(p):
    """The queue: two seed arms for the noise floor, then the grid."""
    space = {k: v for k, v in p.SEARCH_SPACE.items() if v}
    clash = sorted(set(space) & set(p.CARRIED))
    if clash:
        raise SystemExit(
            f"the profile both carries and searches: {', '.join(clash)}\n"
            f"  Those are contradictory claims about the same number. Carried means\n"
            f"  measured once and expected to transfer; searched means it does not.\n"
            f"  Remove each from one of the two sections and the conflict is the record.")
    base = {k: v[0] for k, v in space.items()}
    out = [("seed1", {**base, "seed": 1}), ("seed2", {**base, "seed": 2})]
    keys = sorted(space)
    for combo in itertools.product(*(space[k] for k in keys)):
        d = dict(zip(keys, combo))
        if d == base:
            continue
        out.append(("_".join(f"{k}{v}" for k, v in sorted(d.items())), {**d, "seed": 1}))
    return out


def command(p, name, params, dataset):
    """The exact train.py invocation for one arm. Printed rather than launched: each of
    these is hours, and a chain that starts itself has already deleted 53 GB once."""
    # The seed IS a flag. It used to be excluded here because train.py had nowhere
    # to put it, which made the two noise-floor arms the same command -- a floor
    # that measured only CUDA atomics and so declared every setting significant.
    flags = " ".join(f"--{k} {v}" for k, v in sorted(params.items()))
    train = {**p.TRAIN}
    # THE COUPLING. The run length is epochs x frames, so an arm that changes epochs
    # needs its own schedule. Carrying the schedule derived for another epoch count
    # reintroduces exactly the defect derive.py exists to catch.
    # The epoch count the run length was DERIVED at -- not p.TRAIN, which has no
    # epochs entry at all because epochs is searched rather than carried. Reading it
    # from there silently skipped this whole correction.
    base_ep = p.TRAIN.get("epochs") or p.SEARCH_SPACE["epochs"][0]
    if "epochs" in params and p.DERIVED_ALL.get("_run_length") and base_ep:
        per_epoch = p.DERIVED_ALL["_run_length"] / base_ep
        train["position_lr_max_steps"] = int(round(per_epoch * params["epochs"]))
    # THE DATASET PICKS THE MODEL CLASS, SO THE FLAGS MUST FOLLOW THE DATASET.
    #
    # train.py selects MeshGaussianModel the moment --mesh_assets is present. That model
    # allocates num_timesteps x 24,049 x 3 x 4 bytes TWICE up front -- 23.4 GB on this
    # corpus -- and only skips it when the meshes carry a __path__, which COOKED meshes
    # have and tracked FLAME meshes do not. So --mesh_assets on a tracked dataset is not
    # a setting that gets ignored, it is an OOM kill during loading. uv_size and
    # threshold_xyz are likewise properties of the 24,049-vertex rig head and mean
    # nothing to the 5,143-vertex FLAME one.
    rig = (p.CORPUS / dataset / "canonical.npz").exists()
    # uv_size and threshold_xyz are NOT rig-only: the per-frame nudge network is built
    # at uv_size on both paths (its default is 512, the resolution the recipe says costs
    # 6.8 it/s against 22.7), and FlameGaussianModel binds to its own faces too, so the
    # slide loss reads threshold_xyz there as well. Only the model-class flags are
    # conditional. The slide BUDGET, though, is a median edge and so belongs to whichever
    # topology the dataset implies -- the rig's 1.81 mm is not FLAME's.
    if not rig and p.DERIVED_ALL.get("_threshold_xyz_flame"):
        train["threshold_xyz"] = p.DERIVED_ALL["_threshold_xyz_flame"]
    extra = " ".join(f"--{k} {v}" for k, v in sorted(train.items())
                     if k not in params and k in ("position_lr_max_steps", "uv_size",
                                                  "threshold_xyz"))
    # not_finetune_flame_params belongs to the FLAME path, NOT the rig one: it is what
    # freezes per-frame geometry that inference cannot reproduce. The rig model has no
    # trainable geometry, so it records the flag and reads it. Emitted for both, because
    # dropping it on the FLAME path is exactly the unfrozen run the profile warns about.
    nff = " --not_finetune_flame_params" if p.DECIDED["not_finetune_flame_params"] else ""
    mesh = f"  --mesh_assets {p.HEAD_ASSETS} --pose_mode posed \\\n" if rig else ""
    return (f"cd {p.STAVATAR} && {p.ENV}/bin/python train.py \\\n"
            f"  -s {p.CORPUS / dataset} -m {p.RUNS / ('S_' + name)}{nff} \\\n"
            f"{mesh}  {extra} {flags}".rstrip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--dataset", default="corpus_pred_sel",
                    help="MUST be a _sel dataset: its test split is the validation "
                         "clips. Ranking refuses anything else")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--rank", action="store_true")
    a = ap.parse_args()
    p = load(a.profile)

    # A dataset that is not on disk cannot say which model class it implies, and the
    # plan would silently come out FLAME-shaped for a rig corpus. Refuse rather than
    # emit commands whose meaning depends on a directory appearing later.
    if not (p.CORPUS / a.dataset).is_dir():
        raise SystemExit(
            f"\n{a.dataset} is not on disk under {p.CORPUS}.\n"
            f"  The dataset decides the model class and the flags that go with it,\n"
            f"  so there is no honest plan to print yet. Build it first:\n"
            f"    python {p.PIPE}/gauss/merge_corpus.py --out {a.dataset[:-4]} ...\n")

    if not a.dataset.endswith("_sel"):
        raise SystemExit(
            f"{a.dataset} is not a validation dataset.\n"
            f"  A search arm must be scored on the VALIDATION clips, which is what a\n"
            f"  '_sel' dataset holds out. Scoring on '{a.dataset}' would be selecting\n"
            f"  on test. Build one with:\n"
            f"    python {p.PIPE}/gauss/merge_corpus.py --out {a.dataset[:-4]} ...")

    q = arms(p)
    print(f"searching {len(q)} arms on {a.dataset}, ranked on {RANK_BY} "
          f"({'lower' if LOWER_IS_BETTER else 'higher'} is better)\n")
    print("will vary   : " + ", ".join(sorted(k for k, v in p.SEARCH_SPACE.items() if v)))
    print("will NOT    : " + ", ".join(sorted(p.CARRIED)) + "\n"
          "              (carried means measured once and expected to transfer; moving "
          "one\n               here would invalidate that claim silently -- edit the "
          "profile instead)")
    print("decided     : " + ", ".join(f"{k}={v}" for k, v in p.DECIDED.items()) + "\n")

    if a.plan:
        for i, (name, params) in enumerate(q):
            tag = " <- noise floor" if name.startswith("seed") else ""
            print(f"--- {i+1}/{len(q)}  S_{name}{tag}")
            print(command(p, name, params, a.dataset) + "\n")
        f = p.CACHE / f"sweep_render_{p.SUBJECT}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"dataset": a.dataset, "rank_by": RANK_BY,
                                 "arms": {n: v for n, v in q}}, indent=1))
        print(f"plan recorded at {f}")
        return

    if not a.rank:
        raise SystemExit("pass --plan or --rank")

    # ---- rank what finished, refusing anything not scored on validation ---------
    rows, refused = [], []
    for name, _ in q:
        run = p.RUNS / f"S_{name}"
        ev = run / "evaluation.json"
        if not ev.exists():
            continue
        c = cfg_of(run)
        ds = pathlib.Path(c.get("source_path", "")).name
        if not ds.endswith("_sel"):
            refused.append((name, ds))
            continue
        e = json.load(open(ev))
        last = e[max(e, key=int)]
        rows.append((name, ds, last))

    for name, ds in refused:
        print(f"refused S_{name}: scored on '{ds}', which is not a validation dataset")
    if not rows:
        print("\nno arm has a validation score yet. Run the plan first.")
        return

    rows.sort(key=lambda r: r[2][RANK_BY], reverse=not LOWER_IS_BETTER)
    print(f"\n{'arm':28s}{RANK_BY:>9s}{'PSNR':>8s}{'SSIM':>8s}")
    for name, ds, m in rows:
        print(f"{'S_' + name:28s}{m[RANK_BY]:>9.4f}{m['PSNR']:>8.2f}{m['SSIM']:>8.4f}")

    seeds = [r for r in rows if r[0].startswith("seed")]
    print()
    if len(seeds) < 2:
        print("NO NOISE FLOOR YET. Until both seed arms have finished, no difference\n"
              "below is distinguishable from run-to-run variance -- do not pick a "
              "winner.")
        return
    floor = abs(seeds[0][2][RANK_BY] - seeds[1][2][RANK_BY])
    spread = abs(rows[0][2][RANK_BY] - rows[-1][2][RANK_BY])
    print(f"noise floor (two seeds, same settings): {floor:.4f} {RANK_BY}")
    print(f"best-to-worst spread across arms:       {spread:.4f} {RANK_BY}")
    if spread <= floor:
        print("\nTHE SEARCH FOUND NOTHING. The spread across settings is no wider than\n"
              "the spread between two identical runs. Report that, do not pick the top "
              "row.")
    else:
        best = rows[0]
        margin = abs(best[2][RANK_BY] - rows[1][2][RANK_BY])
        print(f"\nvalidation winner: S_{best[0]}, ahead of the next by {margin:.4f}"
              f"{'  -- inside the noise floor, so treat it as a tie' if margin <= floor else ''}")
        print("Score it on test ONCE, with tools/promote.py, and never re-search after.")


if __name__ == "__main__":
    main()
