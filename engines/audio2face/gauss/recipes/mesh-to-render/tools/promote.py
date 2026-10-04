#!/usr/bin/env python3
"""Assemble a deployable audio-to-render model, from disk, with its faults recorded.

    python tools/promote.py --profile drk --run S_seed1 --name render_v1 \
        --wrong "the lips still do not fully close on plosives"
    python tools/promote.py --profile drk --deployed \
        --wrong "..."                     # the profile's RUN_DEPLOY, signed off by eye

Writes checkpoints/<subject>/face/render/<name>/: the final iteration only, in the
layout the live app and tools/infer.py load (point_cloud/iteration_N, param/iteration_N).

WHAT A RELEASE OF THIS RECIPE IS

Not a checkpoint. The deployed thing is a CHAIN, and every link has to travel with it
or the release is not runnable:

    audio -> controls        the audio-to-mesh release (its own recipe's artefact)
    controls -> geometry     the rig and the corrective layer -- the DECODER, read out
                             of the audio model's checkpoint, never guessed
    geometry -> pixels       this run's Gaussians, its config, its nudge network
    and a camera             deployment has no tracker, so the camera and the head
                             pose have to come from somewhere explicit

A release that records only the last of those is the bug this pipeline already had:
stage B rendered a model trained through one decoder using another, silently, for the
whole of stage B.

WHAT IT REFUSES

  a run with no validation score      Selection happens on validation. A run scored
        only on test cannot be shown to be the winner, so promoting it is selecting on
        test after the fact. The existing drk runs are all in this state, because they
        trained on train PLUS validation.

  a run that is not the validation winner    unless --anyway with a written reason.

  a run whose sanity relations are bad       the numbers cannot be read as intended.

  --wrong missing      Every real release has faults. One with none recorded is one
        nobody checked, and the first face release here shipped with lips that never
        closed while its manifest said nothing.

The manifest is ASSEMBLED FROM DISK, never typed. A number in prose is a number that
has already drifted from the run it describes.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3] / "gauss"))
from _profile import load, run_assets                                  # noqa: E402
from decoder import resolve                                            # noqa: E402


def sha(p, n=16):
    p = pathlib.Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def cfg_of(run):
    y = yaml.safe_load((run / "config.yml").read_text()) or {}
    flat = {}
    for v in y.values():
        if isinstance(v, dict):
            flat.update(v)
    flat.update({k: v for k, v in y.items() if not isinstance(v, dict)})
    return flat


def latest_pc(run):
    d = run / "point_cloud"
    its = sorted((x for x in d.glob("iteration_*")),
                 key=lambda x: int(x.name.split("_")[1])) if d.exists() else []
    return its[-1] if its else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--run", help="a run under gauss/runs (default: the profile's RUN_DEPLOY)")
    ap.add_argument("--name", help="the release's name (default: the run's)")
    ap.add_argument("--deployed", action="store_true",
                    help="release the profile's deployed run. It trained on train+validation "
                         "and was chosen by looking at it live, not by a validation sweep, "
                         "so the sweep checks below do not apply; --wrong still does")
    ap.add_argument("--wrong", action="append", default=[],
                    help="what is still wrong with it. Required, repeatable")
    ap.add_argument("--anyway", default="",
                    help="promote a run that is not the validation winner, and say why")
    a = ap.parse_args()
    p = load(a.profile)
    a.run = a.run or (p.RUN_DEPLOY if a.deployed else None)
    if not a.run:
        raise SystemExit("--run, or --deployed for the profile's RUN_DEPLOY")
    a.name = a.name or a.run
    run = p.RUNS / a.run
    if not run.exists():
        raise SystemExit(f"no run at {run}")
    if not a.wrong:
        raise SystemExit(
            "--wrong is required.\n"
            "  Every real release has faults, and one with none recorded is one nobody\n"
            "  checked. face_v1 shipped with lips that never closed -- 4.06 mm open on\n"
            "  frames where his were shut -- and its first manifest said nothing.")

    c = cfg_of(run)
    src = pathlib.Path(c.get("source_path", ""))

    # ---- was it selected on validation? -----------------------------------------
    if not a.deployed and not src.name.endswith("_sel"):
        raise SystemExit(
            f"{a.run} was scored on '{src.name}', which is not a validation dataset.\n"
            f"  Its numbers are test numbers, so promoting it IS selecting on test.\n"
            f"  Train the candidate on the '_sel' dataset, rank with tools/sweep.py,\n"
            f"  and score the winner on test once -- in that order.")

    # ---- is it the validation winner? -------------------------------------------
    import sweep
    q = sweep.arms(p)
    rows = []
    for name, _ in q:
        r = p.RUNS / f"S_{name}"
        ev = r / "evaluation.json"
        if ev.exists() and pathlib.Path(cfg_of(r).get("source_path", "")).name \
                .endswith("_sel"):
            e = json.load(open(ev))
            rows.append((f"S_{name}", e[max(e, key=int)][sweep.RANK_BY]))
    rows.sort(key=lambda r: r[1], reverse=not sweep.LOWER_IS_BETTER)
    if rows and rows[0][0] != a.run and not a.anyway and not a.deployed:
        raise SystemExit(
            f"{a.run} is not the validation winner. {rows[0][0]} is, on "
            f"{sweep.RANK_BY} {rows[0][1]:.4f} against {a.run}'s "
            f"{dict(rows).get(a.run, float('nan')):.4f}.\n"
            f"  Promote the winner, or pass --anyway with the reason and it goes in the\n"
            f"  manifest where a reader can disagree with it.")

    # ---- do its relations hold? --------------------------------------------------
    s = subprocess.run([sys.executable, str(HERE / "sanity.py"),
                        "--profile", a.profile, "--run", a.run],
                       capture_output=True, text=True)
    if s.returncode != 0 and a.deployed:
        print(s.stdout)
        print("  (a deployed run: released anyway, and the failed relation goes in the manifest)")
    elif s.returncode != 0:
        print(s.stdout)
        raise SystemExit(
            f"{a.run} has a bad sanity relation, so at least one of its numbers cannot\n"
            f"  be read as intended. Fix the run, then promote.")

    # ---- the decoder, from the audio model rather than from memory --------------
    dec = resolve(p.AUDIO_MODEL_DIR)

    # ---- assemble ---------------------------------------------------------------
    dst = p.RELEASE / a.name
    dst.mkdir(parents=True, exist_ok=True)
    pc = latest_pc(run)
    if pc is None:
        raise SystemExit(f"{run} has no point_cloud/iteration_* to release")
    for f in ("config.yml", "cfg_args", "evaluation.json"):
        if (run / f).exists():
            shutil.copy(run / f, dst / f)
    # THE FINAL ITERATION ONLY, in the layout its readers expect: point_cloud/iteration_N
    # and param/iteration_N. Copying pc's contents straight into point_cloud/ (as this
    # did) produced a release nothing could load.
    shutil.copytree(pc, dst / "point_cloud" / pc.name, dirs_exist_ok=True)
    if (run / "param" / pc.name).exists():
        shutil.copytree(run / "param" / pc.name, dst / "param" / pc.name, dirs_exist_ok=True)

    ev = json.load(open(run / "evaluation.json")) if (run / "evaluation.json").exists() else {}
    val = ev[max(ev, key=int)] if ev else {}
    man = {
        "name": a.name,
        "what": "audio -> this person's rendered face. One chain, four links.",
        "subject": p.SUBJECT,
        "promoted_from": {"run": a.run, "dataset": src.name,
                          "iteration": int(pc.name.split("_")[1])},
        "selected_on": ("deployed: trained on train+validation, chosen by looking at it live"
                        if a.deployed else
                        "validation (the dataset's test split is the split's val clips)"),
        "sanity_relations_hold": s.returncode == 0,
        "validation": val,
        "test": "not scored yet -- score the promoted bundle on the non-_sel dataset "
                "ONCE, and do not search afterwards",
        "chain": {
            "audio_to_controls": {"release": p.AUDIO_MODEL,
                                  "best_pt_sha": sha(p.AUDIO_MODEL_DIR / "best.pt")},
            "controls_to_geometry": {"rig": dec.get("rig"),
                                     "rig_sha": dec.get("rig_sha"),
                                     "layer": dec.get("layer"),
                                     "layer_sha": dec.get("layer_sha"),
                                     "read_from": dec.get("_from", "best.pt[decoder]")},
            "geometry_to_pixels": {"run": a.run,
                                   "mesh_assets": pathlib.Path(c.get("mesh_assets") or p.HEAD_ASSETS).name,
                                   "mesh_assets_sha": sha(run_assets(p, c.get("mesh_assets"))),
                                   "metric_xyz": c.get("metric_xyz"),
                                   "uv_size": c.get("uv_size")},
            "camera_and_pose": {
                "camera": "the deploy camera in data/face/gauss/cache/derived_render_<subject>.json; "
                          "on a real body, each body clip's own tracked camera",
                "head_pose": "on a real body, the body clip's own head track, replayed; "
                             "otherwise the profile's INFER pose policy",
            },
        },
        "settings": {"carried": p.CARRIED, "decided": p.DECIDED,
                     "derived": p.DERIVED},
        "known_wrong": a.wrong,
        "promoted_anyway": a.anyway or None,
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True,
                                 cwd=p.PIPE).stdout.strip() or None,
    }
    (dst / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(f"released {dst}")
    if val:
        print(f"  validation {sweep.RANK_BY} {val[sweep.RANK_BY]:.4f}, "
              f"PSNR {val['PSNR']:.2f}, SSIM {val['SSIM']:.4f}")
    print(f"  decoder    layer {dec.get('layer')!r} (from {dec.get('_from', 'best.pt')})")
    print(f"  known wrong: {len(a.wrong)} recorded")
    print("  serve it: ./alive up <creator> reads it from here")


if __name__ == "__main__":
    main()
