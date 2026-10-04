#!/usr/bin/env python3
"""Do this run's numbers make sense for this pipeline?

    python tools/sanity.py --profile drk --run C1_corpus

RELATIONS, NOT THRESHOLDS

"PSNR above 22" is a threshold: it encodes one creator's difficulty as a law, and it
passes a broken run on an easy face while failing a good run on a hard one. Every check
here is instead a relation between numbers this pipeline already produces, so it
transfers to a creator nobody has measured.

  1. the schedule spans the run
        position_lr_max_steps vs epochs x training frames. Short means the positions
        stopped moving early; 30,000 against drk's 242,988 froze them for 88% of it.

  2. the two evaluations were scored under DIFFERENT objectives
        The perceptual term switches on at half the run. So PSNR falling from the
        half-way evaluation to the final one while LPIPS also falls is the expected
        trade and not a regression. BOTH getting worse is a real problem, and this is
        the only reading of those two rows that is correct.

  3. the blob slide budget is smaller than a triangle
        A blob never changes which triangle drives it, so a budget several triangles
        wide charges for nothing. Compared against this topology's own median edge.

  4. training saw every clip the split gave it
        The dataset's frame count against the split's. A cook that produced 124 of 162
        chunks exits 0 and trains happily on the 124.

  5. the run and the dataset agree about geometry freedom
        A number quoted as the renderer's must come from a run that could not move its
        input meshes. The FLAME path's default is to move them.

  6. the run's own unit convention matches what its checkpoint will be read with
        metric_xyz changes what the blob offset MEANS. A checkpoint trained one way and
        rendered the other produced an exploded cloud scoring PSNR 20.7 against a white
        background -- a plausible-looking number for a completely broken render.

EVERY CHECK PRINTS THE TWO NUMBERS IT COMPARED. A check whose reasoning is invisible
gets believed when it is wrong, and this file has been wrong.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _profile import load                                              # noqa: E402

OK, WARN, BAD = "ok  ", "look", "BAD "


def cfg_of(run):
    f = run / "config.yml"
    if not f.exists():
        return {}
    y = yaml.safe_load(f.read_text()) or {}
    # The nested Optimization/Model blocks FIRST, then the top level, because the two
    # disagree: D1_predicted's nested block records mesh_assets as empty while the top
    # level records the real path. Flattening the other way made a rig-mesh run look
    # like a FLAME one, and this check then reported its geometry as trainable when the
    # class it used has no trainable geometry at all.
    flat = {}
    for v in y.values():
        if isinstance(v, dict):
            flat.update(v)
    flat.update({k: v for k, v in y.items() if not isinstance(v, dict)})
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    p = load(a.profile)
    run = p.RUNS / a.run
    if not run.exists():
        raise SystemExit(f"no run at {run}")
    c = cfg_of(run)
    src = pathlib.Path(c.get("source_path", ""))
    verdicts = []

    def say(tag, name, msg):
        verdicts.append(tag)
        print(f"  [{tag}] {name}\n        {msg}")

    print(f"{a.run}   source {src.name or '?'}\n")

    # ---- 1. the schedule spans the run -------------------------------------------
    tf = src / "transforms_train.json"
    frames = len(json.load(open(tf))["frames"]) if tf.exists() else None
    ep, mx = c.get("epochs"), c.get("position_lr_max_steps")
    if frames and ep and mx:
        iters = ep * frames
        if mx >= iters:
            say(OK, "position schedule spans the run",
                f"max_steps {mx:,} >= {ep} epochs x {frames:,} frames = {iters:,}")
        else:
            say(BAD, "position schedule ends inside the run",
                f"max_steps {mx:,} covers {100*mx/iters:.0f}% of {iters:,} iterations; "
                f"blob positions sat at their final rate for the rest")
    else:
        say(WARN, "position schedule", "the run or the dataset did not record enough "
                                       "to compute the run length")

    # ---- 2. the two evaluations, read correctly ---------------------------------
    ev = run / "evaluation.json"
    if ev.exists():
        e = {int(k): v for k, v in json.load(open(ev)).items()}
        ks = sorted(e)
        if len(ks) >= 2:
            half, full = e[ks[-2]], e[ks[-1]]
            dp = full["PSNR"] - half["PSNR"]
            dl = full["LPIPS"] - half["LPIPS"]
            both_worse = dp < 0 and dl > 0
            msg = (f"PSNR {half['PSNR']:.2f} -> {full['PSNR']:.2f} ({dp:+.2f}), "
                   f"LPIPS {half['LPIPS']:.4f} -> {full['LPIPS']:.4f} ({dl:+.4f})")
            if both_worse:
                say(BAD, "the second half made everything worse", msg)
            elif dp < 0:
                say(OK, "PSNR traded for perceptual quality, as the objective intends",
                    msg + "\n        the perceptual term switches on at half the run, "
                          "so these two rows\n        were scored under different "
                          "objectives and PSNR falling is expected")
            else:
                say(OK, "the second half improved both", msg)
        else:
            say(WARN, "evaluations", f"only one row in {ev.name}, nothing to compare")
    else:
        say(WARN, "evaluations", "no evaluation.json; the run has not been scored")

    # ---- 3. the slide budget against this topology ------------------------------
    th, med = c.get("threshold_xyz"), p.DERIVED_ALL.get("_edge_median_m")
    if th is not None and med:
        r = th / med
        if r <= 1.0:
            say(OK, "blob slide budget is within a triangle",
                f"threshold_xyz {th*1000:.2f} mm vs median edge {med*1000:.2f} mm "
                f"({r:.2f}x)")
        else:
            say(WARN, "blob slide budget is wider than a triangle",
                f"threshold_xyz {th*1000:.2f} mm is {r:.1f}x the median edge "
                f"{med*1000:.2f} mm, so the hinge charges for almost no slide")
    else:
        say(WARN, "blob slide budget",
            "run tools/derive.py first; this compares against the derived edge length")

    # ---- 4. training saw every clip the split gave it --------------------------
    sp = p.SPLIT
    seq = src / "sequences_train.txt"
    if sp.exists() and seq.exists():
        s = json.load(open(sp))
        got = {x for x in seq.read_text().split("\n") if x.strip()}
        tr, va, te = set(s["train"]), set(s["val"]), set(s["test"])
        n_tr, n_va, n_te = len(got & tr), len(got & va), len(got & te)
        if n_te:
            say(BAD, "the TEST clips are in the training set",
                f"{n_te} of them. Every number from this run is on seen data")
        elif n_va and n_tr == len(tr):
            say(WARN, "the validation clips are in the training set",
                f"trained on all {n_tr} train + {n_va} val = {len(got)} clips, holding "
                f"out only the {len(te)} test clips.\n        This run therefore has NO "
                f"validation set, so nothing about it can be\n        chosen without "
                f"reading test. Train on the '_sel' dataset to search.")
        elif n_tr < len(tr):
            say(BAD, "the dataset is missing training clips",
                f"{n_tr} of the split's {len(tr)}. A partial cook exits 0 and trains "
                f"on whatever arrived")
        else:
            say(OK, "trained on exactly the split's training clips",
                f"{n_tr} of {len(tr)}, with val and test both held out")
    else:
        say(WARN, "clip count", "no sequences_train.txt or no split.json to compare")

    # ---- 5. geometry freedom ---------------------------------------------------
    nff = c.get("not_finetune_flame_params")
    is_rig = bool(c.get("mesh_assets"))
    if is_rig:
        say(OK, "geometry was frozen",
            "this run binds to the rig mesh, which has no trainable geometry at all "
            f"(the recorded not_finetune_flame_params={nff} is read by nothing)")
    elif nff:
        say(OK, "geometry was frozen", "not_finetune_flame_params=True, so this number "
                                       "is the renderer's own")
    else:
        say(WARN, "geometry was FREE during training",
            "expression, pose and translation were fitted to the training images, so "
            "this\n        number is the renderer plus a tracker repair. It is not "
            "comparable with a\n        frozen run, and the freedom does not exist at "
            "inference")

    # ---- 6. the unit convention -------------------------------------------------
    mxz = c.get("metric_xyz")
    if mxz is None:
        say(WARN, "blob offset unit", "the run does not record metric_xyz")
    elif bool(mxz) == bool(p.METRIC_XYZ):
        say(OK, "blob offset unit matches the profile",
            f"metric_xyz={mxz}; read this checkpoint with the same setting or the "
            f"cloud explodes")
    else:
        say(BAD, "blob offset unit disagrees with the profile",
            f"run says metric_xyz={mxz}, profile says {p.METRIC_XYZ}. These are "
            f"different units, not different settings")

    # ---- 7. the audio model we consume is ITS recipe's validation winner -------
    # A relation between two things already on disk: the release this recipe points at,
    # and whether that release's own manifest records it losing on validation. The
    # correction that face_v2 was test-selected sat in its manifest for a day while
    # every downstream consumer went on reading it, because nothing compared the two.
    man = p.AUDIO_MODEL_DIR / "MANIFEST.json"
    if not man.exists():
        say(WARN, "the audio model records no manifest", f"{man} is absent")
    else:
        mj = json.load(open(man))
        corr = [k for k in mj if k.upper().startswith("CORRECTION")]
        if corr:
            c = mj[corr[0]]
            say(BAD, f"the audio model {p.AUDIO_MODEL} records a correction against "
                     f"itself",
                f"{corr[0]}: {str(c.get('what', c))[:90]}\n        Point the profile's "
                f"RELEASE at the validation winner instead.")
        else:
            say(OK, f"the audio model {p.AUDIO_MODEL} records no correction against "
                    f"itself", f"manifest: {mj.get('what', '')[:70]}")

    n_bad, n_warn = verdicts.count(BAD), verdicts.count(WARN)
    print(f"\n{verdicts.count(OK)} ok, {n_warn} to look at, {n_bad} bad")
    if n_bad:
        print("A bad relation is not a bad model: it is a number that cannot be read "
              "as\nintended. Fix the run, not the check -- unless the check is wrong, "
              "which\nhas happened.")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
