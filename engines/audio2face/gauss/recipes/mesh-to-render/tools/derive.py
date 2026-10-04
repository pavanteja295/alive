#!/usr/bin/env python3
"""Re-derive, for this dataset, the renderer values that have a rule.

    python tools/derive.py --profile drk --dataset corpus_all
    python tools/derive.py --profile drk --dataset corpus_all --write

THREE VALUES, AND WHY EACH HAS A RULE RATHER THAN A DEFAULT

  position_lr_max_steps   The schedule that decays the blob positions. It must span the
        run, and the run is `epochs x training frames` (train.py:137). The shipped
        default is 30,000. drk's corpus run was 242,988 iterations, so the positions
        were at their final rate -- one hundredth of the initial -- for 88% of training,
        and the perceptual term, which switches on at half the run, operated on
        positions that had effectively stopped moving. Nothing warns. This is not a
        tuning choice; a schedule shorter than its run is simply wrong.

  uv_size   The per-frame nudge network reads a UV map, and the region masks it is
        multiplied against exist on disk at fixed resolutions. Running at a size with
        no matching mask file either fails or silently resamples, and running at the
        larger size is four times the work AND makes two runs incomparable. The rule:
        use the resolution a mask file actually exists at, preferring the smallest, and
        say which file was found.

  threshold_xyz   How far a blob may slide inside its triangle before it is charged for
        it. A blob never changes which triangle drives it, so a slide bigger than the
        triangle means it is being animated by the wrong part of the face. In metres
        when METRIC_XYZ. The rule: the triangle scale of THIS topology. drk's 0.01 m
        was typed once against a 5,143-vertex FLAME head and then reused unchanged on a
        24,049-vertex head whose triangles are much smaller, where it permits a slide
        several triangles wide and so charges for almost nothing.

WHAT IS NOT HERE
  epochs, lambda_lpips: each costs a training run per value. That is tools/sweep.py.
  num_clusters: the code derives it from the data and overwrites whatever is passed
        (train.py:123), so it is not a parameter at all.

THREE VERDICTS, the same three the audio half uses:
  fine        the value in the profile already follows the rule
  auto-fixed  the rule gives a different value and the rule is not in doubt
  needs you   the rule cannot be evaluated, or gives something implausible
"""
import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _profile import load                                              # noqa: E402

VERDICT = {"fine": "fine", "fix": "auto-fixed", "ask": "needs you"}


def run_length(p, dataset):
    """epochs x training frames, which is what train.py sets opt.iterations to."""
    src = p.CORPUS / dataset
    tf = src / "transforms_train.json"
    if not tf.exists():
        return None, f"no transforms_train.json under {src}"
    n = len(json.load(open(tf))["frames"])
    ep = p.TRAIN.get("epochs") or p.SEARCH_SPACE["epochs"][0]
    return ep * n, f"{ep} epochs x {n:,} training frames"


def mask_resolutions(p):
    """The resolutions a uv region-mask file actually exists at, beside the assets."""
    out = {}
    for f in sorted(p.HEAD_ASSETS.parent.glob("uv_region_masks*.pkl")):
        try:
            with open(f, "rb") as fh:
                m = pickle.load(fh)
            a = next(iter(m.values())) if isinstance(m, dict) else m
            a = np.asarray(a)
            side = a.shape[-1] if a.ndim >= 2 else None
            if side:
                out[int(side)] = f.name
        except Exception as e:                                  # noqa: BLE001
            out.setdefault(-1, f"{f.name} unreadable: {e}")
    return out


def triangle_scale(p):
    """The median edge length of this topology, in metres. The blob's slide budget is
    a property of the mesh it is bound to, not of the person."""
    z = np.load(p.HEAD_ASSETS)
    V = z["canonical_m"].astype(np.float64)
    # MetaHuman's split-index scheme, which the renderer follows: `faces` indexes the
    # 24,408 UV LAYOUT entries, and pos_idx maps a layout entry to a position. Using
    # faces directly indexes past the end of the vertex array.
    F = z["pos_idx"].astype(np.int64)[z["faces"].astype(np.int64)]
    e = np.concatenate([V[F[:, 0]] - V[F[:, 1]],
                        V[F[:, 1]] - V[F[:, 2]],
                        V[F[:, 2]] - V[F[:, 0]]])
    L = np.linalg.norm(e, axis=1)
    return float(np.median(L)), float(L.mean()), float(np.percentile(L, 90))


def flame_triangle_scale(p):
    """The same rule as triangle_scale, for the OTHER topology this recipe trains on.
    A tracked dataset binds blobs to the 5,143-vertex FLAME head, whose triangles are
    nothing like the 24,049-vertex rig's, so the rig's median edge is the wrong budget
    there. Returns None if the FLAME head cannot be built, because a missing value is
    better than a borrowed one."""
    import os as _os, sys as _s
    cwd = _os.getcwd()
    try:
        # FlameHead resolves its .pkl by a path relative to the STAvatar tree.
        _s.path.insert(0, str(p.STAVATAR)); _os.chdir(p.STAVATAR)
        from flame_model.flame import FlameHead
        fh = FlameHead(300, 100, add_teeth=True)
        V = fh.v_template.detach().cpu().numpy().astype(np.float64)
        F = fh.faces.detach().cpu().numpy().astype(np.int64)
    except Exception as e:
        return None, f"could not build the FLAME head: {type(e).__name__}: {e}"
    finally:
        _os.chdir(cwd)
    e = np.concatenate([V[F[:, 0]] - V[F[:, 1]],
                        V[F[:, 1]] - V[F[:, 2]],
                        V[F[:, 2]] - V[F[:, 0]]])
    L = np.linalg.norm(e, axis=1)
    return (float(np.median(L)), len(V)), None


def deploy_camera(p):
    """The camera to render novel audio through: the one that ACTUALLY EXISTS whose view
    of this face is closest, in pixels, to every other camera in the training set.

    Deployment has no tracker, so it has no camera. Averaging the corpus's cameras is
    the wrong move -- rotations do not average elementwise, and the result is a camera
    nobody ever shot through. A medoid is a real camera, so anything it produces is a
    view this renderer has seen the like of.

    Distance is measured in PIXELS, by projecting this person's resting face through
    each pair and taking the mean disagreement. That avoids inventing a weighting
    between a focal-length difference and a pose difference, which have no common unit.

    TRAIN clips only. Deriving a deployment default from held-out footage would be a
    leak, small but free to avoid.
    """
    sp = json.load(open(p.SPLIT))
    train = set(sp["train"])
    cams = []
    for d in sorted(p.CORPUS.glob("*")):
        if d.name not in train or not (d / "transforms.json").exists():
            continue
        fr = json.load(open(d / "transforms.json"))["frames"][0]
        cams.append(dict(chunk=d.name, fl_x=float(fr["fl_x"]), fl_y=float(fr["fl_y"]),
                         w=int(fr["w"]), h=int(fr["h"]),
                         transform_matrix=fr["transform_matrix"]))
    if not cams:
        return None, "no TRAIN chunk under the corpus has a transforms.json"

    W = np.load(p.PIPE / f"identity/subjects/{p.SUBJECT}/beltrami_wrap.npz")
    V = W["verts_wrapped"].astype(np.float64)
    P = V[np.linspace(0, len(V) - 1, 400).astype(int)]

    def project(c):
        m = np.array(c["transform_matrix"], float)
        m[:3, 1:3] *= -1                      # the renderer's own convention
        w2c = np.linalg.inv(m)
        X = (P @ w2c[:3, :3].T) + w2c[:3, 3]
        z = np.clip(X[:, 2], 1e-6, None)
        return np.stack([c["fl_x"] * X[:, 0] / z + c["w"] / 2,
                         c["fl_y"] * X[:, 1] / z + c["h"] / 2], 1)

    U = np.stack([project(c) for c in cams])
    D = np.linalg.norm(U[:, None] - U[None, :], axis=-1).mean(-1)
    i = int(np.argmin(D.mean(1)))
    fl = np.array([c["fl_x"] for c in cams])
    best = dict(cams[i])
    best["_from_chunk"] = best.pop("chunk")
    return best, (f"{len(cams)} train cameras, focal spread "
                  f"{100*(fl.max()-fl.min())/np.median(fl):.1f}% of median; the medoid "
                  f"disagrees with the rest by {D.mean(1)[i]:.1f} px mean, half of them "
                  f"within {np.median(D[i]):.1f} px, p95 {np.percentile(D[i],95):.1f} px")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--dataset", default=None,
                    help="the merged dataset the run will train on; the run length "
                         "depends on how many frames it has. Default: the profile's "
                         "CORPUS_TRACKED")
    ap.add_argument("--write", action="store_true",
                    help="write cache/derived_render_<subject>.json. Without it this "
                         "only reports, which is the default on purpose")
    a = ap.parse_args()
    p = load(a.profile)
    a.dataset = a.dataset or p.CORPUS_TRACKED

    out, notes, ask = {}, [], []
    print(f"deriving the renderer's rule-bound values for {p.SUBJECT}, "
          f"dataset {a.dataset}\n")

    # ---- 1. the position schedule must span the run ------------------------------
    iters, how = run_length(p, a.dataset)
    print("position_lr_max_steps -- the schedule must span the run")
    if iters is None:
        print(f"  {VERDICT['ask']}: {how}")
        ask.append("position_lr_max_steps")
    else:
        cur = p.TRAIN.get("position_lr_max_steps")
        print(f"  run length  {iters:,}   ({how})")
        print(f"  shipped default 30,000, which covers "
              f"{100*30000/iters:.0f}% of that run")
        if cur == iters:
            print(f"  {VERDICT['fine']}: the profile already says {cur:,}")
            out["position_lr_max_steps"] = iters
        else:
            print(f"  {VERDICT['fix']}: {cur if cur else 'unset'} -> {iters:,}")
            out["position_lr_max_steps"] = iters
        out["_run_length"] = iters

    # ---- 2. uv_size must match a mask file that exists ---------------------------
    print("\nuv_size -- must match a region-mask file that exists on disk")
    res = mask_resolutions(p)
    good = sorted(k for k in res if k > 0)
    for k in sorted(res):
        print(f"  found {res[k]}" + (f" at {k}x{k}" if k > 0 else ""))
    if not good:
        print(f"  {VERDICT['ask']}: no readable uv_region_masks*.pkl beside "
              f"{p.HEAD_ASSETS.name}")
        ask.append("uv_size")
    else:
        pick = min(good)
        cur = p.TRAIN.get("uv_size")
        if cur in good and cur == pick:
            print(f"  {VERDICT['fine']}: {cur}")
            out["uv_size"] = cur
        elif cur in good:
            print(f"  {VERDICT['ask']}: {cur} has a mask file, but so does the smaller "
                  f"{pick}. The larger is four times the work; pick one and keep it, "
                  f"because two runs at different sizes are not comparable")
            ask.append("uv_size")
        else:
            print(f"  {VERDICT['fix']}: {cur if cur else 'unset'} -> {pick}")
            out["uv_size"] = pick

    # ---- 3. the slide budget is a property of the topology ----------------------
    print("\nthreshold_xyz -- a blob should not slide further than its own triangle")
    try:
        med, mean, p90 = triangle_scale(p)
    except Exception as e:                                      # noqa: BLE001
        print(f"  {VERDICT['ask']}: cannot read the topology: {e}")
        ask.append("threshold_xyz")
    else:
        print(f"  edge length on this {p.N_VERT_RIG:,}-vertex head: median "
              f"{med*1000:.2f} mm, mean {mean*1000:.2f} mm, p90 {p90*1000:.2f} mm")
        cur = p.TRAIN.get("threshold_xyz")
        want = round(med, 5)
        if cur is not None and abs(cur - want) / want < 0.25:
            print(f"  {VERDICT['fine']}: {cur} m is within a quarter of the median edge")
            out["threshold_xyz"] = cur
        else:
            print(f"  {VERDICT['fix']}: {cur if cur is not None else 'unset'} -> "
                  f"{want} m (the median edge)")
            if cur:
                print(f"      the carried {cur} m is {cur/med:.1f}x the median edge, so "
                      f"it charges for almost no slide on this topology")
            out["threshold_xyz"] = want
        out["_edge_median_m"] = round(med, 6)
        # ... and the same number for the FLAME topology, kept underscored so it stays
        # out of TRAIN. sweep.py substitutes it when the dataset is a tracked one.
        fl, why = flame_triangle_scale(p)
        if fl is None:
            print(f"  (FLAME slide budget not derived: {why})")
        else:
            fmed, fnv = fl
            print(f"  FLAME topology ({fnv:,} verts): median edge {fmed*1000:.2f} mm "
                  f"-> threshold_xyz {round(fmed, 5)} m on tracked datasets")
            out["_threshold_xyz_flame"] = round(fmed, 5)

    # ---- 4. the camera novel audio is rendered through -------------------------
    print("\ndeploy_camera -- novel audio has no camera, so one is chosen from the data")
    cam, how = deploy_camera(p)
    if cam is None:
        print(f"  {VERDICT['ask']}: {how}")
        ask.append("deploy_camera")
    else:
        print(f"  {how}")
        print(f"  {VERDICT['fix']}: {cam['_from_chunk'][:54]}")
        print(f"      fl {cam['fl_x']:.1f}, {cam['w']}x{cam['h']}")
        print("      head pose is NOT part of this: with no pose source the head sits "
              "where\n      this camera's own framing puts a neutral head. A pose "
              "generator plugs in\n      later, and until it does every inference video "
              "says so.")
        out["deploy_camera"] = cam

    # ---- the one that is a decision, restated so it is not forgotten ------------
    print("\nnot_finetune_flame_params -- decided, not derived")
    print(f"  the profile says {p.DECIDED['not_finetune_flame_params']}, on the rule "
          f"that geometry may\n  only be trained if the same freedom exists at "
          f"inference. It does not: those are\n  per-frame parameters and inference has "
          f"no photograph to fit them to.")
    print(f"  {p.SUBJECT}'s {p.RUN_TRACKED} ran with them FREE, so its number is the renderer "
          f"plus a\n  tracker repair, and is not comparable with a frozen run.")

    # ---- write ------------------------------------------------------------------
    print()
    if ask:
        print(f"{len(ask)} value(s) need you: {', '.join(ask)}")
    if not out or all(k.startswith("_") for k in out):
        print("nothing to change.")
    if a.write:
        f = p.CACHE / f"derived_render_{p.SUBJECT}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out, indent=1))
        print(f"wrote {f}")
    else:
        print("re-run with --write to record these; reporting only by default.")


if __name__ == "__main__":
    main()
