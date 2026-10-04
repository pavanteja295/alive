#!/usr/bin/env python3
"""What is done, what is next, and the scope this recipe may write.

    python tools/status.py --profile drk
    python tools/status.py --profile drk --next     # just the next command

STATUS IS COMPUTED FROM DISK, NEVER WRITTEN INTO PROSE.
    A stale status line is skimmed by a person and believed by a worker. Every row below
    is derived by looking at the artefact itself, so an interrupted run, a deleted cache
    or a half-finished export all show up as what they are rather than as what some note
    claims. Nothing here writes state.

WHAT IT CANNOT DECIDE
    Whether the render looks like him, whether the overlay sits on his face, whether the
    sync is right. Those are the three inspection layers in the recipe and they need eyes.
    Stages that depend on them are reported as `look` rather than `done`.
"""
import argparse, json, pathlib, sys

P = pathlib.Path(__file__).resolve().parents[4]              # pipeline/
RIG = P / "rigfit"
GAUSS = P / "gauss"
VHAP = (pathlib.Path(__file__).resolve().parents[4] / "vhap")
CORPUS = VHAP / "export/corpus"


def n_files(d, pat="*"):
    return len(list(d.glob(pat))) if d.is_dir() else 0


def clips(p):
    """The canonical split, or None if it has not been written yet."""
    f = p.SPLIT
    if not f.exists():
        return None
    s = json.load(open(f))
    return {"train": s["train"], "val": s["val"], "test": s["test"],
            "all": s["train"] + s["val"] + s["test"]}


def merged_ok(d):
    """A merged dataset is usable only if its timesteps are gapless and paths resolve."""
    if not (d / "transforms_train.json").exists():
        return False, "not written"
    if not any((d / m).exists() for m in ("canonical.npz", "canonical_flame_param.npz")):
        return False, "NO CANONICAL MARKER -- the reader would load no meshes at all"
    ts, n = set(), 0
    for split in ("train", "test"):
        f = d / f"transforms_{split}.json"
        if not f.exists():
            return False, f"transforms_{split}.json missing"
        db = json.load(open(f))
        ts |= set(db["timestep_indices"])
        n += len(db["frames"])
        # Name the missing DIRECTORY, not the path with the dot-dots still in it. A
        # merged dataset refers across folders, so when a chunk has been deleted the
        # literal path is unreadable and the real cause -- one folder gone -- is buried.
        for fr in db["frames"][:1] + db["frames"][-1:]:
            for k in ("file_path", "fg_mask_path", "flame_param_path"):
                if not (d / fr[k]).exists():
                    ref = (d / fr[k]).parts
                    owner = pathlib.PurePosixPath(fr[k]).parts[1]
                    if not (d.parent / owner).is_dir():
                        return False, f"the chunk directory {owner} no longer exists"
                    return False, f"{k} does not resolve: {fr[k]}"
    if ts and len(ts) != max(ts) + 1:
        return False, f"timesteps have holes: {len(ts)} distinct, max {max(ts)}"
    return True, f"{n:,} frames, timesteps gapless 0..{max(ts) if ts else -1}"


def run_done(d):
    """A training run is done when it has scored on test at its final iteration."""
    ev = d / "evaluation.json"
    if not ev.exists():
        pcs = d / "point_cloud"
        return False, ("started, no evaluation yet" if pcs.is_dir() else "not started")
    e = json.load(open(ev))
    last = max(int(k) for k in e)
    m = e[str(last)]
    return True, (f"iter {last:,}  PSNR {m['PSNR']:.2f}  SSIM {m['SSIM']:.4f}  "
                  f"LPIPS {m['LPIPS']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--next", action="store_true", help="print only the next command")
    a = ap.parse_args()
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from _profile import load
    p_ = load(a.profile)
    s = p_.SUBJECT
    a.audio_model = p_.AUDIO_MODEL
    a.stage_a, a.stage_b = p_.RUN_TRACKED, p_.RUN_PREDICTED
    rows = []   # (name, state, detail, command-if-not-done)

    def add(name, ok, detail, cmd=None, state=None):
        rows.append((name, state or ("done" if ok else "TODO"), detail, None if ok else cmd))

    # Upstream stages are NOT reported here. They belong to recipes/audio-to-mesh,
    # and checks.py already gates them as preconditions. A status tool that reports
    # on another recipe's stages is the same scope creep it exists to prevent: two
    # tools describing one artifact, and neither owning it.
    sp = clips(p_)
    N = len(sp["all"]) if sp else None

    # ---- the renderer ----------------------------------------------------------
    have = sum(1 for c in (sp["all"] if sp else []) if (CORPUS / c / "transforms_train.json").exists())
    add("chunks exported", sp is not None and have == N,
        f"{have}/{N} of the split" if sp else "split unknown",
        "bash gauss/export_all.sh")

    # this person's placements: cache/rigid holds every creator's chunks
    _rg = GAUSS / "cache/rigid"
    nr = (sum(1 for f in _rg.glob("*.npz") if f.stem.split("__c")[0] in p_.RECORDINGS)
          if _rg.is_dir() else 0)
    add("head placements", sp is not None and nr >= N, f"{nr} chunks",
        f"{p_.ENV_VHAP}/bin/python gauss/head_rigid.py")

    # THE VALUES WITH RULES, before anything is trained. The shipped position
    # schedule covers 12% of a corpus run, and nothing warns.
    dv = GAUSS / f"cache/derived_render_{s}.json"
    add("parameters derived", dv.exists(),
        (f"schedule {json.load(open(dv)).get('position_lr_max_steps')}, "
         f"camera chosen" if dv.exists() else "not derived"),
        f"{p_.ENV}/bin/python gauss/recipes/mesh-to-render/tools/derive.py --profile {a.profile} --write")

    # THE JOINS, before the hours. A misaligned model is still a model.
    # verify_joins stamps WHAT was verified, so the acknowledgement expires by itself
    # when those inputs change. Reading only that the file exists throws that away and
    # reports a statement about yesterday's inputs as if it were about today's.
    ak = GAUSS / f"cache/joins_render_acked_{s}.json"
    if ak.exists():
        from verify_joins import stamp as _joins_stamp
        was = json.load(open(ak)).get("stamp")
        now = _joins_stamp(p_)
        fresh = was == now
        det = ("acknowledged on disk" if fresh else
               "STALE -- acknowledged against inputs that have since changed")
    else:
        fresh, det = False, "nobody has looked"
    add("joins verified and looked at", fresh, det,
        f"{p_.ENV}/bin/python gauss/recipes/mesh-to-render/tools/verify_joins.py --profile {a.profile}")

    # ---- the deployed recipe: the teeth rig ----------------------------------------
    STA = f"{p_.ENV}/bin/python"
    ear = GAUSS / f"cache/ear_{s}.npz"
    add("blinks measured", ear.exists(), ear.name if ear.exists() else "not measured",
        f"{STA} gauss/build_ear.py --subject {s}")

    pre = p_.COOK_PREFIX
    have = sum(1 for c in (sp["all"] if sp else [])
               if (CORPUS / f"{pre}__{c}" / "transforms.json").exists())
    add(f"{pre} meshes cooked from {a.audio_model}", sp is not None and have == N,
        f"{have}/{N} of the split" if sp else "split unknown",
        f"{STA} gauss/cook_predicted.py --subject {s} --out {pre} "
        f"--assets {p_.DEPLOY_ASSETS.relative_to(P)} --blink ear")

    ok, det = merged_ok(CORPUS / p_.CORPUS_DEPLOY)
    add(f"{pre} dataset ({p_.CORPUS_DEPLOY})", ok, det,
        f"{STA} gauss/merge_corpus.py --profile {a.profile} "
        f"--prefix {pre}__ --out {p_.CORPUS_DEPLOY}")

    trn = CORPUS / p_.CORPUS_DEPLOY_TRAIN
    add(f"frontal training set ({p_.CORPUS_DEPLOY_TRAIN})",
        (trn / "transforms_train.json").exists(),
        "written" if (trn / "transforms_train.json").exists() else "not written",
        f"{STA} gauss/filter_frontal.py --subject {s} "
        f"--src {p_.CORPUS_DEPLOY}_sel --out {p_.CORPUS_DEPLOY_TRAIN}")

    ok, det = run_done(GAUSS / "runs" / p_.RUN_DEPLOY)
    add(f"renderer ({p_.RUN_DEPLOY})", ok, det,
        f"cd {p_.STAVATAR} && {STA} train.py -s {trn} -m {GAUSS / 'runs' / p_.RUN_DEPLOY} "
        f"--mesh_assets {p_.DEPLOY_ASSETS} " + " ".join(p_.DEPLOY_TRAIN_ARGS),
        state="look" if ok else None)

    # THE SEARCH, on validation only, and the noise floor before any comparison.
    # this person's: a validation dataset belongs to whoever's clips it was merged from
    def _mine(d):
        f = d / "sequences_train.txt"
        return f.exists() and any(ln.split("__c")[0] in p_.RECORDINGS
                                  for ln in f.read_text().split()[:1])
    sel = [x for x in CORPUS.glob("corpus_*_sel") if _mine(x)]
    add("validation datasets built", bool(sel),
        f"{', '.join(x.name for x in sel)}" if sel else
        "none -- nothing can be selected without reading test",
        f"{STA} gauss/merge_corpus.py --profile {a.profile} "
        f"--prefix {pre}__ --out {p_.CORPUS_DEPLOY}")

    def _run_mine(d):
        cfg = d / "config.yml"
        if not cfg.exists():
            return False
        src = next((ln.split(":", 1)[1].strip() for ln in cfg.read_text().splitlines()
                    if ln.startswith("source_path:")), "")
        return bool(src) and _mine(pathlib.Path(src))
    seeds = [d for d in (GAUSS / "runs").glob("*seed*")
             if (d / "evaluation.json").exists() and _run_mine(d)]
    add("noise floor measured", len(seeds) >= 2,
        f"{len(seeds)}/2 seed arms scored" if seeds else
        "0/2 -- until both land, no comparison is distinguishable from variance",
        f"{STA} gauss/recipes/mesh-to-render/tools/sweep.py --profile {a.profile} --plan")

    # this person's bundles: a manifest names its subject
    rel = [m for m in p_.RELEASE.glob("*/MANIFEST.json")
           if json.load(open(m)).get("subject") == s] if p_.RELEASE.exists() else []
    add("promoted", bool(rel),
        f"{', '.join(x.parent.name for x in rel)}" if rel else "nothing released",
        f"{STA} gauss/recipes/mesh-to-render/tools/promote.py --profile {a.profile} --run <validation winner> "
        f"--name render_v1 --wrong '<what is still wrong>'")

    nxt = next((c for _, st, _, c in rows if st == "TODO" and c), None)
    if a.next:
        print(nxt or "# nothing to do -- every computable stage is done; the inspection layers "
                     "under 'Done criterion' in RECIPE.md still need eyes")
        return 0 if nxt is None else 0

    print(f"\n  SCOPE -- read anything, write only what is listed as produced.\n"
          f"  Anything else belongs to another recipe: fail and name it, do not build it.\n")
    for what, owner in (("tracked chunks and frames", "recipes/face-clips"),
                        (f"the audio model {a.audio_model}", "recipes/audio-to-mesh"),
                        ("split, lag, frame timings", "recipes/audio-to-mesh"),
                        (f"{s}'s rig and wrap", "the identity work")):
        print(f"    required  {what:<30} <- {owner}")
    print(f"    produced  export/corpus/<chunk>/, {p_.COOK_PREFIX}__<chunk>/ (+decoder.json),")
    print(f"              {p_.CORPUS_DEPLOY}(_sel), {p_.CORPUS_DEPLOY_TRAIN}, cache/rigid/,")
    print(f"              cache/derived_render_*, sweep_render_*, joins_*,")
    print(f"              gauss/runs/<name>/, checkpoints/{s}/face/render/<name>/")
    print(f"    foreign   everything else, including anything under rigfit/\n")
    w = max(len(r[0]) for r in rows)
    print(f"  {s}   audio model {a.audio_model}\n")
    for name, st, det, _ in rows:
        mark = {"done": "  ok  ", "look": " LOOK ", "TODO": " TODO ",
                "opt": " opt  "}[st]
        print(f"  [{mark}] {name:<{w}}  {det}")
    print()
    if nxt:
        print(f"  next:\n    {nxt}\n")
    else:
        print("  Every computable stage is done.\n"
              "  Still required by the recipe and NOT computable:\n"
              "    - the mesh overlay has been looked at        (overlay_check.py)\n"
              "    - a held-out clip watched WITH SOUND         (tools/infer.py --clip)\n"
              "  A run is not done until a person or a worker has looked.\n")
    print("  LOOK = trained and scored, but the inspection layers under\n        'Done criterion' in RECIPE.md are not computable.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
