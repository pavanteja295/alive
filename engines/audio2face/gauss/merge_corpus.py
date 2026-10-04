#!/usr/bin/env python3
"""The exported chunks -> one dataset, split exactly the way face_v1 was split.

    ~/miniconda3/envs/stavatar/bin/python pipeline/gauss/merge_corpus.py

WHY NOT THE TRACKER'S OWN COMBINER
    Two reasons, one fatal. It refuses to combine folders whose names disagree before
    the first underscore, and our three recordings are named after their titles, so it
    stops. And it invents its own held-out split by taking one sequence in ten at random.

WHY THE SPLIT MATTERS MORE THAN IT LOOKS
    `rigfit/cache/split.json` is the split face_v1 was trained and judged on: 110 clips
    for training, 26 for validation, 26 held out, and 6 dropped so that a training clip
    and a test clip are never within five seconds of each other in the same recording.
    Reusing it exactly means the renderer's held-out clips are also the ones the audio
    model never saw, which is the only way the eventual photometric finetune stays
    honest. Inventing a second split here would quietly leak clips between the stages.

NOTHING IS COPIED. Frames point at '../<chunk>/...', which is the format's own way of
referring across folders, so the merged dataset is three JSON files.

Per-chunk cameras are preserved because each frame carries its own focal length and the
renderer reads its field of view per frame. See FINDINGS: the tracker fits a lens per
chunk and pinning one is only legal at tracking time.
"""
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = (pathlib.Path(__file__).resolve().parents[1] / "vhap/export/corpus")
# The split comes from the creator's profile (its audio-to-mesh recipe owns it). For
# drk it is rigfit/cache/split.json, the one face_v1 was trained and judged on.


PREFIX = ""


def load(chunk):
    p = CORPUS / (PREFIX + chunk) / "transforms.json"
    return json.load(open(p)) if p.exists() else None


def build(chunks, name):
    """One transforms file, timesteps renumbered so every frame in the corpus is unique."""
    db, base, kept, frames = None, 0, [], 0
    for c in chunks:
        d = load(c)
        if d is None:
            continue
        n = len(d["timestep_indices"])
        for f in d["frames"]:
            f.pop("timestep_index_original", None)
            f.pop("timestep_id", None)
            f["timestep_index"] += base
            for k in ("file_path", "fg_mask_path", "flame_param_path"):
                f[k] = str(pathlib.Path("..") / (PREFIX + c) / f[k])
        d["timestep_indices"] = [t + base for t in d["timestep_indices"]]
        base += n
        kept.append(c)
        frames += len(d["frames"])
        if db is None:
            db = d
        else:
            db["frames"] += d["frames"]
            db["timestep_indices"] += d["timestep_indices"]
    if db is None:
        return None, [], 0
    print(f"  {name:5s}  {len(kept):3d} chunks  {frames:6d} frames")
    return db, kept, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True,
                    help="recipes/mesh-to-render/profiles/<name>.py: whose split")
    ap.add_argument("--out", default=None,
                    help="default: the profile's CORPUS_TRACKED, or CORPUS_PREDICTED with --prefix")
    ap.add_argument("--limit", type=int, default=0, help="first N chunks of each split, for a smoke run")
    ap.add_argument("--prefix", default="", help="e.g. 'predicted__' to assemble the "
                                                 "predicted-mesh dataset instead")
    a = ap.parse_args()
    global PREFIX
    PREFIX = a.prefix

    sys.path.insert(0, str(HERE / "recipes/mesh-to-render/tools"))
    from _profile import load as load_profile
    _p = load_profile(a.profile)
    SPLIT = _p.SPLIT
    a.out = a.out or (_p.CORPUS_PREDICTED if a.prefix else _p.CORPUS_TRACKED)
    split = json.load(open(SPLIT))
    print(f"split.json: {len(split['train'])} train, {len(split['val'])} val, "
          f"{len(split['test'])} test, {len(split['dropped_for_guard'])} dropped for the guard band")

    # ONE SPLIT FOR THE WHOLE PIPELINE. NO STAGE SEES val OR test.
    #
    # This used to fold the validation clips into renderer training, on the argument
    # that they only chose the audio model's checkpoint and did not report its
    # number. That argument is wrong for a multi-stage system: a later stage
    # training on the clips an earlier stage was selected on means the pair of them
    # has, between them, seen the whole set. And it left the renderer with nothing
    # to select on but test, which is how a search ends up fitting the held-out set.
    #
    # The renderer has no third split of its own -- its "val" means a held-out
    # CAMERA, which a one-camera capture does not have -- so instead of inventing
    # one, two datasets are written from the same 110 training clips:
    #
    #   <out>_sel    train 110, held out = the pipeline's VALIDATION clips.
    #                Every search, every hyperparameter, every checkpoint choice.
    #   <out>        train 110, held out = the pipeline's TEST clips.
    #                Scored ONCE, for the model that already won on _sel.
    #
    # Both are gapless within themselves, so no reader changes. Training is 110
    # clips rather than 136: smaller, and the only version in which a number means
    # what it says.
    plans = {a.out + "_sel": {"train": split["train"], "test": split["val"]},
             a.out:          {"train": split["train"], "test": split["test"]}}

    # THE RULE, ASSERTED RATHER THAN INTENDED.
    #
    # No stage of a multi-stage pipeline may train on a clip any stage is selected
    # or scored on. Stated once, it is forgotten; asserted here, a regression stops
    # the merge instead of producing a dataset that looks fine.
    held = set(split["val"]) | set(split["test"])
    for out_name, plan in plans.items():
        leak = held & set(plan["train"])
        if leak:
            raise SystemExit(
                f"\nLEAK -- {out_name} would train on {len(leak)} clip(s) that the "
                f"pipeline holds out:\n"
                + "".join(f"    {c}\n" for c in sorted(leak)[:6]) +
                f"\n  One split serves every stage. A clip used to select or score "
                f"anywhere\n  is training data nowhere.\n")
    for out_name, plan in plans.items():
        _emit(CORPUS / out_name, plan, split, a)
    return


def _emit(out, plan, split, a):
    out.mkdir(parents=True, exist_ok=True)
    base = 0
    written = {}
    print(f"\n  {out.name}")
    for name in ("train", "test"):
        chunks = plan[name]
        if a.limit:
            chunks = chunks[:a.limit]
        db, kept, frames = build(chunks, name)
        if db is None:
            # The reader opens all three files, so an empty split still needs one.
            print(f"  {name:5s}  nothing exported yet -- writing an empty split")
            json.dump({"frames": [], "timestep_indices": [], "camera_indices": [0]},
                      open(out / f"transforms_{name}.json", "w"), indent=1)
            continue
        # timesteps must not collide between splits: they index one shared mesh table
        shift = base
        if shift:
            for f in db["frames"]:
                f["timestep_index"] += shift
            db["timestep_indices"] = [t + shift for t in db["timestep_indices"]]
        base = max(db["timestep_indices"]) + 1
        json.dump(db, open(out / f"transforms_{name}.json", "w"), indent=1)
        if name == "train":   # the reader opens all three; val is always empty here
            json.dump({"frames": [], "timestep_indices": [], "camera_indices": [0]},
                      open(out / "transforms_val.json", "w"), indent=1)
        written[name] = (len(kept), frames)
        (out / f"sequences_{name}.txt").write_text("\n".join(kept) + "\n")

    # The merged directory needs the marker too, or the renderer reads it as a plain
    # Blender dataset and silently loads no meshes at all.
    first = split["train"][0]
    import shutil
    marked = False
    for name in ("canonical_flame_param.npz", "canonical.npz"):
        src = CORPUS / (PREFIX + first) / name
        if src.exists():
            shutil.copy(src, out / name)
            print(f"canonical identity ({name}) taken from {first}")
            marked = True
    if not marked:
        raise SystemExit(f"no canonical marker in {CORPUS/(PREFIX+first)}; the renderer "
                         f"would read this as a plain Blender dataset and load no meshes")

    # COOKED MESHES MUST SAY WHICH DECODER MADE THEM, and every chunk must say the same
    # thing. Controls are only geometry through a particular controls->geometry map; a
    # dataset that mixes maps, or cannot name its own, trains a renderer against
    # geometry the audio model was never optimised to produce. The bare rig is still a
    # face, so nothing about the render would look wrong -- it just carries 0.75 mm of
    # held-out residual where the trained decoder carries 0.29 mm. Asserted here because
    # this is the last point where every chunk is in one place, and because a rule that
    # depends on someone remembering to run a checker is not a rule.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from decoder import one_decoder
    d0 = one_decoder(CORPUS / (PREFIX + c) for c in
                     set(split["train"]) | set(split["val"]) | set(split["test"]))
    if d0:
        json.dump(d0, open(out / "decoder.json", "w"), indent=1)
        print(f"decoder: all cooked chunks agree -- layer {d0.get('layer')!r}, "
              f"from {d0.get('ckpt')} step {d0.get('step')}")

    print(f"wrote {out}")
    for k, (c, f) in written.items():
        print(f"  transforms_{k}.json: {c} chunks, {f} frames")


if __name__ == "__main__":
    main()
