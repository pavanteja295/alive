#!/usr/bin/env python3
"""Run STAR landmark detection over a prepared take, writing landmark2d/STAR.npz.

    ~/miniconda3/envs/vhap/bin/python detect_landmarks.py --seq <seq>

WHY THIS EXISTS AS A SEPARATE STEP
    VHAP normally detects landmarks lazily, inside the tracker (tracker.py:1271-1283), the
    first time a sequence is tracked. The chunking pipeline needs them much earlier than
    that -- `chunk_take.py` reads face box and confidence to classify every frame, long
    before anything is tracked -- so they have to be produced up front.

    This was invisible for the first two takes because both had been through an earlier
    full-video tracking run, which had generated STAR.npz as a side effect. The third take,
    the first prepared from scratch, failed on the missing file immediately. The lesson is
    the pipeline's, not the take's: a step that only works because of an artefact left by
    an abandoned experiment is not a step.

NOTE ON THE OUTPUT
    These landmarks are for CLASSIFICATION only -- which frames hold a usable face, where it
    sits, how big it is. They are deliberately NOT reused for tracking. Each chunk's tracker
    re-detects on the CROPPED frames, which is what fixes STAR picking the wrong face: after
    cropping to 3.2x the face there is nothing else in the picture to lock onto. See
    RECIPE.md step 7.
"""
import argparse, os, pathlib, sys, time

VHAP = pathlib.Path(os.environ.get("VHAP_ROOT",
    str(pathlib.Path(__file__).resolve().parents[1] / "vhap")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--njobs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    seqdir = VHAP / "data/monocular" / a.seq
    out = seqdir / "landmark2d/STAR.npz"
    n = len(list((seqdir / "images").glob("*.jpg")))
    if not n:
        sys.exit(f"no frames at {seqdir/'images'} -- run prepare_take.py first")
    if out.exists() and not a.force:
        print(f"{out} exists ({n} frames in images/), skipping")
        return

    sys.path.insert(0, str(VHAP))
    os.chdir(VHAP)                       # STAR loads its assets by relative path
    from vhap.config.base import DataConfig
    from vhap.data.video_dataset import VideoDataset
    from vhap.util.landmark_detector_star import annotate_landmarks

    cfg = DataConfig(root_folder=VHAP / "data/monocular", sequence=a.seq)
    cfg.use_landmark = False             # or the dataset tries to load what we are creating
    cfg.landmark_source = "star"
    t0 = time.time()
    print(f"{a.seq}: detecting landmarks over {n} frames, njobs={a.njobs}", flush=True)
    annotate_landmarks(VideoDataset(cfg=cfg, batchify_all_views=False), n_jobs=a.njobs)
    dt = time.time() - t0
    print(f"wrote {out}  ({dt:.0f}s, {dt/n*1000:.1f} ms/frame)")


if __name__ == "__main__":
    main()
