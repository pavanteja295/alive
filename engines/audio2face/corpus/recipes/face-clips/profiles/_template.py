"""TEMPLATE. Copy to profiles/<creator>.py and change EVERY declaration.

Nothing here is a safe default. Every value below was read off ONE creator's
histograms (healthygamer) and take 2 proved the gaps move between videos of the
same person, let alone between people.

Working through this file IS the first step of a new creator. Do not run the
pipeline against a half-edited profile: the failure is a plausible shot list,
not an error.

Every value names the file or measurement it was read from. Values marked
RE-DERIVE must be measured on each new creator, not carried over -- every one of
them was read off a gap in a histogram, and the gap moves.
"""
import pathlib

# ------------------------------------------------------------------- machines
# Absolute because the VHAP install is not relocatable: FlameHead loads
# 'asset/flame/flame2023.pkl' by relative path, so the tracker must be run with
# the VHAP root as cwd. Tools do that chdir themselves.
VHAP = (pathlib.Path(__file__).resolve().parents[4] / "vhap")  # machine-level, rarely changes
TAKES = pathlib.Path(__file__).resolve().parents[4] / "takes" / "CHANGEME"   # where source.mkv per take lives

# insightface lives in its OWN conda env, deliberately. The vhap env has a
# hand-built nvdiffrast and pytorch3d; adding onnxruntime-gpu to it is not worth
# the risk. run_face_scan.sh wraps the env switch and the LD_LIBRARY_PATH.
ENV_VHAP = pathlib.Path.home() / "miniconda3/envs/vhap"
ENV_FACEID = pathlib.Path.home() / "miniconda3/envs/faceid"

# ------------------------------------------------------------------ the takes
# 17 videos downloaded; 3 processed. Frame counts are at the extracted rate, not
# the source rate. Formats differ enough that they are listed: a new creator's
# formats will differ again, and the yield column is the thing that varies most.
#
#   Record format and yield per take as you go. Yield varies more than
#   anything else: 93.7% / 69.1% / 56.4% across three takes of one creator.
SUBJECTS = [
    # "<take_dir_name>",   one per video, matching TAKES/<name>/source.mkv
]

# ------------------------------------------------------------- frame extraction
# min(native, MAX_FPS), never a forced constant. The corpus is 23.976 / 30.002 /
# 60 fps across takes; forcing 30 onto the 23.976 take DUPLICATES frames, which
# fabricates zero-motion frames and hands the tracker fake temporal smoothness.
MAX_FPS = 30.0
WIDTH = 1280          # source is 1920 or 2560 wide; 1280 is where the face still
                      # clears 180px in a full-frame shot

# ------------------------------------------------------------------ clustering
# A starting point, not a tuned value. More clusters cost only reading time.
# Per-take k and --subcluster arguments are recorded in each take's
# work/<take>/cluster_config.json, because re-clustering renumbers ids and a
# stale verdicts.json then silently labels the wrong frames.
CLUSTER_K = 28
SUBCLUSTER_K = 3      # when a montage comes back mixed, split just that cluster

# -------------------------------------------------- RE-DERIVE: every one of these
# Read off a distribution on THIS creator. On a new creator, plot the
# distribution and find the gap; do not reuse the number.

# alpha coverage below which a frame holds no real person. Gap on take 1 sat
# between 0.056-0.071 (cartoon faces the detector fires on) and 0.248 (p05 of
# real frames). RE-DERIVE.
ALPHA_GRAPHIC = 0.15

# face centre-x above which a frame is a side panel rather than centred.
# Bimodal on take 1: ~0.55 and ~0.85, nothing between. BROKE ON TAKE 2, which
# has a mode at 0.15 -- and that mode was not a layout at all, it was the
# detector locking onto a bodybuilder in a screenshot. RE-DERIVE, and check the
# montage before believing a mode is a layout.
PANEL_CX = 0.62

# fraction of pixels a dissolving graphic must hold, over the persistence
# window, to count. Calibrated against one known positive (a card at
# f1757-1788: 27.6% of frames flagged) and two known negatives (a
# gesticulation-heavy shot and a clean panel shot: 0.0%). RE-DERIVE.
DISSOLVE_FRAC = 0.15

# robust z on whole-frame difference that counts as a hard cut. 391 found on
# take 1, 2,119 on take 2 -- and take 2's number is ~2x inflated because every
# screenshot swap reads as a cut. RE-DERIVE, and prefer the face-crop-restricted
# variant on a screenshot-heavy creator.
CUT_Z = 8.0

# step in the identity-matched face box that counts as a CAMERA change rather
# than him moving. Verified by rendering frames either side of every split on
# take 2. Median within-shot face-width spread is 8.4% (him leaning), so this
# must sit well above that. RE-DERIVE.
CAMERA_STEP = 0.15

# cosine similarity to the accumulated identity below which a face is not him.
# Enormous gap, so this one is not delicate: good frames p05 0.837-0.870,
# graphic-cluster faces p50 0.052. Anywhere in 0.35-0.60 gives the same answer.
IDENTITY_MIN_SIM = 0.5

# ---------------------------------------------------------------- the crop
# Crop box is CROP_K x the median face width of the shot, square, symmetric
# about the face, edge-padded where it leaves frame. 3.2 was chosen so another
# face falls inside the crop in 0.00% of take-1 and 0.15% of take-2 kept frames
# -- that near-zero is what lets the tracker re-detect landmarks on the crop and
# be certain of picking him.
CROP_K = 3.2
CROP_SIZE = 512       # output square; face lands at ~31% of frame

# Minimum face width, in source pixels, for a layout to be worth keeping at all.
# NOT a threshold any script applies -- it is the number behind a judgement.
# Take 3's largest cluster (19.1%) is a webcam inset at 76-90px against
# 186-223px in takes 1-2; a 3.2x crop of a 76px face upscales ~2x to reach 512
# and is soft. 36% of that take was dropped on this basis. RE-DERIVE, and expect
# to argue about it.
MIN_FACE_PX = 150

# ------------------------------------------------------------ shot assembly
# Frames dropped either side of every boundary. A cut has blend frames and the
# detectors lag one or two, so these are the dirtiest frames in the take and the
# cheapest to lose.
GUARD = 5

# Shortest shot kept. 90 (3s) destroyed 44.6% of take 2, which is cut ~3x faster
# than take 1; 60 (2s) is the current default. A separate, larger floor applies
# to what becomes a VHAP sequence, below. RE-DERIVE per creator's cutting pace.
MIN_SHOT = 60
MIN_SEQUENCE = 90     # too few frames to solve an identity from

# ---------------------------------------------------------------- tracking
# Serial, never parallel: the photometric stage saturates the GPU (13.7 GB of
# 16.3, 94% util measured), so concurrent chunks contend rather than overlap.
GLOBAL_EPOCHS = 30

# Skull anchor for the residual measurement. forehead + scalp + nose, 984 verts.
# NOT the lowest-residual anchor available -- nose alone scores 1.03mm against
# 2.80mm for this set -- because 379 near-coplanar vertices estimate a rotation
# badly however well they fit.
SKULL_ANCHOR = ("forehead", "scalp", "nose")
