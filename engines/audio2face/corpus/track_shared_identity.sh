#!/bin/bash
# Track every chunk of a take against ONE shared identity.
#
#   ./track_shared_identity.sh <seq> <donor_chunk> [--freeze-cam]
#
# THE PROBLEM
#   Tracked independently, N chunks give N identities: `shape` (300) and `static_offset`
#   (5143x3) are optimisable in every stage that matters, so each chunk solves its own face.
#   Fine for head pose, useless for building one avatar.
#
# THE TWO HOOKS, both already in VHAP
#   --model.flame-params-path <ckpt>   tracker.py:79-129 loads shape, static_offset,
#                                      focal_length, tex_extra and lights out of another
#                                      chunk's checkpoint.
#   --pipeline.<stage>.optimizable-params
#                                      get_train_parameters (tracker.py:1495-1505) only
#                                      hands a parameter to the optimiser if its name is in
#                                      this tuple. Drop "shape" and "static_offset" and they
#                                      stay frozen at whatever was loaded.
#
#   Together: solve identity once on a donor chunk, then load-and-freeze it everywhere else.
#
# WHAT STAYS FREE
#   pose, joints, expr, texture, lights. Those are per-chunk by nature -- head motion,
#   expression, and the lighting of that particular shot.
#
# --freeze-cam ALSO pins focal_length. Defensible because every crop is normalised to the
#   same face fraction (31% of frame), so all chunks are meant to be the same virtual
#   camera; pinning it enforces that rather than letting each chunk drift. Off by default
#   because it has not been measured yet.
#
# CAVEAT worth knowing: load_from_tracked_flame_params also copies the DONOR'S per-frame
#   rotation/translation/expr into the first min(N_new, N_donor) frames. Those are the wrong
#   frames, but pose stays optimisable and lmk_init_rigid re-solves frame 0 from scratch over
#   500 steps, so it is an initialisation, not a constraint. If a chunk comes back with a
#   wild pose track, suspect this first.
set -uo pipefail
SEQ="${1:?usage: track_shared_identity.sh <seq> <donor_chunk> [--freeze-cam]}"
DONOR="${2:?need donor chunk name}"
FREEZE_CAM=""
[ "${3:-}" = "--freeze-cam" ] && FREEZE_CAM=1

E=${ENV_VHAP:-$HOME/miniconda3/envs/vhap}
VHAP="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)/vhap"   # the tracker, linked from alive/externals
ROOT="$(cd "$(dirname "$0")" && pwd)"
LEDGER="$ROOT/chunks/$SEQ/shared_ledger.tsv"

CKPT=$(ls "$VHAP/output/chunks/$DONOR"/*/tracked_flame_params_30.npz 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo "donor $DONOR is not tracked yet"; exit 1; }
echo "identity donor: $DONOR"
echo "  $CKPT"
# Recorded beside the ledger: the donor carries the identity every chunk is locked to,
# so recipes/audio-to-mesh builds the person's rig from this chunk's export.
mkdir -p "$ROOT/chunks/$SEQ" && echo "$DONOR" > "$ROOT/chunks/$SEQ/donor.txt"

export CONDA_PREFIX=$E CUDA_HOME=$E PATH=$E/bin:$PATH TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}   # 12.0 = RTX 50xx; set yours
export CC=$E/bin/x86_64-conda-linux-gnu-gcc
export CXX=$E/bin/x86_64-conda-linux-gnu-g++
export NVCC_PREPEND_FLAGS="-ccbin $E/bin/x86_64-conda-linux-gnu-g++"
cd "$VHAP"

# STAR landmark detection spawns this many workers PER CHUNK, each loading its own
# copy of the model: 8 workers measured at 14.5 GB RSS on a 29 GB machine, which is
# what made the system run low on memory mid-run. VHAP's default of 8 is sized for a
# 22,000-frame video; a 200-frame chunk pays eight model loads to process 25 frames
# each. Chunking made the parallelism counterproductive.
LMK_JOBS=${LMK_JOBS:-2}

# Stage parameter tuples with shape and static_offset removed. "cam" is kept unless
# --freeze-cam. Compare against config/base.py:231-294 when VHAP is updated.
if [ -n "$FREEZE_CAM" ]; then C=(); else C=(cam); fi
FREEZE=(
  --pipeline.lmk-init-rigid.optimizable-params        "${C[@]}" pose
  --pipeline.lmk-init-all.optimizable-params          "${C[@]}" pose joints expr
  --pipeline.lmk-global-tracking.optimizable-params   "${C[@]}" pose joints expr
  --pipeline.rgb-init-texture.optimizable-params      "${C[@]}" texture lights
  --pipeline.rgb-init-all.optimizable-params          "${C[@]}" pose joints expr texture lights
  --pipeline.rgb-init-offset.optimizable-params       "${C[@]}" pose joints expr texture lights
  --pipeline.rgb-global-tracking.optimizable-params   "${C[@]}" pose joints expr texture lights
)

[ -f "$LEDGER" ] || printf 'chunk\tframes\tkind\tstatus\tsecs\tresidual_mm\n' > "$LEDGER"
mkdir -p "$ROOT/chunks/$SEQ/pose_shared"

mapfile -t ROWS < <($E/bin/python -c "
import json
for r in json.load(open('$ROOT/chunks/$SEQ/sequences.json')):
    print(r['name'], r['frames'], r['kind'])
")

i=0; n=${#ROWS[@]}
for row in "${ROWS[@]}"; do
  i=$((i+1)); set -- $row; NAME=$1; FR=$2; KIND=$3
  OUT="output/shared/$NAME"
  if ls "$OUT"/*/tracked_flame_params_30.npz >/dev/null 2>&1; then
    echo "[$i/$n] $NAME already done"; continue
  fi
  echo "[$i/$n] $NAME  $FR fr  $KIND  $(date +%H:%M:%S)"
  # Must exist before the redirect to "$OUT.log": the shell evaluates it before VHAP runs,
  # and VHAP only creates $OUT once started. Same fault as track_chunks.sh had -- if you
  # touch one of these two scripts, check the other.
  mkdir -p "$(dirname "$OUT")"
  T0=$(date +%s)
  $E/bin/python vhap/track.py --data.root_folder "$VHAP/data/monocular" \
      --data.sequence "$NAME" --exp.output_folder "$OUT" \
      --model.flame-params-path "$CKPT" "${FREEZE[@]}" \
      --data.landmark-detector-njobs "$LMK_JOBS" \
      --log.interval-media 100000 > "$OUT.log" 2>&1
  RC=$?; DT=$(( $(date +%s) - T0 ))
  CK=$(ls "$OUT"/*/tracked_flame_params_30.npz 2>/dev/null | head -1)
  if [ $RC -ne 0 ] || [ -z "$CK" ]; then
    printf '%s\t%s\t%s\tFAILED\t%d\t\n' "$NAME" "$FR" "$KIND" "$DT" >> "$LEDGER"
    echo "    FAILED rc=$RC -- see $OUT.log"; continue
  fi
  RES=$($E/bin/python "$ROOT/extract_pose.py" --ckpt "$CK" \
        --config "$(dirname "$CK")/config.yml" --vhap-root "$VHAP" \
        --out "$ROOT/chunks/$SEQ/pose_shared/$NAME.npz" 2>/dev/null \
        | grep -oE 'residual [0-9.]+' | grep -oE '[0-9.]+')
  # An empty residual means extract_pose failed. Never let that pass quietly: the tracking
  # is fine, so nothing downstream errors until the donor picker finds nothing to rank.
  [ -z "${RES:-}" ] && echo "    WARNING: residual extraction FAILED for $NAME"
  printf '%s\t%s\t%s\tok\t%d\t%s\n' "$NAME" "$FR" "$KIND" "$DT" "${RES:-}" >> "$LEDGER"
  echo "    ok ${DT}s  residual ${RES:-?} mm"
done
echo "=== shared-identity pass done $(date +%F_%T) ==="
