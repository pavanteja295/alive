#!/bin/bash
# Track every emitted chunk of a take, one at a time, resumable.
#
#   ./track_chunks.sh <seq>
#
# Serial, not parallel: the photometric stage genuinely saturates the GPU (13.7 GB of 16.3,
# 94% util measured), so concurrent chunks would contend for real rather than overlap.
#
# Resumable by marker. A chunk whose output holds tracked_flame_params_30.npz is skipped, so
# a killed run continues where it stopped. VHAP writes no checkpoint during photometric
# sequential tracking, so the worst case is losing one chunk's sequential stage -- a few
# minutes at chunk scale, against ~9 h when the whole video was one sequence. Chunking made
# the pipeline restartable as a side effect.
#
# Landmarks are NOT supplied. VHAP detects them itself (tracker.py:1271-1283) on the CROPPED
# frames, which is what fixes the wrong-face problem by construction: the crop is 3.2x the
# face, so there is nothing else in the picture to lock onto.
set -uo pipefail
SEQ="${1:?usage: track_chunks.sh <seq>}"
E=${ENV_VHAP:-$HOME/miniconda3/envs/vhap}
VHAP="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)/vhap"   # the tracker, linked from alive/externals
ROOT="$(cd "$(dirname "$0")" && pwd)"
LEDGER="$ROOT/chunks/$SEQ/track_ledger.tsv"

export CUDA_HOME=$E PATH=$E/bin:$PATH TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}   # 12.0 = RTX 50xx; set yours
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

[ -f "$LEDGER" ] || printf 'chunk\tframes\tkind\tstatus\tsecs\tresidual_mm\n' > "$LEDGER"

mapfile -t NAMES < <($E/bin/python -c "
import json,sys
for r in json.load(open('$ROOT/chunks/$SEQ/sequences.json')):
    print(r['name'], r['frames'], r['kind'])
")

i=0; n=${#NAMES[@]}
for row in "${NAMES[@]}"; do
  i=$((i+1))
  set -- $row; NAME=$1; FR=$2; KIND=$3
  OUT="output/chunks/$NAME"
  if ls "$OUT"/*/tracked_flame_params_30.npz >/dev/null 2>&1; then
    echo "[$i/$n] $NAME  already tracked, skipping"; continue
  fi
  echo "[$i/$n] $NAME  $FR frames  $KIND  $(date +%H:%M:%S)"
  # The log is redirected to "$OUT.log", so output/chunks/ must exist BEFORE the redirect.
  # VHAP creates $OUT itself, but only once it starts -- and the redirect is evaluated by
  # the shell first. A cleared output/ therefore fails every chunk in 0s.
  mkdir -p "$(dirname "$OUT")"
  T0=$(date +%s)
  $E/bin/python vhap/track.py --data.root_folder "$VHAP/data/monocular" \
      --data.sequence "$NAME" --exp.output_folder "$OUT" \
      --data.landmark-detector-njobs "$LMK_JOBS" \
      --log.interval-media 100000 > "$OUT.log" 2>&1
  RC=$?
  DT=$(( $(date +%s) - T0 ))
  CK=$(ls "$OUT"/*/tracked_flame_params_30.npz 2>/dev/null | head -1)
  if [ $RC -ne 0 ] || [ -z "$CK" ]; then
    printf '%s\t%s\t%s\tFAILED\t%d\t\n' "$NAME" "$FR" "$KIND" "$DT" >> "$LEDGER"
    echo "    FAILED rc=$RC after ${DT}s -- see $OUT.log"; continue
  fi
  CFG=$(dirname "$CK")/config.yml
  RES=$($E/bin/python "$ROOT/extract_pose.py" --ckpt "$CK" --config "$CFG" \
        --vhap-root "$VHAP" --out "$ROOT/chunks/$SEQ/pose/$NAME.npz" 2>/dev/null \
        | grep -oE 'residual [0-9.]+' | grep -oE '[0-9.]+')
  # An empty residual means extract_pose failed. Never let that pass quietly: the tracking
  # is fine, so nothing downstream errors until the donor picker finds nothing to rank.
  [ -z "${RES:-}" ] && echo "    WARNING: residual extraction FAILED for $NAME"
  printf '%s\t%s\t%s\tok\t%d\t%s\n' "$NAME" "$FR" "$KIND" "$DT" "${RES:-}" >> "$LEDGER"
  echo "    ok  ${DT}s  residual ${RES:-?} mm"
done
echo "=== all chunks done $(date +%F_%T) ==="
