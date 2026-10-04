#!/usr/bin/env bash
# Every tracked chunk -> a NeRF-style dataset STAvatar can read.
#
#   bash gauss/export_all.sh        # from engines/audio2face
#
# Idempotent: a chunk whose transforms_train.json exists is skipped, so it can be
# re-run after an interruption. The compiler pinning is not optional -- the tracker's
# rasteriser builds a CUDA extension on first use, and the default toolchain here is a
# CUDA 11.7 nvcc that does not know this GPU plus a gcc newer than nvcc accepts.
set -u
VHAP="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)/vhap"   # the tracker, linked from alive/externals
V="$VHAP"
export CUDA_HOME=$HOME/miniconda3/envs/vhap
export PATH=$CUDA_HOME/bin:$PATH
export CC=${CC:-$HOME/miniconda3/envs/gaussian-avatars/bin/gcc}     # any gcc VHAP's CUDA ops build with
export CXX=${CXX:-$HOME/miniconda3/envs/gaussian-avatars/bin/g++}
cd "$V" || exit 1
n=0; skipped=0; failed=0
for d in output/shared/*/; do
  chunk=$(basename "$d")
  [ -d "$d" ] || continue
  src=$(ls -1dt "$d"*/ 2>/dev/null | while read -r r; do
          [ -f "${r}tracked_flame_params_30.npz" ] && { echo "${r%/}"; break; }; done)
  [ -z "${src:-}" ] && continue
  tgt="export/corpus/$chunk"
  if [ -f "$tgt/transforms_train.json" ]; then skipped=$((skipped+1)); continue; fi
  # Atomic claim, so several copies of this script can share the queue. mkdir either
  # creates the directory or fails; nothing in between, unlike a test-then-touch.
  mkdir "$tgt.lock" 2>/dev/null || { skipped=$((skipped+1)); continue; }
  if ~/miniconda3/envs/vhap/bin/python vhap/export_as_nerf_dataset.py \
        --src_folder "$src" --tgt_folder "$tgt" --background-color white \
        > /tmp/export_$chunk.log 2>&1; then
    n=$((n+1)); echo "OK   $chunk  ($(ls "$tgt/images" | wc -l) frames)"
  else
    failed=$((failed+1)); echo "FAIL $chunk  -> /tmp/export_$chunk.log"
  fi
done
echo "exported $n, skipped $skipped, failed $failed"
