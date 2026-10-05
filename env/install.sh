#!/usr/bin/env bash
# Every Python environment the recipes and the live app need, from the frozen lists here.
#
#   bash env/install.sh                 # all six; any that already exists is left alone
#   bash env/install.sh face tracker    # only these (names below)
#
# Needs conda (Miniconda) and uv, and env/setup_externals.sh run first (two of these
# install code from externals/). About 30 GB and 30-60 minutes; the renderer's and the
# tracker's CUDA code compiles for the card in this machine.
#
#   name             where                     used by
#   voice-f5         ~/.venvs/vc-f5            voice training, the voice server
#   voice-prep       ~/.venvs/vc-prep          voice data preparation
#   speech-encoder   ~/.venvs/mh-offset        the speech encoder pass (motion recipe)
#   face             <conda>/envs/stavatar     face motion, renderer, the face server
#   tracker          <conda>/envs/vhap         face tracking and export
#   faceid           <conda>/envs/faceid       the "is it them" face check
#
# VENVS and CONDA_ENVS move the two roots. TORCH_CUDA_ARCH_LIST defaults to 12.0 (RTX 50
# series); set yours, e.g. 8.9 for an RTX 40. MAX_JOBS caps compile parallelism (RAM).
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"; cd "$ROOT"
E="$ROOT/env"
VENVS=${VENVS:-$HOME/.venvs}
CONDA_ENVS=${CONDA_ENVS:-$(conda info --base)/envs}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0} MAX_JOBS=${MAX_JOBS:-4}
command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v conda >/dev/null || { echo "conda not found: install Miniconda (README, Requirements)"; exit 1; }
[ -d externals/F5-TTS ] && [ -d externals/vhap ] && [ -d externals/stavatar ] ||
  { echo "run bash env/setup_externals.sh first"; exit 1; }

# The +cu128 torch builds live on PyTorch's index, torch_scatter on PyG's.
IDX=(--extra-index-url https://download.pytorch.org/whl/cu128
     --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html
     --index-strategy unsafe-best-match)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

frozen() {   # python, list: installed exactly, in the order building from source needs
  local py=$1 list=$2
  uv pip install --python "$py" setuptools wheel ninja pip
  # pytorch3d 0.7.9 has no wheel for these torch versions: the commit it was built from.
  sed -E 's|^pytorch3d==0\.7\.9.*|pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@3143b3baf8ef8b1023ed76f225af59e2e8a71e06|' \
    "$list" > "$TMP/all.txt"
  # Each list is a complete freeze, so install exactly it (--no-deps): resolving would add
  # packages the working env never had, a second onnxruntime for one. Wheels first, so torch
  # is whole before anything compiles against it; then the packages built from git.
  grep -v ' @ git+' "$TMP/all.txt" > "$TMP/wheels.txt"
  grep    ' @ git+' "$TMP/all.txt" > "$TMP/git.txt" || true
  # chumpy (a FLAME dependency) ships only as source and imports pip in its setup.py.
  uv pip install --python "$py" "${IDX[@]}" --no-deps --no-build-isolation-package chumpy -r "$TMP/wheels.txt"
  # --no-cache: these compile against the env's own torch. A cached build from another env
  # (pytorch3d is the same commit in two envs with different torch) imports, then fails on
  # an undefined symbol at the first GPU call.
  [ -s "$TMP/git.txt" ] && uv pip install --python "$py" --no-deps --no-build-isolation --no-cache -r "$TMP/git.txt"
  return 0
}

# An env this script started and did not finish carries .alive-installing and is rebuilt.
# An env without the marker was made some other way, or finished: it is left alone.
fresh() {    # name, path -> 0 if it should be built now
  if [ -f "$2/.alive-installing" ]; then echo "  $1: $2 was interrupted, rebuilding"; rm -rf "$2"; fi
  if [ -x "$2/bin/python" ]; then echo "  $1: $2 exists, left alone"; return 1; fi
  echo "== $1 -> $2"
}
FAILED=""
done_or_fail() {  # name, path, status
  if [ "$3" = 0 ]; then rm -f "$2/.alive-installing"; echo "  $1: ok"
  else FAILED="$FAILED $1"; echo "  $1: FAILED (rerun this script; it resumes)"; fi
}

venv() {     # name, path, python version, list, then extra commands run in the env
  local n=$1 p=$2 v=$3 l=$4; shift 4
  fresh "$n" "$p" || return 0
  set +e   # so the subshell's own set -e works, and its failure is recorded, not fatal
  ( set -e; uv venv "$p" --python "$v"; touch "$p/.alive-installing"
    frozen "$p/bin/python" "$l"; "$@" ); local st=$?; set -e; done_or_fail "$n" "$p" $st
}

condaenv() { # name, path, list, "conda packages", then extra commands run in the env
  local n=$1 p=$2 l=$3 pk=$4; shift 4
  fresh "$n" "$p" || return 0
  set +e
  ( set -e; conda create -y -q -p "$p" -c conda-forge python=3.10 $pk; touch "$p/.alive-installing"
    # conda keeps CUDA headers in targets/x86_64-linux/include; torch's extension builder
    # looks in $CUDA_HOME/include. Link them across (the original envs had this by hand).
    if [ -d "$p/targets/x86_64-linux/include" ]; then
      for h in "$p"/targets/x86_64-linux/include/*; do
        [ -e "$p/include/$(basename "$h")" ] || ln -s "$h" "$p/include/"; done; fi
    # Build CUDA code with the env's own nvcc 12.8, headers and gcc, never another env's:
    # conda's nvcc finds its headers through CONDA_PREFIX, which a shell left in base
    # points at base's CUDA (11.7 on the machine this was written on).
    export CONDA_PREFIX="$p" CUDA_HOME="$p" CPATH="$p/targets/x86_64-linux/include" CC="$p/bin/x86_64-conda-linux-gnu-gcc" CXX="$p/bin/x86_64-conda-linux-gnu-g++"
    export PATH="$p/bin:$PATH" NVCC_PREPEND_FLAGS="-ccbin $p/bin/x86_64-conda-linux-gnu-g++"
    frozen "$p/bin/python" "$l"; "$@" ); local st=$?; set -e; done_or_fail "$n" "$p" $st
}
tracker_post() {
  # nvdiffrast imports its plugin by name after building it; torch 2.11 no longer puts the
  # build folder on sys.path, so that import fails. The patch uses the module load() returns.
  patch -d "$CONDA_ENVS/vhap/lib/python3.10/site-packages" -p1 -N < "$E/patches/nvdiffrast.patch"
  uv pip install --python "$CONDA_ENVS/vhap/bin/python" --no-deps -e externals/vhap
}
renderer_ext() { for m in diff-gaussian-rasterization simple-knn fused-ssim; do
  uv pip install --python "$CONDA_ENVS/stavatar/bin/python" --no-build-isolation --no-cache "externals/stavatar/submodules/$m"; done; }

WANT="${*:+ $* }"
want() { [ -z "$WANT" ] || [[ "$WANT" == *" $1 "* ]]; }

want voice-f5 && venv voice-f5 "$VENVS/vc-f5" 3.11 "$E/voice-f5.txt" \
  uv pip install --python "$VENVS/vc-f5/bin/python" --no-deps -e externals/F5-TTS
want voice-prep && venv voice-prep "$VENVS/vc-prep" 3.11 "$E/voice-prep.txt" true
want speech-encoder && venv speech-encoder "$VENVS/mh-offset" 3.13 "$E/speech-encoder.txt" true
want face && condaenv face "$CONDA_ENVS/stavatar" "$E/face.txt" \
  "cuda-toolkit=12.8 gxx_linux-64=14" renderer_ext
want tracker && condaenv tracker "$CONDA_ENVS/vhap" "$E/tracker.txt" \
  "cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-nvrtc-dev=12.8 cuda-cccl=12.8 cuda-libraries-dev=12.8 gxx_linux-64=13" \
  tracker_post
want faceid && condaenv faceid "$CONDA_ENVS/faceid" "$E/faceid.txt" "" true

if [ -n "$FAILED" ]; then echo; echo "FAILED:$FAILED. Fix what the log above says, then rerun."; exit 1; fi
echo; echo "done. next: the licensed files (docs/guide/setup.html), then ./alive check <creator>"
