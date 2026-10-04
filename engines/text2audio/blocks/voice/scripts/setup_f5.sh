#!/usr/bin/env bash
# F5-TTS training env. Separate from the prep env on purpose: prep is built once,
# trainer envs get rebuilt as we swap candidates.
set -euo pipefail
VENV="$HOME/.venvs/vc-f5"; VPY="$VENV/bin/python"

uv venv --python 3.11 "$VENV"

# torch first and pinned to cu128, so the f5-tts install below (torch>=2.0.0)
# resolves as already-satisfied instead of pulling a wheel with no sm_120 kernels.
uv pip install --python "$VPY" \
  torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

uv pip install --python "$VPY" -e "$(cd "$(dirname "$(readlink -f "$0")")/../../../../.." && pwd)/externals/F5-TTS"
uv pip install --python "$VPY" tensorboard

"$VPY" - <<'PY'
import torch, f5_tts
print("torch  ", torch.__version__, "| sm_120:", "sm_120" in torch.cuda.get_arch_list())
print("device ", torch.cuda.get_device_name(0))
print("f5_tts ", f5_tts.__file__)
PY
