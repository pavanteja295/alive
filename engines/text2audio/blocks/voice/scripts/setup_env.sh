#!/usr/bin/env bash
# Data-prep environment. Deliberately separate from any trainer env:
# prep runs once, trainers get rebuilt as we swap models.
set -euo pipefail
VENV="$HOME/.venvs/vc-prep"

uv venv --python 3.11 "$VENV"
VPY="$VENV/bin/python"

# cu128 is not optional: the 5080 is sm_120 and older wheels have no kernels for it.
uv pip install --python "$VPY" \
  torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

uv pip install --python "$VPY" \
  faster-whisper==1.2.0 \
  demucs==4.0.1 \
  speechbrain==1.0.3 \
  soundfile librosa pyloudnorm numpy scipy tqdm

"$VPY" - <<'PY'
import torch, sys
print("python     ", sys.version.split()[0])
print("torch      ", torch.__version__, "cuda", torch.version.cuda)
print("sm_120     ", "sm_120" in torch.cuda.get_arch_list())
print("device     ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
