# Environments

Six Python environments, kept apart because their libraries conflict (two torch
versions, two numpy majors, two onnxruntime builds). Each `*.txt` here is the exact
`pip freeze` of the one that built and serves Dr K and Huberman. Install with
`pip install -r` or `uv pip install -r`; the few lines that are comments are packages
built from source, and the comment says how.

| file | default location | Python | used by |
|---|---|---|---|
| `voice-f5.txt` | `~/.venvs/vc-f5` | 3.11 | voice training and the voice server |
| `voice-prep.txt` | `~/.venvs/vc-prep` | 3.11 | voice data prep: separation, transcription, speaker filter, scoring |
| `face.txt` | `~/miniconda3/envs/stavatar` | 3.10 | face motion, renderer training, the face server |
| `tracker.txt` | `~/miniconda3/envs/vhap` | 3.10 | face tracking and export (VHAP) |
| `faceid.txt` | `~/miniconda3/envs/faceid` | 3.10 | the "is it him" face check in the clip recipe |
| `speech-encoder.txt` | `~/.venvs/mh-offset` | 3.13 | the speech encoder pass in the motion recipe |

The answer engine and the viewer need only Python 3.11+ with `anthropic`
(`pip install anthropic`); `./alive` runs them with `python3`, or `$PYTHON` if set.

Every location above can be changed: the live app reads `ENV_F5`, `ENV_STAVATAR` and
`PYTHON`; the recipes read `ENV`, `ENV_VHAP`, `ENV_PREP`, `ENV_F5` and `ENV_FACEID` from
each creator's profile.

## System tools

| tool | used by | note |
|---|---|---|
| `ffmpeg`, `ffprobe` | every recipe, the live app | any recent build |
| `yt-dlp` and a JavaScript runtime (`deno` or `node`) | downloading videos | YouTube extraction needs the JS runtime now |
| CUDA toolkit 12.8 (`nvcc`) and conda's `gxx_linux-64` in the tracker env | face tracking builds VHAP's CUDA ops on first run | `conda install -n vhap -c conda-forge gxx_linux-64 cuda-nvcc=12.8`; set `TORCH_CUDA_ARCH_LIST` for your card (default 12.0, the RTX 50 series) |
| `cmake` | only `--with-riglogic` | to rebuild the shared face assets |
| the `claude` CLI | optional | the answer engine falls back to it when no API key is set |

## Order

`bash env/install.sh` does all of this, and `./alive install` runs it after `setup_externals.sh`.
The commands below are what it runs, for doing one by hand.

```bash
bash env/setup_externals.sh            # tracker, renderer and voice code, at our commits
uv venv ~/.venvs/vc-f5    --python 3.11 && uv pip install --python ~/.venvs/vc-f5    -r env/voice-f5.txt && uv pip install --python ~/.venvs/vc-f5 -e externals/F5-TTS
uv venv ~/.venvs/vc-prep  --python 3.11 && uv pip install --python ~/.venvs/vc-prep  -r env/voice-prep.txt
uv venv ~/.venvs/mh-offset --python 3.13 && uv pip install --python ~/.venvs/mh-offset -r env/speech-encoder.txt
conda create -n stavatar python=3.10 && conda run -n stavatar pip install -r env/face.txt
conda run -n stavatar pip install --no-build-isolation \
    externals/stavatar/submodules/diff-gaussian-rasterization \
    externals/stavatar/submodules/simple-knn externals/stavatar/submodules/fused-ssim
conda create -n vhap python=3.10 && conda run -n vhap pip install -r env/tracker.txt && conda run -n vhap pip install -e externals/vhap
conda create -n faceid python=3.10 && conda run -n faceid pip install -r env/faceid.txt
```

CUDA: everything was built and run against CUDA 12.8 (torch `+cu128`) on an RTX 5080
(compute capability 12.0). The tracker's and renderer's CUDA extensions compile for the
card in front of them; on a different card set `TORCH_CUDA_ARCH_LIST` to match.

`pytorch3d` 0.7.9 has no wheel for these torch versions: build it from source
(`pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@3143b3b"`).
