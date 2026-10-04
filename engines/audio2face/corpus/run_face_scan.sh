#!/bin/bash
# insightface runs in its own conda env: the vhap env has a hand-built nvidia stack that is
# not worth risking. onnxruntime-gpu needs the CUDA 12 wheels on LD_LIBRARY_PATH -- the pip
# package ships the .so files but does not put them on the loader path.
E=${ENV_FACEID:-$HOME/miniconda3/envs/faceid}
export LD_LIBRARY_PATH="$E/lib/python3.10/site-packages/nvidia/cudnn/lib:$E/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$E/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$E/lib/python3.10/site-packages/nvidia/curand/lib:$E/lib/python3.10/site-packages/nvidia/cublas/lib:$E/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$E/lib/python3.10/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH"
exec $E/bin/python "$(dirname "$0")/face_scan.py" "$@"
