#!/usr/bin/env python3
"""The upstream plug: what a baseline must provide, and the concrete xADA teacher.

Delta corrects a baseline, and the baseline will be replaced -- a streaming student
model takes over once the renderer is realtime. So nothing above this file names a
specific model, and the two consequences that are easy to lose:

  * `d_audio` must reach `OffsetNet(d_z=...)`. Hardcoding 512 breaks the swap the
    moment an encoder with a different width arrives. precompute.py asserts it
    against cfg.d_z rather than trusting either side.
  * Swapping the baseline changes `stamp()`, so the cache assert in train.py fires
    by itself. That is the whole point of the stamp: a stale delta is loud instead
    of silent.

Delta is a correction to ONE CONFIGURATION of a frozen model, not to "xADA" in the
abstract. Five things define it -- which model, the chunking policy, the mood
policy, the speaker vector, the time shift -- and changing any one makes a trained
delta stale.
"""

import wave
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class BaselineOut:
    Z: np.ndarray        # [T, d_audio]  audio features at `fps`
    gui81: np.ndarray    # [T, 81]       GUI-space controls, blink ALREADY merged
    head5: np.ndarray    # [T, 5]        (ty, tz, roll, pitch, yaw) -- no X


class Baseline(Protocol):
    name: str
    fps: float
    d_audio: int

    def encode(self, wav: np.ndarray) -> BaselineOut: ...
    def stamp(self) -> dict: ...


def read_wav16k(path):
    """-> float32 [-1, 1] mono at 16 kHz. Asserts rather than resamples."""
    w = wave.open(str(path))
    assert w.getnchannels() == 1, f"{path} is not mono"
    assert w.getframerate() == 16000, f"{path} is {w.getframerate()} Hz, expected 16000"
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return a.astype(np.float32) / 32768.0


def _preload_cuda_libs():
    """Open cuDNN and cuBLAS by absolute path so the CUDA provider can find them.

    They ship inside the environment, with torch, but are not on the loader path,
    and the provider .so lists them as needed libraries -- so it fails to load with
    "libcudnn.so.9: cannot open shared object file" and onnxruntime silently falls
    back to the processor. Setting LD_LIBRARY_PATH from here would not help: glibc
    reads it once at process start. Loading them into this process by hand does,
    and it travels with the code instead of with whichever launcher was edited.

    Silent on failure by design -- the caller falls back to the CPU and says so.
    """
    import ctypes, glob, os, sys
    from pathlib import Path as _P
    root = _P(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" \
           / "site-packages" / "nvidia"
    for sub, pat in (("cudnn", "libcudnn*.so.9"), ("cublas", "libcublas*.so.12")):
        for so in sorted(glob.glob(str(root / sub / "lib" / pat))):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


class XAdaTeacher:
    """Epic's teacher xADA, as the product ships it. Needs onnxruntime.

    Chunking, mood and speaker are all fixed to what Unreal does, because the
    ground-truth curve export was produced under exactly this configuration:

      * 30 s non-overlapping blocks. The encoder hard-fails above 1500 frames,
        which is 30.00 s exactly. Speech2FaceInternal.cpp:282.
      * style_id -1 = AutoDetect, emotion_scalar 1.0. AudioDrivenAnimationConfig.h
        defaults; Speech2FaceInternal.cpp:403-413 encodes AutoDetect as -1.
      * speaker = the graph's own unconditional ReduceMean over the (10, 6) table.
        Picking a row would mean selecting a hyperparameter on the only take we
        have, and this data leaks +0.104 -> -0.046 R2 between a random and a block
        split, so an in-sample margin is not trustworthy.
      * B overwrites R at the two blink slots -- B is authoritative.
        Speech2FaceInternal.cpp:187. Skipping it leaves the eyes on R's weaker
        prediction, with no error anywhere.
    """

    name = "xada_teacher"
    fps = 50.0          # fixed by a 320-sample conv stride, not configurable

    def __init__(self, encoder, decoder, blink_slots, block_sec=30.0,
                 style_id=-1, emotion_scalar=1.0, sr=16000):
        import onnxruntime as ort
        ort.set_default_logger_severity(4)
        # THE CARD IF IT IS THERE, THE PROCESSOR IF IT IS NOT.
        # This was pinned to the CPU because only the CPU build was installed, and
        # it cost 2795 ms of a 5152 ms first frame -- 63% of the wait, in a model
        # that is already ONNX and already the right shape for the card. The fall
        # back is not defensive habit: onnxruntime-gpu raises at session creation
        # when its CUDA libraries are absent, and a face that will not start at all
        # is worse than one that starts slowly.
        _preload_cuda_libs()

        def _session(path):
            if "CUDAExecutionProvider" in ort.get_available_providers():
                try:
                    return ort.InferenceSession(
                        str(path), providers=["CUDAExecutionProvider",
                                              "CPUExecutionProvider"])
                except Exception as e:
                    print(f"[xada] CUDA provider unusable ({type(e).__name__}), "
                          f"staying on the processor", flush=True)
            return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        self.enc = _session(encoder)
        self.dec = _session(decoder)
        self.out_names = [o.name for o in self.dec.get_outputs()]
        self.blink_slots = np.asarray(blink_slots, np.int64)
        self.block_sec, self.sr = block_sec, sr
        self.style_id, self.emotion_scalar = style_id, emotion_scalar
        self.d_audio = self.enc.get_outputs()[0].shape[-1]
        if not isinstance(self.d_audio, int):
            self.d_audio = 512          # dynamic axis; the real width is asserted below

    def encode(self, wav):
        block = int(self.block_sec * self.sr)
        Z, R, B, H = [], [], [], []
        for s in range(0, len(wav), block):
            blk = wav[s:s + block]
            if len(blk) < 320:                  # cannot form even one encoder frame
                break
            blk = blk[: len(blk) // 320 * 320]  # drop the partial tail frame, as UE does
            z = self.enc.run(["Z"], {"audio_signal": blk[None]})[0]
            o = self.dec.run(None, {
                "Z": z.astype(np.float32),
                # int32 is required; int64 raises inside the decoder
                "style_id": np.array([self.style_id], np.int32),
                "emotion_scalar.1": np.array([self.emotion_scalar], np.float32)})
            d = dict(zip(self.out_names, o))
            Z.append(z[0]); R.append(d["R"][0]); B.append(d["B"][0]); H.append(d["H"][0])
        Z = np.concatenate(Z, 0); R = np.concatenate(R, 0)
        R[:, self.blink_slots] = np.concatenate(B, 0)
        self.d_audio = Z.shape[1]
        return BaselineOut(Z=Z, gui81=R, head5=np.concatenate(H, 0))

    def stamp(self):
        return {"model_tag": self.name, "block_sec": self.block_sec,
                "style_id": self.style_id, "emotion_scalar": self.emotion_scalar,
                "speaker": "mean", "ada_fps": self.fps}


def head5_to_6(H5):
    """ADA's (ty, tz, roll, pitch, yaw) -> our (roll, pitch, yaw, tx, ty, tz).

    ADA emits five channels; there is no HeadTranslationX, so slot 3 stays 0 and
    delta learns that channel from scratch. The order matters and is not guessable:
    it must match rig_utils.HEAD_CURVES, which is the order head_pose.root_delta_dna
    consumes.
    """
    H5 = np.atleast_2d(H5)
    H6 = np.zeros((len(H5), 6), H5.dtype)
    H6[:, 0], H6[:, 1], H6[:, 2] = H5[:, 2], H5[:, 3], H5[:, 4]   # roll, pitch, yaw
    H6[:, 4], H6[:, 5] = H5[:, 0], H5[:, 1]                       # ty, tz  (tx stays 0)
    return H6
