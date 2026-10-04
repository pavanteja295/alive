#!/usr/bin/env python3
"""Every constant for the offset model, in one dataclass.

Config lives here rather than in scattered argparse so a run is reproducible from
one logged object, following the GaussianAvatars/STAvatar convention.

THE COUPLING RULE
-----------------
Delta is a correction to ONE SPECIFIC CONFIGURATION of a frozen model, not to
"xADA" in the abstract. Five things define that configuration, and changing any
one of them makes a trained Delta stale:

    1. which model      MODEL_*        teacher xADA, not the xSADA student
    2. chunking policy  BLOCK_SEC      non-overlapping 30 s, as Unreal does
    3. mood policy      STYLE_ID, EMOTION_SCALAR
    4. speaker vector   SPEAKER        the shipped ReduceMean, "mean actor"
    5. the time shift   SHIFT_MS

They are named constants here, stamped into every cache file and logged with
every run, so a mismatch is loud instead of silent. Violating this by accident is
cheap; noticing you have violated it is not.
"""

import math
from dataclasses import dataclass, asdict
from pathlib import Path

PIPE = Path(__file__).resolve().parents[2]


@dataclass
class OffsetConfig:
    # ---- paths ------------------------------------------------------------
    pipe: Path = PIPE
    wav: Path = PIPE / "a2f" / "take4_16k_120s.wav"
    depth_curves: Path = PIPE / "curves" / "AS_Depth_Take4.json"
    encoder: Path = PIPE / "onnx" / "audio_encoder.onnx"
    decoder: Path = PIPE / "onnx" / "animation_decoder.onnx"
    # Built from the IDENTITY DNA (assets/face_identity.dna), not the MetaHuman face
    # DNA -- and the reason is the render target, not the rig's capability.
    #
    # STAvatar displays the Identity mesh, because the MetaHuman mesh comes out of
    # autorigging with scaling artefacts. Delta must be calibrated to the head that is
    # actually shown: the two meshes are NOT a scaled pair (per-axis extent ratios
    # 1.171 / 1.047 / 1.142, and a best-fit similarity still leaves 4.06 mm mean /
    # 23.5 mm max residual on a 564 mm face), so optimising the wrong one means
    # optimising geometry nobody sees. It also changes the per-control sensitivity
    # spread, which is what the vertex loss uses as its weighting.
    #
    # The cost is that this DNA is joints-only: n_bs = 0, so the blendshape path is
    # absent. MEASURED on 64 real take-4 frames, blendshapes contribute 0.059 mm mean
    # / 4.395 mm max -- 3.3% of expression motion, and the same magnitude as the
    # 0.059 mm structural floor. That is 70x smaller than the 4.06 mm mesh difference,
    # so it is the right trade.
    #
    # rig_tables_face.npz is the MetaHuman-DNA version, kept for reference. Every
    # figure quoted in the original design note was measured against IT, so those
    # numbers need re-measuring on this rig.
    rig_tables: Path = PIPE / "offset" / "rig_tables.npz"
    # The DNA rig_tables was BUILT from. These two must always agree: the fixture in
    # cache/rig_fixture.npz is generated from `dna` and checked against `rig_tables`,
    # so a mismatch makes the verification meaningless rather than failing loudly.
    # It surfaced exactly that way -- a fixture with 782 blendshape weights checked
    # against a rig producing 0.
    dna: Path = PIPE / "assets" / "face_identity.dna"
    rig_names: Path = PIPE / "offset" / "cache" / "rig_names.npz"
    cache: Path = PIPE / "offset" / "cache"

    # ---- the five coupled constants (see the module docstring) ------------
    model_tag: str = "xada_teacher"
    block_sec: float = 30.0          # encoder hard-fails above 1500 frames = 30.00 s
    style_id: int = -1               # AutoDetect. Speech2FaceInternal.cpp:403-413 maps
    emotion_scalar: float = 1.0      # AutoDetect -> -1, "which is what the model expects"
    speaker: str = "mean"            # the graph's own ReduceMean. Requires no code.
    shift_ms: float = -66.7          # PER-TAKE. Measure it, do not inherit this value.
    #   ADA trails the depth solve by +66.7 ms on Take 4, so we read ADA at t + 66.7 ms
    #   and emit at t. Calibrated by cross-correlating frame-to-frame activity on a 150 Hz
    #   grid, restricted to the 61 jaw/mouth/lips curves the depth camera actually observes.
    #   EXCLUDE tongue: it is occluded, so the depth solve synthesises it from audio just as
    #   ADA does, and the two agree at 0 ms because neither is watching the face (corr 0.91,
    #   matched standard deviations). Pooling it in pulls the estimate a whole frame.
    #   EXCLUDE brow/eye/nose/cheek: not speech driven, corr 0.01-0.10, inside the noise floor.
    #   Accept only if peak corr > 0.30 and the lag is not pinned at the search-window edge.
    shift_corr: float = 0.6327       # the correlation that produced shift_ms; log it
    shift_channels: str = "jaw|mouth|lips"      # include
    shift_exclude: str = "tongue"               # exclude even though it matches the include

    # ---- rates ------------------------------------------------------------
    # Delta is built at ada_fps and supervised at fps. ONE linear resample sits
    # between its output and the loss, in CONTROL space, with the shift folded into
    # the same interpolation. The latent is NEVER resampled: filtering it costs 6%
    # of its variance and moves the decoder's response by 19% of channel sd, which
    # would pair Delta with a baseline its own input did not produce.
    fps: float = 30.0                # the measurement grid. NOT a resample of anything
    ada_fps: float = 50.0            # fixed by a 320-sample conv stride, not configurable
    sr: int = 16000
    mm_per_cm: float = 10.0          # TorchRig returns CENTIMETRES. Applied ONCE, to
    #   V_pred and V_gt alike, at the loss/metric boundary -- nowhere else. Controls,
    #   c_raw, the cache and the downstream contract stay untouched, and head_pose
    #   keeps its centimetres. Getting this wrong is a factor of 100 on every reported
    #   millimetre and nothing in the run looks wrong.

    # ---- architecture -----------------------------------------------------
    d_z: int = 512                   # audio feature width. Comes from Baseline.d_audio
    #   and must reach OffsetNet(d_z=...). Hardcoding 512 breaks the upstream plug the
    #   moment a baseline with a different encoder is swapped in.
    hidden: int = 128
    layers: int = 2
    d_zp: int = 64                   # audio projection
    d_cp: int = 32                   # baseline projection
    d_pp: int = 0                    # PSD input. 0 = off; 32 enables it
    s_init: float = 1.0              # per-channel tanh bound at init. NOT 0.1:
    #   |c_gt - base| has p95 0.240 / p99 0.866 / max 1.000 on take 4, and 118 of
    #   251 channels exceed 0.1 at their own p99, so 0.1 bounds delta an order of
    #   magnitude below its target and run A plateaus at 0.306 mm. 1.0 is exactly
    #   the control range, so the bound still does its job.

    # ---- optimisation -----------------------------------------------------
    lr_main: float = 3e-4            # gru, projections, head
    lr_affine: float = 1e-4          # a, b, log_s -- start at the LS optimum, refine only
    grad_clip: float = 1.0
    dropout: float = 0.0             # we WANT memorisation of this person
    seed: int = 0

    n_windows: int = 4               # per step
    window: int = 64                 # OUTPUT frames, at `fps`. 2.10 s of span.
    #   Centre-frame context error vs a full-sequence pass: 0.175% at 30 frames,
    #   0.002% at 60, 0.000% at 90. Measured at random init, so re-measure once
    #   Delta is trained. The INPUT width is derived -- see window_in.
    steps_overfit: int = 2000        # run A, one batch, regularisers off
    steps_full: int = 5000           # run B, windows sampled across the take
    steps_affine: int = 2000         # run C, recurrent branch frozen

    # ---- loss -------------------------------------------------------------
    lambda_pos: float = 1.0
    lambda_vel: float = 0.2
    lambda_head: float = 1.0
    lambda_anch: float = 1e-3

    # "ada"  -> base[..., 251:257] = H_ada50, xADA's own head prediction
    # "zero" -> base[..., 251:257] = 0, so delta predicts head DIRECTLY
    #
    # Measured on takes 3 and 4: xADA's head output correlates with the solved head
    # pose at |r| <= 0.294 (largest is NEGATIVE) and using it as a baseline is worse
    # than predicting the take mean on all 12 channel-take pairs -- 13.27 deg vs 1.67
    # on take 4 rx. init_affine independently finds a ~ -1 on every rotation channel,
    # i.e. it already cancels the baseline; "zero" makes that explicit and stops the
    # ill-conditioned tz fit (a = +0.48 on take 3 against -2.06 on take 4, from a
    # predictor whose std is 0.044 against the target's 0.43).
    head_baseline: str = "ada"
    # Nothing is masked. Any channel the performance drives is learnable from it, so
    # masking one is a permanent decision made from a temporary observation -- the same
    # subject speaking or emoting differently exercises a different subset. Gaze is weakly
    # speech-determined and will likely predict near its conditional mean; if that reads as
    # distracting, override those 8 channels at INFERENCE, which is reversible per clip.

    # ---- logging ----------------------------------------------------------
    log_every: int = 10
    probe_every: int = 500           # exact mm through TorchRig on a fixed frame set
    ckpt_every: int = 500
    run_name: str = "v1"

    @property
    def window_in(self) -> int:
        """50 Hz frames one output window reads from. DERIVED, never configured.

        An output window spans (window - 1) / fps seconds. At ada_fps that is
        ceil(span * ada_fps) intervals, so it needs one more sample than that to
        cover them, plus one bracketing sample at each end because the window
        generally starts between two 50 Hz samples. shift_ms translates the whole
        window and therefore does not change the count.

            window=64, fps=30, ada_fps=50  ->  ceil(63/30 * 50) + 2 = 107

        Fixed width, so `i50` can be a plain [W, window_in] index tensor. When a
        window happens to start exactly on a 50 Hz sample only 106 are needed and
        the last column goes unused, which is harmless.
        """
        span = (self.window - 1) / self.fps
        return int(math.ceil(span * self.ada_fps)) + 2

    @property
    def batch_frames(self) -> int:
        return self.n_windows * self.window

    def coupling_stamp(self) -> dict:
        """The five constants, for stamping into caches and checkpoints."""
        return {"model_tag": self.model_tag, "block_sec": self.block_sec,
                "style_id": self.style_id, "emotion_scalar": self.emotion_scalar,
                "speaker": self.speaker, "shift_ms": self.shift_ms,
                "fps": self.fps, "ada_fps": self.ada_fps}

    def as_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v)
                for k, v in asdict(self).items()}
