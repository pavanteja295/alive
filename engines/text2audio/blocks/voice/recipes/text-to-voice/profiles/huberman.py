"""huberman (Huberman Lab). Second creator through this recipe.

Starts from the template. SPK_THRESHOLD is unset until 05_speaker_filter.py has run on
his corpus; the FITTED values are drk's and are the first place to look if he comes
out worse.
"""
import pathlib

# =====================================================================  1 KNOWN
SUBJECT = "huberman"
TAKES = DATA / "takes" / "huberman"

# Five uploads, 4.5 h. Held out: Time Perception (31 min), the one recording the face
# pipeline has tracked for him and splits its test clips from -- so no stage trains on
# a clip another is scored on, and the whole chain can be scored on it.
# The two "Controlling Your Dopamine" uploads (QmOF0crdyRU, the 2 h 16 min episode, and
# XeN6eGO6FVQ, its Essentials re-cut) share sentences and therefore both train.
# The five uploads this voice was built from. The takes folder has since grown.
TAKE_IDS = ['QmOF0crdyRU', 'XeN6eGO6FVQ', 'HiyzzcuaAac', '9gJLWk3W5GQ', 'vXTK0Ac9i1Q']

HELDOUT = [
    "time_perception_memory_focus",
]

VOICE_NAME = f"{SUBJECT}_voice"
PEER_REF = DATA / "voice" / "work/domain_gap/drk_real_ref_16k.wav"

# =====================================================================  2 WHERE
CB = DATA / "voice"                    # this recipe's work, datasets, runs and bundles
F5 = DATA.parent / "externals/F5-TTS"  # the F5-TTS checkout; its data/ and ckpts/ hold training state
ENV_PREP = pathlib.Path.home() / ".venvs/vc-prep"
ENV_F5 = pathlib.Path.home() / ".venvs/vc-f5"

WORK = CB / f"work_{SUBJECT}"
DATASET = CB / "dataset" / SUBJECT
RUNS = CB / f"runs_{SUBJECT}"
F5_DATA = F5 / "data" / f"{SUBJECT}_pinyin"
CKPTS = F5 / "ckpts" / SUBJECT
BUNDLE = DATA.parent / "checkpoints" / SUBJECT / "voice"   # the released voice; the app reads only this
DG_REF = WORK / "domain_gap" / "real_ref_16k.wav"

# ===================================================================  3 DERIVED
# 05_speaker_filter.py, 2026-09-30: p1 0.952, min 0.924, no low tail at all. The bottom
# twelve are all ~3 s clips of him (short clips embed worse), no guest, no ad voice.
# Any threshold up to 0.92 keeps all 1573 chunks; 0.60 is drk's value and cuts nothing.
SPK_THRESHOLD = 0.60

# ===================================================================  4 CARRIED
CARRIED = dict(lr=1e-5, batch_frames=3200, grad_accum=2, max_samples=64, warmup=600,
               save_every=2000, last_every=1000, nfe=32, eval_n=40, target_lufs=-23.0)

# ====================================================================  5 FITTED
# drk's, unchanged. His training audio is ~4 h raw against drk's 4.27 h kept, so the
# epoch count lands in the same range of updates; re-check once 06 reports hours.
FITTED = dict(epochs=60,
              chunk_min_s=3.0, chunk_target_s=11.0, chunk_max_s=14.0, chunk_gap_s=0.28,
              min_asr_conf=0.65,
              ref_min_s=8.0, ref_max_s=12.0, ref_min_asr=0.98,
              ref_rate_tol=0.10, ref_min_spk=0.985)

# First reference (..._0740, "And PEA is found in various foods...") was listened to and
# judged fair, but ran at 13.3 chars/s against his 17.0 and took WER to 4x the ceiling.
# Replaced 2026-09-30 by the pace-aware pick (..._0149, 111.2 Hz, 17.4 chars/s, "You will
# find the rewards..."). NOT YET LISTENED TO. The old choice: work_huberman/reference_slow.json.
REF_TEXT_REJECT = ()
