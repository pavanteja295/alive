"""drk (HealthyGamerGG). The first creator: every value below was measured on him.

Paths are the pre-profile ones, unsuffixed, because the write-up, the bundle and
work/scores.jsonl record them.
"""
import pathlib

# =====================================================================  1 KNOWN
SUBJECT = "drk"
TAKES = DATA / "takes" / "healthygamer"

# The face pipeline's unseen shelf (gauss/live/clips/unseen_*) on 2026-09-22.
# 01_inventory.py used to glob that shelf; it now also holds other creators' clips.
HELDOUT = [
    "ai_therapy_is_making_y",
    "nihilism_the_reason_yo",
    "why_your_healing_makes",
]

VOICE_NAME = "healthygamer_voice"
PEER_REF = DATA / "voice" / "work/domain_gap/peer_huberman_16k.wav"

# =====================================================================  2 WHERE
CB = DATA / "voice"                    # this recipe's work, datasets, runs and bundles
F5 = DATA.parent / "externals/F5-TTS"  # the F5-TTS checkout; its data/ and ckpts/ hold training state
ENV_PREP = pathlib.Path.home() / ".venvs/vc-prep"
ENV_F5 = pathlib.Path.home() / ".venvs/vc-f5"

WORK = CB / "work"
DATASET = CB / "dataset" / SUBJECT
RUNS = CB / "runs"
F5_DATA = F5 / "data" / f"{SUBJECT}_pinyin"
CKPTS = F5 / "ckpts" / SUBJECT
BUNDLE = DATA.parent / "checkpoints" / SUBJECT / "voice"   # the released voice; the app reads only this
DG_REF = WORK / "domain_gap" / "drk_real_ref_16k.wav"

# ===================================================================  3 DERIVED
SPK_THRESHOLD = 0.60     # 05_speaker_filter.py, 2026-09-22

# ===================================================================  4 CARRIED
CARRIED = dict(lr=1e-5, batch_frames=3200, grad_accum=2, max_samples=64, warmup=600,
               save_every=2000, last_every=1000, nfe=32, eval_n=40, target_lufs=-23.0)

# ====================================================================  5 FITTED
FITTED = dict(epochs=60,
              chunk_min_s=3.0, chunk_target_s=11.0, chunk_max_s=14.0, chunk_gap_s=0.28,
              min_asr_conf=0.65,
              ref_min_s=8.0, ref_max_s=12.0, ref_min_asr=0.98,
              ref_rate_tol=0.10, ref_min_spk=0.985)

# The top pitch match was "But don't let it is not something you" -- right pitch,
# broken sentence.
REF_TEXT_REJECT = (" it is not something you",)
