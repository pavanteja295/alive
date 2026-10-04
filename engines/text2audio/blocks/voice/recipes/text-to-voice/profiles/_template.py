"""TEMPLATE. Copy to <creator>.py, change what the sections below tell you to.

**The procedure is never edited.** Only this file.

    1  KNOWN      facts about this person. You know them; nothing derives them.
    2  WHERE      machine paths. Nothing to do with the person.
    3  DERIVED    a script measures a distribution and you read the value off it.
                  Each one names its script. Running that script is not optional.
    4  CARRIED    properties of the model or the method, not of the person.
                  These transfer, and the evidence for each is named.
    5  FITTED     look like CARRIED and are not. Every one was chosen on ONE
                  creator. First place to look when a new person comes out worse.
"""
import pathlib

# =====================================================================  1 KNOWN
# The name everything is keyed by: dataset/<SUBJECT>/, F5's data/<SUBJECT>_pinyin and
# ckpts/<SUBJECT>/. Must equal this file's name.
SUBJECT = "CHANGEME"

# One directory per video, each with Audio/Audio/audio.wav and source.json, as
# MetaHuman's ingest_youtube_take.py writes them. Directories starting with _ are skipped.
TAKES = DATA / "takes" / "CHANGEME"

# Take-name PREFIXES held out as whole recordings. Never a random slice of clips: his
# voice differs by recording, and clips from a trained-on recording measure the room.
#   - Hold out whatever the face pipeline scores this person on, so no stage trains on
#     a clip another stage is scored on (MetaHuman SYSTEM.md).
#   - Two uploads of the SAME talk (a full episode and its re-cut) share sentences and
#     go on the same side. 01_inventory.py flags uploads that share a title.
HELDOUT = [
    # "time_perception_memory_focus",
]

# The directory name under bundle/ and the name voice.json reports as its subject.
VOICE_NAME = f"{SUBJECT}_voice"

# Another person's real speech, 16 kHz mono, ~120 s: the "full identity change" end of
# the domain-gap scale in 09_domain_gap.sh. Any other creator's DG_REF will do. For the
# FIRST creator there is none yet: take two minutes of any other speaker you are allowed
# to use (yourself, read aloud, is enough) and convert it:
#     ffmpeg -i speech.wav -ac 1 -ar 16000 -t 120 data/voice/work/domain_gap/peer_16k.wav
# It only calibrates a diagnostic; nothing is trained on it.
PEER_REF = DATA / "voice" / "work/domain_gap/drk_real_ref_16k.wav"

# =====================================================================  2 WHERE
CB = DATA / "voice"                    # this recipe's work, datasets, runs and bundles
F5 = DATA.parent / "externals/F5-TTS"  # the F5-TTS checkout; its data/ and ckpts/ hold training state
ENV_PREP = pathlib.Path.home() / ".venvs/vc-prep"   # scripts/setup_env.sh
ENV_F5 = pathlib.Path.home() / ".venvs/vc-f5"       # scripts/setup_f5.sh

# Per-person working dirs. drk's predate profiles and keep their unsuffixed names,
# which the write-up and the bundle record.
WORK = CB / f"work_{SUBJECT}"
DATASET = CB / "dataset" / SUBJECT
RUNS = CB / f"runs_{SUBJECT}"
F5_DATA = F5 / "data" / f"{SUBJECT}_pinyin"          # F5 derives this from --dataset_name
CKPTS = F5 / "ckpts" / SUBJECT                        # and this
BUNDLE = DATA.parent / "checkpoints" / SUBJECT / "voice"   # the released voice; the app reads only this
DG_REF = WORK / "domain_gap" / "real_ref_16k.wav"     # 09 builds it from held-out if absent

# ===================================================================  3 DERIVED
# SCRIPT: 05_speaker_filter.py prints the cosine distribution to the dominant voice and
# how many hours each threshold keeps. Read the value where the distribution's low
# tail (other voices: guests, clips, ads read by someone else) separates from the bulk.
# drk: 0.60. Not known to sit there for anyone else.
SPK_THRESHOLD = None     # RE-DERIVE. 06_build_dataset.py refuses to run while None.

# The reference clip is derived too, by 15_pick_reference.py, and lives in
# WORK/reference.json -- not here, because a machine writes it.

# ===================================================================  4 CARRIED
CARRIED = dict(
    lr=1e-5,              # F5's finetune guidance; full finetune of a 336M model
    batch_frames=3200,    # memory on a 16 GB card, not accuracy
    grad_accum=2,
    max_samples=64,
    warmup=600,           # the 20000 default is sized for pretraining from scratch
    save_every=2000,      # every checkpoint kept; the sweep picks, nothing guesses
    last_every=1000,
    nfe=32,               # solver steps at inference. runs/nfe*: 16 audibly worse on drk
    eval_n=40,            # held-out clips per scoring pass, round-robin over recordings
    target_lufs=-23.0,    # one gain per recording, never per clip (06_build_dataset.py)
)

# ====================================================================  5 FITTED
FITTED = dict(
    epochs=60,            # drk: 4.27 h -> 16,260 updates, step 10000 promoted. A
                          # data-volume property; scale with hours of training audio
    chunk_min_s=3.0,      # 04_plan_chunks.py. drk talks fast and leaves few pauses;
    chunk_target_s=11.0,  # a slower speaker may cut cleaner at a longer GAP
    chunk_max_s=14.0,
    chunk_gap_s=0.28,
    min_asr_conf=0.65,    # 06_build_dataset.py
    ref_min_s=8.0,        # 15_pick_reference.py: candidate clips for the reference
    ref_max_s=12.0,
    ref_min_asr=0.98,
    ref_min_spk=0.985,
    ref_rate_tol=0.10,    # pace within 10% of the corpus median (F5 copies it)
)

# Transcripts 15_pick_reference.py must never choose, found by listening. Text-specific
# by nature, so it belongs to the person and starts empty.
REF_TEXT_REJECT = ()
