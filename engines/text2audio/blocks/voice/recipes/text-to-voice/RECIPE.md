# Recipe: text to voice

**Produces** one voice bundle per creator: finetuned F5-TTS weights, the reference clip
and its exact transcript, and the held-out numbers that say how close it is to them.
The bundle is what `voice_server.py` serves and what the face stage hears.

**Validated on** one creator, drk, 17 recordings, 4.27 h of training clips,
2026-09-22/23. Released to `checkpoints/drk/voice/`, step 10000: speaker similarity
**99.2%** of the ceiling his own held-out speech sets, WER 0.023 against 0.034 on his
real speech, domain gap 0.15 under a 0.30 floor.

**One creator is not two.** Every value in a profile marked FITTED was chosen on drk.

**English creators only, by decision (2026-10-01).** Transcription and scoring are fixed to
English on purpose; this is scope, not a gap.

**This file owns the process.** If a script, a prompt or a conversation disagrees
with it, this file is right and the other has drifted.

```
recipes/text-to-voice/
  RECIPE.md            this file
  tools/               -> ../../scripts/, the one set of scripts every creator runs
  profiles/<name>.py   everything that varies by person. The call site
```

Progress, from disk, with the next command: `python3 tools/status.py --profile <name>`

---

## 0. Signature

### Inputs

| input | shape | read from | owned by |
|---|---|---|---|
| the recordings | one dir per video: `Audio/Audio/audio.wav` (48 kHz mono so far), `source.json` with `id`, `title` | the profile's `TAKES` | MetaHuman `ingest_youtube_take.py` |
| the held-out recordings | take-name prefixes | the profile's `HELDOUT` | **you**, matched to what the face pipeline scores this person on |
| a peer's real speech | 16 kHz mono, ~120 s | the profile's `PEER_REF` | any other creator's `DG_REF` |
| the base model | `F5TTS_v1_Base model_1250000` and its pinyin vocab | F5's `finetune_cli.py` fetches it | upstream |

### Outputs, and the only things this recipe may write

| output | what downstream may assume |
|---|---|
| `<WORK>/` | inventory, transcripts, cut plan, speaker scores, reference choice, scores. drk's is `work/`, everyone after `work_<name>/` |
| `<DATASET>/{train,heldout}/` | `wavs/`, `manifest.jsonl`, and F5's `metadata.csv`. One gain per recording |
| `<F5_DATA>`, `<CKPTS>` | F5's packed dataset and every checkpoint, keyed by `SUBJECT` |
| `<RUNS>/f5_s<step>/` | held-out sentences spoken by each checkpoint, round-robin across held-out recordings |
| `<BUNDLE>/` | `model.pt` (EMA + model weights, no optimizer), `reference.wav`, `vocab.txt`, `voice.json`. **The reference is part of the voice**: F5 conditions on it at inference, so weights without it are not reproducible |

### Starting a new creator

**Copy a profile, fill it in, run the same scripts.** No script is written for one
person; a script that has to be written goes in `recipes/NEW-SCRIPTS-LOG.md`.
Every script takes `--profile <name>` (the profile's file name) and none defaults to
anyone.

1. Download the videos: `./alive build <id> --stage videos` (it runs `engines/audio2face/ingest_youtube_take.py`).
2. `cp profiles/_template.py profiles/<name>.py`. Set `SUBJECT`, `TAKES`, `HELDOUT`.
3. `status.py --profile <name>` prints each command in order:
   `01_inventory` -> `02_probe_music` -> `03_transcribe` -> `04_plan_chunks` ->
   `05_speaker_filter` -> **set `SPK_THRESHOLD`** -> `06_build_dataset` ->
   `15_pick_reference` -> **listen to the reference** -> `train_f5.sh` ->
   `overnight.sh` (waits, then `10_sweep.sh`) -> `11_promote.sh <step>`.
4. Serve: `run_voice.sh <name> --port <port>`.

### Composes with

The style engine hands over text; the face service (`engines/audio2face/gauss/live`) takes the
wav and converts it to 16 kHz itself. Neither is described here.

---

## 1. Why this exists

Zero-shot F5 already copies timbre from ten seconds of reference, so a creator's voice
is recognisable without any training. What it cannot copy from one clip is how the
person talks across hours: pitch range and pause density. On drk the finetune closed
30% of the gap in pitch range and 13% in pause density and cut WER ~15% relative. It
did nothing for timing. That is the whole of what training buys, and it is why the
**reference clip is chosen with as much care as the checkpoint**.

---

## 2. The item, and the split

**A whole recording.** The voice differs by recording (room, mic, mood); held-out clips
from a recording that was trained on measure the room. `HELDOUT` names recordings.

- Hold out what the face pipeline scores this person on. **No stage trains on a clip any
  stage is scored on**: one split per person, shared by every recipe.
- **Two uploads of one talk go on the same side.** A full episode and its re-cut share
  sentences. `01_inventory.py` refuses a split where two uploads with the same title
  (before any `|` suffix) land on different sides.
- Scoring takes the held-out clips **round-robin across recordings**. Taking the first N
  once gave 40 clips from one recording, and pitch differs by recording.

---

## 3. Tool inventory

| script | postcondition |
|---|---|
| `status.py` | nothing. Prints what is on disk and the next command |
| `01_inventory.py` | `inventory.json`: every take, its audio, its side of the split |
| `02_probe_music.py` | `music_probe.json`: share of energy outside the vocal stem, six windows per take. **Reports; nothing downstream separates.** drk: 0.0-1.8%, no music bed |
| `03_transcribe.py` | `asr/<take>.json`: Whisper large-v3 word timings. Resumable |
| `04_plan_chunks.py` | `chunks.json`: cut points on real pauses, sentence ends preferred |
| `05_speaker_filter.py` | `speaker.json`: every chunk's cosine to the dominant voice, and a table of what each threshold keeps. **Decides nothing** |
| `06_build_dataset.py` | the dataset, at 48 kHz, one gain per recording. Refuses while `SPK_THRESHOLD` is unset |
| `15_pick_reference.py` | `reference.json`: a clean clip at the corpus median pitch |
| `train_f5.sh` | packs the dataset into F5's format if stale, then finetunes; every checkpoint kept |
| `08_synthesize.py` | held-out sentences spoken by one checkpoint, with the profile's reference |
| `07_score.py` | similarity and WER, each **against the ceiling** real held-out speech sets. Appends to `scores.jsonl` |
| `09_domain_gap.sh` | does the synthesis keep the face encoder on its distribution. Builds `DG_REF` from held-out clips on first use |
| `10_sweep.sh` / `overnight.sh` / `12_rescore.sh` | 08 + 07 + 09 over every checkpoint, then a ranked table |
| `14_prosody.py` | duration, rate, pitch and pause distances to the real held-out sentence. **The instrument that can see the finetune** |
| `11_promote.sh` | the bundle |
| `voice_server.py`, `speak.py`, `run_voice.sh` | serve / call a bundle |


---

## 4. Judgement: unavoidable, forbidden, not needed

| | |
|---|---|
| **unavoidable** | `HELDOUT`; `SPK_THRESHOLD` off 05's table; listening to the reference clip; listening to the promoted voice beside real speech; which checkpoint to promote when the numbers tie |
| **FORBIDDEN** | choosing the reference from held-out recordings; promoting on training loss; quoting similarity without its ceiling |
| **not needed** | cut points, loudness, reference pitch, F5 packing, scoring |

**Every number is quoted against the ceiling.** Similarity of real held-out speech to
its own centroid is not 1.00 (drk: 0.982) and Whisper misreads real speech (drk WER
0.034). A candidate is read as a fraction of those.

**Similarity saturates.** On drk the untrained base and every checkpoint tie at 99.2% of
ceiling: the speaker embedding scores timbre, which F5 copies zero-shot. It confirms a
voice is them; it cannot choose a checkpoint. `14_prosody.py` can.

**The domain gap saturates too, and is blind to lip sync.** All eight drk checkpoints
scored 0.15. It compares distributions of the face model's controls, and a curve
shifted in time has the same distribution.

---

## 5. Instance constants

| kind | examples |
|---|---|
| **known** | `SUBJECT`, `TAKES`, `HELDOUT`, `VOICE_NAME`, `PEER_REF` |
| **where** | `WORK`, `DATASET`, `RUNS`, `F5_DATA`, `CKPTS`, `BUNDLE`, `DG_REF`, the envs |
| **derived** | `SPK_THRESHOLD` (05); the reference clip (15, in `reference.json`, not the profile) |
| **carried** | learning rate, batch, warmup 600, keep every checkpoint, nfe 32, one gain per recording |
| **fitted** | `epochs` (sized to 4.27 h), chunk lengths and pause gap (drk talks fast), reference-candidate filters, `REF_TEXT_REJECT` |

**The reference clip is a hyperparameter.** F5 copies its pitch. Chosen on speaker
similarity it picked a clip at 192 Hz against drk's 165 Hz median; every synthesis came
out at 177 Hz and no timbre metric noticed. Changing that one clip cut duration error
27% and moved median pitch 21 Hz toward him, with no retraining.

---

## 6. Done criterion

**Re-read this, do not carry it.** `status.py` prints it.

1. the corpus is only this person: 05's distribution read, `SPK_THRESHOLD` set from it
2. the reference clip has been **listened to**: its pitch and its transcript both
3. every checkpoint is scored on held-out recordings, and the promoted one is chosen on
   those scores
4. the domain gap of the promoted run is recorded
5. a clip of the promoted voice has been **heard** next to the real held-out speech

`status.py` computes 3 and the existence of 1. **2, 4 and 5 need a person.**

---

## 7. What did not generalise

Second creator: **huberman**, 2026-09-30, five uploads, 4.5 h; 3.52 h train, 0.44 h
held out (Time Perception). Released to `checkpoints/huberman/voice/`, step 8000.

| | drk | huberman | read |
|---|---|---|---|
| music bed | 0.0-1.8% | 0.0% | held |
| other voices | character voices, min 0.73 | **none**, min 0.92 | the threshold cut nothing |
| cuts with no real pause | 46% | 34% | held |
| reference by pitch alone | 192 Hz, fixed to the median | at the median, **22% slow** | new failure: see section 9 |
| WER, ceiling | 0.034 | 0.013 | |
| WER, promoted | 0.023 | **0.004** (0.049 with the slow reference) | |
| similarity, % ceiling | 99.2% | 99.0% | saturated on both |
| base model, untrained | ties every checkpoint on similarity | ties on similarity, WER 0.013 vs 0.004 | the finetune's measurable gain here is WER, on n=40 |
| domain gap: floor / peer / clone | 0.30 / 1.00 / 0.15 | 0.19 / **0.37** / 0.35 | **blind on huberman**: his scale barely separates two real people |
| training | 16,260 updates | stopped at 12,000: **disk full** (5.4 GB a checkpoint) | scores flat across steps, nothing lost |

**Open:** the pace-matched reference pulls pitch down: 112.7 Hz against his held-out
119.8, and pitch spread 40 against 47. The corpus median he is matched to (111.8 Hz)
comes mostly from the dopamine episode; the held-out recording sits higher. Pause rate
went 0.67 -> 0.51 against his 0.60. Not yet ruled a defect or a property.

**Assumed one person, fixed in place:** every script read `work/`, `dataset/drk`,
`ckpts/drk`, `drk_pinyin` or `bundle/healthygamer_voice` as literals; `01_inventory.py`
derived the split from the face shelf's `unseen_*` clips, which now also hold other
creators' clips; `06_build_dataset.py`'s threshold was typed by hand; the F5 packing
step was never scripted; `09_domain_gap.sh` used drk as the reference and huberman as
the peer; `15_pick_reference.py` carried a drk transcript in its reject list.

**Serving** is `./alive up <creator>`, for any creator with a released voice.

---

## 8. Dead ends

- **Music separation on clean speech.** Measured first: drk's non-vocal energy was
  0.0-1.8%. Separation costs fidelity and buys nothing there. The probe stays, because
  a creator with a music bed will read differently.
- **Per-clip loudness.** Flattens the difference between sentences, which is part of the
  voice. One gain per recording removes the 10 dB spread between recordings instead.
- **Cutting on a clock.** Slices mid-word. Insisting on a real pause kept only 53% of
  drk's audio, so the cutter falls back to the widest gap in the window.
- **XTTS-v2 / AllTalk, Fish Speech, Unsloth.** Rejected on this machine: no Blackwell
  kernels, LoRA that learns patterns not timbre, no support for this model family.

---

## 9. Silent failures

| what happened | how it presented |
|---|---|
| reference chosen on speaker similarity | every synthesis 12 Hz too high; similarity unchanged |
| reference at the right pitch, 22% slower than the speaker's pace (huberman) | every sentence ~1.2 s long, WER 4x the ceiling on the base model and every checkpoint alike. F5 sizes output from the reference's chars/s. 15 now filters and ranks on pace too |
| checkpoints filled the disk (5.4 GB each with optimizer state) | a training run that died on a save at 12,000, leaving a truncated checkpoint the sweep could not read |
| reference transcript garbled ("But don't let it is not something you") | nothing in any number; F5 aligns text to audio |
| candidate scored against a centroid that contained its own real sentence | a flattering similarity; fixed by skipping the matching clip, as the ceiling does |
| scoring the first N held-out clips | a ceiling that was a property of one recording |
| a wait loop whose pgrep pattern matched its own argv | waited all night for a finished job |
| the face shelf fed a 24 kHz wav | the worker asserts 16 kHz and does not convert; the failure shows only in the job's `note` |
