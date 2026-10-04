# Recipes

Every model in `checkpoints/` is built by a recipe, and the same recipes build every
creator. This page is where to start: what runs in which order, where each recipe lives,
what it costs, and which steps need a person.

```bash
./alive build <creator>              # download their videos, rebuild everything missing
./alive build <creator> --dry-run    # what would run next in each stage
./alive build <creator> --stage voice
```

`./alive build` (this folder's `rebuild.py`) runs nothing of its own. It asks each recipe's
status command what is next, runs that, and asks again. Steps already on disk are skipped,
so the same command resumes after a stop, a crash or a decision.

## Build order

| stage | makes | recipe | status command | time, one creator |
|---|---|---|---|---|
| videos | `data/takes/<folder>/` | `engines/audio2face/ingest_youtube_take.py` | (the driver checks the folders) | ~1 min per 30 min of video |
| answers | `data/answers/<creator>/` | `engines/knowledge_style/proto/RECIPE.md` | `make status` | minutes, LLM calls |
| voice | `checkpoints/<id>/voice/` | `engines/text2audio/blocks/voice/recipes/text-to-voice/RECIPE.md` | `scripts/status.py --profile <id>` | ~1.5 h |
| clips | tracked clips in `data/face/corpus/chunks/` | `engines/audio2face/corpus/recipes/face-clips/RECIPE.md` | `corpus/status.py --takes takes/<folder>` (from `engines/audio2face`) | **10-20 h per recording** (tracking) |
| motion | `checkpoints/<id>/face/{motion,rig}/` | `engines/audio2face/rigfit/recipes/audio-to-mesh/RECIPE.md` | `tools/status.py --profile <id>` | ~2 h |
| render | `checkpoints/<id>/face/render/` | `engines/audio2face/gauss/recipes/mesh-to-render/RECIPE.md` | `tools/status.py --profile <id>` | ~3-4 h training |
| live app | `creators/<id>/live.env` | this page, *Resting pose* | `./alive check <id>` | minutes, by eye |

Time is on one RTX 5080. Huberman, one tracked recording: about a day end to end. Dr K,
three tracked recordings: about two days. Tracking dominates; everything else is hours.

**Stop the live app before building** (`./alive down`). Building and serving do not share
the card: transcription alone runs out of memory next to it.

## Where a person decides, and where the decision is kept

The driver replays a creator's recorded decisions and stops where there is none.

| stage | decision | recorded in |
|---|---|---|
| videos | which uploads, and that they are him | `creators/<id>/build.json` → `videos.ids` |
| answers | the persona note: eighteen passages of their speech, picked by hand | the note is data (`data/answers/<creator>/creator.md`); the pick is in `creators/<id>/build/answers/persona_passages.txt`. How: `engines/knowledge_style/proto/RECIPE.md`, *The persona note* |
| answers | the promoted retrieval setup | `engines/knowledge_style/proto/harness/profiles/<creator>.json` → `promoted` |
| voice | the speaker-filter threshold; the reference clip (listen to it); the training step to release | voice profile `SPK_THRESHOLD`; `creators/<id>/build/voice/reference.json`; `build.json` → `voice.release_step` |
| clips | which shot clusters are him, framed as wanted | `creators/<id>/build/clips/<take>/verdicts.json` and `cluster_config.json` |
| motion | the joins sheet looked at; which run to release | `creators/<id>/build/motion/joins_acked.json`; `build.json` → `motion.run`, `motion.wrong` |
| render | the joins sheet looked at; release the deployed run | `creators/<id>/build/render/joins_render_acked.json`; `build.json` → `render` |
| live app | the resting pose | `creators/<id>/live.env` → `REST` |

A joins sheet with no recorded look stops the build. `--trust-gates` accepts it, and its
acknowledgement then says that nobody looked. Huberman has no recorded joins looks: his
models were released without them, and they look right live.

## A new creator

1. **Pick the videos.** A single speaker facing the camera, a stable set, 30 fps. Five
   uploads of about 30 minutes each are enough; one of them is held out for testing.
2. **Copy the templates** and fill in every value:
   `creators/_template/` → `creators/<id>/` (`live.env`, `build.json`), and each recipe's
   `profiles/_template.py` → `profiles/<id>.py`. Each template says what each value is.
3. **`./alive build <id>`**. It stops at each decision above; make it, record it where the
   table says, and run the same command again.
4. **Pick the resting pose** (below), then `./alive check <id>` and `./alive up <id>`.

Every recipe's `RECIPE.md` owns its own process: why each step exists, what it checks, and
what went wrong before. Read the one for a stage before changing anything in it.

## Resting pose

Between answers the viewer plays a 4 s loop of the creator's real footage with the
generated head on it. Raised hands, a lean, or a look away read as him doing something,
not as him waiting. A scan cannot tell those apart from rest, so the scan proposes and a
person (or Claude) chooses by looking:

1. `./alive up <id>`, then `curl 'http://127.0.0.1:8730/calm?seconds=4&k=6'` for the six
   stillest windows. Add `&recording=<recording>` if the creator was filmed in more than
   one set, and set `FACE_BODY="random:<recording>"` so the answers stay in that set.
   The windows are frontal in every frame already: the renderer was trained on frontal
   frames only, and a turned head smears.
2. For each, `curl -o c.jpg 'http://127.0.0.1:8730/calm_sheet?clip=<clip>&start=<start>&seconds=4'`
   draws 8 frames across the window. Look at every sheet.
3. Choose the one that reads as listening: hands down or out of frame, upright, facing the
   camera, nothing moving but breathing. The mouth does not matter; the generated head
   replaces it. Rule out anything with text burned in.
4. Write `REST="<clip>@<start>"` into `creators/<id>/live.env` with the reason and the
   runner-up, then `./alive restart app`.
5. If no window qualifies (Dr K: 26 frontal candidates, all gesturing), pick one frontal,
   hands-down frame instead and set `REST_HOLD=1`: the body holds still and only the face
   moves.
6. Look at the built loop (`curl localhost:8800/idle`) before calling it done.

Keep the window short. Huberman never rests his hands for 20 s anywhere in 57 clips, and a
long window picked the quietest gesturing.

## Names you will meet

| name | what it is |
|---|---|
| `drk`, `healthygamer` | Dr K. The face and voice recipes call him `drk`; the clip and answer recipes call him `healthygamer`, after his channel |
| `face_v1`, `huberman_v1` | the released voice-to-motion models |
| `G4_teeth`, `huberman_G4_teeth` | the released renderers: a head with teeth and eyeballs |
| `F_wide`, `huberman_S_seed0` | the training runs those motion models came from |
| `teeth__<chunk>` | a clip's face meshes cooked from the motion model, for training the renderer |
| VHAP, FLAME | the face tracker, and the 3D head model it fits to video |
| xADA | the speech encoder (NVIDIA Audio2Face-3D) the motion model corrects |
| joins sheet | an image a person looks at to confirm sound, camera and mesh line up |

## Checked against the old trees (2026-10-04)

Old code and moved code, run on the same small input, outputs compared:

| step | input | result |
|---|---|---|
| frames and mattes (`prepare_take.py`) | 10 frames of Huberman | byte-identical |
| landmarks (`detect_landmarks.py`) | the same 10 frames | boxes identical; points within 0.00003 px |
| shot signals (`chunk_take.py`) | the same 10 frames | identical |
| speech encoder (`run_xada.py`) | 2 s of Huberman | identical |
| transcript index (`build.py`) | one Huberman take | identical chunks and exemplars |
| the live face, end to end | 8 s of audio on a fixed body clip | Huberman byte-identical; Dr K within JPEG noise |
| download (`./alive build --stage videos`) | one 31 min upload, empty checkout | the original take's layout |

## Not in a recipe yet

- **The answer engine's onboarding.** `make onboard` builds the store and map, but its
  oracle step calls `oracle.py` with arguments it no longer takes, and `harness/tune.py`
  rewrites a creator's profile without its `promoted` block. The driver therefore builds
  only the store and the map, and needs `promoted` already written.
- **Huberman's mouth runs ~0.2 s early.** His source video's sound is late; the motion
  model learned that offset. A per-creator correction is one number, not yet added.
