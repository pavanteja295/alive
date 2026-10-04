# alive

Ask a creator a question and watch them answer it: an answer in their words, from
their own videos, in their cloned voice, on their face and body. Built from public
YouTube videos of the creator. Working today for Dr K and Huberman.

Every frame is labelled AI-generated. The face, voice and words are generated; the
creator did not say them.

```bash
./alive up huberman        # or drk
# open http://127.0.0.1:8800
./alive down
```

## What happens when you ask

```
question -> answer text -> voice -> face motion -> pictures on the creator's real body
            (LLM, their    (cloned  (from the     (renderer trained on their face,
            transcripts)   voice)   audio)        pasted onto real footage)
```

| stage | where | serves on |
|---|---|---|
| answer | `engines/knowledge_style` | :8788, calls the Anthropic API |
| voice | `engines/text2audio/blocks/voice` | :8791 |
| face | `engines/audio2face` (motion, rig, renderer, live server) | :8730 |
| viewer | `app/` | :8800 |

`./alive up <creator>` reads `creators/<creator>/live.env` and starts the four in order.
`./alive check <creator>` says what is missing first.

Measured on the Huberman app: 60-80 s writing the answer (85% of the wait), first sound
about 1 s later, first frame about 3 s after the audio is handed over, then 80 frames a
second (2.7x real time).

## Hardware

| | serving | building a creator |
|---|---|---|
| GPU | NVIDIA, 12 GB (peak use 7 GB on a 90 s answer) | 16 GB (built on an RTX 5080) |
| RAM | 16 GB | 32 GB (face tracking runs 8 workers at ~1.2 GB each) |
| disk | ~3 GB of models per creator | ~150 GB working space per creator |
| other | an Anthropic API key | ffmpeg, yt-dlp, a network connection |

Building and serving do not share the card: `./alive down` before `./alive build`.
Everything was built and run on CUDA 12.8 (torch `+cu128`); other cards are untested.

## Where everything goes

```
alive/
  checkpoints/<creator>/      THE MODELS. Not in git, not released yet.
    voice/                    model.pt, reference.wav, vocab.txt, voice.json
    face/motion/<release>/    voice-to-motion model, with its corrective layer
    face/rig/                 the creator's face rig
    face/render/<run>/        the renderer, final iteration
  data/                       everything else that is not code. Not in git.
    takes/<folder>/<take>/    downloaded videos, audio, subtitles (./alive build fills it)
    answers/<creator>/        indexed transcripts; answers/.protokey holds the API key
    voice/                    the voice recipe's working state
    face/                     the face recipes' working state, plus shared face assets
  externals/                  outside code at our commits: bash env/setup_externals.sh
    vhap/                     face tracker (its data/, output/, export/ hold tracking)
    stavatar/                 face renderer, with our changes applied
    F5-TTS/                   voice model code
```

`data/` and `externals/` can be links to wherever the bytes live. The face engine reaches
them through relative links committed in `engines/audio2face`, so no code names a
machine path.

**Licensed files you supply yourself** (not redistributable, not in git):

| file | goes in | from |
|---|---|---|
| `flame2023.pkl`, `FLAME_masks.pkl` | `externals/vhap/asset/flame/` and `externals/stavatar/flame_model/assets/flame/` | the FLAME head model: register at https://flame.is.tue.mpg.de |
| `audio_encoder.onnx`, `animation_decoder.onnx` | `data/face/onnx/` | NVIDIA Audio2Face-3D (the speech encoder and its decoder) |
| a MetaHuman face rig, `face.dna` and `face_identity.dna` | `data/face/assets/` | Epic's MetaHuman Creator, under Epic's licence. *Shared face assets* below |
| an Anthropic API key | `data/answers/.protokey` | https://console.anthropic.com |

### Shared face assets

Every creator's face rig is the same MetaHuman rig, reshaped to their face. The files
below are built once from that MetaHuman rig and shared by all creators.

The code assumes a MetaHuman face rig of DNA version 2.5: 263 raw controls and 545
corrective shapes. Older rigs (Ada, 476 correctives) do not fit. The one Dr K's and
Huberman's rigs were built from is a MetaHuman of the author's own face and is not
released, so make one in MetaHuman Creator and export its DNA. Building these needs
OpenRigLogic (`bash env/setup_externals.sh --with-riglogic`):

| file, in `data/face/` | what it is | built by |
|---|---|---|
| `offset/rig_tables.npz` | the rig's arithmetic, so it runs without the DNA library | `engines/audio2face/offset/riglogic_numpy.py`, from `face_identity.dna` |
| `offset/cache/rig_names.npz` | control names and the GUI-to-raw map | `engines/audio2face/offset/data/extract_rig_names.py`; also reads one header from an Unreal Engine 5.8 install with the MetaHuman plugin (`UE_ROOT`) |
| `head/head_assets.npz` | the DNA's head mesh: triangles, UVs, vertex map, neutral shape | read out of the DNA's mesh 0; the original builder was not kept |
| `head/head_assets_0134.npz` | the head plus teeth and both eyeballs, appended | `engines/audio2face/extend_head_assets.py` (`--dump` with OpenRigLogic, then `--build`) |
| `head/uv_region_masks_256.pkl` | eye, nose, lips and forehead regions on the UV map | `engines/audio2face/make_region_masks.py` (writes the 512 version; the renderer reads it at 256) |

## Setting up a machine

```bash
bash env/setup_externals.sh      # tracker, renderer, voice code at our commits
# then the six Python environments: env/README.md
./alive check drk                # what is still missing
```

## Building a creator

```bash
./alive build huberman           # download their videos and rebuild every model
```

One command, from YouTube ids to `checkpoints/`. It asks each recipe what is next, runs
it, and replays the decisions recorded for that creator in `creators/<creator>/`. It
resumes where it stopped. About a day for Huberman and two for Dr K, mostly face
tracking. For a new creator, and for every step that needs a person: `recipes/README.md`.

### Reproducing Dr K and Huberman from scratch

Their videos, every decision a person made while building them, and the settings of
every model are in this repo (`creators/drk/`, `creators/huberman/`, the recipes'
profiles). What it cannot hold is the licensed and private input: FLAME, the speech
encoder, the MetaHuman rig and the assets built from it, and each creator's persona
note. With those in place (the tables above), `./alive build drk` and
`./alive build huberman` rebuild both from YouTube. The authors keep those inputs in a
private kit; anyone else supplies their own licensed copies and makes a MetaHuman rig.

## The map

```
engines/      the pipeline. each engine has swappable blocks inside it.
  knowledge_style/   question -> styled text
  text2audio/        styled text -> audio
  audio2face/        audio -> video
recipes/      how every model is built, in order, for any creator
creators/     per creator: live settings, the build record
app/          the viewer
env/          environments, patches to outside code, setup
alive         start, stop, check, build
docs/         write-ups of the answer engine's design (HTML; derived, owns nothing)
corpus/ runs/ agent/ pipelines/ eval/ ingest/ ops/ configs/
              the target layout (MANIFESTO.md); README-only so far
```

Read `MANIFESTO.md` for direction and standing decisions, `CLAUDE.md` for how we work,
`problems.md` for what is unresolved.

## Four rules that carry most of the weight

- **Nothing names a subject internally.** Subject is always an argument; per-creator
  values live in profiles and in `creators/<creator>/`.
- **Engines are pure wiring.** Every block trains and infers alone, on explicit inputs,
  with output you can read.
- **What we observed, worked on and kept are separate.** `data/` is the first two,
  `checkpoints/` is the third.
- **Status is computed from disk, never written.** `./alive check`, and each recipe's
  status command.

## Known limits

- Huberman's mouth runs about 0.2 s ahead of his voice: his source video's sound is late,
  and the motion model learned that offset.
- A hand or the microphone in front of the face is drawn over by the generated head.
- The answer engine's onboarding is only partly scripted (`recipes/README.md`, *Not in
  a recipe yet*).
