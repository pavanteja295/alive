# live — the serving path, held open

Sound in, a face on screen. The models are loaded once and stay loaded, so a request
costs nothing to start.

**For most uses, run the whole app: `./alive up <creator>` (README.md).** This file is
the face service on its own: how to call it, what it costs, and what to know before
building on it.

```
cd engines/audio2face/gauss/live && python server.py --profile <id> --run <RUN_DEPLOY> --compile
#  ~15 s to boot, then 0 s per request  ->  http://127.0.0.1:8730
```

## Three ways to hand it audio

```bash
# the body is audio in any format ffmpeg reads
curl --data-binary @reply.wav 'http://127.0.0.1:8730/speak?label=reply'

# or write a file and it speaks by itself
cp reply.wav engines/audio2face/gauss/live/inbox/

# or use the page: a file picker and a microphone button
```

Everything converges on one conversion to 16 kHz mono before the model sees it.

## Reading it back

| | |
|---|---|
| `GET /frames?job=…` | every frame, newline-delimited JSON, `{"i": 42, "b64": "…"}`. Frame `i` belongs to second `i/30` of the audio. Ends with `{"end": true}`. |
| `GET /audio?job=…` | the converted wav — **the exact samples the model heard**. Play this one. |
| `GET /stats?job=…` | progress and per-stage timing while it runs |
| `GET /latest` | the newest job, so a client notices inbox drops nobody asked for |

Options on the query string of `/speak` or `/run`: `head=generate|rest`, `fp16=1`,
`stream=1&chunk=1.0`.

### On his real body

`body=random` pastes the generated head onto a real clip of the creator instead of the
studio render: a clip long enough for the audio is picked at random, at a random start,
and the head is drawn with that clip's own camera and head movement, so it lands where
his real head was. `body=<chunk>&body_start=<frame>` names one. `body=calm` takes the stillest stretch of his footage, judged from the picture (for a resting loop). `--body` on the server sets the default for jobs that carry no query, such as inbox drops. Any centred clip of 5 s
or more with an export and a head track qualifies; the boot log says how many.

Measured (huberman): 78-80 fps render, first frame 0.50-0.79 s, 2.6x real time. Every
frame matched its real source frame outside the head (mean error ~1 on 0-255).

- The real frames are decoded on CPU threads ahead of the renderer. Decoding them on
  the card from the render thread gave random frames of coloured noise.
- `GET /calm?seconds=4&k=6` lists the stillest non-overlapping windows of his footage;
  `GET /calm_sheet?clip=&start=&seconds=4` draws one as 8 frames. They exist for the
  resting-stretch step: the scan proposes, a person (Claude) chooses by looking.
- `body_hold=1` freezes one real frame with the head held at its placement.
- **Only frontal footage is replayed.** The renderer was trained on frontal frames only
  (filter_frontal.py, above 0.60 on the blink step's frontal score), so a replayed head
  turned further smears. Answers take a stretch where every frame is frontal; if none is
  as long as the answer, they play back and forth inside one of the longest
  (`body.span` in the stats). `/calm` proposes only frontal windows.
- `random:<recording>` / `calm:<recording>` keep to one recording, for creators filmed in
  several sets.
- The pasted head's edge is pulled in ~4 px. Its outermost blobs carry the background
  it was trained against (drk: a white studio), a light rim on a dark room.
- Geometry is built 300 frames at a time on the card. Whole-clip, an 80 s body reply
  took the worker to 12 GB and starved the voice on the same card.
- Not in stream mode yet. A hand or microphone in front of the face is drawn over
  (see the mesh-to-render recipe, *Open*).

## What it costs

| | |
|---|---|
| first frame after the audio lands | 0.83 s |
| each frame after that | 9.7 ms — 103 fps (6.4 ms, 158 fps in fp16) |
| a 10 s reply, start to finish | ~3.9 s, so playback cannot stall |

## The one thing to know before designing around it

The face can only be drawn once the model has heard **three seconds past** the moment
it is drawing — the encoder looks forward inside a fixed 30 s block and the correction
on top of it is bidirectional. Both are structural.

- **Audio you generated before it plays** (text-to-speech, a file): the lookahead is
  free, because the whole waveform already exists. This works today.
- **Audio arriving as someone speaks**: the face lands ~4.8 s behind. Streaming mode
  runs, and measures exactly that. It is not a conversation.

## Two traps

- **Audio must sit at least 5 s into what the encoder reads, and be asked for at
  t − 110 ms.** The worker does both. Bypass it and the mouth stops meaning anything:
  correlation with his real mouth falls from **+0.76 to −0.08**, while still looking
  like a plausibly moving face.
- **`infer.py --audio` had that bug; fixed 2026-10-01.** It now reads a wav exactly as
  the worker does: 5 s lead, driver at the creator's measured lag. Checked against the
  clip path on the same speech: jaw correlation +0.977, aligned at zero shift.

## The files

| | |
|---|---|
| `server.py` | HTTP, the resident renderer, the job queue, the inbox watcher |
| `audio_worker.py` | phase one held open: audio → controls → rig → corrective layer → posed heads, all on the card |
| `build_clips.py` | cuts the demo shelf — held-out test clips, unseen recordings, a different speaker |
| `index.html` | the page |
| `inbox/` | drop audio here (gitignored) |
| `spoken/` | converted audio, per job (gitignored) |

Rebuild the shelf with `python build_clips.py --profile <id>`; it is not committed.
