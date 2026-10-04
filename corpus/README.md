# corpus

What we observed. Append-only.

```
corpus/<subject>/<take_id>/
  source.json      provenance: where it came from, clip window, format
  media/           video, audio at source rate
  transcript/      subtitles with timing
  frames/          extracted stills
  depth/           empty in P1, filled in P2
  layers/          annotations over this take's timeline
    topics.json  shots.json  speakers.json  face_spans.json
```

## Rules

- **Observations only, never solutions.** Curves, meshes, mattes and controls are derived
  and belong in `runs/`.
- **Channels are declared, not assumed.** A take says which it has. P1 fills RGB, P2 fills
  depth. Same object, more channels.
- **Audio at source rate.** Consumers derive what they need. xADA wants 16 kHz mono, the
  voice engine emits 48 kHz. The resample belongs to the consumer.
- **Segments are an index, not copies.** Layers are timestamps over the take. Huberman is
  25 GB; cutting physical segment files is not survivable.
- **A property of the video lives here. A property of an experiment lives in `runs/`.**

## Layers

Each detector writes one independent file. No layer knows about any other. Consumers
compose what they need: the intelligence track takes topics plus transcript, the face
track intersects shots with face-visible spans and ignores topics.

Adding a detector is a new file. Nothing existing changes.

## Status

Not built here. Downloaded takes live in `data/takes/` (`engines/audio2face/ingest_youtube_take.py`,
run by `./alive build`). They are in the Unreal capture-archive shape, which this layout
deliberately does not inherit.
