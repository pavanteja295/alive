# Corpus contract

The one interface where Project 1 and Project 2 could fork the format, so it is specified
now rather than derived from working code.

A **take** is one addressable span of observation about one subject over continuous time.
A YouTube video is a take. A P2 session is a take. Same object, different channels filled
and different provenance.

---

## Layout

```
corpus/<subject>/<take_id>/
  take.json          the manifest. everything below is described here.
  media/
    video.*          source video, as received
    audio.*          source rate, unmodified
  transcript/        text with timing
  frames/            extracted stills. empty until checkpoint 2.
  depth/             empty in P1. filled in P2.
  layers/            annotations over this take's timeline
```

## take.json

```json
{
  "take_id":  "VNtv2SSEzjA",
  "subject":  "healthygamer",
  "source":   {"kind": "youtube", "id": "VNtv2SSEzjA", "url": "...", "uploaded": "20260428"},
  "span":     {"start_s": 0.0, "duration_s": 720.0},
  "state":    "closed",
  "channels": {
    "video":      [{"participant": "subject", "path": "media/video.mkv", "fps": 23.976,
                    "origin": "downloaded", "offset_s": 0.0}],
    "audio":      [{"participant": "subject", "path": "media/audio.wav", "rate": 48000,
                    "origin": "extracted", "offset_s": 0.0}],
    "transcript": [{"participant": "subject", "path": "transcript/en.json3",
                    "origin": "youtube-asr", "lang": "en", "offset_s": 0.0}],
    "frames":     [],
    "depth":      []
  },
  "layers": ["speakers", "topics"]
}
```

### The fields that carry P2

- **`source.kind`** is `youtube` or `upload` in P1, `session` in P2. Nothing downstream
  branches on it; it is provenance.
- **`state`** is `closed` for a finished take, `open` for a session still being written. P1
  takes are born closed. **The corpus is append-only, but an open take is still growing**, so
  a consumer either waits for `closed` or handles a moving tail explicitly.
- **`span.duration_s`** is `null` while `state` is `open`.
- **`channels.*.origin`** records how a channel was produced, not just that it exists. P1
  transcripts are `youtube-asr`; P2 transcripts are ours and name the model. A consumer that
  needs to know how much to trust the text reads this.
- **`depth`** is an empty list in P1. P2 fills it. No format change.

### Every channel is a list of streams

A YouTube video has one stream per modality. **A conversation has two**: the person's mic
and the agent's synthesized speech, the person's camera and the rendered face. A single
`audio` entry cannot express that, and without the split you cannot train a voice clone on
only the person's audio, or feed perception only their side.

So every channel is a **list**, and every stream names its `participant`. P1 is the
one-element case and nothing special-cases it.

```json
"audio": [
  {"participant": "subject", "path": "media/audio.subject.wav", "origin": "mic",  "offset_s": 0.0},
  {"participant": "agent",   "path": "media/audio.agent.wav",   "origin": "tts",  "offset_s": 0.0}
]
```

`depth` is participant-scoped for the same reason: it is depth *of the person*, from their
camera.

**Store what was actually played, not what was generated.** An interrupted agent turn was
cut off mid-utterance, and the person responded to what they heard. The stored stream is
the evidence; the full generated utterance is not.

### One timeline, many sources

**The take owns a single timeline. Every stream carries `offset_s` into it.**

In P1 this is trivial: video, audio and transcript came from one file and all offsets are
zero. In P2 the mic, the camera, the depth sensor and the TTS output are four independent
sources that start at different instants, and the sync between them is the entire value of
the corpus. A stream whose alignment is unknown is not usable evidence, so the offset is
required, not optional.

---

## Channels versus layers

**Channels are modalities.** Video, audio, text, frames, depth. What was observed.

**Layers are annotations over the timeline.** Speaker turns, topics, shot boundaries,
face-visible spans. What we worked out about it.

Each layer is one independent file. No layer knows about any other. Consumers compose what
they need: the intelligence side takes transcript plus topics, the face side intersects
shots with face-visible spans and ignores topics entirely.

**Adding a detector is a new file. Nothing existing changes.**

### speakers, and why it is the layer that matters for P2

P1 takes are solo monologue: one speaker, trivially. P2 sessions are dialogue, and **the
agent's own turns are stored too** because they are evidence of what the person was
responding to.

Same layer, same schema, different content:

```json
{"turns": [{"t0": 0.0, "t1": 4.2, "speaker": "subject"},
           {"t0": 4.2, "t1": 9.8, "speaker": "agent"}]}
```

A P1 take has one speaker and every turn says `subject`. Nothing special-cases it.

---

## Rules

- **Observations only, never solutions.** Curves, meshes, mattes, controls, embeddings and
  indexes are derived. They belong in `runs/`, keyed by the config that produced them.
- **Audio at source rate.** Consumers derive what they need. xADA wants 16 kHz mono; the
  resample belongs to the consumer, not here.
- **Segments are an index, not copies.** Layers are timestamps. One creator take is ~90 MB
  and Huberman is 25 GB; physical segment files are not survivable.
- **A property of the take lives here. A property of an experiment lives in `runs/`.**
- **Channels are declared, not assumed.** Absent is a value. A consumer that needs a channel
  checks `present` and fails clearly, rather than discovering a missing directory.

## Known caveat

YouTube auto-captions carry rolling-window duplication: each cue repeats the tail of the
previous one. Raw word counts are inflated roughly 20-25%. De-duplication is a transcript
prep step, and it runs before any count is trusted.
