# ingest

Anything a person gives us becomes a corpus take. One entry point per source kind.

## Source kinds

| spec | project | shape |
|---|---|---|
| a YouTube channel or list | P1 | batch, finite, pulled once |
| an uploaded file | P1 | batch |
| a live session | P2 | stream, continuous, writes as it goes |

Same verb, same corpus, same everything downstream. The two projects differ only here.

## Perception is a tee

In P2 an ingest stream is also perception. It splits:

```
perception --+--> adapters --> reasoning     a subset, now
             +--> corpus                     everything, always
```

**Store everything. Route selectively. The routing is config.** You can switch
face-to-reasoning on later without touching this layer. You cannot collect retroactively.

## Also here

The layer detectors, since their output is a property of the take. A topic detector will
probably call an LLM; that does not make it part of the intelligence track. This component
produces layers, the intelligence track consumes them.

## Status

Working code: `engines/audio2face/ingest_youtube_take.py`, run by `./alive build` (the channel crawler stayed in the old tree).
Resumable by construction, completion judged from disk, JSONL manifest, paced against
throttling, reasoned for 987 videos. It is the most production-ready code in the old tree
and should be ported nearly as-is, changing only the output layout.

Layer detectors: none exist.
