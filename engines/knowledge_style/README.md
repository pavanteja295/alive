# knowledge + style engine

**question -> styled text.** The only engine that runs in every interface mode.

```
question ----+
             v
knowledge -> llm -> manner -> styled text
  (feed)          (correct)
```

`knowledge` sits **beside** the LLM as a resource, not in the chain. The question does not
pass through it. So the LLM can decide how much it needs, and one-shot retrieval and
multi-turn querying are the same shape here.

`manner` is strictly downstream, text in and text out, and never sees the question or the
retrieved content. Not even the same model: content can be a large API model, manner is a
small local base with an adapter.

`FRAMING.md` is the argument this engine is built on: the problem statement, the curated
related work, and the L0-L4 ladder. Read it before proposing anything here.

External work spanning these blocks is in `REFERENCES.md`; block-specific work in each
block's own.

## Blocks

| block | kind | state |
|---|---|---|
| `llm` | generic | not ported |
| `knowledge` | person-specific, feed | not built. K1 baseline unmeasured. |
| `manner` | person-specific, correct | `ims` exists and is published |
| `reasoning` | person-specific, correct | **P2 only.** Not built, deliberately. |

## Project 2

**No restructure.** One block that already has a directory activates.

- `reasoning` turns on: an adapter for complementation rather than mimicry.
- `knowledge` gains an incremental build, and has to represent **gaps** (positions not
  taken, blind spots). Different content, same slot.
- `manner`'s evaluation changes: sounding like them becomes the delivery vehicle rather
  than the test. The block itself does not change.

**The unsolved thing is not structural.** Complementation has no metric, and mimicry does.
Nothing in this engine can be steered in P2 until that exists.
