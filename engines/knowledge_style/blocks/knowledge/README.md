# knowledge

What they know.

**build**  corpus layers (transcript, topics) -> structured evidence with provenance
**serve**  the LLM queries it during generation

## The one thing to get right

Knowledge sits **beside** the LLM as a resource, not in the chain as a stage. The question
does not pass through it. That means the LLM can decide how much it needs, so one-shot
retrieval and multi-turn querying while reasoning are both the same shape here.

## Build it so P2 can train on it

P1 reads it at inference. P2 partly trains it into weights. If it is built as a retrieval
index and nothing else, P2 rebuilds from scratch and inherits nothing. Built as structured
evidence with links back to source, both consumers work.

This is the one P1 decision with real P2 consequences, and it is cheap now and expensive
later.

## P2

Gains an incremental build, and has to represent **gaps**: positions not taken, blind
spots. Mimicry only needs what they have. Complementation needs what they lack.

## Status

Not built. K1 (baseline control) unmeasured.

Retrieval and memory systems surveyed against K3 in `REFERENCES.md`. Style and authorship literature lives in `Mycode/persona_style_papers`, not here.
