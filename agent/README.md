# agent

Serve orchestration. Online, per session.

Owns the turn: take a question, ask retrieval for what is relevant, assemble, call the LLM,
run the result through manner, emit text plus the expressive channel.

## Where P2 attaches

This is the component that changes most between projects.

| | P1 | P2 |
|---|---|---|
| mode | turn-based: ask, answer | continuous session, always listening |
| objective | reproduce what they would say | contribute what they would miss |
| duplex | no | yes, with interruption |

Same slot, three things moving together. Mode lives here, objective lives in conditioning
plus the reasoning adapter, weights are config.

## Two rules

- **Serve is streaming by default.** Batch is one large chunk. Building batch first and
  retrofitting streaming means writing every serve path twice.
- **Log every interaction from day one**, even though P1 never learns from them. That log
  is P2's ingest path, not an observability nicety. You cannot collect retroactively.

## Open

Whether a session binds one bundle or two. If P2's companion ever wears a creator's face
over the user's mind, two bundles are active at once. Cheap to allow now, awkward later.

## Status

Not built as a block yet. The live chain runs today through `./alive` and `app/server.py`.
