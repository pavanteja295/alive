# text2audio engine

**styled text -> audio at source rate.** Downstream consumers derive what they need; the
resample belongs to them.

## Blocks

| block | kind | state |
|---|---|---|
| `provider` | generic | ElevenLabs. Synthesis exists; **cloning does not** (V3). |
| `voice` | person-specific | a cloned voice id, promoted into the bundle |

The smallest engine, and largely a provider boundary. Swapping to our own model replaces
`provider` and leaves the engine's contract untouched.

## Project 2

**No new blocks. One contract requirement, and it must be honoured from the start.**

- **Streaming.** P1 could get away with whole-utterance synthesis. P2 cannot: the person
  interrupts, and the response has to begin before it is finished. Building batch-first and
  retrofitting means writing the serve path twice.
- **Cancellation.** An interrupted utterance stops mid-stream. **Store what was actually
  played, not what was generated** — the person responded to what they heard.
- `voice` improves from perception: more audio, and audio of them being themselves rather
  than presenting.
