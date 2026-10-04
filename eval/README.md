# eval

Scoring and inspection. Every block supports three modes, and they produce different things.

| mode | produces | who reads it |
|---|---|---|
| **train** | a checkpoint | the next stage |
| **test** | a number, against a held-out set | a loop, or a promotion gate |
| **debug** | artifacts you look at | a person |

**Debug output is block-specific. Having one is not.**

---

## The rule that makes any of it valid

**To test one thing, pin everything else to ground truth.**

The corpus is synced audio, video and text, so every stage has a real target. What is pinned
and what varies is **declared in the run**, otherwise you cannot tell whether a number moved
because of the change or because a pin drifted.

"Improvements to a model" and "a frozen checkpoint" are the same operation with different
pins.

---

## Per block

### knowledge

- **debug**: the answer, plus the passages it drew on, plus which parts of the answer trace
  to which passage. Traceability is the point: it shows the model is grounded rather than
  confabulating.
- **test**: **no score yet.** Citations show grounding, not whether retrieval picked the
  *right* passages. A real metric needs labelled query-to-passage pairs, which do not exist.
  For now this is debug with a person reading it.

### manner

- **debug**: (unstyled input, model output, real target) side by side.
- **test**: parallel pairs held aside, scored. `ims` already emits the right shape, a
  composite of five metrics with fixed weights. Swap the network, re-run the same set.

This is the block that is loop-ready today, and the only one.

### voice

- **debug**: the audio, to listen to.
- **test**: deferred while it stays a provider. The capability exists whenever we want it,
  since the real audio for any transcript is in the corpus.

### audio2face

Three levels, three different questions. Say which one a run is answering.

| compare | tests | needs |
|---|---|---|
| predicted vs solved **controls** | `motion` alone | a solve on the test take |
| predicted vs solved **mesh** | motion + geometry | a solve, plus a bundle |
| rendered vs real **frames** | the whole engine | the full face bundle |

The first is cheap and isolates the block most likely being changed. The third is the only
one that measures the composition.

**Head pose must be pinned to the solve, not predicted.** Otherwise a still head is compared
against a video of a moving one and the number measures head motion rather than articulation.

**Practical constraint:** frame comparison needs all three face assets, and today only one
subject has them. Control comparison needs only a solve.

---

## Held-out comes from the dataset spec

Never chosen by hand, never chosen by the trainer. A loop that picks its own split will
eventually select against the test set. Putting the split in a versioned spec makes it
auditable.

## Scores rank, they do not decide

For the two deep aims the objectives are two-dimensional by definition: fast *and* close,
useful *and* faithful. A loop reports a frontier and must not collapse it. Canonical pose
losing on PSNR while being the right architecture is the standing proof.
