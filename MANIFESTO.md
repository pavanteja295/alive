# Manifesto

Direction and standing decisions for Project 1 and Project 2.
Read this before proposing anything structural. It is not a spec.

---

## What we are building

A system that models a specific person well enough to think, speak and appear as them.

**Project 1.** The person is a creator. Everything known about them comes from their
public archive. You ask, it answers. The model does not change. Started as a rehearsal
for Project 2, but ships to real creators.

**Timing.** Project 1 is built completely first. Project 2 starts roughly two months after
it finishes, so nothing is built for P2 now. What P2 needs is that the structure ports, not
that any of it exists early.

**Project 2.** The person is the user. The system perceives them continuously and keeps
learning. It stops reproducing them and starts complementing them. A companion app.

Same machine. The difference is where data comes from.

---

## The core claim

Every part of this system is either a **generic engine** or a **person-specific asset**.

Generic engines (the LLM, xADA, ElevenLabs, the renderer) are shared across everyone and
replaceable at will. Person-specific assets are the entire product.

> General intelligence provides broad capability. A learned adapter provides local intelligence.

The test for anything new: is it generic or person-specific? Generic goes in an engine.
Person-specific belongs to a subject, is produced by a run, and is promoted.

---

## The shape

```
   PERCEPTION                INTELLIGENCE              INTERFACE

   video --+                                              +--> video
   audio --+--> adapters --> text --> thinks --> text ----+--> audio
   text  --+                                              +--> text
```

Perception collapses modalities into text. Interface expands text back out.
Intelligence is text-native.

**Text-native is a dated simplification, not a principle.** Text is the waist today because
it is the cheapest interface that works; it is a bandwidth choice. In future the
intelligence may read audio and video embeddings directly. That widens the waist without
rearranging the layers, so decision 13 still holds.

Two things follow. It is **where prosody comes back**: open question 7 exists because
prosody is discarded by the adapters on the way in and re-synthesized on the way out, and a
multimodal intelligence stops the inbound loss. And **input-side multimodal is cheap while
output-side would not be**: `manner` works on generated text, so widening what intelligence
reads leaves it untouched, while widening what it emits would leave a text-to-text adapter
with nothing to attach to.

So the perception-to-intelligence contract carries **named channels, of which text is one**,
rather than being literally a string. One unused field now, no format break later.

Three consequences that are forced, not chosen:

- **The interface modes nest.** Video needs audio needs text. Each mode is a prefix of the
  next, so interface selection is only a question of where you stop. Never branch the chain.
- **Perception is the mirror.** Same three rungs, run backwards. P1 has text perception
  only. P2 adds voice and later face.
- **Perception is a tee.** Store everything perceived. Route a subset to reasoning. The
  routing is config and can change later. Collection cannot be done retroactively.

---

## The assets

Six in P1, seven in P2.

| asset | captures | built from |
|---|---|---|
| knowledge | what they know | transcripts |
| manner | how they say it | their text, corrupted into pairs |
| voice | how they sound | their audio |
| geometry | what their face is | solve on video |
| motion | how their face moves | solved controls plus audio |
| appearance | how they render | solved meshes plus real frames |
| reasoning (P2) | how to complement them | interaction |

**Geometry is the chain root.** It constrains motion and appearance, and the three version
together. The other assets are independent.

**Geometry is a scaffold, not a fidelity contributor.** The Gaussians absorb mesh error by
construction, so a wrong mesh does not show up as visible distortion. It shows up as
**overfitting**: the compensation is fitted to the training frames and generalises worse to
unseen expressions and new audio. The fix for geometry error is more data and better motion,
not a better conform.

**The solve is not an asset.** It is a prerequisite three assets share, and the only step
that might need a human.

---

## The three checkpoints

Each is defined by what the person gives, and capability follows from that.

| | they give | we deliver |
|---|---|---|
| **1. Audio** | nothing, just their public archive | basic models, here is where we are |
| **2. Face** | their video | the best face we can make of you |
| **3. Project 2** | continuous presence, including depth | I get better at thinking with you and at being you |

Checkpoint 1 exercises everything P2 needs except perception. Every unverified premise
sits in checkpoint 2, and none of them block checkpoint 1.

In P2 one perception stream improves **both halves**: intelligence and every interface
rung. Checkpoint 2's face is a floor, not a ceiling.

---

## Standing decisions

Do not relitigate these without new evidence.

1. **Nothing names a subject internally.** Subject is always an argument.
2. **Three data tiers.** `corpus` is what we observed and is append-only. `runs` is what we
   tried and is disposable. `creators` is what we chose, versioned and promoted.
3. **A subject is a bundle**, versioned as one thing. Bundles may be partial, and a partial
   bundle is a release tier, not a failure.
4. **Serve is streaming by default.** Batch is one large chunk. Do not write serve twice.
5. **Status is computed from disk, never written.** Written state goes stale. Ours went
   stale in twelve days and claimed nothing had ever been run.
6. **Stages are idempotent and resume from disk.** They do not auto-chain unless asked.
7. **Build what transfers.** P1 is a rehearsal that ships. Do not build P1-only product
   surface, and do not build P2's personalization machinery early.
8. **Ship at the lowest rung that is honest.** Audio before face.
9. **A decision that cannot be a config cannot be auto-explored.** Push decisions into config.
10. **Single scores rank, they do not decide.** Canonical pose lost on PSNR (28.60 against
    29.46) and is still the right architecture for multi-take.
11. **Never train appearance on predicted meshes.** Only solved ones, or the appearance
    model silently absorbs the motion model's error and the two can never be separated.
12. **Learn offsets in the rig's control space, never in vertex space.** Vertex offsets
    bypass whatever corrective machinery the rig has and reintroduce the artifacts it exists
    to prevent.
13. **The three layers are invariant.** Perception, intelligence, interface. Improvements
    happen inside a layer, never by rearranging them. Duplex intelligence, a finetuned
    complementary model, a lighter renderer: all of these change what is in a layer and
    none of them change the shape.
14. **Every asset records what it was built against, and a dependency change invalidates
    it.** Two kinds, one mechanism: a residual on an engine (motion on xADA, manner on the
    helper LLMs, reasoning on the base LLM) and an asset on an upstream asset (motion and
    appearance on geometry). The bundle manifest is a lockfile for a person. Nothing errors
    when a dependency moves, results just go quietly wrong, so the check has to be explicit.
15. **The projects share machinery, not data.** P1's corpus is a creator's archive. P2's is
    the user's own interaction. They never merge, and no creator bundle carries into P2.
    What transfers is code, structure, contracts and the deployment path. So no P1 decision
    should be justified by what P2 might inherit from it.
16. **Every stage runs with its upstream replaced by ground truth.** The corpus is synced
    audio, video and text, so every stage has a real target: the video is ground truth for
    audio-to-face, the real audio for text-to-audio, the transcript for style. Hold
    everything fixed, vary one, measure against truth. A stage that can only run on the
    previous stage's live output cannot be isolated, and the verification property is lost.
    This is an API requirement, not a testing nicety.
17. **Every block trains and infers independently, and engines are pure wiring.** Any block
    can be run alone, on explicit inputs, with an output you can read. It follows that an
    engine contains no logic of its own: the moment a seam normalises or reconciles
    something, running blocks separately stops being equivalent to running the engine and
    every isolated measurement becomes a lie. Logic lives in blocks; a seam that needs
    behaviour means a block is missing. Boundaries are **serializable, not always
    serialized** — passed in memory when serving, dumped to disk when debugging or when a
    downstream block trains against ground truth.
18. **Disabled means not loaded.** Interface and perception modes are set by config, and a
    mode that is off never constructs its model. The serve chain is assembled from the
    selection, with lazy per-stage loading and nothing built at import time. Text mode is
    then genuinely zero-GPU rather than GPU-idle, which is the whole point. Distinguish
    **capability** (what the bundle can do) from **selection** (what was asked for):
    requesting video from a bundle with no face assets is an error, requesting text from a
    complete bundle is normal.
19. **Interface modes nest, perception modes do not.** Each interface mode is rendered from
    the previous, so text is a prefix of audio is a prefix of video, and selection is only a
    question of where to stop. Perception modalities are encoded independently, so they are
    a set of flags. Do not force perception into the interface's shape.
20. **Every residual refit is a single command.** We intend to change baselines on purpose,
    especially chasing real time, and each swap invalidates the residuals sitting on it. If
    refitting is a project rather than a command, the experiments will not happen.

---

## Priorities

Products are the vehicle. Two things are the actual work.

**1. Facial dynamics.** Can we build a facial module that is fast *and* close to the person?
Both at once. Either alone is already solved.

**2. Complementary intelligence.** Can we learn a person well enough to contribute what
general intelligence has and they do not, in a form that is useful *to them*? Not merely
different from them. Better for them.

Everything else is in service of those two.

| deep, this is the research | good enough, use what works |
|---|---|
| motion | manner (`ims` exists and is published) |
| appearance (the "fast" half) | voice (an API call) |
| geometry (the "close" half) | ingest, pipelines, agent, app, ops |
| knowledge (learn a person) | eval infrastructure, though not the metrics |
| reasoning (the complement) | |

### The rule this gives us

When something underperforms, the response depends on its bucket.

- **Deep**: investigate. That is the work, not a distraction from it.
- **Good enough**: accept, route around, or delay. Only investigate when results are
  *visibly* bad.

Two live applications. The TTS distribution shift that `domain_gap.py` measures is a
good-enough concern: recorded, not chased, until a face visibly breaks on synthesized
audio. A rough audio checkpoint is acceptable, because its job is to prove the machinery
and put two composed residuals in front of someone who knows the creator.

### Inside facial dynamics, the order is measured

Not all of the face matters equally, and this is measured rather than assumed:

1. **Lip closure.** Bilabials that fail to close are the most instantly-fake artifact a
   talking head has.
2. **Data volume.** Training so far has used a tiny fraction of the corpus available.
3. **Temporal artifacts.** Jitter within a block. Block boundaries are already cleared.
4. **Mesh identity.** Distant fourth, and only through the overfitting mechanism above.

Before any decomposition work, state which of these four it improves and how that would be
measured. Evidence for the ordering lives in the component READMEs and the register.

### Deep components get a frontier, good-enough components get a gate

A gate is a threshold: acceptable or not, move on. A frontier is a curve being pushed out.

Both aims are explicitly two-objective. Facial dynamics is fast *and* close. Complementary
intelligence is useful *and* faithful to their manner. A single score collapses exactly the
tradeoff we are trying to explore, so the one-number convention holds for the good-enough
bucket and is wrong for the deep one.

### What a playground needs

Cheap, and only for the deep five.

- Room for **competing implementations** side by side. Good-enough assets hold one.
- **One shared eval per asset**, so variants are comparable rather than each carrying its
  own notion of good.
- **A declared search space**, so a loop can explore without knowing what an asset is.
- **A baseline control**: how good is the bare baseline with no residual? Without it you
  cannot tell whether an asset earns its existence.

The facial playground has four real axes already: baseline engine, residual design,
representation (MetaHuman, FLAME, composed), sensor (mono, depth). All comparable through
the same 263-control contract.

The intelligence playground is the L0 to L4 ladder: bare LLM, plus retrieval, plus
structure, plus style, plus adapter. Its problem is not the playground. It is that there is
no scoreboard. Building the way to tell whether complementation worked comes before
climbing any rung.

---

## One half is verifiable, the other cannot be

Run the creator's real audio through audio to face to render, and compare output frames
against the real frames. **The interface half is verifiable end to end.**

The intelligence half is not, and structurally cannot be. An LLM answering a novel question
has no ground truth, and complementation means saying what the person never said, so there
is nothing to compare against.

This is why facial dynamics has measurable everything while complementation has no metric.
Not bad luck: one deep aim has ground truth built into the data, the other does not. The
face work can be driven by measurement. The intelligence work needs a judgment mechanism
invented before it can be steered at all.

---

## Everything is a residual

Not incidental. It is the thesis, instantiated at every layer.

| asset | baseline | the residual is |
|---|---|---|
| knowledge | what a general LLM already knows about them | what it does not |
| manner | generic prose | the rewrite toward their voice |
| reasoning | the base LLM | the adapter |
| geometry | the archetype face | the deviation |
| motion | xADA | the correction toward how they actually move |
| appearance | the mesh | everything the geometry does not explain |

Three consequences:

- **Measure the baseline before building the residual.** The first experiment for every
  asset is how good the baseline is alone.
- **The size of the residual says whether the asset is worth having.** A near-empty residual
  means either the baseline is already the person or the representation cannot capture them.
  Geometry's 4 mm is that signal firing.
- **The dependency rule is definitional, not a quirk.** Swapping a baseline does not
  inconvenience its residual, it makes it undefined.

**The open risk:** personalization is spread across six small deltas. Whether they compound
into someone recognizable or wash out into a slightly-off generic person has never been
measured. Every metric so far is per-asset, and the composition is the product. This is the
strongest argument for shipping the audio checkpoint early.

---

## How we work: commitments and loops

Engineering decisions are driven by loops wherever they can be. The division of labour is
fixed.

- **A person commits.** The structural choice, the direction, the starting point, recorded
  with its reason.
- **A loop varies.** Everything inside the boundary that commitment draws, reported back as
  ranked evidence.
- **A person promotes, or makes a new commitment.** The loop never decides.

### Every decision is one or the other, and it says which

Worked example. Integrating a depth signal into the appearance model:

| | |
|---|---|
| committed | depth enters as an additional supervision term; start at L1 |
| searchable | loss form (L2, weighted L1, Huber), its weight, its schedule |
| needs a new commitment | where depth enters, the architecture, the representation |

The third row is the one that matters. Without it, a loop asking "is L2 better" wanders
into changing what the loss is computed *on*, and the answer addresses a question nobody
asked.

### Commit narrowly, search widely

Commit to the minimum that makes the space explorable and leave the rest open. Every extra
commitment is a dimension the loop cannot help with.

### What each deep asset therefore records

- **its commitments**, with reasons, so the boundary is explicit
- **its search space**, meaning what is open given those commitments
- **train and eval as commands**, so a loop can drive it without knowing what it is
- **a held-out set the loop never sees**

### Three failure modes to design against

- **The loop overfits the selection set.** Picking a loss by scoring on the data it selects
  against makes the winner partly noise. Already seen: `seam_vs_gt.py` records that its
  checkpoint had `split=False` and saw both takes, so the conclusion needs a held-out
  re-run. A search loop turns that from occasional into routine.
- **Loops are cheap to start and expensive to run.** GPU contention is already live: the
  offset training held 56 to 70% of SM while another run got 30 to 45%. A sweep is many
  runs. Budget belongs in the search space, not discovered afterwards.
- **A loop must not collapse a frontier.** For the deep two the aims are two-objective by
  definition. Optimising one number picks wrong, and canonical pose losing on PSNR while
  being the right architecture is the standing proof.

---

## Known coupling

The layers are clean in code and coupled in distribution. `manner` changes word choice,
which changes what the voice engine produces, which changes the audio the face model sees.
`domain_gap.py` exists because the face model is sensitive to exactly that shift.

## Real time is a baseline swap

Going real time is not "use a smaller model". It is swap the baseline, then refit every
residual sitting on it. A lighter audio-to-face engine means refitting motion. A smaller
base LLM means refitting reasoning. A lighter renderer means refitting appearance. Budget
for the refit, not just the swap.

---

## What belongs in this document

**If we swapped one representation, engine or provider for another, nothing here should need
editing.** Anything that would is too low-level and belongs in a component README or the
register.

Principles survive an implementation change. The evidence behind them does not, so evidence
lives with the component it measures.

---

## Open questions

Recorded so they are not forgotten. None of them change the structure.

1. **Can creator video be solved at all?** Podcast footage cuts between cameras. If a
   creator records for us instead, this dissolves. Blocks checkpoint 2 only.
2. **How geometry gets built is unsettled**, and deprioritised by the ordering above. The
   route and its measurements live in `assets/face/geometry/`.
3. **Does the solve need a human?** If yes, full autorun is false for the face assets.
4. **The 33 ms budget.** Decides whether P2 has a face or is voice-only.
5. **Whose face does P2 wear?** If a creator's over the user's mind, a session binds two
   bundles rather than one.
6. **How is complementation measured?** Mimicry has an obvious metric. Complementation
   does not, and it is P2's central claim.
7. **Prosody is lost at the text waist**, twice. Deliberate for now. Note where it would
   re-enter.
8. **Creators as subjects or customers?** Decides whether they can record for us.

---
