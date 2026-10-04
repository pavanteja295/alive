# Answering as the person

Framing for the intelligence half: what we are building, why the obvious formulation is
wrong, what the literature already settles, and the first structure to try.

Written 2026-09-04 against the corpus on disk and the 54 papers in
`Mycode/persona_style_papers`. Component decisions live in the block READMEs and
`problems.md`; this document is the argument that connects them.

---

## 1. Introduction

A person with a large public archive answers the same questions many times. People ask
them anyway, because the answer they want is not a fact but a **position**: how this
particular person frames the problem, what they refuse to concede, where they land. The
archive contains hundreds of hours of exactly that, and it is unsearchable in the way that
matters. You cannot ask it a question.

We want to build the thing you can ask. Given a creator's archive and a question they have
never been asked in those words, produce the answer they would have given: their position,
their reasoning, their words. Not a search over their videos, and not a general assistant
that has read them. A **replacement for the person answering directly**, good enough that
someone who knows their work cannot tell.

The naive formulation of that goal is to match the distribution of the archive. It is
wrong in two ways, and both are worth stating before anything is built.

**It is the wrong conditional.** The archive is the person talking to a camera for twenty
minutes, unprompted and edited. A user asking a question wants ninety seconds back. The
corpus samples `p(text | topic, broadcast)` and we deploy at `p(text | question,
conversation)`. Matching the corpus distribution literally produces a model that opens
every reply with "Today we're going to talk about."

**It targets the half that does not work.** The obvious reading of "sound like them" is
style transfer, and style transfer is where this literature has spent its effort and where
it keeps failing. The best-validated negative result is `09/2509.14543` (EMNLP 2025): over
400 real authors and 40,000+ generations per model, LLMs approximate style in structured
domains like news and email and **fail on informal, conversational writing**, which is the
end of the spectrum spoken monologue sits at. Meanwhile the best-validated positive result
simulates a person from a two-hour interview and predicts their held-out survey responses
at **83% of that person's own two-week test-retest consistency** (`02/2411.10109`, 311
citations), against 74% from demographics alone.

Read together those two results say something specific and, for us, decisive:

> Reproducing **how** a person writes is unsolved and possibly not the valuable part.
> Predicting **what position** a person takes already works, from far less data than we have.

### How much to trust any of this

Most of the recent work in this area is arXiv-only with near-zero citations, and the
temptation is to treat a 2026 preprint's headline number as fact because it is recent and
specific. Citation counts pulled from Semantic Scholar on 2026-09-04:

| tier | papers | how it is used here |
|---|---|---|
| **relied on** | `2307.03172` lost-in-the-middle (TACL, **4899**), `02/2411.10109` (**311**), `01/2010.05700` STRAP (EMNLP, **289**), `2502.05589` SeCom (ICLR, **101**), `08/2204.04907` (**101**), `2410.13070` chunking (NAACL Findings, **69**), `08/2410.12757` (**35**), `11/2306.00539` (ACL, **20**), `09/2509.14543` (EMNLP, **13**), `08/2311.07564` (TACL, **10**) | load-bearing. a design decision may rest on these |
| **corroboration only** | `2604.04853` MemMachine (12), `08/2311.07564`'s newer neighbours, `07/2603.26680` AlpsBench (SIGIR, 6), `2601.15300` (6) | may support a decision already justified by the tier above. never the sole reason |
| **noted, not relied on** | `11/2604.26460` (**0**, single author), `09/2603.29454` (**0**), `09/2603.23219` (**0**), `2606.26185` (1), `09/2603.23069` AuthorMix (1), `11/2508.06374` (2), `09/2505.07888` ZeroStylus (2), `2509.24147` LOT (3) | interesting framings and useful vocabulary. no decision may depend on their numbers |

Two consequences worth stating rather than leaving implicit. **The `2604.26460` calibration
result is in the bottom tier**, so the specific figures 0.484-0.508 against a 0.626
cross-author floor are not evidence, they are a hypothesis; what survives is its *method*,
calibrating a scale before reading it, which costs nothing and is right regardless. And
**every 2026 arXiv number in this document should be assumed unreplicated**, including the
ones that agree with us. `GOLD_STANDARD.md` already set this bar and the first draft of
this document did not meet it.

So this is not a style-transfer project with a knowledge component bolted on. It is a
position-prediction project that has to speak in the right voice at the end. That ordering
is the opposite of where the field's effort sits, and the opposite of where our own paper
shelf sits: six of its twelve buckets feed surface style, two feed evaluation, and
evaluation is the thing that does not exist.

---

## 2. Problem statement

### 2.1 What we are given

A corpus `C` of monologue takes by one person `P`. Today, 17 takes: 6.09 hours, 69,366
words of ASR transcript at 190 wpm, with aligned audio and video. The archive supports
scaling this to several hundred takes, on the order of 1.2M words, which is roughly
1.6M tokens.

That number matters because it crosses a threshold the literature is explicit about.
`01/2212.08986` sets the high-resource bar for authorship style transfer at **>100,000
words** of target-author text and studies the low-resource case below it because that is
where non-famous authors sit. At 300 takes we are an order of magnitude *above* the
high-resource bar. Every mechanism designed to compensate for scarcity, style embeddings,
exemplar conditioning, adapter mixing, soft style prompts, is solving a problem we do not
have. `GOLD_STANDARD.md` already records the regime split from ASTRAPOP (`09/2403.08043`):
below ~16 samples per author, exemplar conditioning wins; with real volume, one model per
author wins, and the boring baseline is also the strongest.

### 2.2 What we want

Given a question `q`, produce `r` drawn from `p_P(r | q)`: what `P` would say if asked.

This factors into four terms, and their difficulty is wildly uneven:

| | term | what it is | status in the literature |
|---|---|---|---|
| 1 | `p(stance \| q, C)` | which position `P` takes | **works**. `02/2411.10109` at 83% of human self-consistency |
| 2 | `p(arc \| stance, q, C)` | how `P` builds to it: what is conceded, in what order | **unstudied at the person level**. no block owns it |
| 3 | `p(words \| arc, C)` | lexicon, rhythm, filler, idiom | **fails**. `11/2604.26460`, `09/2509.14543` |
| 4 | format | the corpus is broadcast, deployment is conversation | not a modelling problem. a data-selection decision |

Term 4 is not learned. It is chosen, and the choice is ours: either find format-matched
archive material (interviews, streams, Q&A, where the person answers a real person) or
declare the product to be "ask it a topic, get a video essay" and accept the corpus as-is.
Nothing downstream is well-posed until that is settled.

### 2.3 The claim being tested

Everything in this system is a residual on a generic engine. For the intelligence half the
baseline is a frontier LLM that has read the public internet, and the residual is whatever
that model does not already get right about `P`.

> **`H`: a frontier LLM plus this person's verbatim archive predicts their positions on
> unseen questions at a level distinguishable from the same model without the archive, and
> the gap is large enough to be worth engineering.**

`H` has never been measured for this person, and no retrieval, adapter or steering result
means anything until it is. It is `problems.md` K1 and it is the first experiment.

### 2.4 The measurement problem, which is worse than it looks

"Indistinguishable" needs an operational definition, and the field does not have a
trustworthy one.

- `11/2604.26460` computes three standard measurement families over the same outputs, a
  trained AV model (LUAR), an LLM-as-judge with decoupled trait matching, and classical
  function-word stylometrics, and finds **near-zero pairwise correlation, `|r| < 0.07`**.
  Metric choice, not method quality, determines what a paper concludes.
- `11/2306.00539` meta-analyses the field and finds not only a standardization gap but a
  **validation gap**: few automated metrics have ever been checked against human judgement.
- `01/2010.05700` (STRAP), surveying 23 style-transfer papers, found the automatic metrics
  gameable and had to propose fixes.
- `08/2311.07564` shows authorship models transfer to speech transcripts but **degrade
  sharply under topic control**, and that plain character n-grams are competitive with
  neural models there. Our takes are topic-separated by construction, so an uncontrolled
  discriminator wins by reading the topic and tells us nothing.

The methodological fix is in `11/2604.26460` and it is cheap: **calibrate the scale before
using it.** Establish the human ceiling and the cross-author floor with the same metric on
the same data, then place every method between them. Uncalibrated, four methods looked like
they were working; calibrated, they were below the floor.

We can do this immediately and better than the paper can, because the two anchors are
already on disk:

- **ceiling**: `P` against `P`, two held-out takes by the same person.
- **floor**: `P` against a different creator. `takes/huberman` is 25 GB of exactly that.

A metric that cannot separate Dr K from Huberman is not a metric. A method that scores
below that floor is not personalizing, whatever its absolute number looks like.

---

## 3. Related work, curated

54 papers were surveyed. Most are not relevant, and saying which and why is the useful
part. Full routing is in the component `REFERENCES` files.

### 3.1 Person simulation: evidence the target is reachable

`02/2411.10109` is the load-bearing result. 1,052 Americans, two-hour semi-structured
interviews on the American Voices Project schedule, agents built from the transcripts,
scored on held-out General Social Survey items. Interview-grounded agents reach 83% of the
participant's own two-week test-retest consistency; demographics-only reach 74%; both
combined reach 86%, a modest gain over either. Three things transfer to us. **Text
transcripts of a person talking are sufficient input.** **Two hours is enough**, and we
have three times that today. **The right ceiling is the person's own self-consistency**,
not perfect agreement, because people do not agree with themselves.

`02/2310.10158` (Character-LLM) and `02/2501.15427` (OpenCharacter) train per-character
models rather than prompting, which matches the high-resource regime, but both build
characters from synthesized profiles rather than a real archive. The recipe is relevant;
the data assumption is not. `02/2406.00627` and `02/2502.03821` are about generating and
scoring role-play dialogue and trait consistency, and are downstream of a scoreboard we do
not have.

### 3.2 The negative results, which are the important ones

- `11/2604.26460`: four inference-time personalization methods at 0.484-0.508 against a
  0.626 cross-author floor and 0.756 human ceiling. **The strongest single argument against
  building surface style first.**
- `09/2509.14543`: 400+ real authors, 40,000+ generations per model. LLMs approximate style
  in **structured** domains (news, email) and fail on **informal, conversational** ones
  (blogs, forums). Spoken monologue is the informal end.
- `09/2603.29454`: GPT-4o impersonation under four prompting conditions across three genres
  does **not** evade forensic authorship verification, neither n-gram tracing, RBI and
  LambdaG nor neural AV.
- `09/2603.23219`: GPT-4o, Gemini 1.5 Pro and Claude Sonnet 3.5 mimicking Whitman,
  Wordsworth, Trump and Obama remain **highly detectable** under BERT plus XGBoost over
  LIWC, perplexity and readability features.

The consistent finding across four independent evaluations is that prompted style
imitation is detectable and calibrated-below-floor. We should expect our style layer to be
detectable too, and should not spend the project's risk budget there.

### 3.3 Where the supervised signal comes from

`01/2010.05700` (STRAP) is the recipe everything else reuses: reformulate style transfer as
paraphrase generation, run the person's text through a paraphraser to strip style, and
train the inverse map from stripped back to original. It manufactures a parallel corpus
from one-sided data, which is our situation exactly. `ims` implements this half and
`DECISION.md` keeps it as the one asset with compounding return.

Its ancestry is bucket `10_translation`: STRAP's trick is back-translation
(`1511.06709`, `1804.07755`, `1711.00043`, `1804.09057`). `1808.09381` is the one to read
if the pair corpus underperforms, because it is the large-data study and finds **sampled or
noised** synthetic sources give a much stronger training signal than beam or greedy ones.
That is a directly applicable knob on how the shadows are generated.

### 3.4 Style-transfer machinery we are skipping, and why

`01/2406.15586` (TinyStyler), `01/2212.08986`, `01/2407.15556` (SETTP) and
`09/2603.23069` (AuthorMix) are all explicitly low-resource methods: authorship embeddings,
transferable prompts, layer-wise adapter mixing. They exist to work around not having data.
`01/2312.17242` (StyleMC) is the honest counter-evidence to our chosen route, finding that
instruction-tuned models struggle to reproduce author style **demonstrated in a prompt**,
but its regime is a small writing sample and ours is 100x that. It names the experiment
rather than settling it.

`01/2505.00679` (register analysis) is the exception worth keeping: a prompt-only method
that describes the exemplar's *register* instead of pasting exemplars, reporting better
style strength and meaning preservation. It is the cheapest available upgrade to a
context-based baseline.

`09/2505.07888` (ZeroStylus) is the only surveyed work operating at the level of term 2
above: it extracts **sentence-level and paragraph-level template repositories** from
reference texts and generates against them, arguing that paragraph-level structural
coherence cannot be reached by sentence-level rewriting. That is the same argument as our
K4, arrived at independently.

### 3.5 Reasoning patterns, the unowned middle

`2509.24147` (LOT) has an LLM propose features that distinguish reasoning traces by their
producer, keeps the ones that predict authorship, and iterates into a **human-readable
taxonomy**, reaching 80-100% discrimination across 12 open reasoning models and then using
the taxonomy to lift a smaller model's GPQA by 3.3-5.7%. It is the closest published method
for building term 2 as something a person can read and edit. Caveat: distinguishing
language models on math and code is not distinguishing humans on speech, and the transfer
is unvalidated.

### 3.6 Retrieval and memory

`07/2304.11406` (LaMP), `07/2407.11016` (LongLaMP) and `07/2406.02888` (HYDRA) are the
retrieval-personalization line. HYDRA's **reranker over retrieved history** is the reusable
part; its cross-user factorization is not, since we serve one subject at a time.
`GOLD_STANDARD.md` notes retrieval is the only component whose quality rises with corpus
size forever, while style saturates.

On the memory side, `2604.04853` (MemMachine) and MemPalace both **preserve verbatim
episodes and refuse extraction**, which is our K3 position; MemMachine's contextualized
retrieval, expanding a nucleus match with its surrounding context, is the right retrieval
unit for a transcript, where a hit at 14:32 is meaningless without the 90 seconds around
it. LightRAG and RAG-Anything index by LLM entity extraction and therefore reverse K3.
`07/2603.26680` (AlpsBench) supplies the outside evidence for that decision: over 2,500
real dialogue sequences, models **fail to reliably extract latent traits**, which is where
an extraction-first index loses what it cannot recover.

`2505.19300` (SituatedThinker) trains a model to interleave reasoning with external lookups
through fixed interfaces. That is the trained version of this engine's standing commitment
that knowledge sits beside the LLM rather than in the chain. Reference for the shape, far
past where we are.

### 3.7 Interventions, deferred

`04/2106.09685` (LoRA), `03/2404.03592` (ReFT, 15-65x more parameter-efficient),
`04/2305.18290` (DPO), `03/2312.06681` (CAA), `05/2401.08565` (proxy-tuning, closing 88% of
the tuning gap on a 70B model using 7B proxies) and `05/2105.03023` (DExperts) are all
sound and all premature: they optimize a thing we have not shown works. One is worth
remembering for deployment rather than training: `03/2605.10664` identifies **KV-cache
contamination** as why persona steering decays over a conversation, and reports coherence
drift improving from -18.6 to -1.9 with turn-10 trait expression from 78.0 to 93.1. If the
product is multi-turn, persona decay is a known failure with a known fix.

### 3.8 Evaluation substrate

`11/2604.26460` for calibration, `11/2508.06374` for which metrics survive scrutiny across
eight tasks and three discrimination settings, `11/2306.00539` for the validation gap,
`08/2311.07564` for the speech-domain caveat and the competitiveness of character n-grams,
`08/2204.04907` (Wegmann) and `08/2410.12757` (StyleDistance, 40 controlled style features,
public weights) as candidate signals. `08/2401.06712` is the inverse framing, detecting
machine-generated text from style representations, and supplies a ready-made adversary if
we ever need a best-of-N scorer.

---

## 4. First structure to try

The goal of this section is **one baseline that is stable and scales**, not the best
achievable system. Everything below is chosen because it is boring, replicated, and does
not change shape when the corpus grows 100x. Improvements attach later, and section 4.5
lists the ones we are deliberately refusing today.

### 4.0 What "stable and scalable" has to mean here

Three tests. A component that fails any of them is not baseline material, however well it
scores.

1. **Invariant to corpus growth.** The corpus goes from 6 hours to several hundred. A
   component whose cost or accuracy is a function of total corpus size is not a baseline,
   it is a demo that expires.
2. **Reproducible enough to detect a 5% change.** If re-running the same configuration
   moves the number more than a real improvement would, the scoreboard cannot rank
   anything and every later decision is noise.
3. **One moving part at a time.** Each rung adds exactly one thing, so a regression has one
   suspect.

### 4.1 The obvious baseline fails test 1

`persona_style_papers/DECISION.md` keeps "frontier model plus large verbatim context" as
the least-action route. That was written against a 50-90k word corpus, roughly 92k tokens,
which is comfortably inside any modern window. It does not survive the scale we are planning
for. Two reasons, and the order matters: the first needs no citation at all.

**Cost and latency are linear in corpus size.** Every query pays for the whole archive. Going
from 17 takes to 300 multiplies the per-query bill by roughly 17x and buys nothing, because
the answer to any one question depends on a small fraction of it. Retrieval is flat in
corpus size. This is arithmetic, it is the whole of test 1, and no paper can overturn it.

**Accuracy is not uniform across a long window.** `2307.03172` (Liu et al., TACL, 4899
citations) is the well-replicated result: performance is highest when the relevant span sits
at the beginning or end of the context and degrades substantially when it sits in the
middle. A larger window does not fix this, it enlarges the middle. Our archive has no
privileged ordering, so relevant material lands mid-context by default.

A weaker preprint, `2601.15300`, reports catastrophic degradation past 40-50% of maximum
context with F1 falling 0.55 to 0.30. It is 6 citations, arXiv only, and tests exactly one
small open model (Qwen2.5-7B), so it is **corroboration and nothing more**. An earlier draft
of this section leaned on it, and on 2026 vendor-blog benchmark tables naming models I could
not verify exist. Both are struck. The argument above does not need them.

The correction to the framing in section 1:

> Putting the whole archive in context is a **measurement instrument**, not a baseline. It
> is available to us exactly now, at 92k tokens, and stops being trustworthy well before the
> archive is complete. Use it while it exists to establish what "all the evidence, perfectly
> retrieved" is worth, then never ship it.

Retrieval is not an optimization over that. It is the only shape whose cost is invariant to
how much of the person we have, which is test 1.

### 4.2 The scoreboard, built first

Two independent problems: the scale is uncalibrated, and the measurements are noisy. Both
have cheap fixes and neither is optional.

**Calibrate the scale** (`11/2604.26460`). Anchor with the two extremes before placing any
method between them.

```
held-out takes by P     ------> CEILING   (P vs P, same metric)
takes by another creator ------> FLOOR    (P vs Huberman, 25 GB already on disk)
                                  |
        every method lands on this segment, or below the floor and is not personalizing
```

**Control the noise.** The protocol below is ordinary measurement discipline and needs no
citation to justify: run it more than once, report the spread, pin the versions. The recent
preprints are useful for *how badly* it goes wrong rather than for whether to do it, and
they sit in the bottom trust tier:

- `2606.19544` (8 cites, arXiv): 21 judge models, nine providers, ~541,000 judgments. Raw
  exact-match agreement reportedly **inflates by 33-41 points** against chance-corrected
  Cohen's kappa, and two production judges pair test-retest reliability >0.95 with position
  bias >0.10. Unreplicated, but the direction matches long-standing practice: consistency is
  not validity, so chance-correct.
- Position bias reaches 75% toward whichever answer is shown first; self-preference runs
  about +10% for one frontier judge and +25% for another. Any judge scoring our own
  system's output is scoring something it partly wrote.
- `2606.26185` and `2407.10457`: temperature control is **necessary but not sufficient**.
  Even forced greedy decoding is not reproducible in production serving, because of
  batch-dependent floating-point reduction, MoE routing and provider load balancing.

The protocol that follows is mechanical:

| | |
|---|---|
| primary metric | character n-grams. `08/2311.07564` finds them competitive with neural models on speech, and they are deterministic and free |
| secondary | one neural AV model, reported separately, never averaged in |
| tertiary | LLM judge, position-randomized, chance-corrected with kappa, never the primary |
| repetition | 5 to 10 rollouts per question, report mean and standard deviation |
| pinning | model version, seed, sampling params, evaluator version, all recorded per run |
| aggregation | none. `MANIFESTO.md` decision 10, and `01/2010.05700` found aggregate style metrics gameable |

Test 2 is satisfied when the spread across rollouts is smaller than the effect we intend to
detect. If it is not, no rung below can be ranked and that is the finding.

### 4.3 The baseline itself, specified boringly

Every choice here is the dull one, and each has a citation saying the clever alternative
did not pay.

| decision | choice | why not the clever thing |
|---|---|---|
| index unit | **fixed-size chunks** with overlap, over `.json3` transcripts | `2410.13070` (NAACL 2025 Findings, 69 cites) evaluated semantic chunking on document retrieval, evidence retrieval and answer generation and found **fixed-size consistently outperformed it**. Note the honest tension: `2502.05589` (ICLR, 101 cites) finds segment-level beats turn-, session- and summary-level on *dialogue*. Different data and different comparison set, so not a contradiction, but fixed-size wins the baseline slot because it is the null hypothesis and the one study that directly tested replacing it came back negative |
| chunk content | verbatim, with take id and timestamp as metadata | K3. `07/2603.26680` finds models fail to reliably extract latent traits, so extraction loses what it cannot recover |
| retrieval | hybrid lexical plus vector, then a reranker | the shape kotaemon and `07/2406.02888` (HYDRA) both settle on. No graph, no entity extraction |
| context assembly | expand each hit with its surrounding transcript window | `2604.04853`'s contextualized retrieval. Small chunks retrieve well and starve the generator: reported cases of ~43-token chunks retrieving correctly and still failing to support an answer |
| generation | frontier model, no adapter, no steering | sections 3.2 and 3.7. Every intervention optimizes something not yet shown to work |
| transcript cleanup | none | `08/2311.07564`: normalized lowercase transcripts work as well or better. Do not prettify ASR |

That is the whole baseline. It is deliberately unremarkable, it is flat in corpus size, and
every part of it has a published negative result protecting it from being replaced by
something fancier.

### 4.4 The ladder, revised

| rung | what is added | what it tests | status |
|---|---|---|---|
| **L0** | bare frontier model, told who the person is | K1: how much is already in the weights | control |
| **L1-oracle** | entire archive in context, no retrieval | the ceiling of perfect retrieval | **instrument, expires at ~40 takes** |
| **L2** | the 4.3 baseline | **this is the baseline.** how much of L1-oracle survives retrieval | ship this |
| **L3** | stance card, always present, never retrieved | term 2, the unowned middle | later |
| **L4** | manner adapter on STRAP pairs (`ims`) | term 3, surface | later, and may not pay |

The important change from a naive reading: **L1 is not a rung, it is a measuring stick.**
The gap between L1-oracle and L2 is the retrieval loss, and it is the only number that says
whether the retrieval design is good enough. Running L1 later is impossible, so it has to
happen while the corpus is still small.

### 4.5 What we are deliberately not building yet

Each of these is real, several are well-evidenced, and none of them belong in a first
baseline. Recorded so the decision is visible rather than forgotten.

| deferred | why not now |
|---|---|
| graph or entity-extraction indexing (LightRAG, RAG-Anything) | reverses K3, and adds an LLM pass over the whole corpus to the build |
| segment-level memory construction (`2502.05589`, SeCom) | **the strongest deferral on this list, and not deferred for weakness.** ICLR, 101 citations, beats turn-, session- and summary-level memory. Deferred on **domain**: it segments multi-session *dialogue* and our takes are continuous monologue with no turn or session boundaries for its segmenter to key on. It is the named first upgrade to L2, not a bell or a whistle |
| adapters, steering, ReFT, DPO, proxy-tuning | optimizing a system not yet shown to work |
| best-of-N with a style scorer | `DECISION.md` gates this on prompting visibly under-committing |
| register-analysis prompting (`01/2505.00679`) | one-paragraph change, genuinely cheap, but it is still a second variable. Add it as the first L2 variant, not inside the baseline |
| multi-turn persona decay fixes (`03/2605.10664`, GCAD) | only matters once the product is multi-turn, and the failure is known and has a known fix |
| trained interleaved retrieval (`2505.19300`) | far past where we are |

The rule this encodes: **the baseline is allowed to be beaten, it is not allowed to be
unstable.** A 2% win that adds a build-time LLM pass, a second failure mode, or a new
dependency on a moving model costs more than it returns at this stage.

### 4.6 The day-one experiment

Unchanged, and still the only step that cannot be skipped. L0 and L1-oracle in one sitting:

1. Hold out 3 to 4 takes. Never index them, never train on them.
2. Pull 30 questions those takes answer.
3. Frontier model, remaining transcripts verbatim in context, no retrieval, no adapter.
4. Place outputs on the calibrated scale from 4.2, with rollouts and variance, and read them
   next to his real answers.

It measures K1, tests `H`, sets the L1-oracle ceiling that L2 will be judged against, and
tells us which of the four terms in 2.2 is broken. It also has to happen before the corpus
grows past the point where it is possible.

## 5. How this extrapolates

The mission is not a creator chatbot. It is a system that models a specific person well
enough to think, speak and appear as them, and then, for the user rather than a creator,
stops reproducing them and starts complementing them. What here survives that transition.

**The ladder ports whole.** Its rungs are about a person's archive, not about whose archive.
In P2 the archive is the user's own accumulating interaction rather than a public
back-catalogue, and the same L0-L4 questions are asked of it. Standing decision 15 holds:
the machinery ports, the data never merges.

**Perception replaces ingest, and nothing else moves.** P1 reads a public archive; P2
perceives its user continuously. Both collapse to the same thing, transcripts of a person
being themselves, which is precisely the input `02/2411.10109` shows is sufficient. The
layer boundary absorbs the change.

**The calibration trick ports, but the target inverts, and this is the hard part.** For
mimicry the ceiling is the person's own self-consistency and the floor is a different
person, both measurable, both on disk. Complementation has neither. There is no held-out
record of what someone would have said about something they never addressed, so the
cross-author floor has no analogue and the human ceiling is undefined. `MANIFESTO.md`
already states this asymmetry and `problems.md` R1 records that no metric exists. **Section
4.1 solves the measurement problem for the half that has ground truth and provides no
route at all for the half that does not.** That should be said plainly rather than papered
over, because the second half is the actual claim.

**The gap representation is the bridge.** Complementation needs the model to represent what
the person does *not* hold: positions not taken, blind spots (`problems.md` R2). The stance
card at L3 is the same object with the sign flipped. Built as "these are his standing
positions," its complement is computable; built as an opaque adapter, it is not. That is
the one design choice here with real P2 consequences, and it argues for L3 being readable
even if a learned version scores better.

**What does not port.** The manner adapter, because P2's register is conversational rather
than broadcast and its evaluation inverts: sounding like the person stops being the test
and becomes the delivery vehicle. And every number in section 3, because they were measured
on Reddit, news and email, not on one person speaking for six hours.

---

## Open, and load-bearing

1. **Monologue or conversational?** Term 4 in 2.2. A commitment, not a measurement.
   Nothing downstream is well-posed until it is made.
2. **K1 is unmeasured.** Everything above is conditional on it.
3. **Does `H` hold at all?** If a frontier model with the archive in context is
   indistinguishable from one without, the intelligence half is a retrieval product and the
   research is entirely in the face.
