# knowledge_style: the system, by its seams

Instantiates the `improve-system` skill for this engine. One row per **seam** — the place an artifact
passes between stages, which is where a measurement attaches and where a component can be
swapped. A stage that cannot be named cannot be blamed.

Written 2026-09-15 from the code as it stands, not from the design.

---

## Stages

```
question
   │
   ▼
[A] knowledge model        retrieval over his transcripts, grounded answering
   │   proto/agent.py, proto/serve.py
   │   contains [A1], the first registered component
   ▼  A-out: an answer in written prose
   │
[B] style renderer         same content, as him speaking
   │   ONE call to an ims LoRA adapter over a frozen base. At most 12s,
   │   measured under GPU contention -- an upper bound, not a cost.
   │   No controller in the path: the loop that was there took 4.7 turns
   │   and 4.7 turns per answer and never beat the bare network by more
   │   than noise. proto/recipes/style-pairs/
   │   instrument: STYLE_JUDGE.py, frozen 2026-09-19, hash-checked.
   │   Scored only via SCORE.py, which refuses to report it without the
   │   four shipped content measures beside it.
   ▼  B-out: text meant to be read aloud
   │
[C] voice                  ElevenLabs, cloned voice
   ▼  engines/text2audio
```

Stage C is a provider boundary and is not modelled here.

## Registered components

A stage is a place in the diagram. A **component** is a part of one with a seam either side, an
instrument, and an owned set of decisions — so it can be swapped, swept and blamed. `blocks/`
is the designed module layout; this is the list of things that can actually be measured alone.

| id | component | inside | isolated | instrument | break test |
|---|---|---|---|---|---|
| `A1` | **chunk + retrieve** | `A` | **yes** | retrieval axis, frozen | runnable now |
| `A2` | **the answering loop** | `A` | **yes** | the four measures, frozen | `harness/selftest.py` |

`A1` registered 2026-09-15, `A2` on 2026-09-18.

**`B` is not registered, and the gap is now one thing rather than three.** It has
seams either side and, since 2026-09-19, a frozen instrument that reads 82.9% on the
creator and 22.4% on the unstyled answer. What it lacks is a break test. A component
needs a seam either side, an instrument, and an owned set of decisions. `B` has the seams
and, since 2026-09-18, an instrument: `METRICS.md`, the style acceptance test, measured on
a testbed locked at the `A`→`B` seam and rebuilt by
`proto/recipes/style-pairs/TESTBED.md`. What it lacks is a **break test** — a change that
must move the number and say which part broke. Until that exists `B` is measured but not
blamable, which is the difference between a stage and a component.

The instrument answers "is it him", and every arm measured so far is not. **Ranking arms
that all fail is a different question with no frozen instrument**, open as `problems.md`
N12.

### `A2` — the answering loop

One intelligence drives the whole thing and one judge scores it afterwards. There is
no planner, no critic and no second model in the path.

```
question
   │
   ▼
┌──────────────────────────────────────────────┐
│ CONTROLLER  one model, accumulating context  │   harness/loop.py
│                                              │   harness/controller_deploy.md
│  turn 1   search × n, batched                │   harness/modules.md
│           one tool. returns FULL TEXT, so    │
│           there is no read round trip        │
│              │                               │
│              ▼                               │
│  turn 2   the answer, every asserting        │
│           sentence carrying [n]              │
└──────────────┬───────────────────────────────┘
               │
               ▼
        ┌──────────────┐   code, not a model. All faults in ONE
        │  GATES       │   message so one retry fixes everything.
        │  cites, cap  │   Reject → back to the controller.
        └──────┬───────┘
               │ pass
               ▼
            answer  ──────────────►  A-out
               │
               ▼  (evaluation only, never in the deploy path)
        ┌──────────────┐
        │  JUDGE       │   harness/judges/answer.md
        └──────────────┘   one call, no history, artifacts only
```

**Two creator-data routes and nothing else.** The verbatim corpus, reached only
through `search`; and `creator.md`, which is 96% his actual speech. Every prompt file
is creator-free and linted for it — `harness/selftest.py` checks all twelve
controllers, for numbers, for his vocabulary, **and for his subject matter**, because
an illustration written in his topics is polluted even when it names nothing. Swapping
creator means swapping the corpus and `creator.md`; no prompt changes.

**Guarantees are code, choices are prompt.** Each `if` statement below exists because
the prompt version of it was measured and lost.

| guarantee | flag | what it prevents |
|---|---|---|
| must search before answering | always on | answering from the model's own knowledge |
| every asserting sentence carries `[n]` | `--require-cites` | the invented join: a connecting sentence has no passage, so no number |
| answer length bounded | `--word-cap` | closing on adjacent material |
| retrieval width floored | `--min-k`, `--expand` | the controller's own default sitting under the material |
| a turn backstop | `--hard-turns` | a loop that never answers |

Sentences marked `[none]` pass the citation gate, so reasoning past the archive stays
possible but must be declared.

**Seams, and what attaches at each.**

| seam | artifact | measured by |
|---|---|---|
| `A2-pool` | the passages a search returned | context recall, context precision |
| `A2-cited` | the `[n]` set in the final answer | context utilization |
| `A2-out` | the answer prose | faithfulness, answer relevancy |

**The judge is not in the deploy path.** At `--max-retries 0` `loop.py` returns the
draft whatever the judge found, so the call only writes into the trace. `--no-judge`
removes it, and is refused unless retries are off — with retries on, the judge is the
thing a retry responds to. Scoring happens offline in `proto/chain.py` from the saved
answers, which is why removing it costs no measurement.

**The deployed configuration.**

```
--controller controller_deploy.md --require-cites --min-k 15 --expand 2
--full-text --word-cap 300 --max-retries 0 --no-judge
```

Results belong to `proto/BASELINES.md`; measure definitions to `METRICS.md`.

### `A1` — chunk + retrieve

Cutting the archive and finding things in it. **One component, because the two cannot be
measured apart:** a chunking is only better or worse *at being retrieved from*.

| | |
|---|---|
| **in, build** | source media per take, plus chunk width and overlap |
| **out, build** | `S-chunks` and a manifest whose `index_version` hashes the parameters **and** every chunk's text |
| **in, serve** | a query string |
| **out, serve** | ranked spans, each carrying take, timestamps and score |

Deterministic on the serve side: same store, same query, same spans, forever. The store's
directory name is derived from the parameters that made it, so a store cannot disagree with its
own contents.

**Owns:** how the text is cut and at what width and overlap; what goes into the index; how a
query is matched; how many spans return and how far a hit expands into its neighbours.

**Must not:**
- **Write the query.** A question arrives already a query. Choosing what to search for belongs
  to `A`, and folding it in here is exactly what makes the two impossible to attribute apart.
- **Judge relevance.** Deciding nothing is on-topic is a separate call with its own failure mode.
- **Return derived text in place of a passage.** Derived pointers may be indexed *beside* a
  passage so it can be found. Never instead of it. `problems.md` K3, now with a measured
  head-to-head behind it in `references/SURVEY.md` §9.

**Open against it**, owned by `problems.md`: K5, K7, K11, K8. Not yet filed there, surfaced
2026-09-15: chunk width and overlap were adopted from `2410.13070` and never tested on this
data, though the instrument to test them is free; and the per-video descriptions cost one model
call each and are read nowhere at serve time.

**Consequence for the break test.** `METRICS.md` is unfrozen because the *style* axis is
contested. That does not reach `A1`, whose axis is mechanical, free and uncontested by either
side of that argument. The break test runs against this component now, and a result from it is
licensed the moment it passes. Nothing else here has that property yet.

## Seams

| id | artifact | produced by | consumed by | from which run / split | frozen |
|---|---|---|---|---|---|
| `S-src` | source audio and video, kept beside the captions | YouTube | nothing yet | one ingest per creator | yes |
| `S-verb` | **absent.** Verbatim transcript with word timing, speaker and disfluency intact | — | would feed `S-chunks`, prosody, disfluency placement | — | — |
| `S-chunks` | chunk store, per take | `ingest/corpus` | `A1`, question corpus | one ingest per creator | yes |
| `S-index` | **absent.** Derived pointers beside each passage: the questions it answers, links across takes | — | would feed `A1` retrieval only, never returned in place of a passage | — | — |
| `S-corpus` | `qcorpus.jsonl`, 10,046 questions + verbatim answers | `recipes/question-corpus` | `recipes/style-pairs` | same ingest as `S-chunks` | yes |
| `S-whole` | `qcorpus_whole.jsonl`, 6,766 rows | `tools/verbatim_trim.py` | `recipes/style-pairs` | derived from `S-corpus`; 41% of verbatims dropped or trimmed | no |
| `S-pairs` | `style_pairs_qualified.jsonl` | `tools/stylepairs.py` + `_qualify.py` | `ims dataset` | oracle-stubbed `A`, so retrieval is pinned to the source passage | no |
| `S-para` | paraphrase pairs, 9,528 after overlap filter | `tools/paraphrase_pairs.py` | `ims dataset` | inputs derived from `S-whole` targets | no |
| `S-dataset` | `~/.inmystyle/datasets/<run>/hf_dataset/{train,eval}` | `ims dataset` | `ims train` | split by `paragraph_id` = video, hashed not encounter-ordered | no |
| `S-adapter` | LoRA checkpoint | `ims train` | `B`, eval | selected by held-out loss today; see `problems.md` | no |
| `S-answer` | `A`'s answer at inference | `A` | `B` | **real retrieval**, unlike `S-pairs` | no |
| `S-styled` | `B`'s output | `B` | `C` | **the locked testbed**, 87 questions at the `A`→`B` seam; 76 scoreable after the judge's reference split | the testbed and the judge are, `B` is not |

## `A1`'s layers: one authored, the rest derived

| layer | seam | state |
|---|---|---|
| source audio and video | `S-src` | **exists**, beside the captions, never moves |
| verbatim transcript — word timing, speaker, disfluency intact | `S-verb` | **absent.** The foundation for K7, prosody, and disfluency placement |
| chunk store | `S-chunks` | exists, and is currently the only layer |
| derived index — questions a passage answers, links across takes | `S-index` | **absent.** The main ingredient was already generated for another purpose |

## `S-src` is not `S-chunks`

The captions are a transcriber's reading of the source, not the source. They have already
dropped disfluency, bleeped words and mis-transcribed others — one judge tell for "that is not
him" turned out to be a transcription error. Every measurement taken off `S-chunks` inherits
that, including the stammer rate in `problems.md` N16.

The source is on disk. The layer between it and the chunk store is the one that does not exist.

## The provenance hazard at `S-pairs` vs `S-answer`

`B` is trained on inputs produced with retrieval **pinned** to the passage the verbatim
came from (`tools/oracle_stub.py`). At inference it receives inputs produced with **real**
retrieval, which returns a median of 42 chunks and finds the source passage 8 times in 10.

Same generator, same prompt, different grounding. Measured gap, content survival:
oracle-stubbed 98%, real-retrieval 94% (n=38, `work/healthygamer/arms/runs_curated18.json`).

## What is deliberately not a stage

**Where disfluency is placed.** No component owns it. Every model tried — trained,
prompted, local and frontier — emits zero stammers. Whether this becomes a stage between
`A` and `B` is an open question in the ledger, not a decision recorded here.

## Break test

Not yet run **system-wide**. The style axis was unlicensed entirely until 2026-09-18;
`METRICS.md` now freezes its **acceptance** test and leaves its **ranking** substrate open,
so a result saying "this is or is not him" is licensed and a result ranking two arms that
are both still machine is not.

**But it is runnable on `A1` today**, and that is the exception the freeze does not cover. Its
instrument is mechanical span overlap against timestamp ground truth: free, deterministic, no
model opinion, and uncontested by either side of the style argument. Breaking the chunker, or
the matcher, must move that number and point at which one broke.

That is the first thing to run on this system, and it is the cheapest.
