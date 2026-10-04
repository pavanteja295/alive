# knowledge_style: the instruments

Instantiates the `improve-system` skill for this engine. One row per instrument: the axis, the seam it
attaches to, where it comes from, and **what it is blind to**. Frozen once agreed; a
change is a commit whose message carries the reason.

**THE STYLE AXIS: its ACCEPTANCE test is frozen 2026-09-18, its RANKING substrate is
not.** The question "is it him yet" now has an instrument, below. The question "which
of two arms that are both obviously not him is closer" does not, and the paper-set
conflict behind it is still open — `problems.md` N12. A number from the acceptance
test is licensed. A ranking of arms by it is not, until that second question is
settled.

**The retrieval axis is frozen, 2026-09-15; SUPERSEDED 2026-09-16 by the chain testbed
below.** Its instrument was span overlap against the one passage each question was
generated from, which is why it is retired: he covers the same ground across 17 videos,
so an answer built from other things he really said scored as wrong. Context recall and
precision against the oracle replace it and keep the property that mattered — mechanical,
free, deterministic, and uncontested by either side of N12.

---

## Agreed

| axis | instrument | seam | source | blind to |
|---|---|---|---|---|
| ~~**retrieval**~~ RETIRED | span overlap against timestamp-labelled ground truth, no model in the loop | `S-chunks` → the spans `A1` returns | the oracle, labelled by hand and corrected three times where the label, not the system, was wrong | **the query**, and everything except the single source passage. Replaced by context recall + precision in the chain testbed below |
| ~~**groundedness**~~ SUPERSEDED | the five-verdict judge, per claim, against the spans that were retrieved | `S-answer` against what `A1` returned | the judge already built for style pairs, `tools/stylejudge.py` | whether the claim is *correct*. Only whether it is supported by what was in front of the model |
| ~~**fidelity to him**~~ RETIRED | the same judge, per claim, against **his verbatim answer** | `S-answer` against `S-corpus` | the reference answers already in the question corpus | wording and voice entirely. Two answers can both be SAME and sound nothing like each other |
| **abstention** | correct refusal when no evidence exists | `S-answer` | the 10 disconnected probes; the corpus has none by construction | everything else. It is the second objective and trades against the first |
| content survival | LLM judge, five verdicts (SAME / ADDED / DROPPED / CONTRADICTS / ELSEWHERE) | `S-styled` against `S-answer` | LLM-as-judge is standard for TST content; `persona_style_papers/11_evaluation` | wording and register entirely, by design. Saturates: it cannot rank two faithful outputs |
| pair fidelity | the same judge, run on `S-pairs` and `S-para` before training | `S-pairs`, `S-para` | — | nothing about whether the pair is *learnable*, only whether it is honest |

The content judge is the one instrument with evidence that it tracks the thing it claims:
the adapter's score matched its training data's score twice, at 40% and at 98%.

## THE CHAIN TESTBED — frozen 2026-09-16, measure 4 added 2026-09-17

**Four measures over the whole chain, question to answer. Owner: `proto/chain.py`.**
Adding another requires a commit whose message says what question the existing ones
could not answer. This rule exists because over one day this engine was ranked on span
overlap, then on DROPPED, then on a holistic "does it answer the question" judge, then
on decomposed completeness — four instruments, four different winners, and three of the
four were later shown to be measuring the wrong thing.

### FINAL CONTENT RESULTS — 100 questions

Per-arm detail and reproduce commands: `proto/BASELINES.md`. Architecture:
`SYSTEM.md`. Both arms below were scored from one cache state.

| | deployed | baseline loop | **shipped** |
|---|---|---|---|
| **questions failing any measure** | **46** | 15 | **4** |
| faithfulness | — | 0.946 | **0.962** |
| context recall | — | 0.876 | **0.913** |
| context utilization | — | 0.869 | **0.883** |
| answer relevancy | — | 0.925 | **0.944** |
| seconds, end to end | — | 122 | **61** |

**The failure count fell further than the means rose.** That is a bound on answer
length, not the averages: it does not lift a mean, it removes the answers that dip
under a threshold.

**Two things no measure here captures.** Unnumbered asserting sentences fell from
283 to ~2 per 20 answers — grounding became an `if` statement rather than a request,
which is deterministic and invisible to all four. And nothing here says whether it
sounds like him; that is the style axis, contested below.

**Do not trust a delta without a cache-bypassed re-score.** An earlier version of
this table was not reproducible. `problems.md` K21.

### The measures attribute error to a WORKFLOW STEP

That is their purpose. A failing question names its owner instead of starting an
argument.

**"Retrieval" is not the BM25 tool.** It is the step: the model writes the queries, the
tool returns passages, the model decides whether to search again. This matters — two
arms at identical `k` score differently because the model wrote different queries, and
a hand-written gap prompt scored *worse* than letting the model choose (recall 79% vs
85%). Better querying is a retrieval improvement; a worse prompt is a retrieval
regression.

| step | what it does | its failures |
|---|---|---|
| **retrieval** | model writes queries → tool returns passages → model decides to search again | **not captured**, and **noise brought in** |
| **assembler** | turn the passages into the answer | **ungrounded**, **dropped**, and its half of **padding** |

```
                   was it retrieved?
                  /                 \
                no                   yes
                 |                    |
       (2) NOT CAPTURED         was it used?
            RETRIEVAL           /          \
                              no            yes
                               |             |
                      (3) DROPPED          working
                        ASSEMBLER

  orthogonal:
  (1) UNGROUNDED  claims no retrieved passage supports         ASSEMBLER
  (4) PADDING     claims that were not asked about             BOTH
                  needs retrieval to bring the noise in AND
                  the assembler to use it. Neither alone does it.
```

| # | our name | established name | asks | blames | needs the oracle |
|---|---|---|---|---|---|
| 1 | ungroundedness | 1 − **faithfulness** | did it assert what it was not given | assembler | no |
| 2 | failure to capture | 1 − **context recall** | are there passages that answer this and were never retrieved | retrieval | **yes** |
| 3 | dropping | 1 − **context utilization** | was relevant material retrieved and then not used | assembler | **yes** |
| 4 | padding | 1 − **answer relevancy** | does it say things that were not asked about | **both** | no |

Measure 4 was added because 1–3 cannot see it: an answer can use every essential
passage (utilization 1.0) and invent nothing (faithfulness 1.0) while padding itself
with grounded but irrelevant claims from the noise it was handed. Utilization is
computed over relevant passages only, so use of noise is invisible to it by
construction.

Established names are used deliberately. Inventing names for standard measures is how
the four-instrument mess above happened.

### Retrieval is deterministic, so it gets counts, not judgements

Given a retrieved set and the oracle, these are exact. No judge, no noise floor beyond
the variance in which queries the model wrote. **Context precision is the free partner
of context recall** — the same two sets divided the other way:

| | asks |
|---|---|
| context **recall** = relevant ∩ retrieved / **relevant** | did the step find what exists |
| context **precision** = relevant ∩ retrieved / **retrieved** | how much of what it brought is noise |

Measured, and it is the worst number in the system:

| arm | shown | recall | precision | noise | essential found of 3.2 |
|---|---|---|---|---|---|
| deployed k=6 | 41 | 67% | 0.057 | **94%** | 2.1 |
| harness v4 (loop) | 80 | 86% | 0.037 | **96%** | 2.7 |
| deployed k=25 | 128 | 89% | 0.023 | **98%** | 2.9 |

**The loop buys recall by making precision worse**, and every arm hands the assembler
40–128 passages of which two or three matter. Two consequences, both measured:

- Widening retrieval **trades retrieval failures for assembler failures**. Questions
  failing at least one measure: deployed 46 (29 retrieval / 17 assembler), v4 21 (7 /
  14), k=25 21 (2 / 19). There is an optimum and k=25 is past it.
- **The assembler cannot be instructed out of it.** `concentrate`, which tells the
  model to use only the two or three most direct passages, went 21 failures → 26 and
  ungrounded 13% → 20%. The noise has to stop arriving, which is what the filter stage
  in `harness/ladder.py` does — measured to cut the pool 62% while keeping 100% of
  essential passages.

### The oracle makes 2 and 3 computable

Owner: `proto/oracle.py`, output `work/<creator>/reports/oracle.jsonl`.

For every question, **every passage in the archive** is judged `ESSENTIAL` /
`SUPPORTING` / `NO`. A full sweep, not TREC-style pooling: pooling judges the union of
what current systems return and accepts unjudged material, which biases the oracle
toward what retrieval already finds — the very thing being measured. The archive is 292
passages and 86k words, so the sweep is affordable and there is no unjudged material.

Without it you can see that an answer used six passages. You cannot see that eleven
others also answered the question and were never retrieved, nor that four were
retrieved and ignored. Those are different failures with different owners.

### No invented numbers

A question can be answered by many passages, so any score of the form "got 6 of 43"
punishes an answer for not quoting everything he ever said on a topic. Each measure is
reported as a **rate over questions against a named threshold**, plus the underlying set
sizes so the threshold can be re-derived. No composite, no weighted sum, no single
number.

| measure | a question fails when |
|---|---|
| ungrounded | under 85% of its claims are supported |
| not captured | under half the relevant passages were retrieved |
| dropped | under a quarter of what it was given was used |
| padding | under 70% of its claims bear on the question |

Thresholds are set at the point a reader would call the answer a failure, **not** at a
percentile of current results — a threshold read off the distribution moves every time
the distribution moves.

### What each is blind to

| measure | blind to |
|---|---|
| ungroundedness | whether the claim is **true**. Only whether a retrieved passage supports it. An answer grounded in irrelevant passages scores well |
| not captured | the query itself. It scores what came back, never whether the query written was the right one — but unlike span overlap it does know what a right query **would** have found |
| dropping | whether dropping was **correct**. Some relevant material genuinely does not belong in a 200-word answer. Read the rate, not the count |
| padding | which side caused it. It needs retrieval noise AND assembler use, so read it against context precision to see which moved |

### The instruments this replaces, and why

| retired | why |
|---|---|
| span overlap against timestamp ground truth | scores against the one passage the question was generated from; he covers the same ground across 17 videos, so an answer built from other things he really said scored as wrong |
| `DROPPED` / `ADDED` vs his verbatim answer | measured, does not detect incompleteness: all 21 DROPPED answers on the deployed arm fully answer the question, and none of the 8 genuinely incomplete ones were flagged. `SAME` has never been returned once across 600+ answers, so it cannot produce its own pass value |
| holistic "does it answer the question" | gave FULLY to an answer covering only part one of a two-part question, and also to one covering only part two |
| decomposed completeness | the split is a model call and over-splits; an independently-worded decomposer agrees on multi-part status in only 57 of 100 cases |

**`CONTRADICTS` is kept** as a separate flag rather than folded in. Reversing what he
said is wrong under any measurement, and it is the one verdict from the retired fidelity
judge that survives unchanged.

### What each surviving measure may and may not decide

| measure | use it to decide | do **not** use it to decide |
|---|---|---|
| ungroundedness | whether a change made the assembler invent more | whether retrieval improved |
| CONTRADICTS | whether it reverses him more often | anything about coverage |
| not captured | whether the retrieval step improved | whether the answer is good |
| context precision | how much noise the retrieval step sends on | whether coverage improved |
| dropping | whether the assembler's selection improved | whether dropping was correct |
| padding | whether the noise reaching the assembler got used | which of the two steps to fix -- check precision |

---

## THE STYLE ACCEPTANCE TEST — frozen 2026-09-19, owner `proto/recipes/style-pairs/STYLE_JUDGE.py`

**It is a classifier and the headline is the pass rate.** One passage at a time,
after transcripts of the creator speaking: is this that person, or a machine
imitating him. The prompt is inlined and hashed into the file; editing it stops the
run and says the numbers are not comparable. Changing it deliberately is `--rebless`
plus a commit saying what the old one could not answer.

| | passed as him |
|---|---|
| **his own real speech** | **82.9%** |
| the unstyled content answer | 22.4% |

**82.9% is the ceiling, not 100.** He sounds composed about a third of the time, and
an arm at the floor is as close as this instrument can see. A move under **2.1
points** is not readable: identical passages judged twice move the mean 0.6, and 12%
flip side.

### What it may not use, each excluded for a measured reason

| forbidden | why |
|---|---|
| **his archive as a lookup** | 93% of a reference answer is verbatim archive text; shown the whole archive the judge scored 1.000 with a floor of 0.0% — it had stopped judging. The reference split is fixed first: some takes are shown, only questions answered from OTHER takes are scored |
| **the mix** | told "some are him, the rest are a machine", it splits the set instead of deciding. Contrast alone moved a reading from 0.852 to 0.999 |
| **word choice, filler rate, catchphrases** | counted for free elsewhere, with a floor. A judge spending judgement there duplicates a cheaper instrument |
| **his argumentative moves** | noun-to-verb reframing, concede-then-narrow, the staged reversal, stacked questions, voiced inner dialogue, the mundane example. All real habits of his, all reproducible by anything built from his material: told to look for them the judge let **51%** of machine output through |
| **length, punctuation, where it starts and stops, topic** | properties of the transcript, not the speaker. Verified: 0 of 174 verdicts named a forbidden cue |
| **fluency** | he is fluent a third of the time. Treating smoothness as a tell is what made an earlier version reject a third of his real speech |

What it is left with is the one thing an imitation cannot fake: **whether the sentence
survived its own delivery.**

### It is a per-creator artifact, not reusable machinery

Content is universal and style cannot be. The content harness keeps its prompts
creator-free because the task is invariant; the thing a style judge measures *is* the
person, so a creator-free style judge can only fall back on speech-versus-writing —
the single axis that misclassified his fluent speech. `problems.md` N29. What
transfers is the **procedure**: fit from the creator's own passages, measure the
floor, verify the discriminator is not copyable.

### Never reported alone

`SCORE.py` runs this and the four shipped content measures together, on the input and
the output, and exits non-zero when content drops. The cheapest way to sound like
someone is to say less than they said.

### What it is blind to

| | |
|---|---|
| whether the content survived | entirely. That is the other half and it is not optional |
| how far a failing arm is from him | it answers "him or not"; a graded reading exists but the verdict is the measurement |
| its own fitting | the prompt was tuned against the same 76 passages it is quoted on. No held-out slice of his speech exists yet, so 82.9% is a fitted number |


---

### The answer stage: what the judge is asked, and what falls out

Two comparisons per answer, never one:

| against | catches | the failure it names |
|---|---|---|
| the spans that were retrieved | invention | it asserted something it was not given |
| **his verbatim answer** | divergence | it did not say what he says |

Five verdicts per claim, reusing the style-pair judge unchanged: **SAME** (the only pass),
**DROPPED** (he says it, the answer does not), **ADDED** (invented), **CONTRADICTS** (the
worst), **ELSEWHERE** (present but misattached).

**The cross-seam box is the point of running both stages on the same questions.** Because
`A1` reports per question whether the evidence was complete, every answer lands in one of
four cells:

| | answer grounded | answer not grounded |
|---|---|---|
| **evidence complete** | working | **the answer stage is at fault** |
| **evidence incomplete** | worked around it, or invented | retrieval is at fault |

Nothing in this system currently produces that top-right number, and it is the only one that
blames a stage rather than describing an outcome.

**Free checks, for the fast tier:** the model's own `direct | extended | none` label against
measured groundedness, which costs nothing and catches overconfidence; length against the
situation directive; stock-phrase rate.

**Excluded deliberately:** any single answer-quality score, any composite, and citation-quote
validity as a scoreboard item — on the first subject every flag it raised was a transcription
artefact, so it stays a diagnostic until that is fixed.

**Open, a person settles:** whether an extension beyond the archive that is *labelled* as one
passes or fails; and whether groundedness counts claims or sentences.

### The retrieval instrument saturates under one definition and not the other

**Found means every required passage, never any of them.** For answers composed from several
passages, "found any" reads near 100% on runs where "found all" reads 0 of 12. The first
definition is pinned at its ceiling and cannot rank anything; the second is the one that
moves. This is settled here so a sweep cannot quietly pick the flattering one.

### Its test sets, and the one that has never been used

| set | size | axis it varies |
|---|---|---|
| hand-banded probes | 38 | directly covered / adjacent / about him / outside his world |
| question corpus | ~10k, each knowing its source passage, each already tagged with whether retrieval reached it | one passage / several in a video / across videos / same answer reworded |

The second is a labelled retrieval benchmark roughly 260 times the size of the one we use. It
was produced as a side effect of building style pairs and has never been scored as a benchmark.
**Filter first:** some questions reuse their own passage's distinctive vocabulary, which tests
word-matching rather than retrieval and inflates the result.

## Contested — the style axis

Two documents in the paper set disagree, both with cited reasons.

| | `GOLD_STANDARD.md` | `DECISION.md` |
|---|---|---|
| LUAR / Wegmann, Toward-Away-Confusion | "non-negotiable evaluation substrate" | kill list: validated on Reddit not speech; every method scores 0.07-0.15; does not track the objective |
| instead | SBERT cosine, CoLA fluency, geometric mean at sentence level | stylometry counters (sentence length, filler rate, top idiom frequencies) + people who know him |
| why | every paper in the set uses it; results are not comparable without it | `2311.07564`: authorship models degrade sharply under topic control on speech; character n-grams competitive |

Our takes are topic-separated by construction, which is exactly the condition
`2311.07564` says breaks the neural authorship models.

## Blind spots already measured, whichever is chosen

- **The blind A/B judge saturates.** 98-100% across four adapters, two base sizes, two
  datasets and five decoding settings. A measure pinned at its ceiling cannot distinguish
  "barely moved" from "did not move", so it cannot drive a search.
- **Corpus-level aggregation is gameable.** STRAP §3.2 shows a naive copy baseline wins
  it. Any aggregate is computed per sentence before averaging.
- **Part of the style gap is unreachable and undesirable.** Targets are ASR output; one
  judge tell was "very very rumitative", a mis-transcription. `2311.07564` additionally
  finds normalised lowercase transcripts work as well or better, so prettifying ASR is
  not the answer either.

## Qualitative substrate

Where no standard instrument survives, the `improve-system` skill requires a rendered artifact and a
named reader rather than an invented score.

| artifact | what it shows | who reads it |
|---|---|---|
| `work/<creator>/reports/audit_*.html` | every failure, plus the hardest-to-call survivals, with the judge's reason beside the text | a person |
| side-by-side input / output / his real words | the only check that caught the truncation error a summary statistic hid | a person |
| `DECISION.md`'s day-1 experiment | 30 held-out questions, frontier model, full transcript, no training, outputs beside his real answers | a person |

## Cost, which the `improve-system` skill requires be decided rather than discovered

| measurement | cost today |
|---|---|
| **retrieval, full corpus** | **seconds, no model calls.** This is why `A1` can be swept and nothing else can |
| groundedness / fidelity, n=100 | 1 answer call + 1 judge call each. Runnable per experiment, too slow to sit inside a revision loop |
| content judge, n=60 | ~8 min (GPU generate + API judge) |
| blind A/B, n=60 | ~25 min |
| one training run to its turn | ~1 h |

At 25 minutes per style measurement a search gets tens of points, not hundreds. Reducing
this is a prerequisite for EXPLORING, not an optimisation.
