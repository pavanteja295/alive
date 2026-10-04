# Recipe: a creator's grounded-answering configuration

Takes a creator who already has a corpus and an adjudicated oracle, and produces a
**promoted configuration** — the flags and prompt that answer a viewer's question
from that creator's own words, measured on four frozen measures.

> **`work/<creator>/` in this recipe is `data/answers/<creator>/`** (`paths.WORK`). It is
> data, not code, and is not in git; the recipe rebuilds it.

**This file owns the process.** If a script, a prompt or a conversation disagrees
with it, this file is right and the other has drifted.

- **Validated on:** one creator (`healthygamer`, 292 chunks, 17 takes, 318
  adjudicated questions). **Never run end to end on a second creator**, so the
  Validate pass this recipe most needs has not happened. A second subject
  (`huberman`) exists as 9 files from the retired toolchain — questions, a map, a
  baseline arm — with **no chunk store and no oracle**, so it is a starting point
  and not a second data point. Read `What did not generalise` before trusting any
  constant here.
- **Measures:** owned by `../METRICS.md`. **Results:** owned by `BASELINES.md`.
  **Unresolved:** `../../../problems.md`. This file owns only the procedure.
- **Runs after** the `question-corpus` recipe. **Runs before** the style recipe
  (the style-model recipe, on branch `question-corpus`; not in this release), which consumes this one's promoted report.

---

## Signature

**Inputs**

| input | shape | read from |
|---|---|---|
| chunk store | `chunk_id take_id ts text` per line | `work/<creator>/store/chunk-w<size>-o<overlap>/chunks.jsonl`. **Resolve it with `paths.Subject(creator, paths.DEFAULT_CONFIG).chunks`, never by hand** — the chunking parameters are in the directory name, so a hardcoded path silently points at the wrong chunking or at nothing |
| question corpus | `q answer ...` per line | `work/<creator>/qcorpus_whole.jsonl` |
| oracle | `q id verdicts{chunk_id: ESSENTIAL\|SUPPORTING\|...}` | `work/<creator>/reports/oracle_strict.jsonl` |
| creator layer | his verbatim speech, ~96% | `data/answers/<creator>/creator.md` (*The persona note*) |

**Preconditions, checked before any work.** `chain.py` raises on the oracle; the
rest are checked by the style recipe's `preflight.py` (on branch `question-corpus`) (correspondence, not
existence — see its own recipe).

1. The oracle exists **for this creator**. `chain.py --creator <name>` resolves
   `work/<name>/reports/oracle_strict.jsonl` and raises if absent, naming
   `question-corpus` as its owner. **It does not build it.**
2. The oracle's questions correspond to the corpus being scored. Existence was
   never the failure mode.
3. Every controller prompt passes the creator-free lint: `python3
   harness/selftest.py`.

**A missing input stops the recipe.** It is not inferred or worked around, and this
recipe never runs the upstream tool even when that tool is one line away.

**Outputs**

| output | consumed by |
|---|---|
| `work/<creator>/reports/<arm>.json` | the promoted one is the style recipe's input |
| `work/<creator>/reports/chain_<arm>.json` | the four measures per question |
| `harness/profiles/<creator>.json` | the per-creator settings |
| the promoted arm **named in `BASELINES.md`** | the style recipe's preflight hashes it |

**Name the promoted arm unambiguously in `BASELINES.md`.** That string is the entire
coupling between the content and style halves. "The newest report in the directory"
is not it, and cost an earlier style run two days.

---

## Why this exists

A frontier model asked to answer as a creator will answer from its own knowledge and
sound approximately right. The failures are invisible without instruments: it
asserts positions the creator does not hold, it answers a bigger question than the
one asked, and it invents the sentence that joins two things he really said.

Measured on 100 benchmark questions, the deployed single-call system failed a
measure on **46** and the promoted configuration on **4**, in half the wall clock.

**On 218 questions unseen by the tuning, the same configuration fails 48 — and its
pre-gate baseline fails 52.** That is the honest size of the win: 11 points of
failure rate where it was tuned, 2 points where it was not. The slice is harder for
everything, so this is not overfitting; but three of the four gains transfer and one
reverses. See `What did not generalise`, and `BASELINES.md` for both columns.

---

## The item

**One question.** Independently answerable, independently scorable, independently
redoable. A run is a list of items; nothing carries between them.

---

## Tool inventory

Read the numbers these print. Do not re-read the corpus to check them.

| tool | does | postcondition once it exits clean |
|---|---|---|
| `harness/loop.py` | generates answers for a question set | `<arm>.json` has one row per requested question; `<arm>_trace/<slug>/` holds the full conversation and the judge's inputs |
| `chain.py` | scores a report on the four measures | `chain_<arm>.json` has `faithfulness context_recall context_utilization relevancy` plus **claim counts** per question |
| `harness/tune.py` | derives retrieval settings from corpus + oracle | `harness/profiles/<creator>.json`, with the grid that produced it |
| `harness/selftest.py` | machinery and creator-free lint | 63 checks; a fail means do not spend questions |
| `oracle.py` | adjudicates passages per question | owned by `question-corpus`, not this recipe |

---

## THE SHARED DISCIPLINE

*Referenced by `recipes/style-model/RECIPE.md` by heading, not by number. 8 rules,
each of which was learned by publishing a wrong number first.*

**Measure the noise floor before any arm.** Re-run the identical configuration on
the identical questions. On the first creator this moved 17 of 20 question scores
and flipped 7 pass/fail. Five prompt variants were run before anyone knew they were
unresolvable by construction.

**Re-score with the cache bypassed, or you are measuring the cache.** `llm.py` is
content-addressed, so a second scoring of the same answers replays the first and
agrees byte-for-byte in under a second. That proves nothing. Use `PROTO_FRESH=1` and
measure the spread. This was believed backwards for a whole session: the cold-cache
run that actually issued the calls disagreed with its cached replays on 64 of 100
questions, and was written off as the outlier.

**Two of the four measures are deterministic and two are not.** Measured on the
first creator, the same 30 answers scored twice with the cache bypassed:

| | mean diff | per-question MAE | questions moved |
|---|---|---|---|
| context recall | 0.0000 | 0.0000 | 0/30 |
| context utilization | 0.0000 | 0.0000 | 0/30 |
| faithfulness | 0.0145 | 0.0237 | 11/30 |
| answer relevancy | 0.0226 | 0.0354 | 16/30 |

Recall is set arithmetic with no model in it, and utilization's verdicts turned out
stable. **Faithfulness and relevancy are model judgements and move.** Scaled to
n=100 a mean's noise falls by about half, so the usable floors there are roughly
0.008 and 0.012 — and the promoted arm's gains on those two are about twice them,
not the comfortable margins the raw deltas suggest. Re-derive this per creator: it
is a property of the judge on that material, not a constant.

**A noise floor and a held-out slice catch DIFFERENT failures, and you need both.**
Noise-checking protects against reading nothing as something. It does nothing about
a real effect that only exists where you measured it. On the first creator a
relevancy gain of +0.019 was real on the benchmark, cleared a properly measured
floor, and then **reversed to −0.038** on unseen questions while the baseline's
relevancy barely moved. Only a held-out slice catches that. A recipe naming only the
noise floor is half-armed.

**Any absolute bound derived from one slice's distribution is a fitted parameter
wearing a setting's clothes.** Make it relative, or re-derive it per slice. A word
cap of 300 was a ceiling where the creator averages 190 words and a licence where he
averages 139: the model fills to whatever it is given, so a bound the baseline does
not have becomes padding the baseline does not produce. Its effect flips with the
distribution it is applied to. *Stated this way by the style work, which hit the
same shape independently — a number that reads as a limit behaves as a target.*

**Read the failure count, not the sum.** The promoted arm has a *lower* sum of four
than an arm it beats, and less than half the failures. A bound does not lift a mean;
it removes the answers that dip under a threshold, which is what a viewer notices.
Report both, promote on failures.

**Report absolute counts beside every ratio.** Faithfulness and relevancy are ratios
over claim count, and claim count tracks answer length. Two arms at the same 0.93
relevancy deliver different absolute amounts of off-question text. A measure that
sums a distribution can be right about the total and wrong about every instance.

**A requested SET meeting a constructed POOL: the list is the requirement, and a
shortfall is a failure.** This is a bug *class*, not three bugs. The shape is: a
caller names what it wants, the callee has its own idea of what exists, the
intersection wins silently, and nobody raises. Found three times in one day across
two independent codebases:

| where | requested | pool | symptom |
|---|---|---|---|
| `loop.py --questions` | 218 held-out | stratified draw | ran 130, reported a smaller set |
| `chain.py` fallback pool | 218 to score | hardcoded `n=100` | scored 0, wrote an empty file, **exit 0** |
| style loop and runner | a question list | rows present | same, both now raise |

Every one produces a plausible artifact from a subset, and the number still reads as
independent. **Anywhere a list meets a pool, assume this exists until you have
checked.** Both now load the missing items or raise and name them.

*The class, and the third instance, were named by the style work on the
other half; the rule belongs in both recipes.*

---

## The loop

```
measure  ──►  diagnose  ──►  change ONE thing  ──►  re-measure
```

**Step 1 — measure.**

```bash
python3 harness/loop.py --bench -n 100 --creator <name> --profile [flags] \
    --json work/<name>/reports/<arm>.json --trace work/<name>/reports/<arm>_trace
python3 chain.py work/<name>/reports/<arm>.json --creator <name> \
    --essential-only --jobs 8 --json work/<name>/reports/chain_<arm>.json
```

**Strip `[n]` citation markers before scoring.** An arm that cites inline and one
that does not are otherwise different artifacts, and a faithfulness judge that can
see citations grades differently.

**Score arms one at a time.** Generation and scoring share one API account; anything
overlapping is contended and its wall clock is meaningless. The four measures are
unaffected, so only timing claims are lost.

**Step 2 — diagnose from the traces, not from statistics over them.**

Read the failing questions and the **judge's** reasoning. The controller never
narrates its own error: it does not write "I am inventing this link", it writes the
link. Counting phrases across reasoning produced a confident 6× that was one
occurrence against one, scaled by a word-count denominator.

| what the traces show | the change it implies |
|---|---|
| claims with no passage behind them, joining two real ones | a **code gate**: every asserting sentence carries `[n]` |
| the answer drifting past the question, usually in its last paragraph | a word cap, and "your last sentence must answer it" |
| essential passages retrieved and not cited | check the cap is not cutting them before widening anything |
| essential passages never retrieved | raise `--min-k` / `--expand`, tuned offline |
| slow | the retry, and the separate `read` turn |

**Step 3 — change one thing, re-measure, accept or retire.** Accept if a measure
improves beyond the floor from `THE SHARED DISCIPLINE` and none regresses beyond it.
Promote by editing `BASELINES.md` to name the arm.

---

## Judgement

| | |
|---|---|
| **unavoidable** | reading failing answers and naming what went wrong; deciding whether a judge finding is right; deciding what a viewer would notice |
| **forbidden** | arguing with a recorded measure. Whether a passage is essential is the oracle's verdict. Repair the procedure, never the measurement |
| **not needed** | chunking, retrieval, the gates, scoring, profile derivation, trace writing — all deterministic |

---

## The persona note

Every answer is written after reading one note about the creator,
`data/answers/<creator>/creator.md`. It is most of what makes the answer sound like
them: the retrieval finds what they said, the note sets how they say it. No tool writes
it. A person, or Claude, picks the passages; it takes about twenty minutes.

1. **Copy the template**: `harness/creators/_template/creator.md` to
   `data/answers/<creator>/creator.md`. The rules under *Who you are* do not change between
   creators; each one is there because a run broke it.
2. **Pick eighteen passages** of their own speech, about 90 seconds each, from the
   transcripts the voice recipe made (`data/voice/work_<id>/asr/`) or the chunk
   store (`data/answers/<creator>/store/`):
   - across as many of their videos as there are, and across different topics;
   - where they are explaining something in their usual register, not reading;
   - **never** a sponsor read, an ad, an intro or an outro: those are how the show is
     paid for, not how they talk;
   - verbatim, with the transcript's errors left in. A cleaned-up passage teaches a
     cleaned-up voice.
3. **Write each one** under *How you talk* with its source:
   `[<take folder> HH:MM:SS-HH:MM:SS]`, then the words, then two blank lines.
4. **Fill in *Your archive***: how many videos and passages the store holds
   (`make status SUBJECT=<creator>` prints both).
5. **Record the pick** in `creators/<id>/build/answers/persona_passages.txt`: one line per
   passage, `<take folder> HH:MM:SS-HH:MM:SS`, no text. That file is in git; the note is not.
6. **Check it**: ask three questions in the live app and read the answers aloud. If an
   answer says "he" about the creator, or sounds like a summary of them rather than them,
   the passages are too formal or too few.

Dr K's and Huberman's picks are in `creators/{drk,huberman}/build/answers/`.

## The promoted block

What the live server runs for a creator is `harness/profiles/<creator>.json` → `promoted`.
Without one, `harness/serve_promoted.py` uses a default (Dr K's deployed arm: the
`controller_deploy.md` prompt, citations required, 15 passages plus 2 neighbours, full
text, no word cap), which is how a new creator serves on day one. Tuning one is the loop
above; when an arm wins, write it in by hand, because `harness/tune.py` rewrites the
file without it:

```json
"promoted": {
  "arm": "deploy100_commit",
  "command": "harness/loop.py --bench -n 100 --creator <c> --controller controller_deploy.md ...",
  "flags": {"controller": "controller_deploy.md", "require_cites": true, "min_k": 15,
            "expand": 2, "full_text": true, "word_cap": null, "max_retries": 0,
            "no_judge": true, "hard_turns": 8, "commit": true},
  "measured": {"n": 100, "failing_any_measure": 38, "faithfulness": 0.898},
  "note": "why these values, and what was tried instead"
}
```

Only `flags` is read by the server. The rest is the record a reader needs to disagree
with it. Huberman's and Dr K's are in `harness/profiles/`.

## Instance constants — re-derive per creator

| constant | first creator | how it was derived |
|---|---|---|
| `--min-k`, `--expand` | 15, 2 | offline grid over the oracle. `harness/tune.py` |
| `--word-cap` | 300 | **not derived.** See `What did not generalise` |
| `--max-passages` | 8 (unused in the promoted arm) | oracle's max essential per question was 4 |
| `GATE_TRIES` | 3 | one per gate |
| scoring noise, faithfulness | 0.0145 @ n=30 | two cache-bypassed scorings |
| scoring noise, relevancy | 0.0226 @ n=30 | same |
| scoring noise, recall / utilization | 0.0000 | same — these two do not move |

**The per-creator surface is these numbers and nothing else.** Every prompt is
creator-free and linted, including subject matter — an example written in the
creator's topics is polluted even when it names no number or term. Swapping creator
is: corpus, `creator.md`, re-derive the profile. No prompt edits.

---

## Done criterion — re-read this, do not carry it

Converged when **no untried candidate has an expected effect above the measured
floor**, and:

- every controller passes the creator-free lint
- the promoted arm is named in `BASELINES.md` with its config string
- the four measures are reported with claim counts beside them
- a held-out slice, genuinely unseen by tuning **and** by the benchmark, has been
  spent once on the promoted arm

Then say plainly what is left, because no further pass reaches it: **none of these
four measures says the voice is right.** That is the style recipe's half.

**A named cost of the citation gate: choppy construction.** The gate at line 217
(every asserting sentence carries `[n]`) is what took unnumbered claims to 0, and
it has a side effect the four measures do not see: a sentence spanning two
different committed passages cannot cleanly carry one citation, so the model
defaults to short, single-claim, citation-anchored sentences rather than prose
that flows between them. Felt directly in --commit answers, where committed
passages are often not adjacent in the source and the model has the least
connective material to work with. This is not the dead-end shape ("asking the
model not to invent joins") — it is the gate correctly refusing to let a
sentence assert a relationship between two passages it cannot cite, which is
exactly what stops an invented join. A later construction/stitching pass should
only ever touch already-gated, already-cited text (never generate before the
gate runs), and must be restricted to pure sequencing words with no causal or
explanatory force ("also", "separately", "that said") — a connective like "so"
or "which means" is a claim wearing filler's clothes, and is the same invented
join the gate exists to catch, just moved one step downstream of it where
nothing is watching. Untried as of 2026-10-01.

**A tried candidate for it: a final plain-LLM polish pass, one real result.**
Not the trained style adapter (that one has its own, separate open question —
see `recipes/style-pairs/tools/hosted_restyle.py` (branch `question-corpus`), which already tests whether
a frontier model shown real exemplars beats it). This is narrower: a plain
call to whatever strong model the content loop itself uses, given the
already-gated, already-cited answer with citation markers stripped, asked to
do nothing but smooth it for being read aloud. **Kept deliberately small** --
this is one more full round trip added to an already turn-cost-sensitive
pipeline (`modules.md`'s own "a turn is expensive"), so the prompt carries
only the one constraint that is actually load-bearing, not a restated essay
about it. Creator-free, reusable as-is for any creator:

```
Rewrite this answer so it reads as one smooth, natural spoken answer: fix any
abrupt jump between sentences, no other change.

CONTENT IS FIXED. Keep every claim, number, and hedge exactly as stated. Add
no new link between two ideas that is not already there.

Output only the rewrite.

=== PASSAGE ===
{t}
```

**One real test, one question, not a held-out batch, against the longer
first draft of this prompt (result kept out of this file deliberately -- see
below).** It read better: smoother spoken cadence, no abrupt jumps. It also
drifted, despite the instruction -- two shapes, both worth watching for on
any creator: a stated claim came back hedged (a certainty word softened,
changing what is being claimed, which the prompt explicitly says must
survive exactly), and a first-person attribution came back as impersonal
narration (a voice slip, not a fact change, but still the model rewriting
more than it was asked to). Same pattern as every other finding in that round: the
instruction alone does not guarantee compliance, on the very first try. The
shrunk prompt above has not itself been run yet -- carries the same risk at
minimum, possibly more with less guardrail text spelled out, until measured.
The actual passage and rewrite are creator content and do not belong in a
creator-free recipe -- if you want the concrete example, it is in this
session's transcript, not reproduced here.

**Before promoting this pass for any creator:** run it across a real batch
(the benchmark 100, or a slice of it), strip `[n]` first, and score the
output with the same content-verdict judge used to qualify style-pairs
training data (SAME/TRIMMED/DROPPED/ADDED/CONTRADICTS, see
`recipes/style-pairs/tools/stylepairs.py`, branch `question-corpus`) plus a faithfulness re-check with
`chain.py` against the ORIGINAL cited answer as ground truth, not just the
oracle. A pass that reads better but silently drops hedges often enough is a
regression wearing a win's clothes -- measure it, do not eyeball one example.

---

## What did not generalise

- **An ABSOLUTE word cap, derived or hand-picked.** It cannot be right, because the
  appropriate length varies per question and a fixed cap does not know it. 300 was
  set where the creator averages 190 words; on a held-out slice where he averages
  139 the same cap licenses nearly twice his length. Measured there, grouping by how
  much longer the answer is than his:

  | answer ÷ his length | relevancy | off-claims | his words |
  |---|---|---|---|
  | shortest third | 0.908 | 1.14 | 205 |
  | middle | 0.894 | 1.35 | 137 |
  | longest third | 0.871 | 1.66 | 75 |

  Monotonic in both columns. **But the Pearson correlation is +0.032** — the ratio
  has a heavy tail, so the groups trend while the linear relationship does not, and
  the honest claim is the grouped one. It also does not explain the whole drop: the
  best third still reads 0.908 against 0.944 on the benchmark. A length rule that
  transfers has to be relative to something known at inference time, and nothing in
  this recipe currently is.

- **A derived word cap.** `tune.py` set it from the archive's own p90 (331). The
  model then wrote to the ceiling — mean 306, max exactly 330 — and gate rejections
  rose from 1.2 to 1.9 per question. **A cap is a target the model fills, not a
  ceiling it occasionally hits.** The archive describes the creator's lengths; a cap
  sets ours. 300, chosen by hand, beat it on every axis including time.
- **Retrieval settings tuned on the wrong trace.** `tune.py` derives `k` and
  `expand` from queries in a trace directory. Fed a trace from an arm issuing 8.5
  searches per question, it picked settings that lost live under a configuration
  issuing 4.5. **Feed it queries from the configuration being tuned.**
- **Comparing arms on a signal-carrying subset.** A set built from known failures is
  far harder than the benchmark — the same baseline scored 0.845 relevancy on one
  and 0.919 on the other. A session of deltas measured there concluded "nothing
  beats the baseline"; the full set reversed it. Use such a subset to find failure
  classes and check deterministic properties. Promote on the full set.

## Dead ends

- **Asking the model not to invent joins.** Tried twice, lost twice; the joins came
  back reworded. A code gate took unnumbered claims from 283 to 0.
- **Tuning BM25's `k1` and `b`.** A full 4×4 grid spans 0.877–0.886 recall. The
  library defaults are fine.
- **Stacking gates without raising the budget.** Three gates on a two-try budget
  shipped 23 ungrounded sentences, because the gate returned at the first fault and
  the citation check never ran. All faults now report together.

## Silent failures — these finish clean and return plausible output

- **Scoring one creator against another's oracle.** `chain.py` hardcoded
  `healthygamer` in two places: every essential passage would be missing, recall
  near zero, run exits 0. Now raises. **Assume this shape exists in every join.**
- **A filtered question list.** See `THE SHARED DISCIPLINE`.
- **A cached re-score read as a reproducibility check.** See the same section.
- **`--no-judge` with retries on.** The judge is what a retry responds to; removing
  it would leave the loop retrying against nothing. Refused in code.
- **Timing read at the first `<answer>`.** With a gate, the first one is a rejected
  draft. Measure at the last. This under-reported by a third: 38s against 59s.
