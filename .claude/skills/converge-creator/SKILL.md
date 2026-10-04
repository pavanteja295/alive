---
name: converge-creator
description: Converge a creator's grounded-answering configuration toward a stated goal by looping measure-diagnose-change-remeasure until no remaining candidate beats the measured noise floor. Use when onboarding a new creator, adding takes to an existing one, or asked to tune, improve, measure or diagnose the knowledge_style prototype. Carries the decision rules and prior results from the first subject so they are not rediscovered.
---

# Converge a creator

You are running a **convergence loop toward a goal**, not a checklist. Each pass
measures, picks the single highest-value change, tries it, and either accepts or
retires it. You stop when nothing left beats noise.

Working directory: `engines/knowledge_style/proto/`.
Reasoning behind every rule: `RECIPE.md`. Prompts: `PROMPTS.md`.

---

## THE GOAL

A promoted configuration for one creator, measured on the four frozen measures.

**`METRICS.md` owns the measures. `proto/BASELINES.md` owns the arm results and the
reproduce commands. This file owns only the decision rules.** If any of them
disagree, they own their half and this file is wrong.

The four, frozen 2026-09-16 in `METRICS.md`, computed by `proto/chain.py`:
faithfulness, context recall, context utilization, answer relevancy. **Adding a
fifth requires a commit saying what the four could not answer.**

> An earlier version of this skill converged against `evidence recall`,
> `correct declining` and `ADDRESSED rate`, driven by `run_arm.py` and
> `verify.py`. Those instruments were retired by the freeze. Both toolchains are
> still on disk, so it is entirely possible to run the old one and get numbers
> that mean nothing against the current baseline. **Score with `chain.py`.**

**Hard constraints. Never trade these.**

1. No failed rows in the scored arm. A crash scores as zero recall and reads as a
   regression.
2. The oracle is not stale against the corpus, and is that creator's own.
   `chain.py --creator <name>` now raises when it is missing; it used to score one
   creator's answers against another's essential passages and finish clean.
3. Every prompt stays creator-free, including subject matter. `harness/selftest.py`
   lints it. A worked example written in this creator's topics is polluted even when
   it names no number.

**Read the failure count, not the sum.** The promoted arm has the *lower* sum of
four than an arm it beats, and less than half the failures. A bound does not lift a
mean; it removes the answers that dip under a threshold, which is what a viewer
notices. Report both, promote on failures.

**They trade against each other, and one pair trades by construction.**
`chain.py` computes utilization as used ÷ *captured*, so **utilization's denominator
is recall's numerator**: finding more essential material lowers utilization unless
the answer lengthens, and lengthening it costs relevancy. Three of the four are one
curve tied by answer length. Do not average them, and do not read a fall in one as
damage without checking the other two.

**Converged when:** no untried candidate has an expected effect above the measured
noise floor, and the deterministic checks hold.

## THE LOOP

Keep state in `work/<subject>/ledger.md`. Write it as you go so the loop can
resume in a later session. One pass:

```
  measure  ──►  diagnose  ──►  pick ONE candidate  ──►  run ONE arm
     ▲                                                      │
     └──────────  accept / retire, update ledger  ◄──────────┘
```

### Step 0 — enter the loop (once)

```bash
cd engines/knowledge_style/proto && make status
```

**Preconditions, and this recipe does not build them.** A creator needs a corpus and
an **oracle** before anything here is interpretable. The oracle comes from the
`question-corpus` recipe, then `oracle.py`. Without it `chain.py` raises. Do not run
the upstream tool from here — name it and stop.

**Order for a new creator:** `question-corpus` → **this** → the style recipe
(`recipes/style-model/`), which takes this one's promoted configuration as an input
and reads `BASELINES.md` to find it. So **name the promoted arm unambiguously in
`BASELINES.md`** — that string is the whole coupling between the two halves.

Then three things that gate everything after them:

- **The control.** An arm with no archive, read by eye. What the base model already
  believes about this creator. Nothing later is interpretable without it.
- **THE NOISE FLOOR, BEFORE ANY ARM.** Re-run the *identical* configuration on the
  *identical* questions and score both. This is not optional and it is not cheap to
  skip: on the first subject it moved **17 of 20 question scores** and flipped 7
  pass/fail, so five prompt variants were run before anyone knew they were
  unresolvable by construction. There is no scoring noise — `llm.py` content-
  addresses every call, so re-scoring one answer is byte-identical. It is all
  generation spread. Record the per-measure band; every accept below uses it.
- **A signal-carrying subset is for reading, never for a delta.** A set built out of
  known failures is far harder than the full benchmark — the same baseline scored
  0.845 relevancy on one and 0.919 on the other. A whole pass of arm comparisons on
  such a set concluded "nothing beats the baseline" and the full 100 reversed it.
  Use it to find failure classes and to check deterministic properties. Promote on
  the full set.

### Step 1 — measure

```bash
python3 harness/loop.py --bench -n 100 --creator <name> --profile [flags] \
    --json work/<name>/reports/<arm>.json --trace work/<name>/reports/<arm>_trace
python3 chain.py work/<name>/reports/<arm>.json --creator <name> \
    --essential-only --jobs 8 --json work/<name>/reports/chain_<arm>.json
```

**Strip `[n]` citation markers before scoring.** An arm that cites inline and one
that does not are otherwise different artifacts, and a faithfulness judge that can
see citations grades differently.

**Score arms one at a time, or their seconds are worthless.** Generation and scoring
share one API account; anything overlapping is contended and its wall clock means
nothing. The four measures are unaffected.

### Step 2 — diagnose, and let it choose the candidate

Do not pick by intuition, and **do not count words in the reasoning.** "The core
mechanism, 6x more frequent in failures" was 1 occurrence against 1, scaled by a
word-count denominator; rates per 1,000 words make single-digit counts look like
structure. Read the failing questions instead — and read the **judge's** reasoning,
because the controller never narrates its own error. It does not write "I am
inventing this link", it just writes the link.

| finding | candidate it implies | measured on the first subject |
|---|---|---|
| ungrounded claims, joins between passages | **a code gate**: every asserting sentence carries `[n]` | unnumbered claims 283 → 0. The single largest win |
| answers drifting past the question | a word cap, and "the last sentence must answer it" | failures 14 → 5 |
| essential passages retrieved but not cited | widen retrieval; check the cap is not cutting them | a 160-word cap used 27 of 59 essential passages |
| essential passages never retrieved | raise `--min-k` / `--expand`, tuned offline against the oracle | recall 0.835 → 0.962 for two integers, zero round trips |
| slow | the retry, and the separate `read` turn | 122s → 61s |

**Guarantees are code, choices are prompt.** Asking the model not to invent a link
was tried twice and lost twice; the joins came back reworded. A gate made it
impossible. A guaranteed change also needs no statistics to verify, which matters
enormously when the noise floor is wide — check it by counting shipped answers.

**Retrieval is tunable offline, for free.** The oracle plus queries already in the
traces make `k` and `expand` a grid search with no model calls. `harness/tune.py`
does it. Two cautions, both learned by getting them wrong: feed it queries from the
configuration you are tuning, since settings the grid calls equivalent are not when
the controller searches half as often; and **do not derive a word cap** — a cap is a
target the model fills, not a ceiling it occasionally hits.

**Fixes that are not candidates, do them immediately and re-measure:** failed rows,
rows that searched zero times, a stale oracle, a wrong band label.

**Verify an instrument against the source before believing it.** Three versions of a
citation checker each reported a crisis that did not exist: bare video ids read as
fabrications, verbatim quotes spanning a chunk boundary, and quotes compared against
a 400-character trace field instead of the passage. A checker that has not been
spot-checked measures its own strictness.

### Step 3 — run exactly one arm

One variable. If you change two, you cannot attribute the result, which is the
mistake that cost a day on the first subject and needed a 2×2 to undo.

```bash
python3 harness/loop.py --bench -n 100 --creator <name> --profile [ONE flag] \
    --json work/<name>/reports/<arm>.json --trace work/<name>/reports/<arm>_trace
python3 chain.py work/<name>/reports/<arm>.json --creator <name> --essential-only \
    --jobs 8 --json work/<name>/reports/chain_<arm>.json
```

An explicit flag beats the profile, so an arm is never silently overridden by the
creator's defaults.

### Step 4 — accept or retire

- **Accept** if a measure improves by more than the band from Step 0 and none
  regresses by more than its band, and every hard constraint holds. Promote on the
  **failure count**, not the sum.
- **Retire** otherwise, and write in the ledger what it measured. A retired
  candidate does not return without new evidence.
- **Thresholded pass/fail is the noisiest readout there is.** A question at 0.71
  changes category on a 0.02 wobble. Read means; use the counts only for promotion,
  on the full set.
- **Promote by editing `BASELINES.md` to name the arm**, because the style recipe
  reads that name and hashes the report. "The newest report in the directory" is not
  the promoted config and cost the style session two days.

### Step 5 — loop or stop

Untried candidate above the floor → back to Step 1. Otherwise converged.

---

## Settled. Do not spend a pass re-deriving these.

Creator-independent, established against literature plus measurement:

- Fixed-size chunks ~300 words, ~20% overlap. Semantic chunking loses.
- Verbatim storage. Never extraction or paraphrase in the store.
- No ASR cleanup, but dedup rolling captions before trusting any count.
- **The model writes several queries, not one.** +0.351. The architecture.
- **At least one search, enforced in code.** Left free, the model skips
  searching on topics it assumes are out of domain, then answers from its own
  knowledge.
- Oracle labels are timestamp spans, never chunk ids.
- Retrieval, not whole-archive context. Flat cost instead of linear.
- **Prompt wording is worth +0.008 on recall.** Add a rule only in response to
  an observed break, never speculatively.
- **The per-creator surface is three numbers, not a prompt.** Every controller is
  creator-free; `harness/profiles/<creator>.json` holds the retrieval settings,
  derived by `harness/tune.py`. Swapping creator is corpus + `creator.md` + a
  re-derive. See `problems.md` K17.
- **The in-loop judge is not the scoring instrument.** At `--max-retries 0` its
  findings are recorded and never acted on, so `--no-judge` removes ~40s and changes
  no answer. `chain.py` scores offline from the saved answers. It is refused with
  retries on, where the judge is what a retry responds to.
- **Report two numbers for time**: to the shipped answer, and including the judge.
  And take the time to the *last* answer turn — with a gate, the first `<answer>` is
  a rejected draft, which under-reports by a third.

---

## On stopping

When converged, say so plainly and name what is left, because it is not
engineering and no further pass will reach it:

- **Does it sound like them?** A person who knows the creator reading ten
  answers. No metric substitutes. Never done for the first subject.
- **One format or several?** If the corpus is a single register, the person
  cannot be separated from the format. A data decision, not a modelling one.

Then `make report SUBJECT=<name>`.

Report honestly, including every candidate you declined to chase because it fell
below the floor. Every number this produces measures grounding and retrieval.
None of them says the voice is right.
