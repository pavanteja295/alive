# Brief: a harness for the style axis

Handoff written 2026-09-18, at the end of the session that built the **content**
harness (`A2` in `SYSTEM.md`). Read `MANIFESTO.md`, `METHOD.md` and the repo
`CLAUDE.md` first — this file only carries what that session learned that a style
session would otherwise rediscover.

---

## 1. The content harness is done. Do not rebuild it.

`A2` in `SYSTEM.md` is the answering loop: one controller, one judge, two creator-data
routes. On 100 questions it beats the previous loop on all four frozen measures and
runs in a third of the turns. Numbers live in `proto/BASELINES.md`, measures in
`METRICS.md`, unresolved items in `problems.md` (K14–K18).

Deployed configuration:

```
python3 harness/loop.py --bench -n 100 --controller controller_deploy.md --profile \
  --require-cites --full-text --max-retries 0 --no-judge
```

**Pending when this brief was written:** the `profile` arm (derived retrieval
settings) was mid-run on 100 questions; it and `deploy.json` still needed scoring,
`METRICS.md` still needed its final content results, and none of it was committed.
Check `git status` in the repo root before assuming any of that landed.

---

## 2. The style axis has no metric that can drive a search. This is the whole problem.

Not an opinion — it is measured and recorded in `METRICS.md` under
"Contested — the style axis". Three findings, each of which kills an obvious approach:

| finding | consequence |
|---|---|
| the blind A/B judge reads **98–100%** across four adapters, two base sizes, two datasets, five decoding settings | a measure at its ceiling cannot tell "barely moved" from "did not move". It cannot rank anything |
| corpus-level aggregation is **gameable** — STRAP §3.2 shows a naive copy baseline wins it | any aggregate must be computed per sentence, then averaged |
| targets are **ASR output**, and one judge tell was "very very rumitative", a mis-transcription | part of the gap is unreachable and undesirable. Chasing 100% similarity means learning transcription errors |

And the paper set disagrees with itself: `GOLD_STANDARD.md` calls LUAR/Wegmann
authorship models a non-negotiable substrate; `DECISION.md` kill-lists them — Reddit
not speech, every method scoring 0.07–0.15. Our takes are topic-separated by
construction, which is the exact condition `2311.07564` says breaks those models.

**So the first task is not a harness. It is settling the measure.** Building a loop
against a saturated judge produces motion and no information.

---

## 3. Cost decides what is even possible

From `METRICS.md`:

| measurement | cost |
|---|---|
| retrieval, full corpus | **seconds, no model calls** — this is the only reason `A1` could be swept |
| blind A/B, n=60 | **~25 minutes** |
| one training run to its turn | ~1 hour |

At 25 minutes a search gets tens of points, not hundreds. `METRICS.md` already states
that reducing this is a **prerequisite for exploring, not an optimisation**.

The asymmetry to exploit: **stylometry counters cost nothing to compute**, like
retrieval. Sentence length, filler rate, idiom frequencies. A style measure built from
counters can be swept; one built on a 25-minute A/B cannot.

---

## 4. Check the data before proposing anything

The repo rule is explicit: ground the data before proposing a loop. There is existing
style-eval output on disk, already paid for:

```
proto/work/healthygamer/reports/basecheck_*.jsonl     fields: model verdict reason source rewrite
proto/work/healthygamer/reports/basecheck_*.html
proto/work/healthygamer/reports/audit_*.html
```

Seven `basecheck` runs across two base models, two checkpoints and three sample sizes
(n=30, n=60, n=210). Each row holds the source, the rewrite, a verdict and the judge's
stated reason.

**The first measurement, and it needs no new runs and no GPU:** compute stylometry
counters over `source` and `rewrite` in those files and ask whether they separate the
arms that the A/B judge could not. If counters discriminate where the judge
saturates, that is the instrument. If they do not, say so — and then the style axis is
a decision for a person, not a measurement.

---

## 5. Seven things the content session learned the hard way

Each cost real time or produced a wrong published number.

1. **Measure the noise floor before any arm.** Rerunning the identical config on the
   identical 20 questions moved 17 of 20 question scores and flipped 7 pass/fail.
   Five prompt variants were run before this was known; all were unresolvable by
   construction. `problems.md` K15.
2. **Never compare arms on a failure-enriched set.** The 20-question iteration set was
   built out of known failures, so it is much harder than the 100 — the same baseline
   scores 0.845 relevancy on it and 0.919 on the 100. A whole session of deltas
   measured there concluded "nothing beats the baseline", and the full 100 reversed it.
   K16. **Use a signal-carrying subset for deterministic checks and for reading
   failures; never for a delta.**
3. **Thresholded pass/fail is the noisiest readout of any measure.** A question at 0.71
   changes category on a 0.02 wobble. Read means.
4. **Guarantees belong in code, choices in the prompt.** Asking the model to avoid
   inventing a link was tried twice and lost twice. A code gate requiring every
   asserting sentence to carry a passage number took unnumbered claims from 283 to 0.
   A guaranteed change also needs no statistics to verify, which matters enormously
   when the noise floor is wide.
5. **Verify every instrument against the source before reporting it.** Three versions
   of a citation checker each reported a crisis that did not exist (bare video IDs read
   as fabrications; verbatim quotes spanning a chunk boundary; quotes compared against
   a 400-char trace field instead of the passage). A checker that has not been
   spot-checked measures its own strictness.
6. **Counting words in the reasoning is not evidence.** "The core mechanism, 6× more
   frequent in failures" was 1 occurrence against 1, scaled by a word-count
   denominator. Read the failures instead; and read the **judge's** reasoning, because
   the controller never narrates its own error.
7. **Creator-free means topics too.** A prompt illustrating a failure with this
   creator's subject matter is polluted even when it names no number or vocabulary
   term. `harness/selftest.py` now lints content-domain words. Examples must describe
   the *shape* of a failure.

---

## 6. The creator-wrapper pattern, reusable as-is

The content harness keeps every prompt creator-free and pushes the per-creator surface
into one derived file:

```
harness/tune.py                 derives the settings from that creator's corpus + oracle
harness/profiles/<creator>.json the only creator-specific file: 3 numbers + the evidence
loop.py --profile               loads it; an explicit flag always wins
```

It immediately beat hand-fitting: the derived `k=10, expand=3` matched the hand-picked
`k=15, expand=2` on recall with a 10% smaller pool, and it corrected a word cap set
from taste (300) to the archive's own p90 (330).

**A style harness should do the same from the start.** Whatever the measure turns out
to be, the thresholds and weights will be creator-specific and must be derived, not
authored. `problems.md` K17.

---

## 7. What to ask before building

`CLAUDE.md` requires these five, agreed with a person, before a loop is set up:

1. **the commitment** — what is fixed, and why
2. **the search space** — what is open given that
3. **what counts as an answer** — the metric, or the frontier if two-objective. *For
   style this is the open question, not a formality.*
4. **the held-out set** the loop never sees
5. **the budget** — runs, hardware, what else is queued

`manner` via `ims` is the only asset that already has a train command, an eval, a
split by `paragraph_id` and a composite score, which is why `MANIFESTO.md` names it
the pilot. Style work starts there, not with new machinery.

---

## 8. Caveats on numbers this brief cites

- The oracle is **318 adjudicated questions** out of a ~10k corpus, plus 250 held out
  and never used. Recall is measured against the oracle's essential set, so a passage
  its sweep never surfaced is invisible to the measure.
- ESSENTIAL-per-question runs min 1, **max exactly 4** across all 318, which looks like
  the adjudication prompt rather than the questions. Two design decisions leaned on
  "no question needs more than 4". K18, open.
- Per-question seconds are only comparable between arms run alone. Several arms in that
  session overlapped on one API account and their wall clock is contended.
