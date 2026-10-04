# The arms that have been measured, and how to get back to any of them

A fallback point. Every arm below has its **answers on disk**, so any of them can be
re-scored on a new instrument without spending a single generation call. If an
exploration goes wrong, nothing here has to be regenerated.

Scored against `work/healthygamer/reports/oracle_strict.jsonl`, 100 questions,
`ESSENTIAL` passages only. Owner of the measures: `chain.py`. Owner of their
definitions: `../METRICS.md`.

## The two to keep handy

| | deployed | best harness |
|---|---|---|
| arm | `abench_k6_live` | `harness_v4_superset` |
| questions failing a measure | 46 of 100 | **21 of 100** |
| ungrounded | 29% | **8%** |
| not captured | 30% | **8%** |
| dropped | 2% | 2% |
| padding | **2%** | 8% |
| essential recall | 67% | **86%** |
| context precision | **0.057** | 0.037 |
| passages to the assembler | 41 | 80 |
| seconds | — | 35 |

**Read the win narrowly.** The deployed arm ran the query writer at `turns=1`. The
deployed function's own default is `max_turns=5` (`agent.py:229`). So this is *the
harness beating the deployed system with its loop switched off*, which is the
configuration every arm has been benchmarked at, but not the deployed default.
**The loop-versus-loop comparison has not been run.** See `docs/three-loops.html`.

## The structural arm, and the noise it has to clear

20-question iteration set. `noise` is the SAME controller on the SAME questions,
changing nothing, so it is the floor: see K15 in `problems.md`.

| the four measures | baseline | `noise` | `narrow` | band | verdict |
|---|---|---|---|---|---|
| faithfulness | 0.929 | 0.933 | **0.978** | 0.005 | +0.047, real |
| answer relevancy | 0.845 | 0.856 | **0.933** | 0.011 | +0.083, real |
| context recall | 0.883 | 0.854 | 0.825 | 0.029 | -0.044, real |
| context utilization | 0.887 | 0.808 | 0.729 | 0.079 | -0.119, real |

`narrow` = `--commit --max-passages 8 --require-cites --word-cap 160 --hard-turns 14`
with `controller_narrow.md`. Scored with the `[n]` markers STRIPPED, so the text is
comparable to arms that carry none -- a faithfulness judge that can see citations
would otherwise be grading a different artifact.

**The deterministic half, which sampling cannot move.** Unnumbered asserting
sentences fell from **283 to 6** across 20 answers, and 4 of the 6 are a sentence
splitter breaking on "Dr." and on transitions. Words per answer 235 to 150.

**Why the two regressions.** Not the commit cap: the oracle says no question needs
more than 4 essential passages (mean 2.9, max 4 over 318), so 8 is already double
the worst case. It is the word cap. The answer cites 3.5 passages, about the right
number, but only **27 of 59 essential passages, against the 48 search had found** --
at 150 words there is no room for three essential passages and an answer, so it
drops material, essential included. `--require-cites` buys faithfulness;
`--word-cap` buys relevancy and pays in recall and utilization. The `cites` arm
separates them.

**Cost.** 7.7 turns and 205s against 5.0 and 122s, from 31 gate rejections at a
round trip each.

## PROMOTED: `nocap` — 39 of 218 failing on unseen questions

```
--controller controller_deploy.md --require-cites --min-k 15 --expand 2
--full-text --max-retries 0 --no-judge
```

Report: `work/healthygamer/reports/nocap.json`. `[n]` markers stripped before
scoring; every pair scored from one cache state.

| held out, 218 unseen | baseline | `deploy` (capped) | **`nocap`** |
|---|---|---|---|
| faithfulness | 0.939 | 0.957 | **0.958** |
| context recall | 0.846 | **0.902** | 0.898 |
| context utilization | 0.732 | **0.754** | 0.749 |
| answer relevancy | **0.929** | 0.891 | 0.897 |
| **failing any measure** | 52 (24%) | 48 (22%) | **39 (18%)** |
| words per answer | — | 267 | **242** |
| gate rejections / question | — | 1.47 | **1.21** |

**Removing the word cap made answers shorter.** 242 words against the capped arm's
267. Told it may write 300, the model writes toward 300; told nothing, it writes
242. That is the "a cap is a target, not a ceiling" claim confirmed directly, and it
is why `--word-cap` is gone from the promoted config.

**The honest size of the win.** On the benchmark 100 the capped arm failed 4 against
the baseline's 15. On 218 unseen questions the promoted arm fails 39 against 52 —
6 points, not 11. The slice is harder for everything; the baseline degrades on it
too. Three of four gains transfer, relevancy roughly recovers to parity.

**Still open:** only 25 of 218 answers pass the citation gate first time, so 1.21
rejections per question at ~13s each is most of the latency. Measurable by a count,
not a judged score.

## turbo on the full 100

`--require-cites --min-k 15 --expand 2 --full-text --max-retries 0` with
`controller_lean.md`. 100 questions, `[n]` markers stripped before scoring.

| | deployed | baseline loop | **turbo** | band@100 |
|---|---|---|---|---|
| faithfulness | — | 0.945 | **0.962** | 0.002 |
| context recall | — | 0.876 | **0.926** | 0.013 |
| context utilization | — | 0.857 | **0.895** | 0.035 |
| answer relevancy | — | 0.919 | **0.933** | 0.005 |
| sum of four | — | 3.597 | **3.716** | |
| coverage (recall x utilization) | — | 0.751 | **0.829** | |
| questions failing any measure | 46 | 14 | **12** | |
| turns | 1 | 5.0 | **3.3** | |
| searches | 1 | 8.5 | **4.5** | |
| to answer | — | 46s | **38s** | |

All four beat the baseline outside their bands, and it is the fastest arm measured.

**Read the earlier 20-question sections with K16 in hand.** That set is
failure-enriched by construction, so it is much harder than the 100 -- the same
baseline scores relevancy 0.845 on the 20 and 0.919 on the 100. Arm deltas measured
there do not transfer, and the conclusions "no configuration beats the baseline" and
"turbo regresses utilization" were both artefacts of it. Utilization in fact rose
0.857 to 0.895.

## turbo at n=20: why this section was misleading

`--require-cites --min-k 15 --expand 2 --full-text --max-retries 0` with
`controller_lean.md`. 20-question iteration set, scored with `[n]` markers stripped.

| | baseline | rerun | narrow | wide | **turbo** |
|---|---|---|---|---|---|
| faithfulness | 0.929 | 0.933 | **0.978** | 0.936 | 0.940 |
| context recall | 0.883 | 0.854 | 0.825 | **0.950** | 0.883 |
| context utilization | 0.887 | 0.808 | 0.729 | 0.625 | 0.775 |
| answer relevancy | 0.845 | 0.856 | **0.933** | 0.880 | 0.867 |
| turns | 5.0 | 5.0 | 7.7 | 4.5 | **3.2** |
| searches | 8.5 | 8.5 | 8.8 | 6.5 | **4.3** |
| to answer | — | 46s | 57s | 47s | **38s** |
| total incl. judge | — | 122s | 205s | 100s | 91s |
| measures worse than the band | — | — | 2 | 1 | **0** |

**Two numbers for time, not one.** `to answer` is wall clock to the first complete
answer. The rest is the judge, which at `--max-retries 0` has its findings recorded
and never acted on -- the measurement instrument, not part of answering. If the judge
must gate output, the number is 91s.

**Three cuts got from 122s to 38s, each from a trace measurement.** The retry fired
on 74 of 100 questions at 69s mean, and the citation gate guarantees in code what the
retry was buying back. 43% of round trips were a lone search or a lone read at ~19s
each, so `--full-text` returns passage text and drops `read` from the schema.
Reasoning costs 8.7s per 100 words, measured -- and was left on deliberately, since
it is where the answers come from.

**The changes interact rather than add.** turbo's recall is 0.883, not `wide`'s
0.950: with full text the controller searches 4.3 times instead of 6.5, because it
stops querying once it can read what it holds. That hands back what the wider `k`
had won.

**What is NOT won.** The four-measure sum is 3.465 against the baseline's 3.544,
inside the +-0.093 noise (K15). No configuration beats the baseline overall, and the
reason is structural: `chain.py:272` makes utilization's denominator recall's
numerator, so finding more essential material lowers utilization unless the answer
grows, and growing it costs relevancy. Three of the four measures are one curve tied
by answer length, and the baseline already sits on a reasonable point of it.

**What IS won.** Faithfulness 0.929 -> 0.978 (`narrow`), 9x its band. Unnumbered
asserting sentences 283 -> 6, and 0 in `stack` -- deterministic, and no measure in
the testbed captures it. And a shape that answers in 38s with nothing worse than
noise.

## Seconds are not comparable across the iteration-set arms

The prompt variants on the 20-question iteration set were run with arms overlapping
in time, against one API account. An arm that shared the account with another paid
for the contention in wall clock and in nothing else. **So compare the four measures
across those arms freely, and do not compare their seconds to each other or to the
100-question runs.** Any timing claim needs an arm run alone, which is why timing was
left until after the variants were chosen.

## Every arm

| arm | what it is | recall | precision | shown | secs |
|---|---|---|---|---|---|
| `abench_k6_live` | deployed, one call | 67% | 0.057 | 41 | — |
| `abench_k25_live` | deployed, wider net | 89% | 0.023 | 128 | — |
| `abench_k25_concentrate` | + "use only the 2-3 most direct passages" | 89% | 0.023 | 128 | — |
| `abench_k25_cover` | + "carry every point they make" | 89% | 0.023 | 128 | — |
| `ladder_L0` | deployed, **through the harness** | 67% | 0.057 | 41 | — |
| `ladder_L1` | + search and read offered | 67% | 0.057 | 41 | 17 |
| `ladder_L3` | + per-part gate | 66% | 0.057 | 41 | 19 |
| `ladder_L4` | + gap search, one round | 74% | 0.032 | 60 | 34 |
| `ladder_L4iter` | + gap search, iterated | 79% | 0.032 | 82 | 62 |
| `ladder_L5` | + relevance filter before assembly | — | — | 33 | 23 |
| `harness_v3` | first loop, starts from nothing | 52% | — | 70 | 42 |
| `harness_v4_superset` | first loop, seeded from deployed | **86%** | 0.037 | 80 | 35 |
| `bench_turns1` | retrieval only, query writer at 1 round | 68%* | 0.080* | 41 | — |
| `bench_turns4` | retrieval only, query writer at 4 rounds | **85%\*** | 0.059* | 71 | — |

\* the two `bench_turns` rows are on the **retired** span-overlap metric — recall
against the single passage each question was generated from. **Not comparable** to
the oracle numbers in the same column. They are here because they are the loop that
already existed, and scoring them on the four measures is the next cheap measurement.

## How to reproduce each

```sh
# deployed, one call
python3 answer_bench.py -n 100 --k 6  --json work/healthygamer/reports/abench_k6_live.json
python3 answer_bench.py -n 100 --k 25 --json work/healthygamer/reports/abench_k25_live.json
python3 answer_bench.py -n 100 --k 25 --variant concentrate --json .../abench_k25_concentrate.json

# the ladder: 0 is the deployed call, each level a strict superset of the one below
python3 harness/ladder.py --level 0 --bench -n 100 --json work/healthygamer/reports/ladder_L0.json
python3 harness/ladder.py --level 5 --bench -n 100 --json work/healthygamer/reports/ladder_L5.json

# the first loop, seeded from the deployed pool -- currently the best arm
python3 harness/run.py --bench -n 100 --json work/healthygamer/reports/harness_v4_superset.json

# retrieval only, the query writer's own loop at 1 and 4 rounds
python3 retrieval_bench.py --set live -n 100 --turns 1 --k 6 --json .../bench_turns1.json
python3 retrieval_bench.py --set live -n 100 --turns 4 --k 6 --json .../bench_turns4.json
```

## How to score, without regenerating anything

```sh
# the four measures, any set of arms, against the strict oracle
python3 chain.py work/healthygamer/reports/abench_k6_live.json \
                 work/healthygamer/reports/harness_v4_superset.json \
                 --essential-only --jobs 6

# why the loop lost, from the model's own narration
python3 harness/why.py work/healthygamer/reports/harness_v4_superset.json --diagnose

# does filtering the pool keep the signal -- no answers generated
python3 prefilter.py -n 40 --arm harness_v4_superset.json
```

The oracle is committed, so scoring is reproducible. The answers are committed, so
re-scoring costs only the judge calls, and those are cached on prompt text.

## The held-out set, untouched

`oracle250.jsonl` is 250 questions with a strict oracle, and **only 39 of them
overlap the 100 above** — the sampler stratifies differently at different `n`. It has
never been scored against any arm. It is spent once, on whichever configuration is
being committed to, to check the gains are not fitted to the iteration set.

## What is not established

- **No noise floor on the four measures.** The only one measured is on a retired
  metric: the same configuration run twice gave DROPPED 24 and 27, CONTRADICTS 14
  and 12. Differences of a few points are not interpretable.
- **`ladder_L5` is unscored on the four measures.** Its first run recorded the whole
  pool instead of the filtered set, so its grounding read high for the wrong reason.
  Fixed; the rerun records 33 passages against a pool of 72, and the filter keeps as
  few as **1** passage on some questions, which is a live risk.
- **The redo-after-judge branch has never been run.** No arm assembles, judges, and
  goes back.
