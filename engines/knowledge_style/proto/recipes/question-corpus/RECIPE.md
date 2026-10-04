# Recipe: the question corpus

**Produces** questions whose answer is the creator's own verbatim words plus
minimal connective fillers, each tagged with whether our retriever can reach it.

**Validated on** one creator: healthygamer (Dr K), 17 videos, 292 passages,
10,046 questions, 2026-09-08. A second creator (huberman) ran 3 passages before
being stopped. **One creator is not two.** Read
[What did not generalise](#7-what-did-not-generalise) before trusting any number
in here.

**This file owns the process.** If a prompt, a script or a conversation
disagrees with it, this file is right and the other has drifted.

```
recipes/question-corpus/
  RECIPE.md          this file
  tools/             the scripts
  profiles/<name>.py everything that varies by creator
work/<subject>/      chunk store, corpus, ledger, reports
```

To run a new creator: copy a profile, change every declaration, run. **The
procedure is not edited.**

---

## 0. Signature

A recipe is a function. It acts on inputs, so the signature comes before the
steps, and every input names the file it is read from rather than being assumed.

### Inputs

| input | shape | where it comes from |
|---|---|---|
| **verbatim subtitles** | YouTube `.json3`, one file per video, `events[].segs[].utf8` with `tStartMs` | a directory you pass as `TAKES=`. Auto-captioned is fine; **do not clean them**, `2311.07564` TACL found normalised lowercase transcripts work as well or better |
| **chunk store** | `work/<subject>/store/<config>/chunks.jsonl`, fields `chunk_id take_id seq ts text title` | `make index`. Not a recipe, shared machinery: `../../build.py` |
| **profile** | `profiles/<subject>.py`, declaring `WHO ASK_YES ASK_NO COINED_TERMS RETRIEVER BUDGET` | `tools/qprofile.py` drafts it, **then a person reads it** |
| **Anthropic API key** | a key in `~/.claude/anthropic_api_key`, or `$ANTHROPIC_API_KEY` | the **file wins** over the environment, deliberately: a stale key in a systemd user manager's environment outlives logins and 401s every unit while the same code works in a shell |
| **`anthropic` SDK** | importable | `pip install anthropic`. The CLI backend also works but hangs inside a systemd unit rather than erroring |

Nothing else. In particular **not** the map, exemplars, probe set or oracle:
those belong to the content-model arms, and an earlier `genq.py` demanded
`exemplars.json` through `agent.load`, which crashes a creator that has only
been indexed.

### Preconditions, all checked before any work starts

```bash
python3 tools/preflight.py --subject <name>
```

Fifteen seconds, and it **raises** rather than warning. It includes one live
model call, because a key that is present and a key that works are different
things. A recipe that discovers a missing input at minute 90 has burned the run
and half-written its outputs.

It also checks the failure the profile loader cannot see: **a profile that was
copied and not edited.** `ASK_YES` identical to another creator's means
generation is steered toward that creator's subject matter, and every output
still reads fine. Editing `WHO` alone does not fix it.

**A missing input stops the recipe.** Nothing is inferred, defaulted or worked
around. That is your decision and it is one sentence to ask.

### Outputs

What is left on disk for something downstream to consume.

| artifact | contents |
|---|---|
| `work/<subject>/qcorpus.jsonl` | one row per question. `q answer used tier source_chunk source_take source_ts retrieval_found_source retrieval_rank_of_source retrieved_takes fact_bearing grounded_trigram filler_ratio answer_words provenance` |
| `work/<subject>/qcorpus_rejects.jsonl` | every rejection with its reason. Not waste: this is the retrieval improvement set |
| `work/<subject>/qcorpus_{bucketA,bucketB,style}.jsonl` | cuts written by `qselect.py`, non-destructive |
| `work/<subject>/reports/qcorpus.html` | searchable, for a person |

**Consumed by** the style-pairs recipe (on branch `question-corpus`; not in this release), which owns
what it does with them. `qcorpus.jsonl` and the reports carry verbatim
transcript and are gitignored; regenerate rather than commit.

---

## 1. Why this exists

Read the creator's verbatim subtitles. Ask: **what would a person using this
model plausibly ask him, that these exact words already answer?**

| | |
|---|---|
| the question is derived **from** the verbatim | not invented and then checked against it |
| the test is **would a real user ask this** | not whether it is hard |
| we are not challenging the creator | the right question is one he answers by reading his own subtitles back |
| we are not challenging our content model | the content model does not run in this recipe at all |

The target is the **mode** of the question distribution. Not its coverage, not
its tail.

**Two things break without this stated, and both happened.**

- A proposer told to *"spread the questions across different material in the
  transcript"* produced coverage, not demand. Cited material came out at mean
  position **0.48** across each video, dead uniform, which is what tail-seeking
  looks like when plotted.
- Retrieval used as an accept **gate** deleted **58** questions whose answers
  were plainly in the subtitles. If the question is plausible and his words
  answer it, the creator could answer it. Our retriever missing it is our
  defect, not grounds to delete the question.

**Never let the measured thing influence what gets produced.** The content model
does not run here. Selecting for questions our system answers well would make
the corpus a measurement of itself. The stage order makes this structural rather
than a matter of discipline.

---

## 2. The item

**One passage.** ~300 words, the unit `build.py` chunks to.

A passage is proposed against, judged, and recorded independently. Nothing
carries between passages except the accumulated output file, so a run resumes at
the first passage absent from it and a failed passage costs only itself.

Tiers are shapes of answer assembly over that unit, not a difficulty ladder, and
are never averaged together:

| tier | assembled from | measured yield |
|---|---|---|
| **1** | one passage | 85% accept, 19 per passage |
| **2** | 2-3 passages, same video | 37% accept, and **793 rejected as "collapsed to one passage"** |
| **3** | passages from different videos | 8% accept, ~0.7 per video pair. **Do not scale it** |
| **4** | re-phrasings of an accepted question | no judge call at all |

**Tier 4 is the cheap one and the sharpest instrument.** The answer is already
known constructible and a re-phrasing asks the same thing, so the answer carries
over unchanged. What it buys is a controlled retrieval experiment: content held
constant, only wording varying.

**A question that collapses is rejected, never relabelled.** If the judge
answers a tier-2 question from one passage it is discarded. Counting it would
inflate the composed total by 70%.

---

## 3. Tool inventory

A postcondition is what may be assumed once the tool exits clean. It exists for
**context economy**: read the twelve numbers, not the corpus.

| tool | intelligence | postcondition |
|---|---|---|
| `tools/preflight.py` | 1 call, to prove the key works | every input in §0 exists and has the declared shape. Raises otherwise |
| `tools/qprofile.py` | 1 call | `profiles/<subject>.py` drafted from the corpus. **A draft, not an answer**: read it |
| `../../build.py` | none | chunk store exists, deterministic from the subtitles, `index_version` stamped. Shared, stays flat in `proto/` |
| `tools/genq.py` | **2 calls per passage** | every accepted question has a verbatim answer, an A/B retrieval verdict, and provenance. Resumable at the first passage absent from the output |
| `tools/genq_tiers.py` | 2 calls (t2/t3), **1** (t4) | tiers 2-4 appended. One failed group is recorded and skipped, never fatal |
| `tools/qdiagnose.py` | 6 free, 2 that call | eight checks. `negctl` is the one that validates the judge |
| `tools/qtag.py` | **1 call per answer** | every answer has `fact_bearing`. Resumable; a re-run costs nothing for answers already tagged |
| `tools/qselect.py` | 1 call per candidate pair, `--dedup` only | a subset by bucket, tier, facts, rank, filler. Non-destructive |
| `tools/qreview.py` | none | searchable HTML for a person |
| `../../llm.py` | shared | every call content-addressed and cached. A replay is free and byte-identical |
| `../../paths.py` | none | what exists and what has gone stale. Run before believing any number |

```bash
make index    SUBJECT=<name> TAKES=/path/to/json3   # the only prerequisite
make qprofile SUBJECT=<name>                        # drafts the profile. READ IT
make preflight SUBJECT=<name>                       # every precondition, raises
make qcal     SUBJECT=<name>                        # 10 passages. READ THEM
make qdiag-full SUBJECT=<name>                      # incl. negctl, on the sample
make qcorpus  SUBJECT=<name> N=20                   # the full run, ~3 h / 292
make qtiers   SUBJECT=<name>                        # tiers 2, 3, then 4
make qtag     SUBJECT=<name>                        # fact_bearing, needed for style
make qreview  SUBJECT=<name>                        # searchable HTML
```

**Order matters in two places.** `qtag` must run before any style cut is taken,
or `qselect` reports untagged rows rather than silently treating them as clean.
Tier 4 must run after tiers 1 to 3, because it re-phrases whatever exists and so
multiplies it, including any duplicates.

**`make onboard` is NOT required.** It builds the map, exemplars, probe set and
oracle, which belong to the content-model arms. This recipe needs the chunk
store and nothing else. An earlier `genq.py` went through `agent.load` and so
demanded `exemplars.json`, which crashes a creator that has only been indexed.

---

## 4. Judgement: unavoidable, forbidden, not needed

| | |
|---|---|
| **unavoidable** | proposing the questions people would actually ask; deciding "minimal fillers"; deciding whether an answer stands alone |
| **FORBIDDEN** | deciding whether a question is *retrievable*. That is BM25's job and its verdict is recorded, never negotiated |
| **not needed** | chunking, indexing, provenance, the A/B split, the tripwires, rendering |

**Nothing in `tools/` decides with a regex.** Four judgements were being made by
pattern match and all four were wrong in ways nothing downstream could see:

| was a regex | measured error | now |
|---|---|---|
| is the answer fact-bearing | **31% wrong both ways.** ~2,000 answers with checkable facts sat inside the style corpus, which is where an adapter learns to invent statistics | `qtag.py` |
| is this question third-person | deleted **78** questions, and not a random 78: *"Why does my boyfriend pull away when he's stressed?"* is almost verbatim a profile YES example. It removed relationship questions as a class | folded into `JUDGE`, no extra call |
| are two questions duplicates | token overlap misses re-wordings, and tier 4 **generates** re-wordings on purpose | blocking, then a model settles each pair |
| why did a passage yield nothing | cannot tell an ad read from a tangent from a real retrieval failure, which is the only distinction that matters | a model call per silent passage, and there are only ever a handful |

Regexes remain for tokenising and parsing. Those are mechanical, not judgements.

**Repair the procedure, never the measurement.** A worker that argues a question
into bucket A has fixed nothing and destroyed the measurement.

### Retrieval partitions, it never filters

| bucket | meaning | for |
|---|---|---|
| **A** | the retriever found the answering passage | deployable now |
| **B** | it did not | the retrieval improvement set |

Both improve the model. B is the only honest large-scale measurement of where
retrieval breaks, because it is thousands of cases where we independently know
the answer exists.

**Bucket A is a lower bound, not the system's reach.** `retrieval_found_source`
is **single-query BM25 on the raw question**. The deployed model
(`agent.py`) writes several queries over several turns: measured at median 42
chunks and the source found 8 times in 10, against 15 chunks and ~5 in 10 for
single-query. So B overstates what the deployed system cannot reach.

**Do not choose a retriever before building this.** Generating against one bakes
its blind spots into the dataset.

### What the judge decides, and that it is subjective on purpose

One question: **can this be answered from these words with only minimal
fillers?** "Minimal" is a judgement call, deliberately left to the model. There
is no threshold and no ratio.

A hand-rolled proxy for exactly this was already caught being wrong: a character
n-gram distance rated a paraphrase *closer* to a passage (0.360) than the
creator's own other passages (0.723), because it was measuring topic and calling
it style. See `2204.04907`.

Scope is what keeps a subjective judge honest, not scoring:

- Asked **constructibility**, never quality.
- **Independent**: a separate call, told nothing about where the question came from.
- **No candidate to compare against**, so no self-preference.
- Told **NO is a normal answer**, because a weak yes becomes training data.

**Enforced, not judged:** every inserted word is bracketed and may only be
connective. No number, study, named entity, claim or position may ever appear in
brackets. A style adapter must learn connective tissue, never to invent
statistics.

**His own citations are wanted.** 27% of answers carry a study, year or number,
and stripping that would make the corpus less like him. The rule is about the
*question* and about *fillers*, never his verbatim. A citation arriving with its
substance is fine; a bare back-reference to one is not.

**The answer must stand alone.** Added after a judged sample found 8 of 40
answers referring to something never introduced. Cause: he defines a term in the
first ten minutes and applies it for the next fifty, so a late passage carries
the back-reference without its referent. 10 of 15 answers using one coined term
never retrieved a passage defining it. After the rule: **2 of 50**.

### How the questions must sound

Addressed **to him**, by someone who wants help. The profile's `ASK_YES` and
`ASK_NO` carry the worked examples, and they are creator-specific: another
creator's examples steer generation toward another creator's subject matter, and
the output still reads fine, so nothing downstream catches it.

- Written as the asker: **I, my, me**. Never *he / him / his*.
- Never refers to a video, transcript or study.
- **Ordinary words, not his vocabulary.** Bridging from an outsider's phrasing to
  his is the whole job; a question in his terms tests string matching.

An early batch produced **18 of 45** third-person questions and 7 that only made
sense to someone who had watched the video, while scoring 100% verbatim. Fluency
was never the failing axis.

---

### Contract: where intelligence was checked and kept

Audited over 13,301 calls, so the next person does not repeat it. **The budget
is already tight; there is no meaningful contraction available.**

| what happened to a candidate | share | scriptable? |
|---|---|---|
| judge said NO on merit | **50%** | no. this is the judgement |
| "collapsed to one passage" | **45%** | no. needs the judge's own output to detect which passages it used |
| caught by regex before any call | 2% | **already free.** `BAD3`, `BADREF` |
| call failure | 1% | fixed, see §8 |

**This audit asked only half the question, and the missing half was the
productive one.** "Where can intelligence be removed" found nothing. "Where is a
script making a judgement badly" found four places, all above, all invisible in
the output because nothing they produced was malformed. Ask both.

Two contractions considered and rejected on measurement:

- **Dedup questions before judging.** Would save the judge calls spent on
  near-identical questions from the same passage. Measured: **48 of 10,046**,
  under 0.5%. Not worth the code.
- **Split the judge into a cheap yes/no then an expensive assemble.** Half of
  all calls end in NO, so the assembly instruction is wasted on them. Rejected:
  `max_tokens` is a ceiling, not spend, so the model already generates only what
  it needs; splitting doubles the calls on accepted items and adds a round trip
  to every one.

**What stays intelligent, and why a script cannot do it:** what a person would
actually ask, what counts as a minimal filler, whether an answer stands on its
own, whether a claim is checkable, whether two questions are the same one, and
why a passage produced nothing. Each is a reading judgement with no defensible threshold, and the one
time a threshold was tried for the third it measured topic instead of style.

---

### No invented constants in a decision path

A threshold nobody validated is a judgement in disguise, and it transfers to a
new creator as a wrong answer rather than an obviously missing one. The ones
that were here are gone: an answer-similarity cutoff now records the overlap
continuously and leaves the call to `qselect --dedup`; answers and passages are
no longer truncated before a model reads them, so nothing is judged on a
fragment; the profile's coined terms are all passed, not the first three.

What remains numeric is either a **budget** (how many to propose, concurrency)
or a **blocking parameter** that only narrows what a model then settles. Neither
decides anything.

---

## 5. Instance constants

**Every number that varies by creator lives in `profiles/<name>.py`**, each
declaration naming the file or measurement it was read from. Nothing in `tools/`
carries a creator-specific default.

Marked **RE-DERIVE** in the profile, meaning measure them per creator and treat
a gap as a signal to look, not a failure:

| | healthygamer |
|---|---|
| accept rate, tier 1 | 85% |
| accepted per passage | median 19, range 3-25 |
| bucket A, tier 1 / tier 2 | 54% / 17% |
| fact-bearing | 27% |
| filler ratio median | 0.006 |
| **negative control, far video** | **2%** |
| negative control, same video | 32% |
| standalone broken | 2% of 50 |
| wall clock, tier 1 | ~3 h, 292 passages, 20-way |

**The number that validates every other one is the negative control.** 85%
accepted on the true passage against 2% on another video's passage is what says
the judge discriminates rather than rubber-stamps. Re-run it per creator: a
judge can go soft on a corpus it has not seen, and no other check would show it.

Read the middle row carefully rather than as a fault. What passes against a far
passage is broad questions he genuinely answers in several videos. The control
assumes one right source per question, and for broad questions that is false.

---

## 6. Done criterion

**Re-read this, do not carry it.** On a long run a worker drifts off what it was
asked, which is exactly how the coverage instruction in §1 got written.

A creator is done when all of these hold:

- the corpus has the target count
- **a person has read a calibration sample**
- the negative-control accept rate is measured on this creator
- both buckets are populated and the A/B split is recorded
- `qdiagnose` reports no unexplained zero-yield passages

**Status is computed from disk**, never written in prose. `paths.py` and
`qselect.py --stats` say what exists.

### Calibrate, and separately check at scale

```bash
python3 tools/genq.py --subject <name> --sample 10 --n 20
```

Ten passages, ~200 questions, ~15 minutes, then a person reads a sample. Only
then launch. **Calibration is necessary and not sufficient**: a class of defect
exists only at volume. 72 exact and 304 near-duplicate questions appeared at
5,000 and were **zero at n=149**. Plan a scale check before any stage that
multiplies output, which means before tier 4.

---

## 7. What did not generalise

**Almost nothing has been tested on a second creator.** This is the largest gap
in this document and it is not a small one.

| | |
|---|---|
| ran on huberman | 3 passages, then stopped |
| what it showed | no creator-specific hardcoding at the time; passages 2 and 3 yielded 24 of 25, matching Dr K |
| what it exposed | `--sample` always picked passage `#0000`, which is always a video opening. It returned 6 of 25 and read like a bad creator when it was a bad sampling rule. Fixed by offsetting half a step |
| never tested | the accept rate, the negative-control baseline, the tier yields, the coined-term problem, the promo regex |

**Do not carry §5's numbers to a new creator as expectations.** A creator who
lectures densely will differ from one who rambles. The negative-control drop is
what distinguishes "different material" from "broken judge".

---

## 8. Silent failures

Things that break with no error. Each was found the hard way.

**Empty model responses cached as successes.** `llm.py`'s API backend had no
emptiness check where the CLI backend always raised; the API path is the one in
use. An empty response never triggered the retry and was written to cache,
making it permanent. Root cause: `stop_reason=max_tokens`, the model spending
its whole budget reasoning and emitting no text. Every tight ceiling produced
them (300, 800, 1500, 2000); the 16000 default never has. **112 in one session**:
37 of 120 tier-3 pairs reported "proposed nothing", ~56 judge calls became
"unparsed". Fixed, and `BUDGET` in the profile now sizes ceilings for reasoning
plus output. Check per creator:

```bash
python3 -c "import json,pathlib,collections; print(collections.Counter(
  json.loads(f.read_text()).get('tag','?')
  for f in pathlib.Path('../../cache').rglob('*.json') if f.name!='_model.txt'
  and json.loads(f.read_text()).get('response','').strip()==''))"
```

**A zero is not evidence of an honest zero.** A stage reporting a
plausible-looking failure, "proposed nothing" or "unparsed", is the shape
silent loss takes. Both times it happened here the reported reason was an
artifact and the real cause was a call that never succeeded.

**`--who` silently invalidates the entire cache.** It sits in every propose
prompt, so editing it changes every key and re-costs the run. Worse, results
from before and after are not comparable: the same passage group returned 3
accepted under one `--who` and 6 under another, which looked like
non-determinism until the prompts were diffed. It now lives in the profile. Set
it once and treat a change as starting over.

**A missing profile must never fall back.** `_profile.py` exits with the copy
command rather than borrowing another creator's values, because borrowed example
questions steer generation toward the wrong subject matter and the output still
reads fine.

**Tier tagged from what search returned, not what the judge used.** Search
returns k spans whatever you ask, so its spread is retrieval noise. 8 of the
first 9 items were labelled cross-video while every sentence came from one.

**Dedup by containment destroys composition.** An answer built from two videos
contains the single-video answer, so it scores ~1.0 and dies. Every cross-video
candidate on the first pair was destroyed this way. Keep an item that draws on a
passage the match does not.

**Near-duplicate answers discarded as waste.** They are the density of the
question distribution. An earlier version deleted 36% of accepted items for
exactly that. Keep and count them: `answer_variants_on_this_passage`.

**`position` on a sampled run is meaningless.** The spread is the sampling
rule's, not the generator's. It reported coverage-seeking on a 7-passage run
where the passages had been picked evenly on purpose.

---

## 8a. Checks, not notes

Every failure in §8 that could be perceived rather than described now raises.
A paragraph is read once; an assertion fails the run. Three tests were applied
before adding each one, because a rule that fires on healthy data is worse than
no rule: is it real (reproduce it), does it fail loudly (raise, do not print),
and what count justified it.

| check | raises when | count that justified it |
|---|---|---|
| `preflight.py`, 11 checks | any input in §0 is absent or the wrong shape, or a live model call fails | a run reached minute 90 before a missing input surfaced |
| profile not copied | `ASK_YES` is identical to another creator's | the `load()` guard catches absence but not laziness, and editing `WHO` alone leaves the steering wrong |
| `_profile.py` fallback | the profile is missing | borrowed examples produce a corpus about the wrong subject matter, and every output reads fine |
| prompt_sha mismatch | appending to a corpus built under different prompts | rows look identical and mean different things |
| index_version mismatch | appending against a re-chunked store | the A/B verdicts in the old rows describe a different corpus |
| zero accepted | a run attempts passages and accepts nothing | this looked exactly like "a creator with nothing to say" while 112 empty responses were being cached as successes |
| untagged answers | a style cut is taken before `qtag` | an untagged answer is not the same as a clean one |
| empty model response | `stop_reason=max_tokens` with no text | 112 in one session, cached permanently because nothing raised |

---

## 9. Dead ends

| tried | outcome |
|---|---|
| retrieval as an accept gate | deleted 58 questions his subtitles plainly answered |
| "spread across different material" | a coverage instruction. Mean position 0.48, dead uniform |
| a cross-video pass for volume | 85 questions from 120 pairs, 400 collapsed. Scarce because he rarely holds two complementary halves in two videos, not because of tooling |
| a hand-rolled style distance | rated a paraphrase closer to a passage than the creator's own other passages. It was measuring topic |
| one call per video | cannot scale past a few hundred and has no reason to ask three things about one paragraph |

---

## 10. Open, and it is a person's call

**The corpus covers each video uniformly, contradicting §1.** Mean cited position
0.50 over 5,134 tier-1 questions. Per-passage proposing guarantees it:
mode-seeking operates *within* a passage, coverage happens *across* them.
Prompting cannot fix it. Weighting toward most-asked material needs a signal for
what gets asked, and none exists. Tiers 2-4 multiply whatever tier 1 produced.
`problems.md` K9.

**Nobody who knows the creator has read the output.** `2306.00539` found across
89 papers that few automated style metrics have ever been checked against human
judgement. Every number here inherits that gap.

**The corpus contains other people's voices.** Viewer letters, quoted tweets, a
comedy clip he reacts to. The judge catches these unprompted and rejects them,
but nothing structural prevents them. `problems.md` K7.
