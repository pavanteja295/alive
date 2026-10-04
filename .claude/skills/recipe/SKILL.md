---
name: recipe
description: Turn finished work into a recipe - the scripts that ran, in order, plus the judgement that turned out to be necessary - so Claude Code can re-run it on new inputs. Invoke ONLY when the user says the work is good and asks for it to be made repeatable; never while the work is still going, and never on your own suggestion. Also use when applying an existing recipe to a new input, or when one breaks.
---

# Recipes

A recipe is the workflow you already built - real scripts doing real work -
written down so that **Claude Code runs them, rather than a runner running
them.**

The aim is good ordinary software. It is not a high-intelligence system. You
have only ever seen a few inputs, so some of what you wrote will be wrong on the
next one. When that happens you want something inspecting the failure, fixing it
if it is a real bug and offering an alternative if it is not, instead of a
script exiting non-zero or, worse, finishing and returning nonsense.

That is the only reason judgement is in here. Not capability. **Coverage you do
not have.**

---

## WHEN THIS RUNS

**Once, at the end, on the user's word.** *Works on these two* - *let's make
this repeatable*. Not before.

Everything up to that point is ordinary Claude Code: logic written, throwaway
code, a look at the output by you or by the user. No structure, no recipe
vocabulary, nothing recorded on purpose. **The session is the record**, so there
is nothing to set up in advance and no reason to load this early.

**Never suggest it.** Do not hint the work looks done, do not offer to write it
up. The user decides when the thing is good enough to keep.

---

## THE FOUR PASSES

Do not design a recipe. Extract it from what happened.

| pass | what happens | what comes out |
|---|---|---|
| **Explore** | **not this skill.** Ordinary Claude Code, user in the loop, nothing written down | a working result |
| **Extract** | on the user's signal, read back over what was actually done. Three steps, below | the recipe |
| **Validate** | re-run from the document alone on a genuinely different input | what did not generalise |
| **Contract** | demote every hole that turned out never to need judgement | fewer holes, more scripts |

**Validate is the pass that gets skipped and the one that pays**, because the
errors it catches are the ones that look like success. Its weakness is
survivorship: you see the interventions that happened, never the ones that
should have. The only fix is a second input.

### Extract, in three steps

1. **Find what is hardcoded, and parameterise what can be.** Do this first.
   Most of what looks like it needs judgement is a value that was never made an
   argument, and skipping this turns missing parameters into fake holes.

2. **Ask what transfers.** Of the scripts written, which hold for a different
   input and which were throwaway - the user knows. Everything that transfers
   gets committed properly: named, wired in order, packaged as software rather
   than left as the pile it was during the run.

3. **Only then name the intelligence that is inescapable.** What is left over,
   that would earn its keep on inputs nobody has seen.

In step 3, **suggest - do not wait to be asked.** Name the places a script would
be brittle on a different input, or where a person would end up looking anyway,
and say plainly that intelligence could unburden that spot. Volunteering
candidates is the job; settling them is not.

Then put the assumptions you both made to the user, one at a time: *we assumed
this - does it hold for other inputs for sure, or does it stay intelligent and
get worked out on the fly?* Sure becomes a script. On the fly becomes a hole.
**Do not decide either way alone.**

**The user's own inspections count.** Anything they looked at and judged during
the run, that you could replicate, is a hole. Those never appear as a tool call,
so they are invisible unless you go looking for them.

**Script what is scriptable, and only that.** If the rule can be stated without
listing the inputs you have seen, write the script. If it cannot, the model
stays. Step 3 comes out short because most steps really are scriptable, not
because holes are scored against you. See `WHAT MINIMAL INTELLIGENCE MEANS`.

---

## WHAT MINIMAL INTELLIGENCE MEANS

It was read as *remove intelligence wherever you find it*, and the result was
model calls inside the workflow - inspection steps, generation steps - rewritten
as functions, then tuned until they passed on the two inputs at hand. That is
not a contraction. It is **overfitting to the coverage you do not have**, which
is the one thing this skill exists to prevent.

The insight was narrower. Three-way, in order:

| the step is | do |
|---|---|
| scriptable, simple, deterministic - you can state the rule | **write the script.** No model. Not even a cheap one |
| not scriptable, but a person settles it once for all inputs | **a decision.** Ask, record it as a profile constant, move on |
| genuinely per-item judgement | **use the model.** This is the job, not a failure to try hard enough |

**A model doing real work is not a hole to be closed.** If the process needs a
model, use one, and size it to the work rather than to a sense that models are a
concession. The concession is unexamined judgement, not judgement.

**The tell.** A replacement built from thresholds, keyword lists or regexes that
were arrived at by adjusting them until the current inputs passed is a model
with worse generalisation and no way to say it is unsure. Swapping a model call
for one of those is a regression that reads as progress, because the run gets
faster and the output still looks right.

**Prefer a relation to a threshold.** That is the constructive half of the tell
above, and the thing to reach for instead. A threshold - *below thirty percent is
bad* - is a number fitted to the inputs you happened to have. A relation is an
ordering or an identity that is true by what the values MEAN: *a model driving
these controls cannot express more than the controls can*; *a stage cannot emit
more items than it was given*; *the fitted thing cannot beat the thing that saw
the answer*. It needs no tuning, it holds on inputs nobody has seen, and when it
breaks something is genuinely wrong rather than merely unusual.

Relations are also where a model earns its place. A checklist covers what was
imagined. A relation covers what is true, and a break leaves a real question -
*why can this beat the thing that could see the answer?* - which usually has two
or three possible causes. **Naming them is the work**, and it is the work a
script cannot do.

**Contract demotes on evidence.** A hole comes out because Validate showed it
never needed judgement across genuinely different inputs. Never because a
plausible-looking function could be written for it.

---

## A RECIPE IS A FUNCTION

A recipe is a function - a workflow is the same thing with more steps. It does
not act on the world in general. It acts on inputs, so **it has a signature, and
the signature is written before the steps.**

**Declare every input: what it is, its shape, and the file it comes from.** Read
those field names, do not assume them. This is the same failure as
`Ask, do not infer, at the top` under SETTLED, one level up: there it was fields
inside an input, here it is which inputs exist at all.

**Preconditions are checked at the top, all of them, before any work starts.** A
recipe that discovers a missing input at minute 90 has burned the run and has
usually half-written its outputs by then. The check raises, per
`A CAUGHT BUG BECOMES A CHECK`.

**A missing input stops the recipe.** It is not inferred, defaulted, or worked
around. That is a decision for the user, and it is one sentence to ask.

**Search properly before declaring it missing, then fail rather than build.** A
precondition that trips on a naive path guess is a bug in the check, not a
finding - look where the artifact actually lives, under the names it actually
takes, before reporting its absence. But once it is genuinely absent the recipe
**fails, and names the recipe that owns it.** It does not run the upstream tool,
even when that tool is one line away and the fix is obvious from here. A recipe
that builds its own inputs has no preconditions - it has a longer body, and the
first time an upstream artifact is subtly wrong it is rebuilt subtly wrong on
every run, with nothing left to compare against.

**A recipe does what its signature declares, and nothing beyond it.** Not the
stage before, not the stage after, not the useful thing noticed along the way.
Scope creeps at the boundary, because the neighbouring work is always cheap from
here. The cost is not the minutes: it is that two recipes now produce the same
artifact and neither owns it, so a change to one is silently undone by the
other. If the work is worth doing it belongs to some recipe - name it, or write
it. Extending this one is the option that looks free and is not.

**Recipes compose.** When an input is another recipe's output, name that recipe
and the artifact. Do not re-describe how the artifact is produced - the other
recipe owns that. Declare your own outputs the same way: what is left on disk
that something downstream may consume. That is the postcondition of the whole
recipe, distinct from the per-tool postconditions in the inventory.

**The profile is the call site.** One per input, holding the arguments. The
document is the body, and running a new input never edits it.

---

## THE DOCUMENT

Write it as a handover to a competent new worker, not as config.

**The recipe owns the process.** Say so in the file: if a prompt, a script or a
conversation disagrees with it, the recipe is right and the other has drifted.
Without that line it decays into a description of whatever the scripts do today.

Open with what it takes in, what it produces, **what it was validated on**, and
a pointer to `What did not generalise` before any constant is trusted.

1. **Why this exists** - what breaks without it, with the measurement. This is
   what lets a worker handle a case you did not anticipate, by reasoning from
   intent instead of pattern-matching the instruction.
2. **Signature** - the inputs it requires, each with its shape and the file it
   is read from; the preconditions checked before anything runs; the outputs a
   downstream recipe may consume.
3. **The item** - the smallest independently redoable unit.
4. **Tool inventory with postconditions** - what may be assumed once each tool
   exits clean. This is for **context economy**: read the twelve numbers, not
   the corpus.
5. **Judgement: unavoidable, forbidden, not needed** - three categories, below.
6. **Instance constants** - every number with the distribution or file it was
   read from, marked re-derive-per-input.
7. **Done criterion** - **re-read, not carried.** On a long run a worker drifts
   off what it was asked.
8. **What did not generalise** and **Dead ends** - so the next input does not
   rediscover them.
9. **Silent failures** - things that break with no error.

**No control flow.** The worker supplies it. A document that pins the order is
wrong the first time an input needs a different one.

**No section numbers in a pointer.** Name the file and the heading. Numbers
rot faster than filenames: `RECIPE_PAIRS.md §5` and `§11b` were both dead
inside one restructure, and a worker following a dead pointer reinvents what it
pointed at.

**No status in prose.** Status is computed from disk. Written status goes stale,
in one case within twelve days, claiming nothing had ever been run. A person
skims a stale status line; a worker reads it, believes it, and stops.

### One folder per recipe

A recipe is a unit. It gets a directory, not a file loose among others.

```
recipes/<name>/
  RECIPE.md          the document. owns the process
  tools/             the scripts this recipe calls
  profiles/          one file per input
  work/<instance>/   ledger, outputs, cache
```

Shared machinery that several recipes call stays outside and is named by path in
the tool inventory. Duplicating it per folder is worse than leaving it flat.

Everything that varies goes in the profile, each declaration carrying a comment
naming the file it was read from. To run a new input: copy a profile, change the
declarations, run. **The procedure is not edited.**

---

## THREE DECISION CATEGORIES

Not two. The middle one is the one that gets missed.

| | |
|---|---|
| **unavoidable** | proposing what a person would actually ask; deciding "minimal fillers"; whether an answer stands alone; whether an input is out of scope |
| **FORBIDDEN** | anything the recipe records in order to measure something. Whether a question is retrievable is BM25's job, and its verdict is recorded, never negotiated |
| **not needed** | chunking, indexing, provenance, the A/B split, tripwires, rendering |

**Repair the procedure, never the measurement.** A worker that argues an item
past a recorded verdict has fixed nothing and destroyed the measurement.

**Never let the measured thing influence what gets produced.** The content model
does not run during question generation at all - selecting for questions the
system answers well makes the corpus a measurement of itself. Order the stages
so this is structural, not a matter of discipline.

**Designing a judge is owned by the component, not by this skill.** How to keep
a subjective judge honest and how to catch one going soft are domain method.
They live beside the recipes that run them, and the recipe names the path. This
skill does not carry them and must not point at them by path: a general skill
that names one project's file is stale the next time that project moves.

---

## A CAUGHT BUG BECOMES A CHECK, NOT A NOTE

Of everything that went wrong building the pairs pipeline, **exactly one thing
raised** - a pip build failure. The rest finished cleanly and returned plausible
output that was wrong: 8 exemplars loaded instead of the deployed 18, producing
good-looking pairs; the model ignoring an injected passage, producing a fluent
answer about other material; 112 empty API responses cached permanently, because
an empty never raised so the retry never fired.

A paragraph saying "watch out for X" is read once. An assertion fails the run.
**Assertions are the sense organs, not the safety rail** - they are how a worker
perceives that the input changed.

Three tests before adding one, because a rule that fires on healthy data is
worse than no rule:

1. **Is it real?** Reproduce it. The first deployed-config check failed on a
   correct profile because Python joins adjacent string literals. That was a bug
   in the check, not a finding.
2. **Does it fail loudly?** Raise, do not print. A warning in a 100-minute run
   gets skimmed.
3. **What count justified it?** Record the number, or the next input cannot tell
   whether to keep it.

**And later, when a check that used to pass starts failing: the check may be what
is wrong.** The three tests above are applied when a check is written, and then
never again, which is how a mis-stated check survives to be believed. A failure
is a claim that two things disagree; it does not say which.

**A check may only be narrowed for a reason that does not mention the
observation.** This is the whole guard, and without it diagnosis becomes
rationalisation: the check is weakened until it passes, the run goes green, and
the sense organ is gone. A legitimate narrowing argues from what the thing
measures or what the stage actually does, and would have been the right statement
*before any run existed*. An illegitimate one argues from the number that broke
it.

Two from one afternoon, both legitimate. A check said the fitted-per-item solver
bounds the model on a particular tail statistic; it did not, and the reason is
that the solver minimises error over the whole object and so bounds whole-object
quantities, saying nothing about a tail it never optimised. Another asserted an
amplitude bound that follows from squared error; the model had an extra shaping
term, so squared error was not what was being minimised and the bound did not
apply. Neither argument mentions the value that failed.

---

## A RULE IS PERCEIVED, NOT REMEMBERED

A long run drifts. Anything stated once, at the top, in prose, is gone by the
time it matters - not because the worker is careless but because it is four
hundred steps and several context windows away from having read it.

This skill already concedes the point twice, in specific cases: the done
criterion is **re-read, not carried**, and status is **computed from disk**
rather than written down. Both are the same rule.

**Any constraint that must hold at minute 400 has to be re-delivered by
machinery the worker already touches.** Not restated in a document it is
expected to revisit - *re-read the instructions* is the forgetting problem one
level up.

Three carriers, in order of how much they survive:

| carrier | why it holds |
|---|---|
| **the failure message** | a check that raises restates the rule at the moment of the mistake. You cannot forget something that stops you and tells you why. Costs nothing to maintain, because the check had to exist anyway |
| **whatever the worker polls** | the status tool is read every time the worker asks what is next. Anything printed there is re-read on the worker's own rhythm, with nobody deciding to re-read it |
| **the harness** | a hook runs outside the model, so it holds when the model is simply wrong. The only carrier that survives drift completely, and the only one worth the configuration it costs |

So a constraint gets written into a check, into the status output, or into a
hook. **Prose is where it is explained, never where it is enforced.**

**The limit, which is real.** These bound what gets *written* and what counts as
*done*. Nothing bounds work that is merely pointless - reading is allowed, so a
worker can spend an hour exploring inside its own scope and violate nothing.
That one is held only by the done criterion being genuinely re-read, which is
why the done criterion belongs in the thing the worker polls rather than in the
document alone.

---

## WHERE THIS IS GOING

Recipes are written in-house now, run by hand, one input at a time. The
direction is that an agent eventually runs them on its own: where it works,
nothing happens; where it breaks, it diagnoses, proposes a fix to a script or a
new instruction, and records what the input taught. What ships then is an agent
and a recipe, not code.

The intelligence share falls as that repeats, and when little is left the task
gets packaged as ordinary deterministic software and the recipe retires.

Nothing about how that gets deployed is settled. Decide it when there is
something to deploy.

---

## SETTLED. Do not re-derive.

- **Pin content by construction, not by retrieval.** Hand the model the passage
  the verbatim came from. Real retrieval returns a median of 42 chunks and finds
  the source passage 8 times in 10 - not content-synchronised, and the adapter
  learns invention. General form: run a stage with its upstream replaced by
  ground truth, so there is something real to measure against.
- **Ask, do not infer, at the top.** Every one of the four inputs in
  `recipes/style-pairs` was wrong at least once when assumed. Field names are
  not standardised. Read the file.
- **The inner loop is always the same:** run a small batch, look at it, change
  one thing, re-run. Reading and looking steps produce no output but a confirmed
  plan, and none is optional - each caught something the numbers did not.
- **Calibration is necessary and not sufficient.** A class of defect exists only
  at volume: 72 exact duplicates and 304 near-duplicates appeared at 5,000 and
  were **zero at n=149**. Plan a scale check too, before any stage that
  multiplies output.
- **Check the accepted items, not just the rejected ones.** Reviewing what you
  threw away finds almost nothing. Contamination sits inside things that looked
  fine, because they looked fine next to real garbage.
- **A zero is not evidence of an honest zero.** A stage reporting a
  plausible-looking failure - "proposed nothing", "unparsed" - is the shape a
  silent-loss bug takes. Both times it happened, the reported reason was an
  artifact and the real cause was a call that never succeeded.

---

**When the recipe's output is a TRAINED MODEL, load `recipe-learning` alongside
this.** A model has failure modes a processed artifact does not - it can be
measured on data it was chosen with, and it can carry a constant fitted to the
first input while reading as a setting. That skill carries the extra structure:
one split for the whole pipeline, parameters sorted by whether they transfer,
relations instead of thresholds, and selection that cannot fit the number it
reports.

---

Worked examples, which own their own specifics:
`engines/knowledge_style/proto/recipes/question-corpus/RECIPE.md`,
`engines/knowledge_style/proto/recipes/style-pairs/RECIPE.md`, and
`engines/knowledge_style/proto/RECIPE.md` (a converged config - loop discipline,
see `converge-creator`).

`question-corpus` is the one to read for shape: it is the only recipe whose
`0. Signature` is written and whose preconditions are a tool that raises.
