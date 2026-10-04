# Working rules for this project

Read `MANIFESTO.md` first. It carries direction, standing decisions and priorities.
Read `METHOD.md` second. It owns **how a learning system gets improved** — the three
states, what must exist before exploring, and what an issue and an experiment are. It
applies to `Mycode/MetaHuman` too.
This file is only about how we work day to day.

## Maintain problems.md

`problems.md` is the register of everything unresolved, grouped by module and ordered by
the priority buckets in the manifesto (deep modules first).

Every entry is exactly one of four kinds, and the kind determines what happens next:

| kind | means | next step |
|---|---|---|
| **defect** | something is wrong and we know it | fix it, quick |
| **decision** | a person has to settle it, no data will | ask, then record it |
| **measurement** | unknown, and one look at existing or new data answers it | measure |
| **exploration** | a measured issue that no single decision fixes | an engineered loop over better modelling choices |

**Only explorations get looped.** Defects are quick. Decisions are a sentence. Measurements
are a look at data. Most things that feel like explorations are one of the other three.
Sort honestly.

Add to the register whenever something unresolved surfaces, whoever notices it. Do not let
it live only in a conversation.

## Ground the data before proposing a loop

**A loop is the last resort, not the first move.**

Before proposing any exploration, check what is already known: old logs, past run outputs,
existing measurements, the notes in the module's README. Say what you found.

Worked example. "Is model capacity enough?" is not an exploration yet. First look at the
training logs for visible underfitting: is the training loss still falling at the end? Is
train and held-out tracking each other? That may answer it outright, and if it does not it
still narrows what a loop should search.

If existing data answers the question, the entry closes as a measurement and no loop runs.

## The cycle

```
build baseline  ->  measure together  ->  identify a real issue  ->  engineer a loop to fix it
```

A loop is a **response to a measured issue**, never a general exploration. No issue behind
it, no loop. This is why the register sorts entries into defect, measurement and
exploration: an exploration is an issue we have measured and cannot fix by a decision or a
patch, so it needs a search over better modelling choices.

Baselines are set by a person. Measurement is done together. The loop is engineered
together once the issue is named.

## Do not set up a loop unilaterally

When an issue is identified as needing a loop, ask before building anything. The
specifics needed are always the same:

1. **The commitment** — what is fixed, and why. This bounds the search.
2. **The search space** — what is open given that commitment.
3. **What would count as an answer** — the metric, or the frontier if the aim is
   two-objective.
4. **The held-out set** the loop never sees.
5. **The budget** — how many runs, on what hardware, against what else is queued.

Then propose the loop and get it approved before it runs.

## Loops report, they do not decide

A loop returns ranked evidence. A person promotes or makes a new commitment. For the deep
modules the aims are two-objective by definition, so a loop reports a frontier and must not
collapse it to one number.

## Per-module tracking

Problems are grouped by module. Effort follows the manifesto's priority split: deep modules
get investigation, good-enough modules get accepted or routed around unless results are
visibly bad.

## The work split

Label every incoming task with its category before starting. It is free, and it is what
makes the practice real rather than aspirational.

| category | owner | examples |
|---|---|---|
| **commitment** | a person | direction, what is fixed, what counts as good |
| **one-shot** | built directly | CLIs, glue, tests, migrations, the deployable stack |
| **loop** | a bounded search inside a commitment | hyperparameters, loss form, architecture within a fixed role |
| **explore** | an agent that reports and never decides | gap-finding, paper review |

### Agent output lands in the register, never in the tree

An exploring agent adds entries to `problems.md`. It does not change code, docs or configs.
A person reads the register and decides what is real. This is what keeps open-ended agents
from producing volume instead of value.

### Budget is part of a loop's definition

Not discovered afterwards. Tokens for standing agents, GPU for sweeps. Contention is a
live problem: an offset training run held 56 to 70% of SM while another got 30 to 45%.

## Sequencing

What is ready depends on prerequisites, not on enthusiasm.

| when | what | why now |
|---|---|---|
| **now** | gap-finding agent against `MANIFESTO.md` and `problems.md` | no metric needed, no infrastructure, compounds |
| **now** | the deployable stack, starting with what the audio checkpoint needs | gates almost everything else |
| **first loop** | hyperparameter search on `manner`, via `ims` | the only asset that is loop-ready today |
| **after** | training loops on the face modules | each needs a one-command train and eval first |
| **last** | architecture search | needs a metric trusted enough to optimise against |

### Why manner is the pilot

A loop needs four things per asset: a train command, an eval, a held-out set, and a budget.
`ims` has all four already: `ims train`, `ims eval`, a split by `paragraph_id` so variants
of one paragraph stay together, and a composite score with fixed weights. No other asset
has any of them yet.

**You cannot sweep what you cannot run with one command.** That is why the deployable stack
comes before the infinite training system, not after it.

## One owner per fact

Do not restate a fact in a second file. Reference it.

| owns | |
|---|---|
| `MANIFESTO.md` | direction, standing decisions, priorities |
| component `README` / `CONTRACT` | everything specific to that component |
| component `REFERENCES` | external work that component could steal from |
| `problems.md` | what is unresolved, and its status |
| `docs/*.html` | **nothing** |

`docs/` is **derived, not authored**. When something changes, edit the owner and rebuild the
narrative — never both. If the two disagree, the owner wins.

The temptation is worst when new evidence arrives mid-conversation: it gets written at
whatever level is currently open, and lands in two or three places. New evidence goes to the
component by default, and only reaches the manifesto if it changes a principle.

## Every component carries a REFERENCES

Each engine and each block has a `REFERENCES.md` beside its `README`. Papers, repos and
prior art that component could steal an idea from. Engine level for anything spanning its
blocks, block level for one block's concern. There is a lot of good work out there per
component and none of it should have to be rediscovered in a conversation.

**Three rules, or it becomes a link dump.**

1. **Every entry carries a verdict against one of our standing decisions**, named. Not a
   summary of the work, a position on it: adopt, conflicts with K3, reference
   implementation only, revisit when X. An entry with no verdict is not an entry.
2. **Verified, not recalled.** Licenses, benchmark numbers, what a repo is built on, whether
   code actually exists. Read the repo or the paper page and date the check. A wrong license
   in a reference file is worse than no reference file.
3. **Group by our concerns, not by the works' own taxonomy.** The grouping is where the
   thinking is. `knowledge/REFERENCES.md` groups by whether a system preserves verbatim text
   or extracts, because that is the axis K3 turns on, not by "graph RAG / vector RAG".

Also record what a component **rejected** and why. The next person to find that repo needs
to know it was already looked at.

**Where the survey rule bends.** An exploring agent normally writes only to `problems.md`
and never to the tree. A reference entry is an observation about the outside world, not a
change to ours, so it may be written directly. Anything it *implies* about our system still
goes to the register. If a surveyed paper suggests we are doing something wrong, that is a
`problems.md` entry, and the reference file only says the paper exists and what it claims.
