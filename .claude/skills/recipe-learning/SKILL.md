---
name: recipe-learning
description: The extra structure a recipe needs when its output is a TRAINED MODEL rather than a processed artifact - one split for the whole pipeline, parameters sorted by whether they transfer, relations instead of thresholds, and selection that cannot fit the number it reports. Load alongside the `recipe` skill when writing or repairing a recipe that trains something, or when a trained pipeline is being moved to a second input and the numbers come out wrong.
---

# Recipes that train something

The `recipe` skill covers any recipe. This is the part that only applies when the
output is a **model**, because a model has failure modes a processed artifact does
not: it can be measured on data it was chosen with, it can carry a constant fitted
to one input, and it can be wrong in a way that looks like a result.

Everything here was derived from one multi-stage pipeline and every step earned
its place by catching something real. **What changes between learning systems is
only what the inputs are and what the relations say. The shape does not.**

---

## THE SHAPE

Eight steps, in order. The order matters: each one is cheap relative to what it
protects, and every one of them is cheaper than the training run it precedes.

| | step | catches |
|---|---|---|
| 1 | **Preconditions: does the input plug in** | a file that exists and cannot be used |
| 2 | **Verify the joins** | inputs and targets that do not correspond |
| 3 | **One split, once, for the whole pipeline** | a later stage training on what an earlier one was judged by |
| 4 | **Sort the parameters** | a constant fitted to the first input, read as a setting |
| 5 | **Derive, ask, or search** | a guess hidden inside a default |
| 6 | **Relations, not thresholds** | numbers that are wrong in a way no threshold notices |
| 7 | **Select on validation, score once, promote with the faults** | a search that fits the number it reports |
| 8 | **Hand over the decoder, not just the weights** | a model used through a different map than it was trained through |

---

## 1. Preconditions ask whether the input will PLUG IN

Not how much of it there is, and not how good it is. **Does it open, does it
carry the fields the model reads, are they the widths the model was BUILT with,
and do the parts line up with each other.**

That is the one question a precondition can settle for an input nobody has seen,
because it does not depend on the input at all: a field is there or it is not.

Existence is never the failure mode. A truncated file exists. A file from a
differently-configured upstream exists. A pinned width matters most: a check that
binds the width to whatever the first file carried will pass a file that loads,
trains, and is wrong.

**Size and quality are REPORTED, never gated.** There is no amount of data that is
*enough* -- it depends on the input, on its variety, and on what the model is for.
A threshold there is fitted to whoever it was fitted to. Print the distribution
and let the reader say what they think; a suggestion can be wrong out loud, a gate
cannot.

## 2. Verify the joins before spending the training run

Every learning system joins things that were produced separately -- an input to a
target, a signal to a clock, a frame to a label. **None of those joins is
self-evidently right, and training does not notice when one is wrong.** A model
fed misaligned data is still a model: it trains cleanly and scores low, and
nothing raises.

Split the joins by what can settle them:

- **A join that is already a measurement** stays a measurement. Check the
  measurement is real -- a correlation peak that is sharp, well above its own
  baseline, and consistent across independent instances -- not merely that a
  number exists.
- **A join that needs eyes** gets rendered, and the tool **does not pass or fail
  it.** Claiming a verdict on something a tool cannot see is inventing a
  threshold for it.

**A verification warns and waits; only a precondition fails.** Conflating them
teaches a worker to route around the check. And the acknowledgement goes on disk
with what the reader said, because a run resumed tomorrow must be able to tell
whether anyone ever looked -- and it should go stale by itself when the thing it
verified changes.

## 3. One split, written once, for the whole pipeline

**No stage may train on anything any stage is selected or scored on.** In a
single-stage system this is obvious. In a multi-stage one it is the rule that
gets broken, because each stage's argument sounds fine on its own: *those clips
only chose the earlier model's checkpoint, they reported nothing.* Between them
the stages have then seen the whole set.

The symptom is a later stage with nothing to select on but the test set.

**Write it before any stage trains, and assert it rather than intending it.** A
split created when a stage happens to need one is a split fitted to whatever has
already been seen.

When a framework offers no validation slot, do not invent one inside the model --
emit two datasets from the same training material, one held out on validation for
every search and choice, one held out on test and scored once.

## 4. Sort the parameters. The dangerous category is the fourth

| kind | meaning |
|---|---|
| **known** | facts about this input; nothing derives them |
| **where** | machine paths; nothing to do with the input |
| **derived** | a tool measures a distribution and the value is read off it |
| **carried** | properties of the model or method, with the evidence named |
| **fitted** | look carried, are not: chosen on ONE input |

**Carried against fitted is the distinction that decides whether a pipeline
survives its second input.** A carried value names its evidence -- *five
architectures landed within a hundredth of each other* -- and a fitted one is a
guess with a sample size of one. Read as carried, it is how something that worked
quietly stops working.

Organise the profile by these, not by topic. Whoever copies it for a new input
should see what to change, what a tool must derive, what transfers, and what only
looks like it transfers.

**Parameterise before you classify.** A value with no flag, living in a config
default, masquerades as needing judgement -- and nobody ever varies it, because
varying it means editing code.

## 5. Derive what has a rule; ask where it is judgement; search what needs a run

Three outcomes per parameter, and the output should be short: the ones that are
fine are moved past silently, or the two that matter are buried.

- **It makes sense** -> move on.
- **It is wrong and the correction is CONCRETE** -> fix it, say so.
- **It is wrong and the correction is a JUDGEMENT** -> stop and ask.

**The line is whether the rule can be STATED.** *The shut end of this input's own
distribution* is a definition and derives itself. *Where the curve flattens* is
not a rule: on a curve that is not smooth, any cut-off picks a different answer,
and the cut-off is a threshold fitted to the one input it was chosen on.

**Prefer a cost argument to a knee.** When a curve is monotone there is no
accuracy reason to stop anywhere, so "find the knee" was the wrong question --
the real one is what the next increment costs. State the trade and choose;
inviting an override is not the same as blocking.

**An auto-fix must never edit a file a person wrote.** Derived values go in their
own file that the profile reads. That split is what makes taking a fix safe to
audit and simple to revert.

**And default to the best case for whoever is running it.** A recipe that stops
at every fork is a recipe nobody finishes.

## 6. Relations, not thresholds -- and a break may be the check's fault

See the `recipe` skill for the general form. For a learning system the relations
are usually these, and they cost nothing to write:

- the model cannot beat **what its output space can express** (measure the
  ceiling; a score without it is uninterpretable, and two inputs with different
  ceilings are not comparable on raw scores)
- the model must beat **predicting the target's mean**
- the model must beat **whatever it was correcting**
- the model cannot beat a **fitted-per-item solve that saw the answer** -- on the
  quantity that solve minimised, and *not* on any sub-measure it never optimised
- validation and test should **agree in ordering**; when they do not, either
  selection is contaminated or the spread is noise

**Establish the noise floor before believing any ordering.** Repeated seeds of
one configuration bound how large a difference has to be before it is a
difference. Without it, a spread across variants cannot be told from repeated
samples of the same model -- and a search will confidently rank noise.

## 7. Selection is the part that gets got wrong by hand

**Rank on validation. Read test once, for the configuration that already won, and
never to choose.** A search that selects on test does not measure a model, it
fits the held-out set, and the more variants the better the winner looks and the
less it means.

**Two runs are comparable only if they were scored the same way.** Record what
each was measured on -- the scope, the yardstick, the context, the target -- beside
its history. A run that does not record it **cannot be shown** comparable, so
leave it out of the ranking rather than assuming it in. Numbers from different
yardsticks are not one quantity measured twice.

**Promotion assembles the manifest from disk** -- the command, the step chosen on
validation, the held-out numbers, the ceiling and the fraction of it reached --
and **requires what is still wrong with the model.** Every real release has
faults; one with none recorded is one nobody checked. Refuse to promote a
checkpoint that is not the validation winner among comparable runs, unless
somebody says in writing why.

---

## 8. A model hands over its decoder, not just its weights

A model's outputs are almost never the thing anyone wants. They are an index into
something else: control values that become a face through a rig, tokens that become
text through a vocabulary, latents that become an image through a decoder, logits
that become a class through a label order. **The weights only mean something through
that map.**

So the map is part of the artefact. Use the same weights with a different map and
nothing errors, nothing looks broken, and the output is wrong by a margin no
downstream metric will attribute to the cause. This is the worst failure shape a
multi-stage system has: silent, systematic, and present in every frame.

**Write the map inside the weights file.** Not beside it. Anything beside it is one
copy step away from being lost, and the copy step will be written by somebody who
is thinking about checkpoints, not about maps. Record what identifies the map -- its
path and a content hash, so it is identified by what it is and not by where it sat.

**The consumer reads the map off the artefact and never takes it as an argument.** An
argument is a thing that can be forgotten, and a forgotten map has a default, and the
default is wrong half the time with no sign. An artefact that cannot say which map it
used is **refused**, not assumed.

**Then check it at the seam, with a relation.** The producing recipe already publishes
numbers on its own scale; the consuming recipe reproduces one of them before trusting
the handover. Drive the model's own solved or oracle inputs through the map in use and
ask which published reference point the result lands on. That needs no threshold and
no new metric, and it catches every kind of map drift rather than the one that was
imagined.

**The claim and the evidence must come from different objects.** Take the claim from
the recorded map and the evidence from the measurement. A check that asks the
thing-in-use what it is will get a consistent answer from a broken object and pass:
a decoder that silently drops a correction also reports itself as having none. The
first version of this exact check passed the exact bug it was written to catch.

---

## WHAT TO ASK WHEN STARTING ONE

In this order, and the answers become the recipe's Signature:

1. What does the model read, and **at what widths was the model built**?
2. Which joins exist between things produced separately, and for each: is it a
   measurement, or does it need eyes?
3. What is the unit that gets split -- and is it big enough that holding one out
   holds out its neighbours too?
4. What is the ceiling: what can the output space express at all for this input?
5. Which constants came from the first input you tried?
6. What must be true of the numbers if everything works?
7. What is still wrong with the model you would ship?
8. **What map turns this model's outputs into the thing anyone wants, and does the
   checkpoint itself say which one?**

**Six is the one that gets skipped and the one that pays.** It is the only
question whose answer catches a fault nobody imagined in advance.

**Eight is the one that is assumed rather than answered.** Whoever trains the model
knows the map, so it never feels like a question -- and that is exactly why it is not
written down, and why the stage that consumes the model gets it wrong.
