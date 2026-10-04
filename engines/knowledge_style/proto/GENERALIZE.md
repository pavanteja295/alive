# Generalizing this pipeline to a new creator

**The goal of onboarding a second (and later, third) creator is to prove the
pipeline generalizes, not just to produce one more working model.** A creator
onboarded through one-off, creator-specific scripts has not tested the
pipeline even if their model trains successfully.

## The rule

Before writing any new script for a step: check whether an existing recipe
tool already does it, or could with a small, creator-agnostic change (adding
a profile/creator parameter is a generalization, not a new script).

**Allowed without discussion — these are meant to vary per creator by
design:**
- `profiles/<creator>.py`, `harness/profiles/<creator>.json` and equivalents
- processing that is genuinely about *this creator's* material: their marker
  set, their coined terms, their `ASK_YES`/`ASK_NO` examples, thresholds
  re-derived from their own data

**Not allowed without logging it here first:** a script that only works for
one creator, covering a step an existing tool already covers for another
creator, or a step with no existing tool where a shared one should have been
written instead.

## Log of exceptions

*Nothing logged yet as of 2026-09-29 (Huberman onboarding). If a script gets
written that isn't a per-creator profile/config and isn't a small
generalization of an existing tool, it goes here: what it does, why nothing
existing covered it, and whether it should be generalized later instead of
staying a one-off.*
