# pipelines

Build orchestration. Offline, batch.

The counterpart to `agent/`, which is the same job online.

## What it does

Resolves build order from dependencies. Solve before geometry, geometry before motion and
appearance, the text assets in parallel with all of it. **Nobody specifies order.**

An asset whose input is missing is **skipped and reported, not failed**. That is how partial
bundles arise naturally rather than by special-casing.

## The trigger is the only P1/P2 difference

- P1 fires on an upload finishing. Batch, finite, once.
- P2 fires on accumulated interaction. Same pipeline, same promotion, different condition.

No new orchestration component for P2.

## Rules

- Stages are idempotent and resume from disk.
- Completion is judged from disk, never from a checkpoint file. An interrupted run, a killed
  process and a reboot all resume by running it again.
- Stages do not auto-chain unless asked, so a single stage can be re-run after a config tweak.

## Status

Not built. The idiom is proven: see `ingest_channel.py` and `prepare_stavatar_take.py`.
