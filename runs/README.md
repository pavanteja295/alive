# runs

What we tried. Disposable.

```
runs/<run_id>/
  config.yaml      the manifest that produced this run
  outputs/
  score.json       what eval said
```

A run is its config plus its outputs. That is the whole of experiment tracking and model
versioning, so neither needs a separate system.

## Rules

- **Keyed by config.** Same config, same run identity.
- **Nothing outside reads a run path directly.** Consumers read `creators/`. That is what
  makes rollback a pointer change.
- **Safe to delete.** Anything that must survive gets promoted.

## Status

Empty.
