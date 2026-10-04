# Experiments

One row in `index.tsv` per idea, one file per idea. Schema and rules: the `improve-system` skill.

- `tag` is a **git tag**, so the tag recovers the code state.
- `prediction` is written **before** the run.
- `sanity` names what was checked, on a case small enough to verify by eye.
- `verdict` is `keep` | `drop` | `instrument blind`. The third sends you back to
  `METRICS.md` and must not be recorded as a failed idea.

Nothing is recorded here yet. Experiments require frozen metrics, and the style axis in
`METRICS.md` is unresolved.
