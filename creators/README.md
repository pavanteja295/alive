# creators

What we chose, per creator. Everything here is small and committed; the models and data
it names are not (README.md, *Where everything goes*).

```
creators/<id>/
  live.env               the live app: which voice, face run, body, resting pose, examples
  build.json             the rebuild: which videos, which profiles, which runs were released
  build/                 decisions a person made, replayed by ./alive build
    clips/<take>/        cluster settings and keep/drop verdicts
    voice/               the reference clip choice
    motion/ render/      joins-sheet acknowledgements
    answers/             which passages the persona note quotes (no text)
creators/_template/      copy this for a new creator
```

The trained models are in `checkpoints/<id>/`. Today: `drk` (Dr K) and `huberman`.
