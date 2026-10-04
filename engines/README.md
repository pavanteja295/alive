# engines

The pipeline is a series of engines. Each engine is a subsystem with a stable external
contract and **swappable blocks inside it**.

```
[perception]  ->  knowledge+style  ->  text2audio  ->  audio2face
   P2 only          question              styled        audio
                    -> styled text        text -> audio  -> video
```

## Interface modes are just how far you run

- **text** stops after knowledge+style
- **audio** stops after text2audio
- **video** runs all three

Mode selection is not a mechanism layered on top. It is the count of engines you run, which
is why a mode that is off never constructs its models.

## Two swap granularities

- **Swap a block.** A GRU for a transformer inside audio2face. Same engine, retrained.
- **Swap the whole engine.** A provider for our own model. The external contract holds,
  everything inside changes.

## Blocks come in two kinds

- **Person-specific**: trained, promoted into a bundle, one per subject. `knowledge`,
  `manner`, `voice`, `geometry`, `motion`, `appearance`.
- **Generic**: wraps an external service or a fixed component, shared by every subject.
  `llm`, `provider`, `audio_encoder`, `rig`, `renderer`.

A person-specific block usually **attaches to** a generic one, in one of two ways:

- **correct** — takes the generic block's output and adjusts it. `manner` on `llm`,
  `motion` on `audio_encoder`. Fitted to that block's output distribution, so swapping the
  generic block means **refit, not port**.
- **feed** — supplies input to it. `knowledge` sits beside `llm` as a resource rather than
  in the chain.

## Internal structure of a block

Only the blocks we train. The training loop reads both from config and knows neither.

```
<block>/
  architectures/   same signature, declaring streaming capability
  losses/          take the batch, not just (pred, target)
  train.py  eval.py
  space.yaml       what is searchable
```

**Every block and every engine also carries a `README` and a `REFERENCES`**, trained or not.
The engine's covers what spans its blocks, a block's covers that block alone. Rules for what
belongs in a `REFERENCES` are in `CLAUDE.md`.

**Declare the seam, defer the abstraction.** Write down in the contract where a swap will
happen. Do not build the registry until there are two things to register: an interface
designed for one implementation is a guess.
