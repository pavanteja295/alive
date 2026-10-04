# perception engine

**the world -> text and signals.** The mirror of the interface side.

```
face  --+
voice --+--> adapters --> text --> knowledge+style
text  --+
```

## It is not a ladder

Interface modes nest because each is rendered from the previous. Perception modalities are
**encoded independently**, so they are a set of flags, not a chain. Voice-in without
face-in is normal. Do not force perception into the interface's shape.

## It is a tee, not a path

```
perception --+--> adapters --> reasoning     a subset, now
             +--> corpus                      everything, always
```

**Store everything. Route selectively. The routing is config.** Face-to-reasoning can be
switched on later without touching this layer, because the data is already there. Nothing
can be collected retroactively.

## Text is the waist today, not forever

The adapters exist to collapse everything into text because that is the cheapest interface
that works. In future the intelligence may read audio and video embeddings directly, and the
adapters become optional rather than mandatory.

So the contract out of this engine is a **set of named channels**, of which `text` is one.
Empty channels are normal. Declaring them now costs one field; adding them later is a format
break across two engines.

This is also where prosody re-enters. It is lost twice today, once here and once at
synthesis, and a multimodal intelligence stops the first loss.

## Blocks

| block | kind | state |
|---|---|---|
| `asr` | generic | **P2.** voice -> text |
| `face_encoder` | generic | **P2, later.** video -> expression signals |

Note that reading a face into controls and rendering a face from controls are inverse
operations over the same representation. `face_encoder` and audio2face's blocks are two
directions of one thing, which is why the face blocks stay one family.

## Project 1 and Project 2

**P1 has text perception only**, and it is not zero: the person types, and that is sensing.
No block is needed for it.

**This is the only engine that is genuinely empty**, and therefore the only real
restructuring P2 requires. Everything else in the pipeline scales by activating a block
that already has a home, or by retraining.
