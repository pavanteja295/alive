# configs

**Global names *which*. The component defines *what*.**

## Global

`configs/<name>.yaml` — true of the whole run, never architecture-specific.

```yaml
subject: healthygamer
bundle:  v3
mode:    audio            # text | audio | video

engines:
  knowledge_style:
    knowledge: verbatim_index
    manner:    ims_r8
  text2audio:
    provider:  elevenlabs
    voice:     cloned
  audio2face:
    audio_encoder: xada
    motion:        gru_v2
    appearance:    stavatar
```

Every value is a **name**, resolving to a file inside that block.

## Component

`engines/audio2face/blocks/motion/configs/gru_v2.yaml`

```yaml
architecture: gru
hidden: 512
layers: 3
loss:  weighted_l1
loss_args: {w_mouth: 3.0}
```

Swapping architecture is one line in global, because `transformer_v1.yaml` sits beside it
with entirely different keys and nothing outside the block ever sees them.

## Three rules that stop this rotting

1. **The resolved config is written into the run.** Whatever the precedence rules, the run
   records the final merged result. Without it you cannot tell which value took effect, and
   a run stops being reproducible. This matters more than any other rule here.
2. **Global never holds a value only one architecture understands.** The moment
   `hidden_size` appears at the top level, swapping architectures breaks.
3. **Each architecture's config shape lives with the architecture.** GRU and transformer do
   not share a schema and should not be forced into one. A block's contract is its inputs
   and outputs, not its hyperparameters.

## The search space is a separate file

`space.yaml`, beside the configs. A config says what this run uses; a space says what a loop
may vary and within what bounds. Keeping them apart is what stops a loop wandering past a
commitment.

## What never goes in a config

*How* a component works. Research logic stays in code, specific to the project and expected
to be replaced. A key only one experiment will ever set belongs in that experiment's code.
