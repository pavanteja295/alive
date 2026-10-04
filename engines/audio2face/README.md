# audio2face engine

**audio -> video.** The deepest engine, the entire GPU cost, and every unverified premise
in the project.

```
audio -> audio_encoder -> motion -> rig -> appearance -> renderer -> frames
                                     ^
                                 geometry
```

## Blocks

| block | kind | state |
|---|---|---|
| `audio_encoder` | generic | xADA, two ONNX models. **The one non-differentiable link.** |
| `_solve` | generic | fit the rig to video. Not deployable, a prerequisite three blocks share. |
| `geometry` | person-specific | chain root. Route unsettled, deprioritised. |
| `motion` | person-specific, correct | a residual on `audio_encoder` |
| `rig` | generic | controls to mesh. `TorchRig` is differentiable. |
| `appearance` | person-specific | Gaussians. All the GPU. |
| `renderer` | generic | |

## The chain inside

`geometry` constrains `motion` and `appearance`. Change it and both are invalidated,
silently, because neither will error. The three version together in a bundle.

## Priorities inside this engine, measured

1. **Lip closure.** Bilabials that fail to close are the most instantly-fake artifact.
2. **Data volume.** Training has used a tiny fraction of what is available.
3. **Temporal artifacts** within a block. Boundaries are already cleared.
4. **Mesh identity.** Distant fourth, and only via overfitting: the Gaussians absorb mesh
   error, but that compensation is fitted to training frames and generalises worse.

Before any decomposition work, say which of the four it improves and how that is measured.

## Differentiability

Every link is torch except `audio_encoder`, which is ONNX. Closing that one gap makes the
whole engine differentiable, which is the only route to optimising the composition rather
than each block locally.

Differentiable does not oblige end-to-end training. It makes it possible.

## Project 2

**No new blocks. Everything gets better rather than different.**

- **Depth arrives**, because the person is finally in front of a sensor. It improves
  `geometry`, `motion` and `appearance` by retraining, not by restructuring.
- **One prerequisite P1 does not have:** the depth solve is identity-dependent, so the
  improved path needs a person-specific identity first. The mono path P1 uses needs none.
- **Real time.** Lighter blocks, which is a baseline swap, which invalidates the residuals
  on them. Budget the refit, not just the swap.
- **Phone rendering.** Either `renderer` swaps for a lighter block, or the render runs
  server-side and streams. Either way it is a baseline swap.
