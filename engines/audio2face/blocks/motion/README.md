# motion

How their face moves.

**infer**  audio -> `c_raw [T, 263]` + `head [T, 6]` at 30 Hz
**train**  (baseline features, baseline controls) -> solved controls, supervised in vertex space

## The baseline, as of now

A **residual with the null model folded in**. ~578k parameters wrapped around a frozen
audio-to-face baseline, initialised so step zero sits exactly on the closed-form linear
solution — neither branch can make the system worse than what it wraps.

```
Z50   [B, 107, d_z]   baseline encoder features, native 50 Hz, untouched
base  [B, 107, 257]   251 raw controls + 6 head, the frozen baseline
psd   [B, 107, 545]   optional: the rig's corrective activations

  Linear d_z->64  +  Linear 257->32  ->  concat 96
      -> biGRU x2, hidden 128 -> 256
      -> Linear 256->257, ZERO init
  plus an affine branch  a*base + b  (514 params, closed-form least squares)

delta [B, 107, 257] at 50 Hz
```

`|affine| / |delta|` reads out directly whether the other 577k parameters earned their place.

**`d_z` is never hardcoded.** It comes from the baseline's own feature width, so a baseline
with different dimensions can be plugged in without touching the network.

## Two things that look contradictory and are both right

**Predict in control space. Supervise in vertex space.**

Offsets live in control space so the rig's correctives fire (standing decision 12). The
*loss* is in vertex space, on a measurement rather than a principle: control space is
anisotropic by ~90x (p95/p5 of millimetres moved per unit control). Under equal weighting
the ten most face-moving controls receive **0.5%** of the gradient; weighted by millimetres
actually moved they receive **73.3%**. A plain control-space MSE spends most of the model's
capacity on controls that barely move the face.

This only works because the rig is differentiable. Do not "fix" one to match the other.

## Identity conditioning

The network is never told whose face it is. A constant input cannot inform anything — it
merges into the bias — and the static half of the residual is already absorbed by the affine
branch's 257 biases.

What *can* inform is the **PSD input**: 545 corrective activations, identity-specific and
time-varying, each a clamped monomial of degree 2 to 6. A tanh GRU cannot cheaply represent
a degree-6 product of its own inputs, so supplying them is reach rather than duplication.
Compute them from the baseline, never from base + delta, which is circular.

## Non-causal, and what that costs

The GRU is **bidirectional** and the baseline is already non-causal, so the window is a cost
parameter rather than a quality one. Measured: 30 s of context changes the centre frame by
0.175%, 60 s by 0.002%.

**Consequence: this path cannot stream.** Real time is therefore a baseline swap, not tuning,
and by standing decision 14 a swap invalidates this residual and forces a refit.

## Rates and the one resample

Built at 50 Hz, supervised at 30 Hz, with a single resample in control space. The window is
64 output frames (2.1 s); the 107-frame input width is **derived** from the two rates, never
chosen.

Time shift is **per-take and measured**, never inherited. The sign is the negation of the
measured lag. Getting it backwards costs twice the shift and reads as a bad model.

## Status

Trained for one subject, from depth. Checkpoint exists. See `paths.yaml` under `legacy`.
