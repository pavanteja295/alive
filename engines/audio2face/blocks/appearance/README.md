# appearance

How they render.

**build**  (solved meshes, real frames) -> Gaussians bound to the mesh, plus a UNet
**serve**  controls + geometry -> rendered frames

## Two traps, both already paid for

- **Never train on predicted meshes.** Only solved ones. Otherwise this model silently
  absorbs the motion model's error and the two can never be separated again.
- **Never put learned offsets in vertex space.** Control space, so the rig's correctives
  fire. Measured worst case for bypassing them: 40.72%.

## The coupling worth knowing

The UNet takes the control vector, not just the mesh. So at inference, control error reaches
the pixels **twice**: through the geometry and through the conditioning. The two stages
share nothing at training time and are coupled at serve time.

## The representation swap already happened here

This block was adapted from a FLAME-based avatar to MetaHuman: the mesh Gaussian model
replaced the FLAME one, and the dataloader reads vertices rather than FLAME parameters. So
representation swappability in this block is demonstrated rather than claimed.

Conditioning is on **257** channels (251 selected raw controls + 6 head), the same vector
`motion` emits internally, and clustering is on that control vector.

## Status

Built for one subject first (the author's own face: 12,743 frames, iteration 133,812).
This is the entire GPU cost of the system.
