# _solve

Fit the rig to video, per frame. The prerequisite geometry, motion and appearance all share.

Not an asset. Nothing deploys it. It is underscored so the assets stay countable.

## Why it is the riskiest thing in the project

- **It may not work on creator video at all.** Podcast footage cuts between cameras, so
  there is no static camera and no head-locked framing. The old ingest script says outright
  that such takes feed the audio branch only.
- **It may need a human.** If it means driving Unreal on Windows by hand, full autorun is
  false for all three face assets, and P1's "upload a list and it happens" is false too.
- **A consenting creator dissolves both problems.** Someone who wants this will sit for a
  five-minute capture: neutral pose, teeth visible, slow head rotation, expressive range.
  That is the difference between blocked and merely unbuilt.

## Variants

Depth solve and mono solve produce the same 263-control output. Sensor changes the input,
not the interface.

## Status

Works on iPhone depth captures. Unverified on creator footage.
