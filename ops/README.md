# ops

How we run it and watch it.

Logging, deployment, cost rollup.

## Several concerns need no code here

They fall out of the three data tiers instead:

- **model versioning** from runs being config-keyed and creators holding promoted pointers
- **data versioning** from corpus being append-only
- **experiment tracking** because a run *is* its config plus its outputs

## Status is a command, never a file

Written state describes the world on the day someone typed it. The old tree's state
document went stale in twelve days and claimed nothing had ever been run, while a trained
model and a 12,743-frame dataset sat on disk.

Every component reports status by computing it from disk.

## Cost

Metered at the two provider layers (`engines/llm`, `engines/tts`) plus GPU seconds for face
sessions. That is the whole of it.

## Status

Not built.
