# app

The surface. Chat, API endpoints, and in P2 a phone client.

Shared infrastructure: P2 inherits it rather than rebuilding.

## Interface selection

The three modes are prefixes of one chain, so selection is only a question of where to stop.

| mode | stops after | needs a GPU |
|---|---|---|
| text | manner | no |
| audio | voice | no |
| video | appearance | **yes, and this is the entire GPU cost** |

Text and audio are API-shaped and scale cheaply. Face sessions carry all of it.

Requesting a mode the bundle cannot serve is not a special case: status reports capability,
not just presence.

## Open

Where the phone renders. Locally on a lighter appearance model, or server-side and streamed
down. That is a baseline swap either way, so it invalidates appearance.

## Status

Not built.
