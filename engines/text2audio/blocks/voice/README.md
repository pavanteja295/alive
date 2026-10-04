# voice

How they sound.

**build**  their audio -> provider clone -> a voice id
**serve**  styled text -> audio at source rate

Small, and largely a provider boundary. Swapping providers is a new directory under
`engines/tts/`, not a change anywhere upstream.

## Status

Built: a cloned voice per creator, trained and served from here.
`recipes/text-to-voice/RECIPE.md` builds one; `scripts/voice_server.py` serves it (`./alive up`).

Whether the checkpoint uses a cloned creator voice or a stock voice is unsettled.

External work in `REFERENCES.md`. Nothing there is on the path; this block is good-enough.
