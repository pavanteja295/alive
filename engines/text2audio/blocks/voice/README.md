# voice

How they sound: a finetuned F5-TTS voice per creator.

**build**  their audio -> cleaned clips -> F5-TTS finetune -> `checkpoints/<id>/voice/`
**serve**  text -> streamed audio (`scripts/voice_server.py`, started by `./alive up`)

The recipe is `recipes/text-to-voice/RECIPE.md`; the plain-language version is
`docs/guide/voice.html`. External work considered is in `REFERENCES.md`.
