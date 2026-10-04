# manner

How they say it.

**build**  their paragraphs, corrupted through helper LLMs into (shadow -> original) pairs,
           then a LoRA on that map
**serve**  rewrites the LLM's answer. text in, text out, no prompt.

## Why the separation from content is clean

The adapter never sees the question, the retrieved knowledge or the entity model. It sees a
paragraph. They are not even the same model: content can be a large API model, manner is a
0.5B base with a rank-8 adapter, 15 to 80 MB, running locally.

Two different models, in series, joined by plain text. Swap either and the other is
untouched.

## Status

External work and bucket routing into `Mycode/persona_style_papers` is in
`REFERENCES.md`.

**Exists and is published.** `Mycode/inmystyle` (`ims`), arXiv 2607.29238, PolyForm NC.
11 CLI commands, stages do not auto-chain, composite eval already implemented
(authorship 0.40, content-F1 0.25, target-F1 0.15, ai-tells 0.15, stylometric 0.05).

Adaptation needed: it is built for written style from academic PDFs. Its `[prep]` filters
assume affiliations, math ratios and citation markers. Transcripts need different filters.
That is config, not redesign. Spoken style is arguably the more correct target here, since
the output is going to be spoken anyway.

## P2

Improves from perception. Conversational register differs from broadcast delivery, so the
model moves closer to the person and further from the performance.
