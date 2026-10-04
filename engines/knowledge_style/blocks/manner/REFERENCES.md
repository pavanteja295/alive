# manner: external work

**The survey already exists.** `Mycode/persona_style_papers` holds 54 papers with abstracts
in `index.json` and Semantic Scholar citation counts in `citations.json`. Its `DECISION.md`
carries the standing verdicts and its `GOLD_STANDARD.md` carries the credibility ranking.
Neither is restated here.

This file exists because that tree is organised by **the literature's taxonomy, not ours**,
so its buckets cut across four of our blocks. This is the routing, plus what the tree holds
that its own two summary documents do not cover.

Checked 2026-09-04.

---

## Which buckets belong to which of our components

| bucket | ours |
|---|---|
| `01_style_transfer`, `09_current`, `10_translation` | **manner** |
| `03_learned_layer`, `04_peft_baselines`, `05_decoding_time` | **manner** now, `reasoning` in P2. Same interventions, different objective |
| `02_persona_agents` | `knowledge_style` engine level. Per-person role-play is the whole engine, not one block |
| `07_personalization` | `knowledge`. See its own `REFERENCES` |
| `08_style_embeddings`, `11_evaluation` | the **scoreboard**, which no block owns. Engine level |
| `06_voice` | `text2audio/blocks/voice` |

**The counts in that tree disagree**: `README.md` says 29 papers, `GOLD_STANDARD.md` says 44,
`index.json` has 54 and `citations.json` has 49. Treat `index.json` as authoritative and the
two prose files as written against an earlier snapshot. Their conclusions still stand; their
inventories do not.

---

## What manner already committed to, in one line

`DECISION.md` keeps the `(neutral -> him)` pair corpus, frontier model with large verbatim
context, and best-of-N only if prompting under-commits. It kills TinyStyler as the main
engine, the Away/Towards/Sim/Joint suite as an optimisation target, and training for now.
`ims` implements the pair-corpus half of that. Go there for the reasoning, not here.

## What the tree holds that neither summary covers

| | what it is | verdict |
|---|---|---|
| **StyleMC** <br> `01/2312.17242` | Finds instruction-tuned LMs **struggle to reproduce author-specific style demonstrated in a prompt**, and steers instead with contrastively-trained stylometric representations plus sequence-level inference. | **The one piece of direct counter-evidence to our chosen route**, and it is not in either summary document. Read precisely: the finding is about a *small* writing sample in the prompt. We have 100-200x that, which is the untested regime `DECISION.md` names. Not a refutation, but the thing the day-1 experiment is actually testing. |
| **Register-analysis prompting** <br> `01/2505.00679` | Prompting method: describe the exemplar's style via register analysis rather than handing over raw exemplars. Reports better style strength and meaning preservation than other prompting strategies. | **Cheapest available upgrade to the day-1 experiment.** Prompt-only, no training, no weights. If a plain verbatim-context prompt under-commits, this is the first thing to try, before best-of-N. |
| `10_translation` (6 papers) | Back-translation and unsupervised MT: `1511.06709`, `1711.00043`, `1804.07755`, `1804.09057`, `1808.09381`, plus the TST survey `2011.00416`. | **Ancestry, not method.** STRAP's neutralise-then-restyle trick is back-translation, so this bucket is why the pair corpus works at all. Read `1808.09381` (back-translation at scale) if the pair corpus underperforms: our regime is large-data and that paper is the large-data one. |

---

## Standing read

- **The tree is over-supplied for manner and under-used for evaluation.** Six buckets feed a
  block whose method is already settled, while `11_evaluation` feeds the one thing the
  project does not have. Effort should move the other way.
- **Nothing here needs reading before the day-1 experiment.** `DECISION.md` already says
  reading more papers is on the kill list. Its two additions above are both single-paragraph
  changes to that experiment, not prerequisites for it.
