# knowledge+style: external work

Work that spans this engine's blocks. Verified against the arXiv page or the repo on
2026-09-04, not from recall.

**Scope.** Block-specific work lives in the block's own `REFERENCES`: retrieval and memory
systems in `blocks/knowledge/`. The style and authorship-transfer literature is already
surveyed in `Mycode/persona_style_papers` and is not restated here.

---

## The unowned middle: how they reason

This engine splits a person into what they know (`knowledge`) and how they sound
(`manner`). Neither owns **how they move through an argument**: the recurring discourse
arc, the order in which a position is built, what gets conceded before it is refused.
`manner` structurally cannot own it, because it trains a paragraph-to-paragraph map and
never sees a whole answer. Work on that gap goes here until a block owns it.

| | what it is | verdict |
|---|---|---|
| **LOT** (LLM-proposed Open Taxonomy) <br> `arXiv 2509.24147` <br> Chen, Mao, Yang, Ge, Bi, Liu, Hosseini, Tan, Nie, Nie | An LLM reads reasoning traces from different producers, proposes candidate distinguishing features, and keeps the ones that predict which producer wrote a trace. Iterating yields a **human-readable taxonomy** of reasoning patterns rather than a score. Second half uses the taxonomy to align a smaller model's reasoning with a larger one's. | **The closest published method to building the unowned middle**, and the only one surveyed that produces something a person can read and edit. Two distinct uses, and the second is the weaker claim. |

**Use 1: discover the taxonomy instead of writing it.** Contrast the creator's transcripts
against a generic LLM answering the same questions and let the procedure propose what
separates them. That is a discovered stance-and-moves description rather than a hand-written
one, and it stays auditable, which a steering vector or an adapter does not.

**Use 2: as a scoreboard.** The discriminator is the interesting half. `MANIFESTO.md` states
the intelligence half is not verifiable end to end, and for **complementation** that holds.
For **mimicry** it does not, and this is a candidate for the metric that `problems.md` R1
says does not exist. Its features are LLM-proposed and human-readable, so a failure is
legible: you can read which move the clone is missing. That is worth more here than the
LUAR / Wegmann authorship embeddings, which `GOLD_STANDARD.md` records as degrading sharply
on topic-controlled speech.

**Caveats, and they are load-bearing.**

- Reported 80-100% discrimination on **12 open-source reasoning models** over math, science
  and coding. Distinguishing two language models is not distinguishing two humans, and
  transfer to ASR speech transcripts is unvalidated.
- `2311.07564` in our own paper set is the direct warning: authorship models transfer to
  speech but collapse under topic control. Dr K's takes are topic-separated by construction,
  so a discriminator can win by reading the topic. **Any use as a scoreboard has to be
  topic-controlled or the number is meaningless.**
- The alignment result is GPQA +3.3-5.7%, on reasoning accuracy, not on sounding like
  anyone. It is evidence the taxonomy carries signal, not evidence it transfers style.
- No public code repository stated on the abstract page.

## The scoreboard

No block owns this, which is why it is at engine level. `MANIFESTO.md` says the intelligence
half is not verifiable end to end and that is true of **complementation**; for **mimicry**
there is ground truth in held-out takes and no metric has been chosen. `problems.md` R1.

`persona_style_papers` buckets `08_style_embeddings` and `11_evaluation` feed this and are
referenced by neither of that tree's summary documents. Checked 2026-09-04.

| | what it is | verdict |
|---|---|---|
| **Evaluating Style-Personalized Text Generation** <br> `11/2508.06374` | Critically examines the metrics the field actually uses (BLEU, embeddings, LLM-as-judge) against a purpose-built style discrimination benchmark spanning eight tasks and three settings: domain discrimination, authorship attribution, and personalized vs non-personalized discrimination. | **Read this before choosing any metric.** It is the only surveyed work whose subject is whether these metrics work at all, and its framing (discrimination across three settings) is a usable design for our scoreboard. |
| **A Call for Standardization and Validation of TST Evaluation** <br> `11/2306.00539` | Meta-analysis of the TST literature. Finds a standardization gap and, more damagingly, a **validation gap**: few automated metrics have ever been checked against human judgement. | Confirms `DECISION.md`'s position from the other direction. That document rejects the Away/Towards/Sim/Joint suite as an optimisation target on the grounds that every method scores badly on it; this paper says most such metrics were never validated in the first place. |
| **StyleDistance** <br> `08/2410.12757` | Content-independent style embeddings trained on LLM-generated near-exact paraphrases with controlled variation across 40 style features. Public weights on HuggingFace. | **The best-motivated style embedding in the tree**, because its contrastive pairs vary style while holding content fixed, which is the exact failure mode (`content leakage`) that makes the others unreliable here. Still unvalidated on speech, so it inherits `08/2311.07564`'s warning. Candidate signal, not a decided metric. |
| **Same Author or Just Same Topic?** <br> `08/2204.04907` <br> **Speech transcripts** `08/2311.07564` | The Wegmann style embedding, and the TACL paper measuring whether authorship models survive on speech. | Already load-bearing across this repo. `2311.07564` is the reason topic control is mandatory: our takes are topic-separated by construction, so any discriminator can win by reading the topic. It also finds **character n-grams competitive with neural models on speech**, which makes the cheap metric the more valid one. |
| **Detecting machine-generated text** <br> `08/2401.06712` | Few-shot detection using style representations. | Inverse framing of our problem: they detect that text is not a given human's, we want text that passes as one. Its discriminator is a ready-made adversary for a best-of-N scorer. Not on the path until prompting is shown to under-commit. |

## Measurement noise

Separate from whether the scale is calibrated: whether a number is stable enough to rank
anything. This decides the protocol in `FRAMING.md` 4.2. Checked 2026-09-04.

| | what it says | verdict |
|---|---|---|
| **Reliability without Validity** <br> `2606.19544` | 21 judge models, nine providers, three benchmarks, ~541,000 judgments. Raw exact-match agreement **inflates by 33-41 points** against chance-corrected Cohen's kappa. Judge rankings **shift by 14 positions** across benchmarks. Two production judges pair test-retest reliability >0.95 with position bias >0.10. Verbosity bias minimal (<0.011). | **Kills the naive LLM-judge scoreboard.** A judge can be perfectly self-consistent and still wrong in a fixed direction, which is exactly the failure that looks like success. Chance-correct everything; never let a judge be the primary metric. |
| **Temperature Control and Reproducibility** <br> `2606.26185` <br> **Non-Determinism** <br> `2407.10457` | Temperature control is **necessary but not sufficient**. Across 690 API calls over two providers, three model tiers and five sampling configurations, 1-2 of 7 borderline items stay non-reproducible **even under forced greedy decoding**. Causes are batch-dependent floating-point reduction, MoE routing, and provider-side load balancing. | **Sets the repetition protocol.** 5 to 10 rollouts per question, report mean and standard deviation, pin model version and evaluator. A single-run number cannot detect the size of improvement we care about. |

Position bias toward the first-shown answer is reported as high as 75%, and self-preference
at roughly +10% and +25% for two frontier judges. Any judge scoring our output is partly
scoring its own writing, so position randomization is mandatory and a judge from the same
family as the generator is disqualified.

---

## Standing read

- Take the **procedure**, not the result. LOT is a recipe for turning traces into a readable
  taxonomy, and the recipe is cheap to run at our scale. The published numbers are about
  language models and do not transfer.
- Running it needs a **held-out split that does not exist yet**, and it needs K1 first. A
  taxonomy separating the creator from a generic LLM is uninteresting if the generic LLM
  already does a passable version of him.
