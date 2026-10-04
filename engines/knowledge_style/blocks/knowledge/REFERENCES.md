# knowledge: external work

Retrieval and memory systems considered for this block. Verified against the repo or the
arXiv page on 2026-09-04, not from recall.

**Scope split.** `Mycode/persona_style_papers` owns the style and authorship-transfer
literature, which is `manner`'s concern. This file owns the knowledge, retrieval and memory
side and does not restate it.

**The decision everything here is measured against** is K3 in `problems.md`: index over
**verbatim transcripts with provenance, not extracted claims**. A system that indexes by
LLM extraction is not a drop-in for this block, it is a reversal of K3 and needs its own
argument.

---

## Verbatim-preserving. Aligned with K3.

| | what it is | license | why it is relevant |
|---|---|---|---|
| **MemPalace** <br> `github.com/MemPalace/mempalace` | Local memory system. Explicitly "does not summarize, extract, or paraphrase." Hierarchical scoping: entities are *wings*, topics are *rooms*, verbatim content sits in *drawers*. ChromaDB default, pluggable to SQLite / Milvus / Qdrant / pgvector. Local embeddings, no API key. Optional temporal entity-relation graph with validity windows, on the side, not as the store. | MIT | The only surveyed system whose storage contract **is** K3. The wings/rooms/drawers hierarchy is a scoping mechanism over verbatim text rather than a replacement for it. Reports 96.6% R@5 on LongMemEval retrieval recall with raw semantic search, no LLM in the loop. Python 3.9+, ~300 MB for the embedding model. |
| **MemMachine** <br> `arXiv 2604.04853` | Three layers: short-term, long-term episodic, profile. Stores **entire conversational episodes** rather than extractive summaries. "Contextualized retrieval" expands a nucleus match with its surrounding context. A Retrieval Agent routes queries by complexity. | paper CC BY 4.0. Claims open source; no repo URL on the abstract page, unverified. | Contextualized retrieval is the right **retrieval unit for transcripts**: a hit at 14:32 is meaningless without the 90 seconds around it. Reports LoCoMo 0.9169 (gpt-4-mini), LongMemEvalS 93.0%, HotpotQA-hard 93.2%, ~80% fewer input tokens than Mem0. The three-layer split is a P2 shape (continuous perception), not a P1 one. |

## Extraction-based. Conflicts with K3.

| | what it is | license | why it is relevant |
|---|---|---|---|
| **LightRAG** <br> `github.com/HKUDS/lightrag` | Knowledge-graph RAG positioned as a lighter alternative to Microsoft GraphRAG. **Extracts entities and relations from chunks into a graph.** Dual-level retrieval over the KG plus vector embeddings. Incremental update and selective deletion. Claims to work with 30B open-weight models. | MIT | Reverses K3. Recorded because the incremental-update and deletion story is the one thing P2 will need and MemPalace/MemMachine do not clearly provide. |
| **RAG-Anything** <br> `github.com/HKUDS/RAG-Anything` | Multimodal RAG built **on top of LightRAG**. Text, images, tables, equations, charts. MinerU for document extraction, VLMs for image analysis. Multimodal knowledge graph with cross-modal relations, vector search plus graph traversal, modality-aware ranking. | MIT | Inherits LightRAG's extraction, so it inherits the K3 conflict. Its multimodal indexing addresses a corpus we do not have: our takes are ASR text over talking-head video with no slides, tables or figures. Revisit only if the corpus gains document-like material. |
| **kotaemon** <br> `github.com/Cinnamon/kotaemon` | RAG document-QA product: multi-user login, collections, hybrid full-text plus vector retrieval with re-ranking, citations with PDF preview. Question decomposition, ReAct and ReWOO agents. GraphRAG indexing is one optional pipeline, not the default. Pluggable doc stores (Elasticsearch, LanceDB) and vector stores (ChromaDB, Milvus, Qdrant). | Apache-2.0 | A **product shell**, not a retrieval design. Hybrid retrieval plus re-ranking plus citations is the correct default shape and is worth copying as a reference implementation. The UI, multi-user and collections layer is `app/`'s problem, not this block's, and we do not have it yet. |

## Reasoning-time retrieval

| | what it is | license | why it is relevant |
|---|---|---|---|
| **SituatedThinker** <br> `arXiv 2505.19300` <br> `github.com/jnanliu/SituatedThinker` | RL-trained interleaving of internal reasoning with calls to external information through predefined interfaces. Gains on multi-hop QA and math; claims generalisation to KBQA, table QA and text games. RL algorithm, interfaces and base models not stated on the abstract page. | arXiv default license | This is the **mechanism behind this block's central commitment**: knowledge sits beside the LLM as a resource, so the model decides how much it needs and one-shot retrieval and multi-turn querying are the same shape. SituatedThinker is what that looks like when the deciding is trained rather than prompted. Training it is far past where we are; the paper is the reference for the shape, not a dependency. |

## Personalization benchmarks

`Mycode/persona_style_papers/07_personalization` is this block's bucket, not `manner`'s.
Six papers. Checked 2026-09-04.

| | what it is | verdict |
|---|---|---|
| **LaMP** `2304.11406` (ACL, 523 cites) <br> **LongLaMP** `2407.11016` | The standard retrieval-personalization benchmarks. `GOLD_STANDARD.md` Direction 3 names this as the component whose quality rises monotonically with corpus size forever, where style saturates. | The **task shape** matches ours: adapt to one individual from their history. The **data** does not: user profiles and short-form tasks, not a 6-hour speech archive. Useful as a harness design, not as a target number. |
| **HYDRA** `2406.02888` (NeurIPS) | Black-box personalization by factorization: trains a reranker over top-retrieved history plus an adapter, capturing user-specific patterns and shared cross-user knowledge separately. | The **reranker over retrieved history** is directly applicable and is the part kotaemon also gets right. The cross-user factorization is not: P1 is one subject at a time, and standing decision 15 says P1 and P2 share machinery, not data, so there is no user population to factor against. |
| **PersonaConvBench** `2505.14106` <br> **AlpsBench** `2603.26680` | Multi-turn personalized conversation benchmarks. AlpsBench is 2,500 real long-term sequences from WildChat with human-verified structured memories, scoring extraction, updating, retrieval and utilization. | **P2's benchmarks, not P1's.** Both score memory lifecycle over continuing dialogue, which is exactly the axis P1 does not have. AlpsBench's finding that models fail to reliably extract latent user traits is the strongest external support for K3: extraction is where the loss happens. |
| **Survey** `2502.11528` | Map of the personalized-LLM field. | Entry point if this block ever needs re-surveying. Not on the path now. |

## Retrieval engineering

The boring decisions, and the evidence that the clever alternative did not pay. These set
the `FRAMING.md` 4.3 baseline. Checked 2026-09-04.

| | what it says | verdict |
|---|---|---|
| **Is Semantic Chunking Worth the Computational Cost?** <br> `2410.13070`, NAACL 2025 Findings <br> Qu, Tu, Bao (Vectara, UW-Madison) | Evaluates semantic chunking against fixed-size on document retrieval, evidence retrieval and retrieval-based answer generation. **Fixed-size consistently outperformed semantic chunking** on realistic document sets; the computational overhead was not justified. | **Decides our chunking.** Fixed-size with overlap. Peer-reviewed, and it is a negative result, which is the kind that survives. |
| **Intelligence Degradation in Long-Context LLMs** <br> `2601.15300` | Catastrophic degradation past **40-50% of maximum context length**; F1 0.55-0.56 to 0.30, a 45.5% drop. Mixed dataset of 1,000 samples spanning 5-95% of context, five-method cross-validation. | Read as a **shape, not a constant**: only Qwen2.5-7B was tested. Combined with the 2026 frontier-model reports of measurable degradation past 200K on multi-fact retrieval, enough to disqualify whole-archive context as a baseline. |
| **On Memory Construction and Retrieval for Personalized Conversational Agents** <br> `2502.05589` (Pan et al.), **ICLR, 101 cites**, SeCom | Compares turn-level, session-level and summarization-based memory granularity, finds each limited, and proposes **segment-level** memory via a conversation segmentation model plus compression-based denoising. Gains on LOCOMO and Long-MT-Bench+. | **The best-validated memory-construction result in this file**, and the named first upgrade to the baseline. Deferred on **domain, not quality**: its segmenter keys on multi-session dialogue structure, and our takes are continuous monologue with no turn or session boundaries. Test it against fixed-size once fixed-size has a number. `FRAMING.md` 4.5. |

---

## Standing read

- **All five memory systems solve an axis P1 does not have.** They exist to remember what
  happened across sessions. P1's corpus is static, read-only and fully known in advance.
  Their machinery is P2's problem. Recorded now so P2 does not re-survey.
- **Retrieval is the baseline, not an optimization.** An earlier version of this file said
  the corpus fits in a frontier context window and retrieval was therefore only a cost
  choice. That is wrong at the scale we are planning for: see `FRAMING.md` 4.1 and the
  retrieval-engineering table below. Whole-archive context is a measuring stick that expires
  at roughly 40 takes. Retrieval is the only shape flat in corpus size.
- **Nothing here is adoptable until K1 is measured.** What a bare LLM already knows about the
  creator is the baseline, and no retrieval result means anything against an unmeasured
  baseline.
