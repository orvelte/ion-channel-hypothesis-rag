# CLAUDE.md

## project overview

This is a transformer-based RAG pipeline for automated hypothesis generation in neuro drug discovery, built around voltage-gated sodium channels (SCN1A–SCN9A) as the target class. It integrates ChEMBL, UniProt, PDB, and PubMed with LLM inference.

**This is a deliberate learning project.** Data is small-scale by design. The primary goal is rigorous understanding of each stage — ML reasoning, biological validity, and evaluation — not throughput or production readiness.

---

## working principles

**Checkpoints before code.** Every pipeline stage has a corresponding evaluation checkpoint in `src/evaluation/` and `notebooks/checkpoints/`. Before implementing a new stage, confirm the checkpoint script for the previous stage runs and produces interpretable output. Do not skip forward.

**Explain the reasoning, not just the implementation.** When writing code for any pipeline component, include a short comment block (3–5 lines) explaining *why* this design decision was made — what failure mode it prevents, what the biological or ML intuition is, what a simpler approach would miss.

**Fail loudly at data boundaries.** Any function that touches raw database records should raise explicit, descriptive errors for: missing required fields, unexpected schema, SMILES parse failure, sequence length outliers, or ID cross-reference gaps. Never silently drop records — log them to a dedicated `data/rejected/` path with the reason.

**Preserve evaluation artifacts.** Checkpoint outputs (score distributions, retrieval logs, parse failure rates) should be saved to `data/evaluation/stage_{n}/` with a timestamp. These are part of the project record, not throwaway diagnostics.

---

## architecture

```
query
  └─► query encoder (PubMedBERT)
        └─► FAISS flat index
              └─► top-k chunks (k=5–10)
                    └─► re-ranker (optional cross-encoder)
                          └─► prompt builder
                                └─► LLM (GPT-4o / Claude API)
                                      └─► structured hypothesis output
```

**Three tiers:**
- Tier 1 (data): ingestion + preprocessing — stages 1–2
- Tier 2 (retrieval): embedding + vector store + retriever — stages 3–4
- Tier 3 (generation): prompt engineering + LLM inference — stages 5–6

---

## data sources and scale

| source | scope | scale |
|--------|-------|-------|
| ChEMBL | CNS compounds, Na⁺/K⁺ channel assays | ~500 compounds |
| UniProt | SCN1A–SCN9A canonical sequences + annotations | 9 proteins |
| PDB | Na⁺/K⁺ channel structures — metadata only, no 3D parsing | 10–20 structures |
| PubMed | Channel pharmacology and neuro drug discovery abstracts | 100–200 abstracts |

Raw data lives in `data/raw/{source}/`. Never modify raw files — all transformations produce new files in `data/processed/`.

---

## evaluation checkpoints

These are the primary deliverables at each stage. Each has a corresponding script in `src/evaluation/` and a notebook in `notebooks/checkpoints/`.

**Check 1 — data integrity** (`stage_1_integrity.py`)
What to verify: schema conformance across all four sources, null/missing field rates, ChEMBL–UniProt–PDB ID cross-reference coverage, SMILES parseable by RDKit.
Flag if: >5% null rate on any required field, or <80% cross-reference coverage between sources.

**Check 2 — preprocessing quality** (`stage_2_preprocessing.py`)
What to verify: chunk length distribution (target: 150–250 tokens, flag outliers), token budget per source, SMILES→Morgan FP conversion success rate, metadata tag completeness.
Flag if: >10% of chunks fall outside the target length window, or FP conversion rate <95%.

**Check 3 — embedding and retrieval quality** (`stage_3_embeddings.py`)
What to verify: nearest-neighbour spot check (manually inspect top-5 neighbours for 10 seed queries), known-pair retrieval test (e.g. lamotrigine should retrieve SCN1A-related chunks in top-3), cosine score distribution (should not be uniformly high — indicates embedding collapse).
Flag if: known pairs fail to retrieve in top-5, or cosine score distribution has mean >0.95.

**Check 4 — retriever performance** (`stage_4_retrieval.py`)
What to verify: Recall@k on a 20-query synthetic test set, cosine score distribution per query, context relevance scored by LLM-as-judge (prompt: "does this retrieved chunk answer the query?"), precision@k.
Flag if: Recall@5 <0.6 on the synthetic set, or >30% of retrieved chunks judged irrelevant by LLM-as-judge.

**Check 5 — prompt ablation** (`stage_5_prompts.py`)
What to verify: run each of the two prompt variants (CoT vs direct answer) against 5 fixed queries with identical retrieved context. Compare: answer length, citation rate, hallucination rate (manually check 3 outputs per variant against source chunks).
Flag if: either variant produces hypotheses with no traceable grounding in retrieved context.

**Check 6 — output evaluation** (`stage_6_outputs.py`)
What to verify: structured output parse success rate (target field extraction), factual grounding (trace each claimed mechanism to a specific retrieved chunk), mechanistic plausibility (use domain knowledge — does the proposed ion channel modulation mechanism make pharmacological sense?), novelty check (is the output verbatim from a chunk, or genuinely synthesised?).
Flag if: parse success rate <90%, or >20% of outputs have no traceable grounding.

**Check 7 — end-to-end benchmark** (`stage_7_e2e.py`)
Run 5 held-out queries through the full pipeline. For each output, classify any failure as: retrieval failure (right answer not in top-k), generation failure (right chunks retrieved but hypothesis incorrect/ungrounded), or data gap (information not in corpus). Record the taxonomy. This is the primary diagnostic artifact.

---

## configuration

All tunable parameters live in `configs/pipeline.yaml`. Do not hardcode values in source files.

Key parameters:
- `embedding_model`: default `"pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-sst2"` or `"NLP4Science/PubMedBERT-abstract-fulltext"`
- `chunk_size_tokens`: default `200`
- `chunk_overlap_tokens`: default `20`
- `retrieval_k`: default `7`
- `morgan_radius`: default `2`
- `morgan_nbits`: default `2048`
- `llm_model`: default `"gpt-4o"` or `"claude-opus-4-6"`
- `llm_temperature`: default `0.3` (low — we want grounded, not creative)

---

## conventions

- Python 3.11+. Type hints on all function signatures.
- One module per data source in `src/ingestion/`. Each exposes a `fetch()` function returning a typed dataclass or Pydantic model.
- Logging via the standard `logging` module, not `print`. Log level INFO for pipeline steps, DEBUG for per-record operations.
- Tests in `tests/` mirror the `src/` structure. At minimum, one unit test per ingestion client and one integration test per checkpoint script.
- Notebooks are for exploration and checkpoint reporting only — no pipeline logic lives in notebooks.
- All LLM calls go through a single client wrapper in `src/generation/llm_client.py` so the model can be swapped without touching prompt logic.

---

## scope boundaries

The following are explicitly out of scope for this project:

- 3D structure parsing or molecular docking (PDB used for metadata only)
- Fine-tuning any embedding model
- Local LLM inference
- ANN index tuning (flat index only)
- Multi-target generalisation beyond sodium/potassium channels
- Production API serving or containerisation

If any of the above becomes relevant, create a new branch and document the reasoning before expanding scope.
