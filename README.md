# ion-channel-hypothesis-rag

A transformer-based retrieval-augmented generation (RAG) pipeline for automated mechanistic hypothesis generation targeting voltage-gated sodium channels (Nav1.1–Nav1.7 / SCN1A–SCN9A). The pipeline integrates structured biological databases (ChEMBL bioactivity, UniProt channel sequences, PDB structural metadata, PubMed abstracts) with BiomedBERT embeddings, BM25/dense hybrid retrieval, and Claude LLM inference to generate grounded, citation-backed drug–channel interaction hypotheses.

The clinical anchor is epilepsy and neuropathic pain: the corpus covers channel-blocking antiepileptics (lamotrigine, carbamazepine), gain-of-function channelopathies (Dravet syndrome, erythromelalgia), and isoform-selective compounds relevant to SCN9A/Nav1.7-targeted analgesia.

---

## target scope

**Biological focus:** voltage-gated sodium channels (SCN1A–SCN9A) as the primary target class, with CNS-relevant compound libraries. Clinically anchored in epilepsy and neuropathic pain pharmacology.

**Data scale (intentional):**
- ChEMBL: ~500 CNS compounds filtered to Na⁺/K⁺ channel assays
- UniProt: SCN1A–SCN9A canonical sequences + functional annotations
- PDB: 10–20 sodium/potassium channel crystal structures (metadata only, no 3D parsing)
- PubMed: 100–200 abstracts on channel pharmacology and neuro drug discovery

---

## pipeline stages

| stage | description |
|-------|-------------|
| 0 | project scoping and target class definition |
| 1 | data ingestion from ChEMBL, UniProt, PDB, PubMed |
| 2 | preprocessing, normalisation, chunking, featurisation |
| 3 | embedding with PubMedBERT + FAISS/ChromaDB vector store |
| 4 | retriever — query encoding, top-k fetch, optional re-ranking |
| 5 | prompt engineering — system role, context injection, hypothesis template |
| 6 | LLM inference — hypothesis generation + structured output parsing |
| 7 | end-to-end evaluation — RAGAS-style metrics, error taxonomy |

---

## checkpoints (primary deliverables)

Each stage has an explicit evaluation checkpoint. These are treated as the primary output of each stage — not the code.

- **Check 1** — data integrity: schema validation, null rates, cross-database ID coverage
- **Check 2** — preprocessing quality: chunk length distribution, SMILES parse rate, token budget
- **Check 3** — embedding quality: nearest-neighbour spot check, known-pair retrieval (e.g. lamotrigine → SCN1A), semantic coherence
- **Check 4** — retrieval performance: Recall@k, cosine score distribution, context relevance via LLM-as-judge
- **Check 5** — prompt ablation: context-present vs absent, CoT vs direct, hallucination rate
- **Check 6** — output evaluation: factual grounding via citation tracing, mechanistic plausibility, novelty vs verbatim retrieval, structured parse success rate
- **Check 7** — end-to-end: 5-query benchmark, retrieval fail vs generation fail vs data gap diagnosis

---

## project structure

```
neuro-rag-pipeline/
├── data/
│   ├── raw/              # downloaded database records
│   ├── processed/        # chunked, normalised, featurised
│   └── embeddings/       # vector store artifacts
├── src/
│   ├── ingestion/        # ChEMBL, UniProt, PDB, PubMed clients
│   ├── preprocessing/    # chunking, normalisation, featurisation
│   ├── retrieval/        # embedding, vector store, retriever
│   ├── generation/       # prompt templates, LLM client, output parser
│   └── evaluation/       # checkpoint scripts for each stage
├── notebooks/
│   └── checkpoints/      # one notebook per evaluation checkpoint
├── configs/
│   └── pipeline.yaml     # model names, k values, chunk sizes, etc.
├── tests/
├── README.md
└── CLAUDE.md
```

---

## key design decisions

- PDB structures used for metadata and sequence context only — no 3D coordinate parsing. Simplifies the pipeline without sacrificing the pedagogical value of the retrieval and generation stages.
- Embedding model is PubMedBERT for text chunks; Morgan fingerprints (radius=2, 2048-bit) used as dense vectors for compounds.
- Vector store uses a flat index (no ANN tuning) — correct for this data scale and avoids premature optimisation.
- LLM inference via API (GPT-4o or Claude) — no local GPU required.
- Hybrid retrieval (BM25 + semantic) implemented at Stage 4 as an optional extension once baseline retrieval is evaluated.

---

## notable findings and fixes

Things that broke, required investigation, or produced non-obvious results during the build. Recorded here because they are the most instructive parts of the project.

**BERT anisotropy (Check 3)**
Non-contrastive BERT pooling compresses all cosine scores into a narrow band (~0.90–0.998), making absolute scores useless as confidence signals. Retrieval can only use relative ranking — the top-k chunk scores cannot be interpreted as "how relevant is this chunk." Documented in `data/evaluation/stage_3/KNOWN_LIMITATION.md`. Fix if needed: contrastive fine-tuning (SimCSE) or a natively contrastive model.

**Dense retrieval hit rate 65% → 90% after adding BM25 (Check 4)**
Seven of 20 synthetic queries failed with dense-only retrieval — all on low-frequency specific terms (gene names like SCN3A, drug names like mexiletine). BM25's exact term matching fixed all seven with no regressions. RRF fusion (k=60) was used to merge ranked lists without requiring score calibration across the two arms. BM25 tokeniser intentionally has no stemmer to preserve gene name and drug name spelling.

**`max_tokens=1024` caused 100% parse failure at Check 5**
The LLM was truncating JSON mid-response at 4011 characters — the parsed output was invalid in every case. Increasing to `max_tokens=2048` in `configs/pipeline.yaml` fixed all parse failures. This was invisible until Check 5 because earlier stages didn't call the LLM.

**Explicit JSON schema in system prompt fixed field name mismatches**
Before adding the schema to `SYSTEM_PROMPT`, the model returned `"mechanism"` instead of `"mechanism_type"`, omitted `"data_gaps"`, and occasionally added undeclared fields. Adding the exact expected schema as a literal block inside the system prompt resolved all field name errors and brought parse success to 100%.

**Embedding model substitution (Stage 3)**
The originally specified model (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-sst2`) was unavailable on HuggingFace. Substituted `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (canonical PubMedBERT, same pre-training corpus and architecture). No downstream impact on retrieval quality was observed.

**"Data gap" vs "depth gap" — a model reporting distinction (Check 6 spot check)**
The model reported `state-dependent binding` and `isoform selectivity` as data gaps in Q3/Q4. Corpus search found 5 and 20 matching chunks respectively — both terms were present in the retrieved and cited context. The model was conflating "concept mentioned but not quantified" with "absent from corpus." The two confirmed genuine corpus absences were `binding constant` (0 matches) and `dose-response` (0 matches). This distinction matters at Check 7: misclassifying a depth gap as a data gap would mask a generation failure.

**Inconsistent refusal behaviour on data gaps (Check 7)**
Q5 (lidocaine, Check 6 control) correctly refused to fabricate — the model said context was insufficient. H3 (DEKA selectivity filter, Check 7) had zero corpus matches but the model extrapolated a pore mutation hypothesis from loosely related content rather than refusing. The difference appears to be whether the topic is entirely absent versus tangentially adjacent. A stricter grounding instruction in the system prompt would likely narrow this gap.

**Single-modality citation pattern (Checks 6–7)**
Across all benchmark and held-out queries, the model cited only PubMed chunks — UniProt sequence annotations and PDB structural metadata chunks were retrieved but never cited. The most likely cause is a prompt framing issue: the system prompt emphasises mechanism and pharmacology, which maps naturally to abstract text rather than sequence records. Addressed by labelling chunk types in the prompt context; not yet resolved.

---

