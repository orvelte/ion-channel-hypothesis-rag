"""
Check 4 — Retriever performance checkpoint.

Run after Stage 4. Measures Hit Rate@k (fraction of queries with ≥1 relevant
chunk in top-k) and Mean Precision@k on a 20-query synthetic test set, plus
LLM-as-judge context relevance on the top-3 chunks per query.

Relevance is determined by keyword matching against expected terms derived from
biological knowledge of the corpus. A chunk is relevant if it contains ANY of
the expected_terms (case-insensitive) in its text or metadata fields.

Flag conditions:
  - Hit Rate@5 <0.60 on the synthetic test set
  - >30% of top-3 chunks judged irrelevant by LLM-as-judge
"""

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from src.generation.llm_client import LLMClient

logger = logging.getLogger(__name__)

# 20-query synthetic test set.
# expected_terms: case-insensitive substrings that must appear in a relevant chunk.
# A chunk is relevant if ANY term matches (text, gene_name, or chunk_id field).
SYNTHETIC_QUERIES: list[dict] = [
    # --- Gene/protein function (UniProt + PubMed) ---
    {
        "query": "SCN1A Nav1.1 voltage-gated sodium channel function",
        "expected_terms": ["SCN1A"],
        "category": "gene_function",
    },
    {
        "query": "SCN2A Nav1.2 brain sodium channel expression",
        "expected_terms": ["SCN2A"],
        "category": "gene_function",
    },
    {
        "query": "SCN3A Nav1.3 developing brain neonatal sodium channel",
        "expected_terms": ["SCN3A"],
        "category": "gene_function",
    },
    {
        "query": "SCN4A Nav1.4 skeletal muscle sodium channel excitability",
        "expected_terms": ["SCN4A"],
        "category": "gene_function",
    },
    {
        "query": "SCN5A Nav1.5 cardiac sodium channel myocardium",
        "expected_terms": ["SCN5A"],
        "category": "gene_function",
    },
    {
        "query": "SCN8A Nav1.6 neuronal excitability action potential threshold",
        "expected_terms": ["SCN8A"],
        "category": "gene_function",
    },
    {
        "query": "SCN9A Nav1.7 nociceptor pain signalling",
        "expected_terms": ["SCN9A"],
        "category": "gene_function",
    },
    # --- Disease/clinical (PubMed + UniProt) ---
    {
        "query": "Dravet syndrome SCN1A loss of function epileptic encephalopathy",
        "expected_terms": ["SCN1A", "Dravet"],
        "category": "disease",
    },
    {
        "query": "long QT syndrome SCN5A cardiac arrhythmia mutation",
        "expected_terms": ["SCN5A", "long QT"],
        "category": "disease",
    },
    {
        "query": "erythromelalgia congenital insensitivity to pain SCN9A sodium channel",
        "expected_terms": ["SCN9A", "erythromelalgia"],
        "category": "disease",
    },
    # --- Drug mechanisms (PubMed) ---
    {
        "query": "lamotrigine sodium channel anticonvulsant mechanism of action",
        "expected_terms": ["lamotrigine"],
        "category": "drug_mechanism",
    },
    {
        "query": "phenytoin voltage-gated sodium channel inactivation antiepileptic",
        "expected_terms": ["phenytoin"],
        "category": "drug_mechanism",
    },
    {
        "query": "lidocaine local anesthetic sodium channel block",
        "expected_terms": ["lidocaine"],
        "category": "drug_mechanism",
    },
    {
        "query": "carbamazepine use-dependent sodium channel block epilepsy",
        "expected_terms": ["carbamazepine"],
        "category": "drug_mechanism",
    },
    {
        "query": "mexiletine sodium channel cardiac ventricular arrhythmia",
        "expected_terms": ["mexiletine"],
        "category": "drug_mechanism",
    },
    # --- Biophysical mechanism (PubMed) ---
    {
        "query": "sodium channel fast inactivation ball and chain mechanism",
        "expected_terms": ["inactivation", "sodium channel"],
        "category": "mechanism",
    },
    {
        "query": "action potential depolarization voltage-gated channel gating kinetics",
        "expected_terms": ["action potential"],
        "category": "mechanism",
    },
    {
        "query": "sodium channel IC50 patch clamp electrophysiology assay",
        "expected_terms": ["IC50", "patch clamp"],
        "category": "mechanism",
    },
    # --- Structure (PDB + PubMed) ---
    {
        "query": "cryo-EM structure voltage-gated sodium channel resolution",
        "expected_terms": ["cryo-EM", "sodium channel"],
        "category": "structure",
    },
    {
        "query": "calmodulin sodium channel C-terminal domain crystal structure",
        "expected_terms": ["calmodulin", "sodium channel"],
        "category": "structure",
    },
]

_JUDGE_SYSTEM = (
    "You are a biomedical literature relevance judge for a RAG (retrieval-augmented "
    "generation) system. Given a search query and a retrieved text chunk, output exactly "
    "one token: YES if the chunk contains information relevant to answering the query, "
    "NO if it does not. Do not explain your answer."
)

_JUDGE_USER_TMPL = (
    "Query: {query}\n\n"
    "Chunk {idx} (source: {source}, id: {chunk_id}):\n"
    "{text}\n\n"
    "Is this chunk relevant? Answer YES or NO only."
)


def _is_relevant_keyword(chunk: dict, expected_terms: list[str]) -> bool:
    """Return True if any expected term appears (case-insensitive) in the chunk."""
    haystack = " ".join([
        chunk.get("text", ""),
        chunk.get("gene_name", ""),
        chunk.get("chunk_id", ""),
        chunk.get("source", ""),
    ]).lower()
    return any(term.lower() in haystack for term in expected_terms)


def _judge_relevance(
    llm: LLMClient,
    query: str,
    chunk: dict,
    idx: int,
) -> Optional[bool]:
    """Ask LLM to judge relevance of a single chunk. Returns True/False/None on parse error."""
    prompt = _JUDGE_USER_TMPL.format(
        query=query,
        idx=idx,
        source=chunk.get("source", ""),
        chunk_id=chunk.get("chunk_id", ""),
        text=chunk.get("text", "")[:400],
    )
    try:
        response = llm.complete(_JUDGE_SYSTEM, prompt).strip().upper()
        if response.startswith("YES"):
            return True
        if response.startswith("NO"):
            return False
        logger.warning("Unexpected LLM judge response: %r", response)
        return None
    except Exception as exc:
        logger.warning("LLM judge failed for chunk %s: %s", chunk.get("chunk_id"), exc)
        return None


def run(config_path: str = "configs/pipeline.yaml") -> dict:
    """
    Execute Check 4 and save artifact to data/evaluation/stage_4/.

    Returns results dict with Hit Rate@k, Mean Precision@k, and LLM-as-judge verdicts.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    artifact_dir = Path(config["paths"]["evaluation"]) / "stage_4"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Load cached FAISS index and embedder from Stage 3
    embeddings_dir = Path(config["paths"]["embeddings"])
    index_path = embeddings_dir / "faiss_index"
    if not (index_path / "index.faiss").exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. Run Stage 3 first."
        )

    logger.info("Loading cached FAISS index from %s", index_path)
    store = VectorStore.load(index_path)
    model_name: str = config["embedding"]["model"]
    embedder = Embedder(model_name=model_name)
    embedder.load()

    k_eval = config["retrieval"]["k"]
    retriever = Retriever(embedder=embedder, store=store, k=k_eval)

    llm = LLMClient.from_config(config)

    recall_threshold: float = config["evaluation"]["recall_at_k_threshold"]
    irrelevant_threshold: float = config["evaluation"]["irrelevant_chunk_threshold"]

    # --- Check 4a: Hit Rate@k and Mean Precision@k (keyword-based) ---
    per_query_results = []
    for item in SYNTHETIC_QUERIES:
        query = item["query"]
        expected_terms = item["expected_terms"]

        hits = retriever.retrieve(query)
        scores = [round(h["score"], 4) for h in hits]

        relevant_flags = [_is_relevant_keyword(h, expected_terms) for h in hits]
        hit_at_5 = any(relevant_flags[:5])
        relevant_in_top_k = sum(relevant_flags)
        precision_at_k = relevant_in_top_k / len(hits) if hits else 0.0

        top_chunks_summary = [
            {
                "chunk_id": h.get("chunk_id", ""),
                "source": h.get("source", ""),
                "score": round(h["score"], 4),
                "relevant_keyword": relevant_flags[i],
                "text_snippet": h.get("text", "")[:120],
            }
            for i, h in enumerate(hits)
        ]

        per_query_results.append({
            "query": query,
            "category": item["category"],
            "expected_terms": expected_terms,
            "hit_at_5": hit_at_5,
            "relevant_in_top_k": relevant_in_top_k,
            "precision_at_k": round(precision_at_k, 4),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "top_k_chunks": top_chunks_summary,
        })

    hit_rate_at_5 = sum(1 for r in per_query_results if r["hit_at_5"]) / len(SYNTHETIC_QUERIES)
    mean_precision_at_k = sum(r["precision_at_k"] for r in per_query_results) / len(SYNTHETIC_QUERIES)

    # --- Check 4b: LLM-as-judge on top-3 chunks per query ---
    # 20 queries × 3 chunks = 60 judge calls; each is a short prompt.
    judge_results = []
    judge_irrelevant_count = 0
    judge_total = 0

    logger.info("Running LLM-as-judge on top-3 chunks for %d queries...", len(SYNTHETIC_QUERIES))
    for item in SYNTHETIC_QUERIES:
        query = item["query"]
        hits = retriever.retrieve(query)[:3]

        for idx, chunk in enumerate(hits, start=1):
            verdict = _judge_relevance(llm, query, chunk, idx)
            is_irrelevant = (verdict is False)
            judge_results.append({
                "query": query,
                "chunk_id": chunk.get("chunk_id", ""),
                "score": round(chunk["score"], 4),
                "verdict": "YES" if verdict is True else ("NO" if verdict is False else "PARSE_ERROR"),
                "irrelevant": is_irrelevant,
            })
            if verdict is not None:
                judge_total += 1
                if is_irrelevant:
                    judge_irrelevant_count += 1

    llm_irrelevant_rate = judge_irrelevant_count / judge_total if judge_total > 0 else 0.0

    # --- Flags ---
    flags = []
    if hit_rate_at_5 < recall_threshold:
        flags.append(
            f"Hit Rate@5 {hit_rate_at_5:.2f} < threshold {recall_threshold:.2f} "
            f"({sum(1 for r in per_query_results if r['hit_at_5'])}/{len(SYNTHETIC_QUERIES)} queries)"
        )
    if llm_irrelevant_rate > irrelevant_threshold:
        flags.append(
            f"LLM irrelevant rate {llm_irrelevant_rate:.1%} > threshold {irrelevant_threshold:.0%} "
            f"({judge_irrelevant_count}/{judge_total} chunks judged irrelevant)"
        )

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": config["embedding"]["model"],
        "index_size": store._index.ntotal if store._index else 0,
        "k": k_eval,
        "n_queries": len(SYNTHETIC_QUERIES),
        "hit_rate_at_5": round(hit_rate_at_5, 4),
        "mean_precision_at_k": round(mean_precision_at_k, 4),
        "llm_irrelevant_rate": round(llm_irrelevant_rate, 4),
        "judge_calls": judge_total,
        "per_query": per_query_results,
        "llm_judge_detail": judge_results,
        "flags": flags,
        "passed": len(flags) == 0,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = artifact_dir / f"check_4_{timestamp}.json"
    artifact_path.write_text(json.dumps(results, indent=2))
    logger.info("Check 4 artifact saved to %s", artifact_path)
    return results


def compare_retrievers(config_path: str = "configs/pipeline.yaml") -> list[dict]:
    """
    Run dense-only and hybrid retrievers against all 20 SYNTHETIC_QUERIES and
    produce a side-by-side comparison.

    Saves hybrid_vs_dense_comparison.csv to data/evaluation/stage_4/.
    Returns list of row dicts.

    Regression check: raises AssertionError if the hybrid retriever makes any
    previously passing (dense HIT) query into a MISS — the hybrid must never
    regress a query that dense-only already answered.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    artifact_dir = Path(config["paths"]["evaluation"]) / "stage_4"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = Path(config["paths"]["processed"])
    embeddings_dir = Path(config["paths"]["embeddings"])
    index_path = embeddings_dir / "faiss_index"

    chunks: list[dict] = json.loads((processed_dir / "chunks.json").read_text())

    store = VectorStore.load(index_path)
    model_name: str = config["embedding"]["model"]
    embedder = Embedder(model_name=model_name)
    embedder.load()

    k = config["retrieval"]["k"]
    dense = Retriever(embedder=embedder, store=store, k=k)
    hybrid = HybridRetriever(chunks=chunks, store=store, embedder=embedder, config=config)

    rows = []
    regressions = []

    for item in SYNTHETIC_QUERIES:
        query = item["query"]
        expected_terms = item["expected_terms"]

        # Dense arm
        dense_hits = dense.retrieve(query)
        dense_hit5 = any(_is_relevant_keyword(h, expected_terms) for h in dense_hits[:5])
        dense_first_rank = next(
            (i + 1 for i, h in enumerate(dense_hits)
             if _is_relevant_keyword(h, expected_terms)),
            None,
        )

        # Hybrid arm
        hybrid_hits = hybrid.retrieve(query, k=k)
        hybrid_hit5 = any(
            _is_relevant_keyword(
                {"text": h.text, "chunk_id": h.chunk_id, "gene_name": h.metadata.get("gene_name", "")},
                expected_terms,
            )
            for h in hybrid_hits[:5]
        )
        hybrid_first_rank = next(
            (i + 1 for i, h in enumerate(hybrid_hits)
             if _is_relevant_keyword(
                 {"text": h.text, "chunk_id": h.chunk_id, "gene_name": h.metadata.get("gene_name", "")},
                 expected_terms,
             )),
            None,
        )

        # BM25 rank: position of first relevant chunk in full BM25 ranking
        query_tokens = re.split(r"[\s\W]+", query.lower())
        bm25_scores = hybrid._bm25.get_scores(query_tokens)
        bm25_sorted_idx = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_first_rank = next(
            (rank + 1 for rank, idx in enumerate(bm25_sorted_idx)
             if _is_relevant_keyword(chunks[idx], expected_terms)),
            None,
        )

        row = {
            "query": query[:60],
            "category": item["category"],
            "dense_hit5": dense_hit5,
            "hybrid_hit5": hybrid_hit5,
            "bm25_rank": bm25_first_rank,
            "dense_rank": dense_first_rank,
            "rrf_rank": hybrid_first_rank,
        }
        rows.append(row)

        if dense_hit5 and not hybrid_hit5:
            regressions.append(query)
            logger.error("REGRESSION: hybrid missed a query that dense-only hit: %r", query)

    # Save CSV
    csv_path = artifact_dir / "hybrid_vs_dense_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Comparison saved to %s", csv_path)

    dense_hits_total = sum(1 for r in rows if r["dense_hit5"])
    hybrid_hits_total = sum(1 for r in rows if r["hybrid_hit5"])

    print("\n=== HYBRID vs DENSE COMPARISON ===")
    print(f"{'Query':<42} {'Dense':>6} {'Hybrid':>7} {'BM25':>6} {'Dense':>7} {'RRF':>5}")
    print(f"{'':42} {'Hit@5':>6} {'Hit@5':>7} {'rank':>6} {'rank':>7} {'rank':>5}")
    print("-" * 75)
    for r in rows:
        d = "HIT" if r["dense_hit5"] else "MISS"
        h = "HIT" if r["hybrid_hit5"] else "MISS"
        bm25 = str(r["bm25_rank"]) if r["bm25_rank"] else "-"
        dr = str(r["dense_rank"]) if r["dense_rank"] else "-"
        rrf = str(r["rrf_rank"]) if r["rrf_rank"] else "-"
        print(f"{r['query']:<42} {d:>6} {h:>7} {bm25:>6} {dr:>7} {rrf:>5}")
    print("-" * 75)
    print(f"  Dense  Hit Rate@5 : {dense_hits_total}/20 = {dense_hits_total/20:.0%}")
    print(f"  Hybrid Hit Rate@5 : {hybrid_hits_total}/20 = {hybrid_hits_total/20:.0%}")

    if regressions:
        print(f"\n  REGRESSIONS ({len(regressions)}):")
        for q in regressions:
            print(f"    [REGRESSED] {q[:60]}")
        raise AssertionError(
            f"Hybrid retriever caused {len(regressions)} regression(s): {regressions}"
        )
    else:
        print(f"\n  No regressions — hybrid >= dense on all queries")

    return rows


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    results = run()

    print("\n=== CHECK 4 SUMMARY ===")
    print(f"  Embedding model     : {results['model']}")
    print(f"  Index size          : {results['index_size']} vectors")
    print(f"  k                   : {results['k']}")
    print(f"  Queries evaluated   : {results['n_queries']}")
    print()
    print(f"  Hit Rate@5          : {results['hit_rate_at_5']:.0%}")
    print(f"  Mean Precision@k    : {results['mean_precision_at_k']:.0%}")
    print(f"  LLM irrelevant rate : {results['llm_irrelevant_rate']:.0%}  ({results['judge_calls']} judge calls)")
    print()
    print("  Per-query hit@5:")
    for r in results["per_query"]:
        status = "HIT " if r["hit_at_5"] else "MISS"
        print(f"    [{status}] [{r['category']:14s}] {r['query'][:60]}")
    print()
    if results["flags"]:
        print(f"RESULT: FAIL ({len(results['flags'])} flag(s))")
        for flag in results["flags"]:
            print(f"  [FLAG] {flag}")
    else:
        print("RESULT: PASS — no flags raised")

    print()
    compare_retrievers()
