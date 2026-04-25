"""
Check 3 — Embedding quality checkpoint.

Run after Stage 3. Performs nearest-neighbour spot checks on 10 seed queries,
validates known-pair retrieval (e.g. lamotrigine → SCN1A chunks in top-3),
and inspects cosine score distribution for embedding collapse.

Flag conditions:
  - Known pairs fail to appear in top-5
  - Mean cosine score across all pairs >0.95 (collapse)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Known pairs: query string → gene/keyword that should appear in top-5 retrieved chunks.
# These test whether the embedding space aligns pharmacological language with target biology.
KNOWN_PAIRS = [
    ("lamotrigine sodium channel block epilepsy", "SCN1A"),
    ("phenytoin Nav1.2 inactivation anticonvulsant", "SCN2A"),
    ("lidocaine voltage-gated channel cardiac arrhythmia", "SCN5A"),
    ("Nav1.7 inhibitor neuropathic pain SCN9A", "SCN9A"),
    ("Dravet syndrome sodium channel mutation", "SCN1A"),
]

# Seed queries for nearest-neighbour spot check (manual inspection in artifact)
SEED_QUERIES = [
    "voltage-gated sodium channel mechanism of action",
    "SCN1A loss of function epilepsy mutation",
    "Nav1.5 cardiac sodium channel structure",
    "sodium channel blocker IC50 pharmacology",
    "action potential depolarization channel gating",
    "SCN9A Nav1.7 pain nociception",
    "anticonvulsant drug sodium channel target",
    "potassium channel voltage sensor domain",
    "ion channel selectivity filter pore",
    "epilepsy genetic mutation channelopathy",
]


def _build_index(config: dict) -> tuple:
    """Embed all chunks and build the FAISS index. Returns (store, embedder)."""
    processed_dir = Path(config["paths"]["processed"])
    embeddings_dir = Path(config["paths"]["embeddings"])
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    chunks = json.loads((processed_dir / "chunks.json").read_text())
    logger.info("Loaded %d chunks for embedding", len(chunks))

    model_name: str = config["embedding"]["model"]
    embedder = Embedder(model_name=model_name)
    embedder.load()

    texts = [c["text"] for c in chunks]
    logger.info("Encoding %d chunks (batch_size=32)...", len(texts))
    embeddings = embedder.encode(texts, batch_size=32)
    logger.info("Encoded — shape %s", embeddings.shape)

    store = VectorStore(dim=embedder.hidden_size)
    metadata = [
        {
            "chunk_id": c["chunk_id"],
            "source": c["source"],
            "record_id": c["record_id"],
            "text": c["text"],
            **{k: v for k, v in c.get("metadata", {}).items()},
        }
        for c in chunks
    ]
    store.add(embeddings, metadata)
    store.save(embeddings_dir / "faiss_index")

    # Persist raw embeddings matrix for distribution analysis
    np.save(str(embeddings_dir / "chunk_embeddings.npy"), embeddings)

    return store, embedder


def run(config_path: str = "configs/pipeline.yaml") -> dict:
    """
    Execute Check 3 and save artifact to data/evaluation/stage_3/.

    Returns results dict with known-pair recall, cosine distribution stats,
    and nearest-neighbour examples for manual inspection.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    artifact_dir = Path(config["paths"]["evaluation"]) / "stage_3"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    embeddings_dir = Path(config["paths"]["embeddings"])
    index_path = embeddings_dir / "faiss_index"

    # Re-use a cached index if present to avoid re-embedding on every run
    if index_path.exists() and (index_path / "index.faiss").exists():
        logger.info("Loading cached FAISS index from %s", index_path)
        store = VectorStore.load(index_path)
        model_name: str = config["embedding"]["model"]
        embedder = Embedder(model_name=model_name)
        embedder.load()
    else:
        store, embedder = _build_index(config)

    k_retrieve = config["retrieval"]["k"]
    collapse_threshold: float = config["evaluation"]["embedding_collapse_cosine_threshold"]

    # --- Check 3a: Known-pair retrieval ---
    known_pair_results = []
    pair_failures = []
    for query, expected_gene in KNOWN_PAIRS:
        qvec = embedder.encode_query(query)
        hits = store.search(qvec, k=5)
        scores = [h["score"] for h in hits]

        # Check if expected gene appears in any retrieved chunk (text or metadata)
        found_in_top5 = any(
            expected_gene in h.get("text", "") or
            expected_gene in h.get("gene_name", "") or
            expected_gene in h.get("chunk_id", "")
            for h in hits
        )
        found_at_rank = next(
            (i + 1 for i, h in enumerate(hits)
             if expected_gene in h.get("text", "") or
                expected_gene in h.get("gene_name", "") or
                expected_gene in h.get("chunk_id", "")),
            None,
        )

        known_pair_results.append({
            "query": query,
            "expected_gene": expected_gene,
            "found_in_top5": found_in_top5,
            "rank": found_at_rank,
            "top5_scores": [round(s, 4) for s in scores],
            "top5_chunks": [
                {"chunk_id": h.get("chunk_id", ""), "score": round(h["score"], 4),
                 "text_snippet": h.get("text", "")[:120]}
                for h in hits
            ],
        })
        if not found_in_top5:
            pair_failures.append(f"{expected_gene} not in top-5 for: {query!r}")

    known_pair_recall = sum(1 for r in known_pair_results if r["found_in_top5"]) / len(KNOWN_PAIRS)

    # --- Check 3b: Cosine score distribution (collapse detection) ---
    # Sample all pairwise similarities between a random set of 100 chunk embeddings
    embeddings = np.load(str(embeddings_dir / "chunk_embeddings.npy"))
    rng = np.random.default_rng(42)
    sample_size = min(100, len(embeddings))
    idx = rng.choice(len(embeddings), size=sample_size, replace=False)
    sample = embeddings[idx]
    # All pairwise cosine sims (dot products of L2-normalised vectors)
    sims = (sample @ sample.T).ravel()
    # Exclude self-similarities (diagonal = 1.0)
    off_diag = sims[sims < 0.9999]

    cosine_mean = float(np.mean(off_diag))
    cosine_std = float(np.std(off_diag))
    cosine_min = float(np.min(off_diag))
    cosine_max = float(np.max(off_diag))
    cosine_p95 = float(np.percentile(off_diag, 95))

    # --- Check 3c: Nearest-neighbour spot check (informational) ---
    nn_examples = []
    for query in SEED_QUERIES[:10]:
        qvec = embedder.encode_query(query)
        hits = store.search(qvec, k=k_retrieve)
        nn_examples.append({
            "query": query,
            "top_k": [
                {"chunk_id": h.get("chunk_id", ""), "score": round(h["score"], 4),
                 "source": h.get("source", ""), "text_snippet": h.get("text", "")[:150]}
                for h in hits
            ],
        })

    # --- Flags ---
    flags = list(pair_failures)
    if cosine_mean > collapse_threshold:
        flags.append(
            f"Embedding collapse: mean cosine similarity {cosine_mean:.3f} > {collapse_threshold}"
        )

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": config["embedding"]["model"],
        "index_size": store._index.ntotal if store._index else 0,
        "known_pair_recall_at_5": round(known_pair_recall, 4),
        "known_pair_results": known_pair_results,
        "cosine_distribution": {
            "sample_size": sample_size,
            "mean": round(cosine_mean, 4),
            "std": round(cosine_std, 4),
            "min": round(cosine_min, 4),
            "max": round(cosine_max, 4),
            "p95": round(cosine_p95, 4),
            "collapse_threshold": collapse_threshold,
        },
        "nn_spot_check": nn_examples,
        "flags": flags,
        "passed": len(flags) == 0,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = artifact_dir / f"check_3_{timestamp}.json"
    artifact_path.write_text(json.dumps(results, indent=2))
    logger.info("Check 3 artifact saved to %s", artifact_path)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    results = run()

    print("\n=== CHECK 3 SUMMARY ===")
    print(f"  Model               : {results['model']}")
    print(f"  Index size          : {results['index_size']} vectors")
    print(f"  Known-pair recall@5 : {results['known_pair_recall_at_5']:.0%}")
    print()
    print("  Known pairs:")
    for r in results["known_pair_results"]:
        status = f"rank {r['rank']}" if r["found_in_top5"] else "MISS"
        print(f"    [{status}] {r['expected_gene']:7s} ← {r['query'][:55]}")
    print()
    cd = results["cosine_distribution"]
    print(f"  Cosine sim (off-diag, n={cd['sample_size']}²):")
    print(f"    mean={cd['mean']:.3f}  std={cd['std']:.3f}  "
          f"min={cd['min']:.3f}  max={cd['max']:.3f}  p95={cd['p95']:.3f}")
    print()
    if results["flags"]:
        print(f"RESULT: FAIL ({len(results['flags'])} flag(s))")
        for flag in results["flags"]:
            print(f"  [FLAG] {flag}")
    else:
        print("RESULT: PASS — no flags raised")
