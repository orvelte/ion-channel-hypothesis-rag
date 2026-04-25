"""
Check 7 — End-to-end benchmark.

Runs 5 held-out queries through the full pipeline (retrieval → generation → parse).
For each output, failure is classified as one of:
  - retrieval_failure: correct answer not in top-k retrieved chunks
  - generation_failure: correct chunks retrieved but hypothesis incorrect/ungrounded
  - data_gap: information genuinely absent from the corpus
  - pass: retrieval and generation both succeeded

This taxonomy is the primary diagnostic artifact of the project. It tells us
where to invest — better retrieval, better prompts, or more data — and prevents
misattributing data gaps to model failure.

Failure classification algorithm (in priority order):
  1. If ALL primary expected_terms have zero corpus matches → data_gap
  2. If expected_terms exist in corpus but none appear in retrieved chunks → retrieval_failure
  3. If expected_terms retrieved but output implausible or ungrounded → generation_failure
  4. Otherwise → pass

Pre-run corpus coverage analysis (from Check 7 setup):
  h1 SCN9A/selectivity/SCN5A : present  → retrievable
  h2 persistent sodium/neuropathic : present → retrievable
  h3 DEKA/selectivity filter  : 0 matches → definitive data gap
  h4 SCN2A/autism             : present  → retrievable
  h5 spider (2)/Nav1.7 (0)   : partial  → Nav1.7 indexed as SCN9A; "spider" barely covered
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from src.evaluation.stage_6_outputs import (
    CitationTrace,
    flag_citation_trace,
    trace_cited_chunks,
    verbatim_overlap,
    _judge_plausibility,
)
from src.generation.llm_client import LLMClient
from src.generation.output_parser import HypothesisOutput, parse as parse_output
from src.generation.prompt_templates import build_prompt
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

FailureType = str  # "retrieval_failure" | "generation_failure" | "data_gap" | "pass"

# 5 held-out queries with domain-calibrated expected_terms for failure classification.
# primary_terms: terms that MUST appear in retrieved chunks for retrieval to be considered
#   successful; chosen against confirmed corpus vocabulary (Nav1.7→SCN9A, etc.)
# all_terms: full set searched for corpus coverage check
HELD_OUT_QUERIES: list[dict] = [
    {
        "query_id": "h1",
        "query": "What compounds have demonstrated selectivity for SCN9A over SCN5A?",
        "primary_terms": ["SCN9A", "SCN5A"],   # both in corpus
        "all_terms": ["SCN9A", "selectivity", "SCN5A"],
    },
    {
        "query_id": "h2",
        "query": "How does persistent sodium current contribute to neuropathic pain signalling?",
        "primary_terms": ["persistent", "neuropathic"],   # both in corpus
        "all_terms": ["persistent sodium", "neuropathic"],
    },
    {
        "query_id": "h3",
        "query": "Describe the role of the DEKA selectivity filter in ion channel pharmacology.",
        "primary_terms": ["DEKA"],              # 0 corpus matches — data gap
        "all_terms": ["DEKA", "selectivity filter"],
    },
    {
        "query_id": "h4",
        "query": "What is the evidence for SCN2A gain-of-function in autism spectrum disorder?",
        "primary_terms": ["SCN2A", "autism"],   # both in corpus
        "all_terms": ["SCN2A", "autism"],
    },
    {
        "query_id": "h5",
        "query": "How do spider toxins targeting Nav1.7 compare to small-molecule blockers?",
        "primary_terms": ["spider"],            # 2 corpus matches; Nav1.7 indexed as SCN9A
        "all_terms": ["spider", "Nav1.7"],
    },
]


def _corpus_coverage(terms: list[str], chunks: list[dict]) -> dict[str, int]:
    """Return match count per term across all chunk texts (case-insensitive)."""
    return {
        term: sum(1 for c in chunks if term.lower() in c["text"].lower())
        for term in terms
    }


def _retrieval_coverage(terms: list[str], retrieved: list[dict]) -> dict[str, bool]:
    """Return whether each term appears in any retrieved chunk."""
    return {
        term: any(term.lower() in c.get("text", "").lower() for c in retrieved)
        for term in terms
    }


def _classify(
    primary_terms: list[str],
    corpus_hits: dict[str, int],
    retrieval_hits: dict[str, bool],
    parsed: Optional[HypothesisOutput],
    grounding_rate: Optional[float],
    plausibility_verdict: str,
) -> tuple[FailureType, str]:
    """
    Apply the failure taxonomy. Returns (failure_type, reasoning).
    """
    # Step 1 — data gap: no primary term has any corpus presence
    all_zero = all(corpus_hits.get(t, 0) == 0 for t in primary_terms)
    if all_zero:
        return "data_gap", (
            f"Primary terms {[t for t in primary_terms if corpus_hits.get(t,0)==0]} "
            f"have zero corpus matches — information genuinely absent from corpus"
        )

    # Step 2 — retrieval failure: terms exist in corpus but none reached top-k
    any_in_corpus = any(corpus_hits.get(t, 0) > 0 for t in primary_terms)
    none_retrieved = not any(retrieval_hits.get(t, False) for t in primary_terms)
    if any_in_corpus and none_retrieved:
        return "retrieval_failure", (
            f"Terms {[t for t in primary_terms if corpus_hits.get(t,0)>0]} present in corpus "
            f"({[corpus_hits[t] for t in primary_terms if corpus_hits.get(t,0)>0]} chunks) "
            f"but none appeared in top-k retrieved set"
        )

    # Step 3 — generation failure: terms retrieved but output is implausible or ungrounded
    if parsed is None:
        return "generation_failure", "Parse failed — no structured output produced"
    if grounding_rate is not None and grounding_rate == 0:
        return "generation_failure", "Output has zero grounded citations"
    if plausibility_verdict == "IMPLAUSIBLE":
        return "generation_failure", (
            "Relevant chunks retrieved but mechanism judged pharmacologically IMPLAUSIBLE"
        )

    return "pass", "Relevant chunks retrieved; output parsed, grounded, and plausible"


def run(config_path: str = "configs/pipeline.yaml") -> dict:
    """
    Execute Check 7 end-to-end benchmark and save artifact to data/evaluation/stage_7/.

    Runs all 5 held-out queries through the full pipeline. Failure taxonomy is the
    primary output — aggregate pass/fail counts are secondary.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    artifact_dir = Path(config["paths"]["evaluation"]) / "stage_7"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = str(Path(config["paths"]["processed"]) / "chunks.json")

    chunks: list[dict] = json.loads(Path(chunks_path).read_text())
    store = VectorStore.load(Path(config["paths"]["embeddings"]) / "faiss_index")
    embedder = Embedder(model_name=config["embedding"]["model"])
    embedder.load()
    retriever = HybridRetriever(chunks=chunks, store=store, embedder=embedder, config=config)
    k = config["retrieval"]["k"]
    llm = LLMClient.from_config(config)

    per_query = []
    taxonomy_counts: dict[str, int] = {
        "pass": 0, "retrieval_failure": 0, "generation_failure": 0, "data_gap": 0
    }

    for item in HELD_OUT_QUERIES:
        qid = item["query_id"]
        query = item["query"]
        primary_terms = item["primary_terms"]
        all_terms = item["all_terms"]
        logger.info("Running %s: %s", qid, query[:65])

        # --- Corpus coverage ---
        corpus_hits = _corpus_coverage(all_terms, chunks)
        logger.info("  corpus coverage: %s", corpus_hits)

        # --- Retrieval ---
        hits = retriever.retrieve(query, k=k)
        context_chunks = [
            {"chunk_id": h.chunk_id, "text": h.text, "source": h.source}
            for h in hits
        ]
        retrieved_ids = [h.chunk_id for h in hits]
        retrieval_hits = _retrieval_coverage(primary_terms, context_chunks)
        logger.info("  retrieval coverage: %s", retrieval_hits)

        # --- Generation ---
        system_prompt, user_prompt = build_prompt(query, context_chunks, variant="cot")
        try:
            raw = llm.complete(system_prompt, user_prompt)
            call_error = None
        except Exception as exc:
            raw = ""
            call_error = str(exc)
            logger.warning("  LLM call failed: %s", exc)

        # --- Parse ---
        parsed, parse_error = parse_output(raw)

        # --- Citation trace ---
        trace: Optional[CitationTrace] = None
        trace_flags: list[str] = []
        if parsed:
            trace = trace_cited_chunks(parsed, chunks_path, query_id=qid)
            trace_flags = flag_citation_trace(trace)

        # --- Novelty ---
        novelty = verbatim_overlap(parsed.hypothesis, context_chunks) if parsed else None

        # --- Plausibility ---
        plausibility = {"verdict": "UNCERTAIN", "reasoning": "", "error": "not_parsed"}
        if parsed and not call_error:
            plausibility = _judge_plausibility(
                llm, parsed.hypothesis, parsed.mechanism_type, expected_target=None
            )

        # --- Classify ---
        failure_type, reasoning = _classify(
            primary_terms=primary_terms,
            corpus_hits=corpus_hits,
            retrieval_hits=retrieval_hits,
            parsed=parsed,
            grounding_rate=trace.grounding_rate if trace else None,
            plausibility_verdict=plausibility["verdict"],
        )
        taxonomy_counts[failure_type] += 1
        logger.info("  → %s: %s", failure_type.upper(), reasoning[:80])

        per_query.append({
            "query_id": qid,
            "query": query,
            "primary_terms": primary_terms,
            "corpus_coverage": corpus_hits,
            "retrieval_coverage": retrieval_hits,
            "retrieved_chunk_ids": retrieved_ids,
            "call_error": call_error,
            "parse_error": parse_error,
            "parsed": parsed.model_dump() if parsed else None,
            "grounding_rate": trace.grounding_rate if trace else None,
            "unresolvable_ids": trace.unresolvable_ids if trace else [],
            "citation_flags": trace_flags,
            "verbatim_overlap_5gram": novelty,
            "plausibility": plausibility,
            "failure_type": failure_type,
            "failure_reasoning": reasoning,
            "raw_response": raw,
        })

    results = {
        "timestamp": datetime.now().isoformat(),
        "embedding_model": config["embedding"]["model"],
        "llm_model": config["llm"]["model"],
        "prompt_variant": "cot",
        "retrieval_k": k,
        "n_queries": len(HELD_OUT_QUERIES),
        "taxonomy": taxonomy_counts,
        "per_query": per_query,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = artifact_dir / f"check_7_{timestamp}.json"
    artifact_path.write_text(json.dumps(results, indent=2))
    logger.info("Check 7 artifact saved to %s", artifact_path)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    results = run()

    print("\n=== CHECK 7 — END-TO-END BENCHMARK ===")
    print(f"  Model: {results['llm_model']}  k={results['retrieval_k']}")
    print()

    tax = results["taxonomy"]
    total = results["n_queries"]
    print("  Failure taxonomy:")
    print(f"    pass              : {tax['pass']}/{total}")
    print(f"    generation_failure: {tax['generation_failure']}/{total}")
    print(f"    retrieval_failure : {tax['retrieval_failure']}/{total}")
    print(f"    data_gap          : {tax['data_gap']}/{total}")
    print()

    print("  Per-query:")
    for r in results["per_query"]:
        cov = {t: v for t, v in r["corpus_coverage"].items()}
        ret = {t: ("✓" if v else "✗") for t, v in r["retrieval_coverage"].items()}
        parsed = "PARSED" if r["parsed"] else "FAIL  "
        grnd = f"{r['grounding_rate']:.0%}" if r["grounding_rate"] is not None else "N/A"
        plaus = r["plausibility"]["verdict"]
        ftype = r["failure_type"].upper()
        print(f"  [{r['query_id']}] {ftype:<20} {parsed}  grnd={grnd}  plaus={plaus}")
        print(f"       {r['query'][:72]}")
        print(f"       corpus={cov}  retrieved={ret}")
        if r["parsed"]:
            print(f"       hypothesis: {r['parsed']['hypothesis'][:100]}...")
        print(f"       {r['failure_reasoning'][:90]}")
        if r["citation_flags"]:
            for f in r["citation_flags"]:
                print(f"       [FLAG] {f}")
        print()
