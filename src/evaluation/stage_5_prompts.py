"""
Check 5 — Prompt ablation checkpoint.

Run after Stage 5. Compares CoT vs direct-answer prompt variants on 5 fixed
queries with identical retrieved context. Measures answer length, citation rate,
and parse success rate. Hallucination rate requires manual inspection — slots
are reserved in the artifact for manual annotation.

Identical context is critical: both variants receive the exact same top-k chunks
for a given query so that any difference in output quality is attributable to the
prompt structure, not retrieval variance.

Flag condition:
  - Either variant produces hypotheses with no traceable grounding in retrieved context.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from src.generation.llm_client import LLMClient
from src.generation.output_parser import parse as parse_output
from src.generation.prompt_templates import build_prompt
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# 5 fixed queries used for ablation — held constant across all prompt experiments.
# These are distinct from the Stage 7 benchmark queries so Stage 7 results remain
# held-out and uncontaminated by the ablation process.
ABLATION_QUERIES = [
    "How does lamotrigine modulate SCN1A gating kinetics?",
    "What structural features of Nav1.7 make it a target for neuropathic pain?",
    "Compare the selectivity profiles of carbamazepine and phenytoin on sodium channels.",
    "What is the mechanism of use-dependent block by local anaesthetics?",
    "How do SCN1A loss-of-function mutations lead to Dravet syndrome?",
]

VARIANTS = ("direct", "cot")


def _citation_rate(response_text: str, chunk_ids: list[str]) -> float:
    """Fraction of retrieved chunk_ids that appear literally in the response."""
    if not chunk_ids:
        return 0.0
    cited = sum(1 for cid in chunk_ids if cid in response_text)
    return cited / len(chunk_ids)


def _word_count(text: str) -> int:
    return len(text.split())


def _has_any_citation(response_text: str, chunk_ids: list[str]) -> bool:
    return any(cid in response_text for cid in chunk_ids)


def run(config_path: str = "configs/pipeline.yaml") -> dict:
    """
    Execute Check 5 and save artifact to data/evaluation/stage_5/.

    Runs both prompt variants on all 5 queries with identical retrieved context
    and saves raw outputs + metrics for manual hallucination inspection.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    artifact_dir = Path(config["paths"]["evaluation"]) / "stage_5"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Load hybrid retriever (re-uses cached FAISS index from Stage 3)
    processed_dir = Path(config["paths"]["processed"])
    embeddings_dir = Path(config["paths"]["embeddings"])
    index_path = embeddings_dir / "faiss_index"

    chunks: list[dict] = json.loads((processed_dir / "chunks.json").read_text())
    store = VectorStore.load(index_path)
    model_name: str = config["embedding"]["model"]
    embedder = Embedder(model_name=model_name)
    embedder.load()

    retriever = HybridRetriever(chunks=chunks, store=store, embedder=embedder, config=config)
    k = config["retrieval"]["k"]

    llm = LLMClient.from_config(config)

    query_results = []
    ungrounded_queries: dict[str, list[str]] = {"direct": [], "cot": []}

    for query in ABLATION_QUERIES:
        logger.info("Query: %s", query[:70])

        # Retrieve once — both variants get identical context
        hits = retriever.retrieve(query, k=k)
        context_chunks = [
            {"chunk_id": h.chunk_id, "text": h.text, "source": h.source}
            for h in hits
        ]
        chunk_ids = [h.chunk_id for h in hits]

        variant_outputs = {}
        for variant in VARIANTS:
            system_prompt, user_prompt = build_prompt(query, context_chunks, variant=variant)

            try:
                raw = llm.complete(system_prompt, user_prompt)
                call_error = None
            except Exception as exc:
                raw = ""
                call_error = str(exc)
                logger.warning("LLM call failed for variant=%s query=%r: %s", variant, query[:40], exc)

            parsed, parse_error = parse_output(raw)

            citation_rt = _citation_rate(raw, chunk_ids)
            has_citation = _has_any_citation(raw, chunk_ids)
            words = _word_count(raw)

            if not has_citation and not call_error:
                ungrounded_queries[variant].append(query)

            variant_outputs[variant] = {
                "raw_response": raw,
                "call_error": call_error,
                "parse_error": parse_error,
                "parsed": parsed.model_dump() if parsed else None,
                "word_count": words,
                "citation_rate": round(citation_rt, 4),
                "has_any_citation": has_citation,
                # Slots for manual annotation during Check 6
                "hallucination_manual": None,
                "hallucination_notes": "",
            }

        query_results.append({
            "query": query,
            "retrieved_chunk_ids": chunk_ids,
            "context_snippets": [
                {"chunk_id": c["chunk_id"], "source": c["source"],
                 "text_snippet": c["text"][:120]}
                for c in context_chunks
            ],
            "variants": variant_outputs,
        })

    # --- Aggregate metrics per variant ---
    variant_metrics: dict[str, dict] = {}
    for variant in VARIANTS:
        outputs = [r["variants"][variant] for r in query_results]
        n_calls = len(outputs)
        n_call_errors = sum(1 for o in outputs if o["call_error"])
        n_parse_ok = sum(1 for o in outputs if o["parsed"] is not None)
        n_ungrounded = len(ungrounded_queries[variant])

        variant_metrics[variant] = {
            "n_queries": n_calls,
            "call_errors": n_call_errors,
            "parse_success_rate": round(n_parse_ok / max(n_calls - n_call_errors, 1), 4),
            "mean_word_count": round(
                sum(o["word_count"] for o in outputs) / n_calls, 1
            ),
            "mean_citation_rate": round(
                sum(o["citation_rate"] for o in outputs) / n_calls, 4
            ),
            "ungrounded_count": n_ungrounded,
            "ungrounded_queries": ungrounded_queries[variant],
        }

    flags = []
    for variant in VARIANTS:
        m = variant_metrics[variant]
        # Only flag ungrounded outputs for successful calls (errors are separate)
        successful_calls = m["n_queries"] - m["call_errors"]
        if successful_calls > 0 and m["ungrounded_count"] == successful_calls:
            flags.append(
                f"Variant '{variant}': all {successful_calls} successful outputs "
                f"have no traceable grounding (zero citations)"
            )

    results = {
        "timestamp": datetime.now().isoformat(),
        "embedding_model": config["embedding"]["model"],
        "llm_model": config["llm"]["model"],
        "retrieval_k": k,
        "n_queries": len(ABLATION_QUERIES),
        "variants": VARIANTS,
        "variant_metrics": variant_metrics,
        "query_results": query_results,
        "flags": flags,
        "passed": len(flags) == 0,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = artifact_dir / f"check_5_{timestamp}.json"
    artifact_path.write_text(json.dumps(results, indent=2))
    logger.info("Check 5 artifact saved to %s", artifact_path)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    results = run()

    print("\n=== CHECK 5 SUMMARY ===")
    print(f"  LLM model      : {results['llm_model']}")
    print(f"  Retrieval k    : {results['retrieval_k']}")
    print(f"  Queries        : {results['n_queries']}")
    print()
    print(f"  {'Metric':<28} {'direct':>10} {'cot':>10}")
    print(f"  {'-'*50}")
    for metric, label in [
        ("call_errors",        "Call errors"),
        ("parse_success_rate", "Parse success rate"),
        ("mean_word_count",    "Mean word count"),
        ("mean_citation_rate", "Mean citation rate"),
        ("ungrounded_count",   "Ungrounded outputs"),
    ]:
        d = results["variant_metrics"]["direct"][metric]
        c = results["variant_metrics"]["cot"][metric]
        fmt_d = f"{d:.0%}" if isinstance(d, float) and metric.endswith("rate") else str(d)
        fmt_c = f"{c:.0%}" if isinstance(c, float) and metric.endswith("rate") else str(c)
        print(f"  {label:<28} {fmt_d:>10} {fmt_c:>10}")
    print()

    print("  Per-query citation presence:")
    for r in results["query_results"]:
        d_cite = "CITED" if r["variants"]["direct"]["has_any_citation"] else "NONE "
        c_cite = "CITED" if r["variants"]["cot"]["has_any_citation"] else "NONE "
        d_err = " [ERR]" if r["variants"]["direct"]["call_error"] else ""
        c_err = " [ERR]" if r["variants"]["cot"]["call_error"] else ""
        print(f"    direct={d_cite}{d_err}  cot={c_cite}{c_err}  {r['query'][:55]}")
    print()

    if results["flags"]:
        print(f"RESULT: FAIL ({len(results['flags'])} flag(s))")
        for flag in results["flags"]:
            print(f"  [FLAG] {flag}")
    else:
        print("RESULT: PASS — no flags raised")
