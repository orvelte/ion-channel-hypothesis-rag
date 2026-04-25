"""
Check 6 — Output evaluation checkpoint.

Run after Stage 6. Measures structured output parse success rate, factual grounding
(each mechanism claim traced to a specific retrieved chunk), mechanistic plausibility,
and novelty vs verbatim retrieval.

Flag conditions:
  - Parse success rate <90%
  - >20% of outputs have no traceable grounding in retrieved context

Preparatory functions (run before Stage 6 inference):
  - verify_data_gaps(): confirms whether model-reported data gaps are genuine corpus
    absences or retrieval failures — critical for distinguishing the two at Check 7
  - trace_cited_chunks(): resolves cited chunk IDs to full records and flags hallucinated
    citations (IDs that don't exist in the corpus) for every Stage 6 output
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from src.generation.output_parser import HypothesisOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spot Check 4 — citation trace data models
# ---------------------------------------------------------------------------

class ResolvedChunk(BaseModel):
    chunk_id: str
    source: str       # pubmed | uniprot | pdb
    record_id: str
    text: str
    token_count: int
    metadata: dict


class CitationTrace(BaseModel):
    query_id: str
    cited_chunk_ids: list[str]
    resolved_chunks: list[ResolvedChunk]
    unresolvable_ids: list[str]   # cited IDs absent from chunks.json → hallucination signal
    grounding_rate: float          # resolved / total cited; <0.8 is flagged


# ---------------------------------------------------------------------------
# Spot Check 1 — data gap verification
# ---------------------------------------------------------------------------

def verify_data_gaps(gap_terms: list[str], chunks_path: str) -> dict:
    """
    For each term, search all chunk texts (case-insensitive) and return match stats.

    Returns dict keyed by term with: match_count, matching_chunk_ids, sources.
    Zero matches → confirmed data gap.
    Matches in sources not represented in retrieved context → retrieval failure
    masquerading as a data gap.
    """
    chunks: list[dict] = json.loads(Path(chunks_path).read_text())

    results = {}
    for term in gap_terms:
        term_lower = term.lower()
        matches = [c for c in chunks if term_lower in c["text"].lower()]
        results[term] = {
            "match_count": len(matches),
            "matching_chunk_ids": [c["chunk_id"] for c in matches],
            "sources": sorted({c["source"] for c in matches}),
            # 'chunk_type' field does not exist in this schema; 'source' is the closest proxy
            "chunk_types": sorted({c["source"] for c in matches}),
        }
    return results


# ---------------------------------------------------------------------------
# Spot Check 4 — chunk ID tracing
# ---------------------------------------------------------------------------

def trace_cited_chunks(
    output: HypothesisOutput,
    chunks_path: str,
    query_id: str,
) -> CitationTrace:
    """
    Resolve every chunk ID cited by the model to its full record.

    Unresolvable IDs (not present in chunks.json) are a hallucination signal —
    the model fabricated a chunk ID that does not exist in the corpus.
    """
    chunk_lookup: dict[str, dict] = {
        c["chunk_id"]: c
        for c in json.loads(Path(chunks_path).read_text())
    }

    resolved = []
    unresolvable = []
    for cid in output.supporting_chunk_ids:
        if cid in chunk_lookup:
            c = chunk_lookup[cid]
            resolved.append(ResolvedChunk(
                chunk_id=c["chunk_id"],
                source=c["source"],
                record_id=c["record_id"],
                text=c["text"],
                token_count=c["token_count"],
                metadata=c.get("metadata", {}),
            ))
        else:
            unresolvable.append(cid)

    n_cited = len(output.supporting_chunk_ids)
    grounding_rate = len(resolved) / n_cited if n_cited > 0 else 0.0

    return CitationTrace(
        query_id=query_id,
        cited_chunk_ids=output.supporting_chunk_ids,
        resolved_chunks=resolved,
        unresolvable_ids=unresolvable,
        grounding_rate=round(grounding_rate, 4),
    )


def flag_citation_trace(trace: CitationTrace) -> list[str]:
    """
    Return a list of flag strings for the given trace. Empty list means no flags.
    """
    flags = []
    if trace.unresolvable_ids:
        flags.append(
            f"hallucinated_citation: {len(trace.unresolvable_ids)} cited ID(s) "
            f"not in corpus — {trace.unresolvable_ids}"
        )
    if trace.grounding_rate < 0.8:
        flags.append(
            f"low_grounding_rate: {trace.grounding_rate:.0%} of cited chunks resolved "
            f"({len(trace.resolved_chunks)}/{len(trace.cited_chunk_ids)})"
        )
    sources = {rc.source for rc in trace.resolved_chunks}
    if len(sources) == 1 and trace.resolved_chunks:
        flags.append(
            f"single_modality: all resolved chunks are from '{next(iter(sources))}' only"
        )
    return flags


# ---------------------------------------------------------------------------
# Novelty helpers
# ---------------------------------------------------------------------------

def _ngrams(text: str, n: int = 5) -> set[tuple]:
    words = text.lower().split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def verbatim_overlap(hypothesis: str, context_chunks: list[dict], n: int = 5) -> float:
    """
    Fraction of hypothesis n-grams that appear verbatim in any retrieved chunk.
    0.0 = fully novel; 1.0 = entirely copied from context.
    Returns 0.0 if the hypothesis has fewer than n words.
    """
    hyp_grams = _ngrams(hypothesis, n)
    if not hyp_grams:
        return 0.0
    corpus_grams: set[tuple] = set()
    for c in context_chunks:
        corpus_grams |= _ngrams(c.get("text", ""), n)
    overlap = hyp_grams & corpus_grams
    return round(len(overlap) / len(hyp_grams), 4)


# ---------------------------------------------------------------------------
# Plausibility judge
# ---------------------------------------------------------------------------

_PLAUSIBILITY_SYSTEM = (
    "You are a senior pharmacologist specialising in voltage-gated ion channels. "
    "You evaluate whether a proposed drug-channel mechanism is pharmacologically sound."
)

_PLAUSIBILITY_TMPL = (
    "Hypothesis: {hypothesis}\n"
    "Mechanism type: {mechanism_type}\n"
    "Expected target: {expected_target}\n\n"
    "Is this mechanism pharmacologically plausible for the stated target and drug class? "
    "Reply with exactly one of: PLAUSIBLE / IMPLAUSIBLE / UNCERTAIN — then one sentence of reasoning."
)


def _judge_plausibility(llm, hypothesis: str, mechanism_type: Optional[str],
                        expected_target: Optional[str]) -> dict:
    prompt = _PLAUSIBILITY_TMPL.format(
        hypothesis=hypothesis[:600],
        mechanism_type=mechanism_type or "unspecified",
        expected_target=expected_target or "unspecified",
    )
    try:
        raw = llm.complete(_PLAUSIBILITY_SYSTEM, prompt).strip()
        verdict = "PLAUSIBLE" if raw.upper().startswith("PLAUSIBLE") else (
            "IMPLAUSIBLE" if raw.upper().startswith("IMPLAUSIBLE") else "UNCERTAIN"
        )
        return {"verdict": verdict, "reasoning": raw, "error": None}
    except Exception as exc:
        return {"verdict": "UNCERTAIN", "reasoning": "", "error": str(exc)}


# ---------------------------------------------------------------------------
# Stage 6 inference and evaluation
# ---------------------------------------------------------------------------

def run(config_path: str = "configs/pipeline.yaml") -> dict:
    """
    Execute Check 6: run LLM inference on all 5 benchmark queries then evaluate.

    Uses the CoT prompt variant throughout — Check 5 showed CoT achieves higher
    citation rates (86% vs 57%) with no parse-rate penalty, making it the better
    choice for grounding-critical hypothesis generation.

    Saves three artifacts:
      check_6_{ts}.json   — full per-query results + aggregate metrics
      citation_traces.json — overwritten with Stage 6 traces (smoke test wrote Check 5 traces)
      citation_flags.json  — overwritten with Stage 6 flags
    """
    from src.generation.llm_client import LLMClient
    from src.generation.output_parser import parse as parse_output
    from src.generation.prompt_templates import build_prompt
    from src.retrieval.embedder import Embedder
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.vector_store import VectorStore

    with open(config_path) as f:
        config = yaml.safe_load(f)

    artifact_dir = Path(config["paths"]["evaluation"]) / "stage_6"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = str(Path(config["paths"]["processed"]) / "chunks.json")

    # Build retriever
    chunks: list[dict] = json.loads(Path(chunks_path).read_text())
    store = VectorStore.load(Path(config["paths"]["embeddings"]) / "faiss_index")
    embedder = Embedder(model_name=config["embedding"]["model"])
    embedder.load()
    retriever = HybridRetriever(chunks=chunks, store=store, embedder=embedder, config=config)
    k = config["retrieval"]["k"]

    llm = LLMClient.from_config(config)
    benchmark_queries = config["benchmark"]["queries"]

    parse_threshold: float = config["evaluation"]["parse_success_rate_threshold"]
    ungrounded_threshold: float = config["evaluation"]["max_ungrounded_output_rate"]

    per_query = []
    all_traces: list[dict] = []
    all_flags: list[dict] = []

    for bq in benchmark_queries:
        qid = bq["id"]
        query_text = bq["text"]
        expected_target = bq.get("expected_target")
        is_control = bq.get("control", False)
        control_type = bq.get("control_type")
        logger.info("Running inference: %s — %s", qid, query_text[:60])

        # Retrieve
        hits = retriever.retrieve(query_text, k=k)
        context_chunks = [
            {"chunk_id": h.chunk_id, "text": h.text, "source": h.source}
            for h in hits
        ]
        retrieved_ids = [h.chunk_id for h in hits]

        # Infer
        system_prompt, user_prompt = build_prompt(query_text, context_chunks, variant="cot")
        try:
            raw = llm.complete(system_prompt, user_prompt)
            call_error = None
        except Exception as exc:
            raw = ""
            call_error = str(exc)
            logger.warning("LLM call failed for %s: %s", qid, exc)

        # Parse
        parsed, parse_error = parse_output(raw)

        # Citation trace
        trace: Optional[CitationTrace] = None
        trace_flags: list[str] = []
        if parsed:
            trace = trace_cited_chunks(parsed, chunks_path, query_id=qid)
            trace_flags = flag_citation_trace(trace)
            all_traces.append(trace.model_dump())
            if trace_flags:
                all_flags.append({"query_id": qid, "flags": trace_flags})

        # Novelty
        novelty_overlap = None
        if parsed:
            novelty_overlap = verbatim_overlap(parsed.hypothesis, context_chunks)

        # Mechanistic plausibility judge
        plausibility = {"verdict": "UNCERTAIN", "reasoning": "", "error": "not_parsed"}
        if parsed and not call_error:
            plausibility = _judge_plausibility(llm, parsed.hypothesis,
                                               parsed.mechanism_type, expected_target)

        has_grounding = bool(parsed and trace and trace.grounding_rate > 0)

        per_query.append({
            "query_id": qid,
            "query": query_text,
            "expected_target": expected_target,
            "is_control": is_control,
            "control_type": control_type,
            "retrieved_chunk_ids": retrieved_ids,
            "call_error": call_error,
            "parse_error": parse_error,
            "parsed": parsed.model_dump() if parsed else None,
            "raw_response": raw,
            "grounding_rate": trace.grounding_rate if trace else None,
            "unresolvable_ids": trace.unresolvable_ids if trace else [],
            "citation_flags": trace_flags,
            "verbatim_overlap_5gram": novelty_overlap,
            "plausibility": plausibility,
            "has_grounding": has_grounding,
        })

    # Aggregate metrics (exclude control queries from thresholds)
    non_control = [r for r in per_query if not r["is_control"]]
    n_total = len(non_control)
    n_parsed = sum(1 for r in non_control if r["parsed"] is not None)
    n_grounded = sum(1 for r in non_control if r["has_grounding"])
    n_call_errors = sum(1 for r in non_control if r["call_error"])

    parse_success_rate = n_parsed / (n_total - n_call_errors) if (n_total - n_call_errors) > 0 else 0.0
    ungrounded_rate = 1 - (n_grounded / n_parsed) if n_parsed > 0 else 1.0

    flags = []
    if parse_success_rate < parse_threshold:
        flags.append(
            f"parse_success_rate {parse_success_rate:.0%} < threshold {parse_threshold:.0%}"
        )
    if ungrounded_rate > ungrounded_threshold:
        flags.append(
            f"ungrounded_rate {ungrounded_rate:.0%} > threshold {ungrounded_threshold:.0%} "
            f"({n_parsed - n_grounded}/{n_parsed} outputs have no grounding)"
        )

    results = {
        "timestamp": datetime.now().isoformat(),
        "embedding_model": config["embedding"]["model"],
        "llm_model": config["llm"]["model"],
        "prompt_variant": "cot",
        "retrieval_k": k,
        "n_benchmark_queries": len(benchmark_queries),
        "n_control_queries": sum(1 for r in per_query if r["is_control"]),
        "aggregate": {
            "parse_success_rate": round(parse_success_rate, 4),
            "ungrounded_rate": round(ungrounded_rate, 4),
            "n_parsed": n_parsed,
            "n_grounded": n_grounded,
            "n_call_errors": n_call_errors,
        },
        "per_query": per_query,
        "flags": flags,
        "passed": len(flags) == 0,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_path = artifact_dir / f"check_6_{timestamp}.json"
    artifact_path.write_text(json.dumps(results, indent=2))
    logger.info("Check 6 artifact saved to %s", artifact_path)

    # Overwrite Stage 6 citation traces (smoke test wrote Check 5 traces)
    (artifact_dir / "citation_traces.json").write_text(json.dumps(all_traces, indent=2))
    (artifact_dir / "citation_flags.json").write_text(json.dumps(all_flags, indent=2))

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    results = run()

    agg = results["aggregate"]
    print("\n=== CHECK 6 SUMMARY ===")
    print(f"  LLM model          : {results['llm_model']} (variant: {results['prompt_variant']})")
    print(f"  Benchmark queries  : {results['n_benchmark_queries']} "
          f"({results['n_control_queries']} control)")
    print()
    print(f"  Parse success rate : {agg['parse_success_rate']:.0%}  "
          f"({agg['n_parsed']}/{results['n_benchmark_queries'] - results['n_control_queries']} non-control)")
    print(f"  Ungrounded rate    : {agg['ungrounded_rate']:.0%}")
    print(f"  Call errors        : {agg['n_call_errors']}")
    print()
    print("  Per-query:")
    for r in results["per_query"]:
        parsed_ok = "PARSED" if r["parsed"] else "FAIL  "
        grounded = f"grnd={r['grounding_rate']:.0%}" if r["grounding_rate"] is not None else "grnd=N/A"
        novelty = f"novel={1-r['verbatim_overlap_5gram']:.0%}" if r["verbatim_overlap_5gram"] is not None else "novel=N/A"
        plaus = r["plausibility"]["verdict"] if r["plausibility"] else "N/A"
        ctrl = " [CONTROL]" if r["is_control"] else ""
        print(f"    [{r['query_id']}]{ctrl} {parsed_ok} {grounded}  {novelty}  plaus={plaus}")
        print(f"          {r['query'][:65]}")
        if r["citation_flags"]:
            for flag in r["citation_flags"]:
                print(f"          [FLAG] {flag}")
    print()
    if results["flags"]:
        print(f"RESULT: FAIL ({len(results['flags'])} flag(s))")
        for flag in results["flags"]:
            print(f"  [FLAG] {flag}")
    else:
        print("RESULT: PASS — no flags raised")
