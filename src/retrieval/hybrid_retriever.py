"""
Hybrid BM25 + dense retriever with Reciprocal Rank Fusion — Stage 4 extension.

BM25 exact-term matching complements the dense FAISS retriever where BERT
anisotropy causes low-frequency terms (specific gene names like SCN3A, drug
names like mexiletine) to be outranked by high-frequency sodium channel content.
RRF fuses both ranked lists without requiring score calibration across the two
retrieval paradigms — a crucial property because BM25 and cosine scores live
on incomparable scales.

No stemmer is used on purpose: gene names (SCN3A, SCN8A) and drug names
(mexiletine, lidocaine) must match exactly. A stemmer would corrupt them
(e.g. "lidocaine" → "lidocain" fails BM25 lookup against the corpus token).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from rank_bm25 import BM25Okapi

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> list[str]:
    """Lowercase, split on whitespace and punctuation. No stemming."""
    return re.split(r"[\s\W]+", text.lower())


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    score: float          # RRF score (higher is better)
    bm25_rank: Optional[int] = None    # 1-based; None if not in BM25 top-2k
    dense_rank: Optional[int] = None   # 1-based; None if not in dense top-2k
    rrf_score: float = 0.0
    metadata: dict = field(default_factory=dict)


class HybridRetriever:
    """
    Combines FAISS dense retrieval and BM25 sparse retrieval via Reciprocal
    Rank Fusion (RRF). The BM25 index is built in-memory at construction time
    from the same chunk list used to build the FAISS index — no separate
    persistence needed at this data scale (562 chunks).
    """

    def __init__(
        self,
        chunks: list[dict],
        store: VectorStore,
        embedder: Embedder,
        config: dict,
    ) -> None:
        retrieval_cfg = config.get("retrieval", {})
        self.rrf_k: int = retrieval_cfg.get("rrf_k", 60)
        self.bm25_weight: float = retrieval_cfg.get("bm25_weight", 1.0)
        self.dense_weight: float = retrieval_cfg.get("dense_weight", 1.0)
        self.top_k_mult: int = retrieval_cfg.get("bm25_top_k_multiplier", 2)

        self._chunks = chunks          # original chunk dicts, parallel to BM25 index
        self._store = store
        self._embedder = embedder

        # Build BM25 index over tokenised chunk texts
        tokenised = [_tokenise(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenised)
        logger.info("HybridRetriever: BM25 index built over %d chunks", len(chunks))

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """
        Return top-k chunks fused by RRF from BM25 and dense retrieval.

        Each retriever contributes its top 2k candidates. Chunks absent from
        one list receive a penalty rank of 2k+1, preserving their RRF
        contribution without inflating the fused score for single-source hits.
        """
        candidate_k = k * self.top_k_mult

        # --- BM25 arm ---
        query_tokens = _tokenise(query)
        bm25_scores = self._bm25.get_scores(query_tokens)
        # argsort descending, take top candidate_k
        bm25_ranked_idx = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:candidate_k]
        bm25_rank_map: dict[str, int] = {
            self._chunks[idx]["chunk_id"]: rank + 1
            for rank, idx in enumerate(bm25_ranked_idx)
        }

        # --- Dense arm ---
        qvec = self._embedder.encode_query(query)
        dense_hits = self._store.search(qvec, k=candidate_k)
        dense_rank_map: dict[str, int] = {
            h["chunk_id"]: rank + 1 for rank, h in enumerate(dense_hits)
        }

        # --- RRF fusion over the union of both candidate sets ---
        all_chunk_ids = set(bm25_rank_map) | set(dense_rank_map)
        penalty = candidate_k + 1  # rank assigned to chunks absent from one arm

        rrf_scores: dict[str, float] = {}
        for cid in all_chunk_ids:
            r_bm25 = bm25_rank_map.get(cid, penalty)
            r_dense = dense_rank_map.get(cid, penalty)
            rrf_scores[cid] = (
                self.bm25_weight / (self.rrf_k + r_bm25)
                + self.dense_weight / (self.rrf_k + r_dense)
            )

        # Build a lookup from chunk_id → chunk dict
        chunk_lookup: dict[str, dict] = {c["chunk_id"]: c for c in self._chunks}

        top_k = sorted(all_chunk_ids, key=lambda cid: rrf_scores[cid], reverse=True)[:k]

        results = []
        for cid in top_k:
            chunk = chunk_lookup[cid]
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=chunk.get("text", ""),
                    source=chunk.get("source", ""),
                    score=rrf_scores[cid],
                    bm25_rank=bm25_rank_map.get(cid),
                    dense_rank=dense_rank_map.get(cid),
                    rrf_score=rrf_scores[cid],
                    metadata={
                        k: v for k, v in chunk.items()
                        if k not in ("text", "chunk_id", "source")
                    },
                )
            )
        return results
