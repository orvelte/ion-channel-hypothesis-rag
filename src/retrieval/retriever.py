"""
Retriever — Stage 4.

The retriever is a thin orchestration layer between the embedder and vector store.
Keeping it separate from both makes it easy to swap in BM25 or a cross-encoder
re-ranker at Stage 4 without touching the embedding or index code.
Hybrid retrieval (BM25 + semantic) is an optional Stage 4 extension, not part
of the baseline — baseline evaluation must run first to establish a retrieval
failure rate worth improving.
"""

import logging

from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, k: int = 7) -> None:
        self.embedder = embedder
        self.store = store
        self.k = k

    def retrieve(self, query: str) -> list[dict]:
        """
        Encode query and return top-k chunks with scores.

        Each returned dict contains chunk text, metadata, and cosine score.
        Raises RuntimeError if the vector store is empty.
        """
        qvec = self.embedder.encode_query(query)
        return self.store.search(qvec, k=self.k)

    def retrieve_batch(self, queries: list[str]) -> list[list[dict]]:
        """Retrieve for multiple queries. Returns a list parallel to queries."""
        return [self.retrieve(q) for q in queries]
