"""
FAISS flat vector store — Stage 3.

A flat index (IndexFlatIP) is used instead of an approximate index (IVF, HNSW)
because at ~600 chunks this corpus is small enough that exact search is both
fast and correct. Approximate indexes introduce recall error that would confound
the Stage 4 evaluation — we want to know if the *embeddings* are good, not
whether the ANN search is tuned correctly.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around a FAISS IndexFlatIP with metadata side-table."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index = None
        self._metadata: list = []   # parallel to FAISS internal IDs

    def add(self, embeddings: np.ndarray, metadata: list, config: Optional[dict] = None) -> None:
        """
        Add L2-normalised embeddings with associated metadata.

        Raises ValueError if embeddings.shape[1] != self.dim, or if
        len(embeddings) != len(metadata).
        """
        assert config is None or not config.get("retrieval", {}).get("compound_index_enabled", False), (
            "Compound fingerprint indexing is disabled for the text-only baseline. "
            "See CLAUDE.md scope boundaries before enabling."
        )
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} != index dim {self.dim}"
            )
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"embeddings length {len(embeddings)} != metadata length {len(metadata)}"
            )

        import faiss
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dim)

        self._index.add(embeddings.astype(np.float32))
        self._metadata.extend(metadata)
        logger.info("VectorStore: added %d vectors (total %d)", len(embeddings), self._index.ntotal)

    def search(self, query_vector: np.ndarray, k: int) -> list:
        """
        Return top-k chunks as list of metadata dicts, each with added 'score' key.
        Raises RuntimeError if index is empty.
        """
        if self._index is None or self._index.ntotal == 0:
            raise RuntimeError("VectorStore is empty — call add() before search()")

        q = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self._metadata[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results

    def save(self, path: Path) -> None:
        """Persist index and metadata to disk for reuse across runs."""
        import faiss
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        (path / "metadata.json").write_text(json.dumps(self._metadata, indent=2))
        logger.info("VectorStore saved to %s (%d vectors)", path, self._index.ntotal)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """Restore a previously saved index."""
        import faiss
        path = Path(path)
        index = faiss.read_index(str(path / "index.faiss"))
        metadata = json.loads((path / "metadata.json").read_text())
        store = cls(dim=index.d)
        store._index = index
        store._metadata = metadata
        logger.info("VectorStore loaded from %s (%d vectors)", path, index.ntotal)
        return store
