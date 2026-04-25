"""
Query and chunk embedding — Stage 3.

PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) is used
instead of a general-purpose encoder because it was pre-trained on biomedical abstracts,
giving it better representation of pharmacological terminology, gene names, and assay
language. Mean-pooling over the last hidden state is used rather than [CLS] pooling
because it better captures full-span semantics for the long, complex sentences common
in biology literature.
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class Embedder:
    """Wraps a HuggingFace transformer for encoding text chunks and queries."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        """Load tokeniser and model. Called once before encoding."""
        from transformers import AutoTokenizer, AutoModel
        logger.info("Loading tokenizer and model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # use_safetensors=True avoids the torch.load CVE-2025-32434 restriction
        # that applies to torch < 2.6 loading .bin weight files.
        self._model = AutoModel.from_pretrained(self.model_name, use_safetensors=True)
        self._model.eval()
        self._model.to(self.device)
        logger.info("Model loaded on %s (hidden_size=%d)", self.device, self._model.config.hidden_size)

    @property
    def hidden_size(self) -> int:
        if self._model is None:
            raise RuntimeError("Call load() before accessing hidden_size")
        return self._model.config.hidden_size

    def encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of strings into L2-normalised float32 embeddings.

        Returns shape (len(texts), hidden_dim). Raises RuntimeError if model not loaded.
        L2 normalisation means cosine similarity reduces to dot product, which FAISS
        IndexFlatIP can exploit without extra computation.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call load() before encode()")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                out = self._model(**inputs)

            # Mean-pool over token dimension, masking padding tokens
            token_embeddings = out.last_hidden_state  # (batch, seq_len, hidden)
            attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (token_embeddings * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed / counts  # (batch, hidden)

            embeddings = mean_pooled.cpu().float().numpy()

            # L2-normalise so FAISS IndexFlatIP gives cosine similarity directly
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms < 1e-9, 1.0, norms)
            embeddings = embeddings / norms

            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings).astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns shape (hidden_dim,)."""
        return self.encode([query])[0]
