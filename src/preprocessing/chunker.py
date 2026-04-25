"""
Text chunking — Stage 2.

Sentence-aware chunking with overlap is used instead of hard token splits
because mid-sentence breaks destroy the syntactic context that transformers
use to form good contextualised embeddings. The 150–250 token target window
is calibrated to PubMedBERT's 512-token limit, leaving budget for the query
token overhead during retrieval.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Split on sentence-ending punctuation followed by whitespace and an uppercase letter.
# Simple enough for well-formed scientific abstracts; abbreviations like "et al."
# and "Fig." will occasionally cause incorrect splits but don't materially affect
# retrieval quality at this corpus scale.
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# Lazy-loaded tokenizer singleton — avoids import overhead at module load time.
# bert-base-uncased WordPiece is used for BPE token counting; it has the same
# vocabulary structure as BioBERT, so counts are consistent with the embedder.
_tokenizer: Any = None


def _get_tokenizer() -> Any:
    global _tokenizer
    if _tokenizer is None:
        try:
            from tokenizers import Tokenizer
            _tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        except Exception as e:
            logger.warning("Could not load tokenizer (%s); falling back to word count", e)
            _tokenizer = False  # sentinel: don't retry
    return _tokenizer


@dataclass
class Chunk:
    chunk_id: str          # "{source}_{record_id}_{chunk_index}"
    source: str            # "pubmed" | "uniprot" | "chembl" | "pdb"
    record_id: str
    text: str
    token_count: int
    metadata: dict         # source-specific fields (pmid, uniprot_id, etc.)


def _count_tokens(text: str) -> int:
    """Count BPE tokens, minus [CLS]/[SEP], with word-count fallback."""
    tok = _get_tokenizer()
    if tok:
        # subtract 2 for the [CLS] and [SEP] tokens added by the encoder
        return max(0, len(tok.encode(text).ids) - 2)
    return len(text.split())


def _split_sentences(text: str) -> list:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    record_id: str,
    source: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    metadata: Optional[dict] = None,
    min_chunk_tokens: int = 0,
) -> list:
    """
    Split text into overlapping chunks respecting sentence boundaries.

    Raises ValueError if text is empty after stripping whitespace.
    Logs a warning (does not raise) for any chunk outside the 150–250 token window —
    these will be flagged by the Check 2 evaluation script.

    If min_chunk_tokens > 0, a short trailing chunk (below the minimum) is merged
    into the preceding chunk rather than emitted separately. Single-chunk records
    that are inherently short are left as-is (no preceding chunk to absorb them).
    """
    text = text.strip()
    if not text:
        raise ValueError(f"Empty text for record {record_id!r}")

    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    sent_lens = [_count_tokens(s) for s in sentences]

    chunks = []
    chunk_idx = 0
    start = 0

    while start < len(sentences):
        # Greedily accumulate sentences until the BPE token budget is exceeded
        end = start
        total = 0
        while end < len(sentences):
            if total + sent_lens[end] > chunk_size_tokens and end > start:
                break
            total += sent_lens[end]
            end += 1

        chunk_text_str = " ".join(sentences[start:end])
        actual_tokens = _count_tokens(chunk_text_str)

        if actual_tokens < 150 or actual_tokens > 250:
            logger.warning(
                "Chunk %s_%s_%d: %d tokens outside 150–250 window",
                source, record_id, chunk_idx, actual_tokens,
            )

        chunks.append(Chunk(
            chunk_id=f"{source}_{record_id}_{chunk_idx}",
            source=source,
            record_id=record_id,
            text=chunk_text_str,
            token_count=actual_tokens,
            metadata=metadata or {},
        ))
        chunk_idx += 1

        if end >= len(sentences):
            break

        # Find the next chunk start by rewinding from `end` until we accumulate
        # enough tokens to provide the desired overlap with the current chunk.
        overlap_accum = 0
        next_start = end
        for k in range(end - 1, start, -1):
            overlap_accum += sent_lens[k]
            if overlap_accum >= chunk_overlap_tokens:
                next_start = k
                break

        start = max(next_start, start + 1)  # always advance to prevent infinite loop

    # Merge a short trailing chunk into the previous chunk so we don't emit
    # tiny slivers that contribute nothing to retrieval quality.
    if min_chunk_tokens > 0 and len(chunks) >= 2:
        last = chunks[-1]
        if last.token_count < min_chunk_tokens:
            prev = chunks[-2]
            merged_text = prev.text + " " + last.text
            chunks[-2] = Chunk(
                chunk_id=prev.chunk_id,
                source=prev.source,
                record_id=prev.record_id,
                text=merged_text,
                token_count=_count_tokens(merged_text),
                metadata=prev.metadata,
            )
            chunks.pop()

    return chunks
