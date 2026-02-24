"""
app/utils/chunker.py

Chunking strategy for feeding long papers to the LLM without truncation.

Problem: LLM context windows are limited. A 20-page paper is ~40,000 chars.
We can't send it all at once, but we also can't just take the first 6000 chars
(that throws away 85% of the paper).

Solution — hierarchical chunking:
  1. If the text is short enough (< DIRECT_THRESHOLD), send it directly.
  2. Otherwise split into overlapping chunks, extract a structured JSON from
     each chunk, then MERGE the results intelligently:
     - Text fields: take the longest non-empty value across chunks
     - Equations: deduplicate by LaTeX similarity
     - Figures: deduplicate by caption similarity

This means every page of the paper contributes to the final extraction.

Chunk size is chosen to fit comfortably within llama-3.3-70b's 32k context:
  - ~6000 chars per chunk ≈ ~1500 tokens (well within limits)
  - 400-char overlap ensures section boundaries are not cut mid-sentence
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
DIRECT_THRESHOLD = 5_000    # chars — send directly if shorter than this
CHUNK_SIZE       = 6_000    # chars per chunk
OVERLAP          = 500      # chars overlap between consecutive chunks
MAX_CHUNKS       = 12       # hard cap — never send more than 12 LLM calls


@dataclass
class Chunk:
    index: int
    text:  str
    start: int   # byte offset in original text
    end:   int


def chunk_text(text: str) -> list[Chunk]:
    """
    Split text into overlapping chunks.

    If text fits in one chunk, returns a single-element list.
    Tries to split on paragraph boundaries (double newline) to avoid
    cutting in the middle of a sentence.
    """
    if len(text) <= DIRECT_THRESHOLD:
        return [Chunk(index=0, text=text, start=0, end=len(text))]

    chunks: list[Chunk] = []
    start = 0
    idx   = 0

    while start < len(text) and idx < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, len(text))

        # Try to end on a paragraph boundary
        if end < len(text):
            para_break = text.rfind("\n\n", start + CHUNK_SIZE // 2, end)
            if para_break != -1:
                end = para_break + 2
            else:
                # Fall back to sentence boundary
                sent_break = text.rfind(". ", start + CHUNK_SIZE // 2, end)
                if sent_break != -1:
                    end = sent_break + 2

        chunks.append(Chunk(index=idx, text=text[start:end], start=start, end=end))

        # Next chunk starts OVERLAP chars before the end of this one
        start = max(end - OVERLAP, end - CHUNK_SIZE // 4)
        idx  += 1

    logger.info(
        "Chunked %d chars into %d chunks (max %d chars each).",
        len(text), len(chunks), CHUNK_SIZE,
    )
    return chunks


def is_single_chunk(text: str) -> bool:
    return len(text) <= DIRECT_THRESHOLD


# ── Section merging ──────────────────────────────────────────────────────────

def merge_section_dicts(dicts: list[dict]) -> dict:
    """
    Merge multiple JSON section extractions into one coherent result.

    For each text field, pick the longest non-empty value.
    This handles the case where the abstract is only in chunk 0
    but the conclusion is only in chunk 5.
    """
    if not dicts:
        return {}
    if len(dicts) == 1:
        return dicts[0]

    TEXT_FIELDS = [
        "title", "authors", "abstract", "introduction",
        "methodology", "results", "conclusion", "limitations", "future_work",
    ]

    merged: dict = {}
    for field in TEXT_FIELDS:
        candidates = [
            d.get(field, "")
            for d in dicts
            if isinstance(d.get(field), str)
            and d.get(field, "").strip()
            and d.get(field, "").strip().lower() != "not found"
        ]
        if not candidates:
            merged[field] = "Not found"
        else:
            # Pick the longest (most detailed) value
            merged[field] = max(candidates, key=len)

    return merged
