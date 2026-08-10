"""Structure-aware chunker for C0.

Priority: ``page -> paragraph -> sentence fallback``.

Rules
-----
1. Page number is preserved on every chunk.
2. Paragraphs are blank-line-separated blocks; soft line wraps inside a
   paragraph (single ``\\n``) are joined with a space.
3. A paragraph that fits within ``CHUNK_MAX_CHARS`` becomes one chunk.
4. A larger paragraph is split by sentences; sentences are packed
   greedily toward ``CHUNK_TARGET_CHARS`` without exceeding the hard max.
5. If sentence splitting fails (no sentence boundaries), split at word
   boundaries; if a single token alone exceeds the max, hard-slice it.
6. ``chunk_index`` is 0-based and sequential across the whole document.
7. ``section_path`` is always ``None`` in C0 (no heading detection yet).

``token_count`` is an approximation (whitespace token count) — no
tokenizer dependency, as required.

Chunking is intentionally non-semantic: no embeddings, no overlap
windows, no heading awareness. Reliable retrieval + provenance first.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

from docpipe import config
from docpipe.normalize import normalize_content
from docpipe.parser import ParsedPage

logger = logging.getLogger(__name__)

# A sentence boundary is a sentence-ending punctuation mark followed by
# whitespace. Best-effort: splits "e.g." and "3.14" too — acceptable for
# C0 (documented limitation); the word/char fallback guarantees progress.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

# Soft wrap: a single newline between two non-blank lines inside a paragraph.
_WITHIN_LINE_RE = re.compile(r"(?<=\S)\n(?=\S)")


@dataclass(frozen=True)
class Chunk:
    """A searchable section of extracted document text."""

    chunk_index: int
    content: str  # extracted text for display/provenance
    normalized_content: str  # cleaned text for retrieval
    page_number: int | None
    section_path: str | None  # always None in C0
    token_count: int  # approximation: len(content.split())


def _split_paragraphs(page_text: str) -> list[str]:
    """Split page text into paragraphs (blank-line separated blocks).

    Soft line wraps inside a paragraph are joined with a single space —
    this is the one structural transformation applied to *content*.
    """
    paragraphs: list[str] = []
    for block in page_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        paragraphs.append(_WITHIN_LINE_RE.sub(" ", block))
    return paragraphs


def _split_sentences(paragraph: str) -> list[str]:
    """Best-effort sentence split; returns [paragraph] if it fails."""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(paragraph)]
    parts = [p for p in parts if p]
    return parts if len(parts) > 1 else [paragraph]


def _pack_sentences(sentences: list[str], hard_max: int) -> list[str]:
    """Greedily pack sentences into chunks not exceeding ``hard_max``.

    A single sentence longer than the hard max is passed through
    untouched — the caller falls back to word splitting.
    """
    packed: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > hard_max:
            if current:
                packed.append(current)
                current = ""
            packed.append(sentence)  # caller will hard-split it
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= hard_max:
            current = candidate
        else:
            packed.append(current)
            current = sentence
    if current:
        packed.append(current)
    return packed


def _split_long_text(text: str, hard_max: int) -> list[str]:
    """Split text that exceeded the hard max, without sentence cuts.

    First at word boundaries, then raw character slices for tokens that
    alone exceed the limit (e.g. a base64 blob). Guaranteed to terminate.
    """
    pieces: list[str] = []
    remaining = text
    while len(remaining) > hard_max:
        cut = remaining.rfind(" ", 0, hard_max)
        if cut <= hard_max // 2:  # no usable word boundary in the window
            cut = hard_max
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return [p for p in pieces if p]


def _chunk_paragraph(paragraph: str, target: int, hard_max: int) -> list[str]:
    """Split one paragraph into chunk texts respecting the size limits."""
    if len(paragraph) <= hard_max:
        return [paragraph]

    # Too large: split by sentences, pack toward the target.
    sentences = _split_sentences(paragraph)
    if len(sentences) > 1:
        packed = _pack_sentences(sentences, hard_max)
        # Anything still over the hard max (a mega-sentence) gets the
        # word/char fallback; anything under is kept as-is.
        result: list[str] = []
        for piece in packed:
            if len(piece) <= hard_max:
                result.append(piece)
            else:
                result.extend(_split_long_text(piece, hard_max))
        return result

    # Sentence splitting failed: word-boundary split, then char slices.
    return _split_long_text(paragraph, hard_max)


def chunk_document(
    pages: Iterable[ParsedPage],
    *,
    target_chars: int = config.CHUNK_TARGET_CHARS,
    hard_max: int = config.CHUNK_MAX_CHARS,
) -> list[Chunk]:
    """Chunk normalized pages into provenance-carrying chunks.

    ``target_chars`` / ``hard_max`` are overridable for tests.
    """
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        for paragraph in _split_paragraphs(page.text):
            for text in _chunk_paragraph(paragraph, target_chars, hard_max):
                chunks.append(
                    Chunk(
                        chunk_index=index,
                        content=text,
                        normalized_content=normalize_content(text),
                        page_number=page.page_number,
                        section_path=None,
                        token_count=len(text.split()),
                    )
                )
                index += 1
    logger.info("chunking complete: %d chunks", len(chunks))
    return chunks
