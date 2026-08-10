"""FTS5 retrieval for C0: safe query building + BM25-ranked search.

Query safety
------------
User text is never passed into ``MATCH`` verbatim. ``build_fts_query``
extracts word tokens (unicode letters/digits), drops a small English
stopword list and single characters, and re-assembles the tokens as
double-quoted phrases joined with ``AND``. Because every term is a
quoted single token, no FTS5 syntax (``AND``/``OR``/``NOT``/``NEAR``/
``*``/``:``/``"``/parentheses) can leak into the query.

Matching semantics: case-insensitive (unicode61 default), diacritic-
sensitive (documented limitation). ``AND`` of content terms keeps
precision high; stopword removal keeps natural-language questions from
matching nothing.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

from docpipe import config

logger = logging.getLogger(__name__)

# Small English stopword list so natural questions ("what is the method
# used in section 2?") don't AND-match themselves into silence. Content
# words are never removed.
STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "not", "but", "of", "to", "in",
        "on", "for", "with", "at", "by", "from", "as", "is", "are",
        "was", "were", "be", "been", "being", "do", "does", "did",
        "can", "could", "would", "should", "will", "may", "might",
        "what", "how", "why", "when", "where", "which", "who", "whom",
        "whose", "this", "that", "these", "those", "it", "its", "they",
        "their", "them", "we", "you", "your", "our", "i", "he", "she",
        "his", "her", "there", "here", "about", "into", "than", "then",
    }
)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


class InvalidSearchQueryError(ValueError):
    """The query produced no usable FTS terms (empty/stopword-only)."""


def build_fts_query(user_text: str) -> str:
    """Convert a natural-language query into a safe FTS5 MATCH expression.

    Raises :class:`InvalidSearchQueryError` when no usable terms remain.
    """
    if not user_text or not user_text.strip():
        raise InvalidSearchQueryError("empty query")

    terms: list[str] = []
    for raw in _WORD_RE.findall(user_text):
        term = raw.lower()
        if term in STOPWORDS:
            continue
        if len(term) > 1 or term.isdigit():
            terms.append(term)

    if not terms:
        raise InvalidSearchQueryError(
            "query contains no searchable terms (only stopwords/symbols)"
        )

    # Every term is a quoted phrase of a single token: FTS syntax cannot
    # survive this reconstruction.
    return " AND ".join(f'"{term}"' for term in terms)


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk with provenance, per the C0 contract."""

    chunk_id: int
    document_id: int
    page_number: Optional[int]
    section_path: Optional[str]
    content: str


def search(
    conn: sqlite3.Connection,
    document_id: int,
    query: str,
    limit: int = config.SEARCH_DEFAULT_LIMIT,
) -> list[SearchResult]:
    """BM25-ranked FTS5 search over one document's chunks.

    ``limit`` is clamped to ``[1, config.SEARCH_MAX_LIMIT]``. Raises
    :class:`InvalidSearchQueryError` for unusable queries. A missing
    document simply returns no results (the caller decides on 404).
    """
    fts_query = build_fts_query(query)
    limit = max(1, min(int(limit), config.SEARCH_MAX_LIMIT))

    rows = conn.execute(
        """
        SELECT c.id AS chunk_id, c.document_id, c.page_number,
               c.section_path, c.content,
               bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ? AND c.document_id = ?
        ORDER BY score
        LIMIT ?
        """,
        (fts_query, document_id, limit),
    ).fetchall()

    results = [
        SearchResult(
            chunk_id=int(row["chunk_id"]),
            document_id=int(row["document_id"]),
            page_number=row["page_number"],
            section_path=row["section_path"],
            content=row["content"],
        )
        for row in rows
    ]
    logger.info(
        "search executed: document_id=%s query=%r -> %d results",
        document_id,
        query,
        len(results),
    )
    return results
