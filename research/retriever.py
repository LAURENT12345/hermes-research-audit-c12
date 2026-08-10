"""LocalRetriever — the C2 retrieval port backed by C0 search + C1 evidence.

Orchestration ONLY. This module never reimplements search, chunking,
storage, or evidence creation:

* search:        ``docpipe.search.search`` (FTS5/BM25, C0)
* evidence:      ``evidence.builder.create_evidence`` (identity, dedupe,
                 snapshot — C1)

Flow::

    query
      -> C0 search -> chunks
      -> C1 create_evidence per chunk (existing evidence is returned,
         never duplicated)
      -> RetrievalResult(query, document_id, evidence[], skipped[])

Error policy:

* C0 search failures (invalid query, database error) propagate as
  :class:`RetrievalError` — a failed search is a hard failure.
* C1 per-chunk failures (chunk vanished between search and evidence
  creation, etc.) are recorded in ``skipped`` — one bad chunk does not
  kill the whole retrieval.
* C1 database failures propagate as :class:`RetrievalError`.

No LLM, no web, no new database, no new schema.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import docpipe.db as c0db
import docpipe.search as c0search
from evidence import builder as evidence_builder
from evidence.model import Evidence

from research import config

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """A hard retrieval failure (C0 search or database level)."""


class EvidenceCreationError(RetrievalError):
    """C1 failed at the database level while creating evidence."""


@dataclass(frozen=True)
class RetrievalResult:
    """Outcome of one retrieval query."""

    query: str
    document_id: int
    evidence: list[Evidence] = field(default_factory=list)
    #: Per-chunk failures: [{"chunk_id": int, "reason": str}, ...]
    skipped: list[dict] = field(default_factory=list)

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)


class LocalRetriever:
    """C0 search + C1 evidence orchestration for local documents."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = db_path

    def retrieve(
        self,
        document_id: int,
        query: str,
        limit: int = config.SEARCH_LIMIT_PER_QUERY,
    ) -> RetrievalResult:
        """Search one document and promote every hit to C1 Evidence."""
        conn = c0db.connect(self.db_path)
        try:
            try:
                chunks = c0search.search(conn, document_id, query, limit=limit)
            except c0search.InvalidSearchQueryError as exc:
                raise RetrievalError(f"invalid search query: {exc}") from exc
            except sqlite3.Error as exc:
                raise RetrievalError(
                    f"database failure during search: {exc}"
                ) from exc
        finally:
            conn.close()

        evidence_list: list[Evidence] = []
        skipped: list[dict] = []
        for chunk in chunks:
            try:
                evidence = evidence_builder.create_evidence(
                    chunk.chunk_id, db_path=self.db_path
                )
                evidence_list.append(evidence)
            except evidence_builder.EvidenceError as exc:
                # Chunk-level C1 failure: skip and record, keep going.
                logger.warning(
                    "evidence creation skipped for chunk %s: %s",
                    chunk.chunk_id,
                    exc,
                )
                skipped.append({"chunk_id": chunk.chunk_id, "reason": str(exc)})
            except sqlite3.Error as exc:
                raise EvidenceCreationError(
                    f"database failure during evidence creation: {exc}"
                ) from exc

        logger.info(
            "retrieval executed: document=%s query=%r -> %d evidence, %d skipped",
            document_id,
            query,
            len(evidence_list),
            len(skipped),
        )
        return RetrievalResult(
            query=query,
            document_id=document_id,
            evidence=evidence_list,
            skipped=skipped,
        )

    def retrieve_many(
        self,
        document_id: int,
        queries: list[str],
        limit: int = config.SEARCH_LIMIT_PER_QUERY,
    ) -> list[RetrievalResult]:
        """One :class:`RetrievalResult` per query (order preserved)."""
        return [
            self.retrieve(document_id, query, limit=limit) for query in queries
        ]
