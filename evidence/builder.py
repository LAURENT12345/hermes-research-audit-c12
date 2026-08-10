"""Evidence Builder: the small service that turns a C0 chunk into Evidence.

Flow::

    create_evidence(chunk_id)
        → load chunk (C0)
        → load document (C0)
        → verify relationship (chunk.document_id is used; mismatch impossible
          through this service by construction)
        → construct Evidence (content EXACTLY chunk.content — no summarize,
          no paraphrase, no re-cleaning)
        → validate shape via dataclass
        → persist (dedupe: an existing Evidence for the same chunk is
          returned, never duplicated)
        → return Evidence

The service performs no semantic reasoning and calls no LLM.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from hermes_cli.sqlite_util import write_txn

import docpipe.db as c0db

from evidence import model as evidence_model
from evidence import store

logger = logging.getLogger(__name__)


class EvidenceError(Exception):
    """Base class for C1 evidence-creation failures."""


class ChunkNotFoundError(EvidenceError):
    """The requested chunk does not exist in C0."""


class DocumentNotFoundError(EvidenceError):
    """The chunk's document does not exist (corrupt C0 state)."""


class ChunkDocumentMismatchError(EvidenceError):
    """Defensive: the chunk belongs to a different document than expected."""


def create_evidence(
    chunk_id: int,
    *,
    db_path: Path | str | None = None,
    evidence_type: str = evidence_model.CHUNK_EVIDENCE_TYPE,
) -> evidence_model.Evidence:
    """Create (or return the existing) Evidence for one C0 chunk.

    ``db_path`` overrides the default ``<HERMES_HOME>/docpipe.db``
    (tests and unusual deployments). Raises :class:`EvidenceError`
    subclasses on invalid input; raises ``sqlite3.Error`` on database
    failure (nothing is persisted partially).
    """
    conn = store.connect(db_path)
    try:
        chunk = c0db.get_chunk(conn, chunk_id)
        if chunk is None:
            raise ChunkNotFoundError(f"chunk {chunk_id} not found in C0 store")

        document = c0db.get_document(conn, int(chunk["document_id"]))
        if document is None:
            raise DocumentNotFoundError(
                f"document {chunk['document_id']} for chunk {chunk_id} not found"
            )

        with write_txn(conn):
            # Deterministic dedupe: one Evidence per chunk.
            existing = store.get_evidence_by_chunk(conn, chunk_id)
            if existing is not None:
                logger.info(
                    "duplicate evidence creation for chunk %s -> %s (returning existing)",
                    chunk_id,
                    existing.evidence_id,
                )
                return existing

            locator = {"page": chunk["page_number"], "chunk_id": chunk_id}
            return store.create_evidence(
                conn,
                document_id=int(document["id"]),
                chunk_id=chunk_id,
                evidence_type=evidence_type,
                source_type=evidence_model.LOCAL_PDF_SOURCE_TYPE,
                source_title=document["title"],
                content=chunk["content"],  # verbatim — never rewritten
                page_number=chunk["page_number"],
                section_path=chunk["section_path"],
                locator=locator,
            )
    finally:
        conn.close()
