"""Evidence validation and tamper detection for C1.

Every persisted Evidence is re-checked against the *current* C0 state
(the chunk and document rows are the integrity anchor). Validation never
repairs anything — a corrupted Evidence is reported, never silently fixed.

Checks, in order:

1. evidence exists
2. document exists
3. chunk exists
4. relationship: chunk.document_id == evidence.document_id
5. page provenance: evidence.page_number == chunk.page_number
   (only where page information exists on either side)
6. content integrity: evidence.content == chunk.content (exact)
7. source type: evidence.source_type == "local_pdf"
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import docpipe.db as c0db

from evidence import store
from evidence.model import LOCAL_PDF_SOURCE_TYPE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    """Structured validation outcome."""

    evidence_id: str
    valid: bool
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        out = {"evidence_id": self.evidence_id, "valid": self.valid}
        if self.reason is not None:
            out["reason"] = self.reason
        return out


def validate_evidence(
    conn: sqlite3.Connection, evidence_id: str
) -> ValidationResult:
    """Validate one Evidence against the C0 store. Never mutates data."""
    evidence = store.get_evidence(conn, evidence_id)
    if evidence is None:
        return ValidationResult(evidence_id, False, "evidence_not_found")

    document = c0db.get_document(conn, evidence.document_id)
    if document is None:
        return ValidationResult(evidence_id, False, "document_not_found")

    chunk = c0db.get_chunk(conn, evidence.chunk_id)
    if chunk is None:
        return ValidationResult(evidence_id, False, "chunk_not_found")

    if int(chunk["document_id"]) != evidence.document_id:
        return ValidationResult(evidence_id, False, "document_chunk_mismatch")

    if (
        evidence.page_number is not None
        and chunk["page_number"] is not None
        and evidence.page_number != chunk["page_number"]
    ):
        return ValidationResult(evidence_id, False, "page_mismatch")

    if evidence.content != chunk["content"]:
        return ValidationResult(evidence_id, False, "content_mismatch")

    if evidence.source_type != LOCAL_PDF_SOURCE_TYPE:
        return ValidationResult(evidence_id, False, "source_type_mismatch")

    logger.info("evidence validated: %s -> valid", evidence_id)
    return ValidationResult(evidence_id, True)


def validate_evidence_by_id(
    evidence_id: str, *, db_path=None
) -> ValidationResult:
    """Convenience wrapper that opens/closes the store connection."""
    conn = store.connect(db_path)
    try:
        return validate_evidence(conn, evidence_id)
    finally:
        conn.close()
