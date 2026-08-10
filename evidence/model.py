"""Evidence model for C1.

An Evidence object is a structured reference to a C0 chunk used as
research evidence. It retains enough provenance to trace back to the
exact document, chunk, page and verbatim content.

Evidence IDs are deterministic, sequential and human-readable
(``E-000001``, ``E-000002``, ...). They are assigned from the persisted
MAX(id) so they never change across server restarts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

#: Canonical evidence ID format: E- followed by six zero-padded digits.
EVIDENCE_ID_RE = re.compile(r"^E-\d{6}$")

#: The only source type C1 supports (C0 local PDF documents).
LOCAL_PDF_SOURCE_TYPE = "local_pdf"

#: The only evidence type C1 produces: a verbatim chunk excerpt.
CHUNK_EVIDENCE_TYPE = "chunk"


def is_valid_evidence_id(value: str) -> bool:
    return bool(EVIDENCE_ID_RE.match(value))


def format_evidence_id(sequence_number: int) -> str:
    """``1 -> "E-000001"``."""
    return f"E-{int(sequence_number):06d}"


@dataclass(frozen=True)
class Evidence:
    """A validated, persisted reference to one C0 chunk."""

    evidence_id: str
    document_id: int
    chunk_id: int
    evidence_type: str
    source_type: str
    source_title: Optional[str]
    content: str
    page_number: Optional[int]
    section_path: Optional[str]
    locator: dict[str, Any]
    created_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Full serializable form (``locator`` as a dict, per the API)."""
        return {
            "evidence_id": self.evidence_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "evidence_type": self.evidence_type,
            "source_type": self.source_type,
            "source_title": self.source_title,
            "content": self.content,
            "page_number": self.page_number,
            "section_path": self.section_path,
            "locator": self.locator,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        """Rebuild from a serialized dict (used by tests and callers)."""
        return cls(
            evidence_id=str(data["evidence_id"]),
            document_id=int(data["document_id"]),
            chunk_id=int(data["chunk_id"]),
            evidence_type=str(data["evidence_type"]),
            source_type=str(data["source_type"]),
            source_title=data.get("source_title"),
            content=str(data["content"]),
            page_number=data.get("page_number"),
            section_path=data.get("section_path"),
            locator=dict(data.get("locator") or {}),
            created_at=data.get("created_at"),
        )

    @classmethod
    def from_row(cls, row: Any) -> "Evidence":
        """Build from a ``sqlite3.Row`` (``locator`` is stored as JSON text)."""
        import json

        return cls(
            evidence_id=str(row["id"]),
            document_id=int(row["document_id"]),
            chunk_id=int(row["chunk_id"]),
            evidence_type=str(row["evidence_type"]),
            source_type=str(row["source_type"]),
            source_title=row["source_title"],
            content=str(row["content"]),
            page_number=row["page_number"],
            section_path=row["section_path"],
            locator=json.loads(row["locator"]),
            created_at=row["created_at"],
        )
