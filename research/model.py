"""C2 data model: sessions, queries, claims, links, iterations, events.

ID conventions follow C1: deterministic sequential, human-readable IDs
(``S-000001`` for sessions, ``C-000001`` for claims), assigned from
persisted MAX(id) so they never change across restarts.

``statement_normalized`` is a deterministic fold of the claim statement
(lowercase + collapsed whitespace) used by the UNIQUE constraint to
prevent duplicate claims within a session. The original statement is
always preserved verbatim in ``statement``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

SESSION_ID_RE = re.compile(r"^S-\d{6}$")
CLAIM_ID_RE = re.compile(r"^C-\d{6}$")

SESSION_STATUSES = (
    "draft", "active", "complete", "complete_with_gaps", "failed", "aborted",
)
CLAIM_STATUSES = (
    "proposed", "supported", "insufficient", "contradicted", "invalidated",
    "withdrawn",
)
CLAIM_ROLES = ("supporting", "contradicting")
QUERY_ORIGINS = ("user", "derived")
QUERY_STATUSES = ("pending", "executed", "failed")
ITERATION_PHASES = (
    "planning", "retrieving", "evidencing", "claiming", "evaluating",
    "synthesizing", "done", "failed",
)
ITERATION_STATUSES = ("running", "completed", "failed")


def is_valid_session_id(value: str) -> bool:
    return bool(SESSION_ID_RE.match(value))


def is_valid_claim_id(value: str) -> bool:
    return bool(CLAIM_ID_RE.match(value))


def format_session_id(sequence_number: int) -> str:
    """``1 -> "S-000001"``."""
    return f"S-{int(sequence_number):06d}"


def format_claim_id(sequence_number: int) -> str:
    """``1 -> "C-000001"``."""
    return f"C-{int(sequence_number):06d}"


def normalize_statement(statement: str) -> str:
    """Deterministic fold for claim deduplication.

    Lowercases and collapses all whitespace runs to single spaces. This is
    the ONLY transformation applied; the original statement is preserved
    separately in ``Claim.statement``.
    """
    return " ".join(statement.split()).lower()


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResearchSession:
    """One research question and its loop state."""

    session_id: str
    question: str
    status: str  # one of SESSION_STATUSES
    min_supporting: int
    max_iterations: int
    final_answer: Optional[str]  # JSON ResearchAnswer, NULL until synthesized
    created_at: Optional[str]
    updated_at: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "question": self.question,
            "status": self.status,
            "min_supporting": self.min_supporting,
            "max_iterations": self.max_iterations,
            "final_answer": self.final_answer,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> "ResearchSession":
        return cls(
            session_id=str(row["id"]),
            question=str(row["question"]),
            status=str(row["status"]),
            min_supporting=int(row["min_supporting"]),
            max_iterations=int(row["max_iterations"]),
            final_answer=row["final_answer"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class ResearchQuery:
    """One retrieval query issued for a session."""

    id: int
    session_id: str
    query_text: str
    origin: str  # 'user' | 'derived'
    status: str  # 'pending' | 'executed' | 'failed'
    iteration: Optional[int]
    created_at: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "query_text": self.query_text,
            "origin": self.origin,
            "status": self.status,
            "iteration": self.iteration,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> "ResearchQuery":
        return cls(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            query_text=str(row["query_text"]),
            origin=str(row["origin"]),
            status=str(row["status"]),
            iteration=row["iteration"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class Claim:
    """One atomic factual claim within a research session.

    ``parent_claim_id`` (C3): set when this claim is an atomic
    decomposition of another claim (NULL for top-level claims).
    ``context`` (C3): optional JSON object of scope qualifiers
    (population/dataset/task/...); NULL means "no context recorded" —
    deliberately NOT interpreted as universal applicability.
    """

    claim_id: str
    session_id: str
    statement: str
    statement_normalized: str
    status: str  # one of CLAIM_STATUSES
    confidence: Optional[float]
    notes: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    parent_claim_id: Optional[str] = None
    context: Optional[str] = None  # JSON dict text

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "session_id": self.session_id,
            "statement": self.statement,
            "statement_normalized": self.statement_normalized,
            "status": self.status,
            "confidence": self.confidence,
            "notes": self.notes,
            "parent_claim_id": self.parent_claim_id,
            "context": self.context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> "Claim":
        return cls(
            claim_id=str(row["id"]),
            session_id=str(row["session_id"]),
            statement=str(row["statement"]),
            statement_normalized=str(row["statement_normalized"]),
            status=str(row["status"]),
            confidence=row["confidence"],
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            parent_claim_id=row["parent_claim_id"],
            context=row["context"],
        )


@dataclass(frozen=True)
class ClaimEvidenceLink:
    """A claim <-> evidence relationship with an explicit role."""

    claim_id: str
    evidence_id: str
    role: str  # 'supporting' | 'contradicting'
    created_at: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "evidence_id": self.evidence_id,
            "role": self.role,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> "ClaimEvidenceLink":
        return cls(
            claim_id=str(row["claim_id"]),
            evidence_id=str(row["evidence_id"]),
            role=str(row["role"]),
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class LoopIteration:
    """One recorded phase execution of the research loop."""

    id: int
    session_id: str
    iteration: int
    phase: str  # one of ITERATION_PHASES
    status: str  # one of ITERATION_STATUSES
    started_at: Optional[str]
    completed_at: Optional[str]
    notes: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "iteration": self.iteration,
            "phase": self.phase,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "notes": self.notes,
        }

    @classmethod
    def from_row(cls, row: Any) -> "LoopIteration":
        return cls(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            iteration=int(row["iteration"]),
            phase=str(row["phase"]),
            status=str(row["status"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            notes=row["notes"],
        )


@dataclass(frozen=True)
class Contradiction:
    """A detected contradiction between a claim and evidence (or another
    claim), with its resolution state.

    ``kind``: ``explicit`` (valid contradicting evidence link) or
    ``structural`` (deterministic token-overlap heuristic — advisory).
    ``status``: ``open`` | ``resolved`` | ``dismissed``.
    ``detail`` / ``resolution`` are JSON text (metadata only).
    """

    id: int
    session_id: str
    claim_id: str
    kind: str  # 'explicit' | 'structural' | 'semantic'
    status: str  # 'open' | 'resolved' | 'dismissed'
    detail: Optional[str]
    resolution: Optional[str]
    created_at: Optional[str]
    resolved_at: Optional[str]
    observation_key: Optional[str] = None  # C3 semantic observation identity

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "claim_id": self.claim_id,
            "kind": self.kind,
            "status": self.status,
            "detail": self.detail,
            "resolution": self.resolution,
            "observation_key": self.observation_key,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> "Contradiction":
        return cls(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            claim_id=str(row["claim_id"]),
            kind=str(row["kind"]),
            status=str(row["status"]),
            detail=row["detail"],
            resolution=row["resolution"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
            observation_key=row["observation_key"],
        )


@dataclass(frozen=True)
class ResearchEvent:
    """One append-only audit event for a session."""

    id: int
    session_id: str
    event_type: str
    detail: Optional[str]  # JSON text
    created_at: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "detail": self.detail,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> "ResearchEvent":
        return cls(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            event_type=str(row["event_type"]),
            detail=row["detail"],
            created_at=row["created_at"],
        )
