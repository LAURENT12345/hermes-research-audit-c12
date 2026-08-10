"""C2 synthesis: deterministic ``ResearchAnswer`` construction.

Synthesis is the ONLY place a research session's findings become an
answer. Everything here is deterministic and derived from persisted
state — no LLM, no free text, no hallucination:

* every non-terminal claim is re-evaluated FIRST via
  ``claims.evaluate_claim`` (live C1 validation through
  ``evidence.validation.validate_evidence``; persisted statuses are
  never trusted),
* ``withdrawn`` / ``invalidated`` claims are skipped entirely (terminal
  statuses are final by design — never re-evaluated, never surfaced),
* a claim becomes a *statement* only when it is ``supported`` AFTER
  evaluation AND has no open explicit contradiction record; its
  citations derive from ``claim_evidence`` links with role
  ``supporting`` whose evidence validates right now (invalid evidence
  never becomes a citation),
* every other non-terminal claim becomes a *gap* with a deterministic
  reason — never a silent drop,
* the summary is a deterministic count, never prose.

The session's status is NEVER changed here: the loop owns lifecycle
(``complete`` / ``complete_with_gaps``), synthesis owns only
``final_answer`` + the ``synthesized`` audit event. ``synthesize`` is
idempotent — a rebuild that produces byte-identical JSON performs no
write and no event.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from evidence import store as c1store
from evidence import validation as c1validation

from research import claims, store

logger = logging.getLogger(__name__)

#: Audit event written when a new answer is persisted.
EVENT_SYNTHESIZED = "synthesized"

#: Deterministic gap reasons.
GAP_REASON_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
GAP_REASON_CONTRADICTED = "contradicted"

#: Session statuses that allow synthesis (draft/failed/aborted do not).
ALLOWED_SYNTHESIS_STATUSES = ("active", "complete", "complete_with_gaps")

#: Claim statuses that are terminal: never re-evaluated, never surfaced.
_TERMINAL_CLAIM_STATUSES = ("withdrawn", "invalidated")


class SynthesisError(Exception):
    """Base class for synthesis-domain failures."""


class SynthesisNotAllowedError(SynthesisError):
    """The session status does not allow synthesis."""


# ---------------------------------------------------------------------------
# Answer model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Citation:
    """One provenance pointer from a statement to a validated evidence."""

    evidence_id: str
    document_id: int
    page: Optional[int]
    locator: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "document_id": self.document_id,
            "page": self.page,
            "locator": self.locator,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Citation":
        return cls(
            evidence_id=str(data["evidence_id"]),
            document_id=int(data["document_id"]),
            page=data.get("page"),
            locator=dict(data.get("locator") or {}),
        )


@dataclass(frozen=True)
class Statement:
    """One supported claim with its derived citations."""

    claim_id: str
    statement: str
    citations: tuple[Citation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "citations": [c.to_dict() for c in self.citations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Statement":
        return cls(
            claim_id=str(data["claim_id"]),
            statement=str(data["statement"]),
            citations=tuple(
                Citation.from_dict(c) for c in (data.get("citations") or [])
            ),
        )


@dataclass(frozen=True)
class ResearchGap:
    """One claim that could not be stated, with a deterministic reason."""

    claim_id: str
    statement: str
    reason: str  # GAP_REASON_INSUFFICIENT_EVIDENCE | GAP_REASON_CONTRADICTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchGap":
        return cls(
            claim_id=str(data["claim_id"]),
            statement=str(data["statement"]),
            reason=str(data["reason"]),
        )


@dataclass(frozen=True)
class ResearchAnswer:
    """The deterministic synthesis output of one research session.

    ``summary`` is a pure count derived from the statements/gaps — it
    never contains factual content beyond what the statements carry.
    Ordering is deterministic: statements by ``claim_id``, citations by
    ``evidence_id``, gaps by ``claim_id``.
    """

    session_id: str
    question: str
    summary: str
    statements: tuple[Statement, ...]
    gaps: tuple[ResearchGap, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "question": self.question,
            "summary": self.summary,
            "statements": [s.to_dict() for s in self.statements],
            "gaps": [g.to_dict() for g in self.gaps],
        }

    def to_json(self) -> str:
        """Deterministic JSON (``sort_keys`` — key order never varies)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "ResearchAnswer":
        data = json.loads(raw)
        return cls(
            session_id=str(data["session_id"]),
            question=str(data["question"]),
            summary=str(data["summary"]),
            statements=tuple(
                Statement.from_dict(s) for s in (data.get("statements") or [])
            ),
            gaps=tuple(
                ResearchGap.from_dict(g) for g in (data.get("gaps") or [])
            ),
        )


# ---------------------------------------------------------------------------
# Answer construction
# ---------------------------------------------------------------------------

def build_answer(conn, session_id: str) -> ResearchAnswer:
    """Re-validate every claim and derive the deterministic answer.

    Rules (see module docstring): terminal claims are skipped; every
    other claim is re-evaluated live; ``supported`` claims without an
    open explicit contradiction become statements (citations re-validated
    per evidence); everything else becomes a gap with a deterministic
    reason. A claim is never silently dropped.
    """
    session = store.get_session(conn, session_id)
    if session is None:
        raise SynthesisError(f"session {session_id} not found")

    statements: list[Statement] = []
    gaps: list[ResearchGap] = []

    for claim in store.list_claims(conn, session_id):
        if claim.status in _TERMINAL_CLAIM_STATUSES:
            # Terminal statuses are final: never re-evaluated, never
            # included in the answer (not even as a gap).
            continue
        fresh = claims.evaluate_claim(conn, claim.claim_id)

        if fresh.status == "supported":
            if store.find_open_contradiction(
                conn, fresh.claim_id, "explicit"
            ) is not None:
                # Stale state cannot slip through: an open explicit
                # contradiction blocks the statement even when the claim
                # currently validates as supported.
                gaps.append(
                    ResearchGap(
                        claim_id=fresh.claim_id,
                        statement=fresh.statement,
                        reason=GAP_REASON_CONTRADICTED,
                    )
                )
                continue
            citations = _derive_citations(conn, fresh.claim_id)
            if not citations:
                # Defensive: a supported claim always has valid support,
                # so this can only happen if evidence vanished between
                # evaluation and citation derivation. Surface it as a gap
                # instead of emitting an uncited statement.
                gaps.append(
                    ResearchGap(
                        claim_id=fresh.claim_id,
                        statement=fresh.statement,
                        reason=GAP_REASON_INSUFFICIENT_EVIDENCE,
                    )
                )
                continue
            statements.append(
                Statement(
                    claim_id=fresh.claim_id,
                    statement=fresh.statement,
                    citations=tuple(citations),
                )
            )
        elif fresh.status == "insufficient":
            gaps.append(
                ResearchGap(
                    claim_id=fresh.claim_id,
                    statement=fresh.statement,
                    reason=GAP_REASON_INSUFFICIENT_EVIDENCE,
                )
            )
        elif fresh.status == "contradicted":
            gaps.append(
                ResearchGap(
                    claim_id=fresh.claim_id,
                    statement=fresh.statement,
                    reason=GAP_REASON_CONTRADICTED,
                )
            )
        # 'proposed' cannot survive evaluation: the evaluator always
        # decides supported / insufficient / contradicted.

    # Deterministic ordering — never depend on set/dict iteration.
    statements.sort(key=lambda s: s.claim_id)
    gaps.sort(key=lambda g: g.claim_id)
    summary = f"{len(statements)} supported statement(s), {len(gaps)} gap(s)."

    return ResearchAnswer(
        session_id=session_id,
        question=session.question,
        summary=summary,
        statements=tuple(statements),
        gaps=tuple(gaps),
    )


def _derive_citations(conn, claim_id: str) -> list[Citation]:
    """Valid supporting evidence of a claim, as sorted citations.

    Each supporting link is re-validated via C1; invalid evidence never
    becomes a citation. Ordering by ``evidence_id`` is deterministic.
    """
    citations: list[Citation] = []
    for link in store.get_links_for_claim(conn, claim_id):
        if link.role != "supporting":
            continue
        if not c1validation.validate_evidence(conn, link.evidence_id).valid:
            continue  # invalid evidence is never cited
        evidence = c1store.get_evidence(conn, link.evidence_id)
        if evidence is None:  # pragma: no cover - validation just read it
            continue
        citations.append(
            Citation(
                evidence_id=evidence.evidence_id,
                document_id=evidence.document_id,
                page=evidence.page_number,
                locator=evidence.locator,
            )
        )
    citations.sort(key=lambda c: c.evidence_id)
    return citations


# ---------------------------------------------------------------------------
# Persistence + access
# ---------------------------------------------------------------------------

def synthesize(conn, session_id: str, *, force: bool = False) -> ResearchAnswer:
    """Build and persist the answer for one session (idempotent).

    Allowed session statuses: ``active``, ``complete``,
    ``complete_with_gaps``. ``draft`` / ``failed`` / ``aborted`` raise
    :class:`SynthesisNotAllowedError`.

    When the freshly built JSON is byte-identical to the persisted
    ``final_answer`` (and ``force`` is False), NOTHING is written and no
    event is emitted — the stored answer is returned as-is. Session
    status is never modified (the loop owns lifecycle).
    """
    session = store.get_session(conn, session_id)
    if session is None:
        raise SynthesisError(f"session {session_id} not found")
    if session.status not in ALLOWED_SYNTHESIS_STATUSES:
        raise SynthesisNotAllowedError(
            f"cannot synthesize session {session_id} in state "
            f"{session.status!r} (allowed: "
            f"{', '.join(ALLOWED_SYNTHESIS_STATUSES)})"
        )

    answer = build_answer(conn, session_id)
    answer_json = answer.to_json()
    if session.final_answer == answer_json and not force:
        logger.info("synthesis idempotent: session %s answer unchanged",
                    session_id)
        return answer  # no write, no event

    store.set_final_answer(conn, session_id, answer_json)
    store.add_event(
        conn,
        session_id=session_id,
        event_type=EVENT_SYNTHESIZED,
        detail=json.dumps(
            {
                "claim_count": len(answer.statements),
                "citation_count": sum(
                    len(s.citations) for s in answer.statements
                ),
                "gap_count": len(answer.gaps),
            },
            ensure_ascii=False,
        ),
    )
    logger.info(
        "synthesized answer for session %s: %d statement(s), %d gap(s)",
        session_id, len(answer.statements), len(answer.gaps),
    )
    return answer


def get_answer(conn, session_id: str, *, refresh: bool = True) -> ResearchAnswer:
    """Return the answer, re-synthesizing when ``refresh`` is True.

    ``refresh=True`` delegates to :func:`synthesize`, which rebuilds the
    answer from current state and persists it only when it changed (stale
    answers self-heal). ``refresh=False`` reads the stored answer and
    raises :class:`SynthesisError` when none exists.
    """
    if refresh:
        return synthesize(conn, session_id)
    answer = load_answer(conn, session_id)
    if answer is None:
        raise SynthesisError(f"no stored answer for session {session_id}")
    return answer


def load_answer(conn, session_id: str) -> Optional[ResearchAnswer]:
    """The stored answer, or ``None`` when the session/answer is absent."""
    session = store.get_session(conn, session_id)
    if session is None or session.final_answer is None:
        return None
    return ResearchAnswer.from_json(session.final_answer)
