"""C2 sufficiency: claim-level and session-level assessment.

This module only ASSESSES state — it never changes claim status, never
writes events, and never runs the loop. The loop (a later commit) decides
what to do next based on these results.

Claim-level: every linked evidence is re-validated via
``evidence.validation.validate_evidence`` (persisted evidence state is
never trusted). A claim is *sufficient* when valid supporting evidence >=
the session's ``min_supporting`` and there is no valid contradicting
evidence.

Session-level: reads the persisted claim statuses (callers should run
``claims.evaluate_claim`` per claim first so statuses are fresh). Gaps are
claims that are insufficient or unresolved-contradicted; the session is
*synthesis-ready* when at least one claim is supported and no gaps exist.

``build_plan_inputs`` produces the deterministic input structure for
``ReasonerPort.plan_queries`` — the Gap dataclass is imported as a TYPE
only; the reasoner is never called from here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from evidence import validation as c1validation

from research import store
from research.reasoner import Gap

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaimSufficiency:
    """Structured claim-level sufficiency assessment."""

    claim_id: str
    sufficient: bool
    supporting_valid_count: int
    supporting_invalid_count: int
    contradicting_valid_count: int
    required_support: int
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SessionSufficiency:
    """Structured session-level assessment (statuses as persisted)."""

    session_id: str
    question: str
    total_claims: int
    supported_count: int
    insufficient_count: int
    contradicted_count: int
    unresolved_count: int  # claims that are not supported
    has_gaps: bool
    synthesis_ready: bool
    needs_more_retrieval: bool
    gaps: list[dict] = field(default_factory=list)  # [{claim_id, statement, status}]


@dataclass(frozen=True)
class PlanInputs:
    """Deterministic input for ``ReasonerPort.plan_queries`` (loop commit)."""

    question: str
    gaps: list[Gap] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Claim-level
# ---------------------------------------------------------------------------

def check_claim_sufficiency(conn, claim_id: str) -> ClaimSufficiency:
    """Pure assessment of one claim against its CURRENT evidence state.

    Does NOT change claim status and does NOT write events.
    """
    claim = store.get_claim(conn, claim_id)
    if claim is None:
        raise ValueError(f"claim {claim_id} not found")
    session = store.get_session(conn, claim.session_id)
    if session is None:  # pragma: no cover - FK prevents this
        raise ValueError(f"session {claim.session_id} not found")

    supporting_valid = 0
    supporting_invalid = 0
    contradicting_valid = 0
    for link in store.get_links_for_claim(conn, claim_id):
        result = c1validation.validate_evidence(conn, link.evidence_id)
        if link.role == "supporting":
            if result.valid:
                supporting_valid += 1
            else:
                supporting_invalid += 1
        elif result.valid:
            contradicting_valid += 1

    reasons: list[str] = []
    sufficient = supporting_valid >= session.min_supporting and contradicting_valid == 0
    if supporting_valid < session.min_supporting:
        reasons.append(
            f"valid supporting evidence {supporting_valid} < required "
            f"{session.min_supporting}"
        )
    if contradicting_valid > 0:
        reasons.append(
            f"{contradicting_valid} valid contradicting evidence present"
        )
    if supporting_invalid > 0:
        reasons.append(f"{supporting_invalid} supporting evidence invalid")

    return ClaimSufficiency(
        claim_id=claim_id,
        sufficient=sufficient,
        supporting_valid_count=supporting_valid,
        supporting_invalid_count=supporting_invalid,
        contradicting_valid_count=contradicting_valid,
        required_support=session.min_supporting,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Session-level
# ---------------------------------------------------------------------------

def check_session_sufficiency(conn, session_id: str) -> SessionSufficiency:
    """Assess a session from its persisted claim statuses.

    Callers (the loop) should run ``claims.evaluate_claim`` for each claim
    first so the statuses reflect the current evidence state.
    """
    session = store.get_session(conn, session_id)
    if session is None:
        raise ValueError(f"session {session_id} not found")

    claims = store.list_claims(conn, session_id)
    supported = [c for c in claims if c.status == "supported"]
    insufficient = [c for c in claims if c.status == "insufficient"]
    contradicted = [c for c in claims if c.status == "contradicted"]
    gaps = [
        {"claim_id": c.claim_id, "statement": c.statement, "status": c.status}
        for c in [*insufficient, *contradicted]
    ]

    return SessionSufficiency(
        session_id=session_id,
        question=session.question,
        total_claims=len(claims),
        supported_count=len(supported),
        insufficient_count=len(insufficient),
        contradicted_count=len(contradicted),
        unresolved_count=len(claims) - len(supported),
        has_gaps=bool(gaps),
        synthesis_ready=bool(supported) and not gaps,
        needs_more_retrieval=bool(insufficient),
        gaps=gaps,
    )


def build_plan_inputs(conn, session_id: str) -> PlanInputs:
    """The deterministic question+gaps structure for plan_queries.

    Gap = insufficient or unresolved contradicted claim. The reasoner is
    NOT called here (and is not imported for execution).
    """
    session = store.get_session(conn, session_id)
    if session is None:
        raise ValueError(f"session {session_id} not found")

    gaps: list[Gap] = []
    for claim in store.list_claims(conn, session_id):
        if claim.status in ("insufficient", "contradicted"):
            gaps.append(Gap(claim_id=claim.claim_id, statement=claim.statement))
    return PlanInputs(question=session.question, gaps=gaps)
