"""C2 claims: creation, status transitions, evidence linking, evaluation.

This module owns the CLAIM side of the research loop. The reasoner only
PROPOSES; this module's deterministic evaluator decides status.

Status transition matrix (evaluator + operator use the same rules)::

    proposed      -> supported | insufficient | contradicted | invalidated
    supported     -> insufficient | contradicted | invalidated
    insufficient  -> supported | contradicted | invalidated
    contradicted  -> supported | withdrawn | invalidated
    withdrawn     -> invalidated

(Compared with the approved design diagram this adds
``supported -> contradicted`` and ``insufficient -> supported /
contradicted`` — the evaluator requires them: a valid contradicting
evidence must move a *supported* claim to contradicted, and a claim with
fresh valid support must be promotable from insufficient.)

Every transition:

1. validates the source/target pair,
2. updates the claim row,
3. appends a ``claim_status_changed`` research_event with
   ``{claim_id, from, to, reason}`` (metadata only — never content),
4. stores reason/notes when provided.

Evaluation rule (``evaluate_claim``): persisted evidence state is NEVER
trusted — ``evidence.validation.validate_evidence`` is called for every
linked evidence, and the target status is recomputed from scratch:

* any VALID contradicting evidence  -> contradicted
* valid supporting >= session.min_supporting -> supported
* otherwise                         -> insufficient
  (reason ``evidence_invalid`` when demoting a supported claim)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from research import store
from research.model import Claim, ClaimEvidenceLink
from research.reasoner import ClaimProposal
from research import config
from evidence import store as c1store
from evidence import validation as c1validation

logger = logging.getLogger(__name__)

#: Allowed status transitions (deterministic; see module docstring).
TRANSITIONS: dict[str, frozenset[str]] = {
    "proposed": frozenset({"supported", "insufficient", "contradicted", "invalidated"}),
    "supported": frozenset({"insufficient", "contradicted", "invalidated"}),
    "insufficient": frozenset({"supported", "contradicted", "invalidated"}),
    "contradicted": frozenset({"supported", "withdrawn", "invalidated"}),
    "withdrawn": frozenset({"invalidated"}),
}

#: Event types used by this module.
EVENT_CLAIM_PROPOSED = "claim_proposed"
EVENT_CLAIM_STATUS_CHANGED = "claim_status_changed"
EVENT_EVIDENCE_LINKED = "evidence_linked"

#: Deterministic reason codes for evaluator-driven transitions.
REASON_EVIDENCE_INVALID = "evidence_invalid"
REASON_CONTRADICTING_EVIDENCE = "contradicting_evidence"
REASON_SUFFICIENT_EVIDENCE = "sufficient_evidence"
REASON_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
REASON_MANUAL = "manual"


class ClaimError(Exception):
    """Base class for claim-domain failures."""


class ClaimNotFoundError(ClaimError):
    """The claim does not exist."""


class EvidenceNotFoundError(ClaimError):
    """The evidence does not exist (checked at link time)."""


class InvalidRoleError(ClaimError):
    """role must be 'supporting' or 'contradicting'."""


class ClaimTransitionError(ClaimError):
    """The requested status transition is not allowed."""


class InvalidClaimError(ClaimError):
    """Statement is empty or exceeds MAX_STATEMENT_CHARS."""


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def create_claim(
    conn,
    *,
    session_id: str,
    statement: str,
    notes: Optional[str] = None,
) -> Claim:
    """Create a proposed claim (manual/operator path).

    Deduplicates on ``(session_id, statement_normalized)``: a duplicate
    returns the existing claim and creates NO new event.
    """
    statement = statement.strip()
    if not statement:
        raise InvalidClaimError("claim statement must not be empty")
    if len(statement) > config.MAX_STATEMENT_CHARS:
        raise InvalidClaimError(
            f"claim statement exceeds {config.MAX_STATEMENT_CHARS} characters"
        )
    existing = _find_duplicate(conn, session_id, statement)
    if existing is not None:
        return existing
    claim = store.create_claim(
        conn, session_id=session_id, statement=statement, notes=notes
    )
    store.add_event(
        conn,
        session_id=session_id,
        event_type=EVENT_CLAIM_PROPOSED,
        detail=json.dumps({"claim_id": claim.claim_id}),
    )
    logger.info("claim proposed: %s (session %s)", claim.claim_id, session_id)
    return claim


def create_claim_from_proposal(
    conn,
    *,
    session_id: str,
    proposal: ClaimProposal,
    notes: Optional[str] = None,
) -> Claim:
    """Create a claim from a reasoner proposal.

    The proposal's evidence is NOT attached here (linking is explicit) and
    the claim is NEVER marked supported just because the proposal carried
    evidence — the evaluator decides.
    """
    return create_claim(
        conn, session_id=session_id, statement=proposal.statement, notes=notes
    )


def _find_duplicate(conn, session_id: str, statement: str) -> Optional[Claim]:
    from research.model import normalize_statement

    normalized = normalize_statement(statement)
    for claim in store.list_claims(conn, session_id):
        if claim.statement_normalized == normalized:
            return claim
    return None


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def transition_claim(
    conn,
    claim_id: str,
    to_status: str,
    *,
    reason: Optional[str] = None,
    notes: Optional[str] = None,
) -> Claim:
    """Apply one validated status transition (with audit event).

    Raises :class:`ClaimTransitionError` for invalid transitions — status
    is never silently mutated.
    """
    claim = _require_claim(conn, claim_id)
    allowed = TRANSITIONS.get(claim.status, frozenset())
    if to_status not in allowed:
        raise ClaimTransitionError(
            f"transition {claim.status} -> {to_status} is not allowed "
            f"(allowed: {sorted(allowed)})"
        )
    store.update_claim_status(conn, claim_id, to_status, notes=notes)
    store.add_event(
        conn,
        session_id=claim.session_id,
        event_type=EVENT_CLAIM_STATUS_CHANGED,
        detail=json.dumps(
            {
                "claim_id": claim_id,
                "from": claim.status,
                "to": to_status,
                "reason": reason or REASON_MANUAL,
            }
        ),
    )
    logger.info(
        "claim %s: %s -> %s (reason=%s)", claim_id, claim.status, to_status, reason
    )
    updated = store.get_claim(conn, claim_id)
    assert updated is not None
    return updated


def _require_claim(conn, claim_id: str) -> Claim:
    claim = store.get_claim(conn, claim_id)
    if claim is None:
        raise ClaimNotFoundError(f"claim {claim_id} not found")
    return claim


# ---------------------------------------------------------------------------
# Evidence linking
# ---------------------------------------------------------------------------

def link_evidence(
    conn,
    *,
    claim_id: str,
    evidence_id: str,
    role: str = "supporting",
) -> ClaimEvidenceLink:
    """Link evidence to a claim with an explicit role.

    Rules: claim exists, evidence exists, role is supporting|contradicting,
    duplicate links are idempotent (returned as-is). Linking NEVER changes
    claim status. Evidence is never modified. The audit event carries
    metadata only (no evidence content).
    """
    _require_claim(conn, claim_id)
    if c1store.get_evidence(conn, evidence_id) is None:
        raise EvidenceNotFoundError(f"evidence {evidence_id} not found")
    if role not in ("supporting", "contradicting"):
        raise InvalidRoleError(f"role must be 'supporting' or 'contradicting', got {role!r}")

    existing = [
        l for l in store.get_links_for_claim(conn, claim_id)
        if l.evidence_id == evidence_id and l.role == role
    ]
    if existing:
        return existing[0]

    claim = store.get_claim(conn, claim_id)
    link = store.link_evidence(
        conn, claim_id=claim_id, evidence_id=evidence_id, role=role
    )
    store.add_event(
        conn,
        session_id=claim.session_id,
        event_type=EVENT_EVIDENCE_LINKED,
        detail=json.dumps(
            {"claim_id": claim_id, "evidence_id": evidence_id, "role": role}
        ),
    )
    logger.info("evidence linked: %s %s -> %s", evidence_id, role, claim_id)
    return link


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_claim(conn, claim_id: str) -> Claim:
    """Deterministic evaluator: recompute the claim's status from the
    CURRENT validated state of every linked evidence.

    Never trusts persisted statuses: ``evidence.validation.
    validate_evidence`` is called per link. When the recomputed target
    differs from the persisted status, the transition is applied with a
    ``claim_status_changed`` event (reason ``evidence_invalid`` when a
    supported claim loses its support); when it matches, nothing changes.
    """
    claim = _require_claim(conn, claim_id)
    if claim.status in ("withdrawn", "invalidated"):
        # Terminal statuses are final by design: never re-evaluated (a
        # transition out of them is not part of the state machine).
        return claim
    session = store.get_session(conn, claim.session_id)
    if session is None:  # pragma: no cover - FK prevents this
        raise ClaimError(f"session {claim.session_id} missing for claim {claim_id}")

    links = store.get_links_for_claim(conn, claim_id)
    supporting_valid = 0
    supporting_invalid = 0
    contradicting_valid = 0
    for link in links:
        result = c1validation.validate_evidence(conn, link.evidence_id)
        if link.role == "supporting":
            if result.valid:
                supporting_valid += 1
            else:
                supporting_invalid += 1
        else:  # contradicting
            if result.valid:
                contradicting_valid += 1

    if contradicting_valid > 0:
        target = "contradicted"
        reason = REASON_CONTRADICTING_EVIDENCE
    elif supporting_valid >= session.min_supporting:
        target = "supported"
        reason = REASON_SUFFICIENT_EVIDENCE
    else:
        target = "insufficient"
        reason = (
            REASON_EVIDENCE_INVALID
            if claim.status == "supported"
            else REASON_INSUFFICIENT_EVIDENCE
        )

    if target == claim.status:
        return claim  # no change -> no event

    logger.info(
        "evaluating claim %s: %d valid support / %d invalid support / "
        "%d valid contradicting -> %s",
        claim_id, supporting_valid, supporting_invalid, contradicting_valid, target,
    )
    return transition_claim(
        conn, claim_id, target, reason=reason,
        notes=f"evaluator: {supporting_valid} valid supporting, "
              f"{supporting_invalid} invalid supporting, "
              f"{contradicting_valid} valid contradicting",
    )
