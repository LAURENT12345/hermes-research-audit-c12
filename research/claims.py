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

from hermes_cli.sqlite_util import write_txn

from research import store
from research.model import (
    Claim,
    ClaimEvidenceLink,
    format_claim_id,
    normalize_statement,
)
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


class ClaimHierarchyError(ClaimError):
    """An invalid claim parent/child relationship was requested."""


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
    # Status + audit event commit atomically: a crash cannot leave a
    # changed status without its event (or an event without the status).
    with write_txn(conn):
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
# Claim hierarchy (decomposition parent/child) — C3 hardening
# ---------------------------------------------------------------------------

def validate_parent_relation(
    conn, child_id: str, parent_id: str, *, child: Optional[Claim] = None
) -> None:
    """Deterministic parent/child integrity rules (raises ClaimHierarchyError).

    * parent must exist,
    * a claim cannot be its own parent,
    * parent and child must belong to the same session,
    * the parent chain must not contain a cycle.
    """
    if child_id == parent_id:
        raise ClaimHierarchyError("a claim cannot be its own parent")
    child = child or _require_claim(conn, child_id)
    parent = store.get_claim(conn, parent_id)
    if parent is None:
        raise ClaimHierarchyError(f"parent claim {parent_id} not found")
    if parent.session_id != child.session_id:
        raise ClaimHierarchyError(
            f"parent {parent_id} and child {child_id} must belong to the "
            "same session"
        )
    # Cycle detection: walk up the parent chain.
    seen = {child_id}
    current = parent_id
    while current is not None:
        if current in seen:
            raise ClaimHierarchyError(
                f"parent cycle detected involving claim {current}"
            )
        seen.add(current)
        row = store.get_claim(conn, current)
        current = row.parent_claim_id if row is not None else None


def create_decomposed_claims(
    conn,
    *,
    session_id: str,
    parent_claim_id: str,
    statements: list[str],
    evidence_ids: list[str],
    notes: Optional[str] = None,
) -> list[Claim]:
    """Atomically create (or reuse) child claims of a decomposition.

    Everything happens inside ONE write_txn: parent hierarchy validated,
    children deduplicated by normalized statement, parent assigned,
    evidence linked, audit events written. A failure at any point rolls
    back the whole decomposition — no partial children, no partial
    links, no partial audit state.

    Idempotent re-application: children whose normalized statement
    already exists are reused (with their existing parent — re-parenting
    to a DIFFERENT parent is rejected deterministically).
    """
    parent = store.get_claim(conn, parent_claim_id)
    if parent is None:
        raise ClaimHierarchyError(f"parent claim {parent_claim_id} not found")
    if parent.session_id != session_id:
        raise ClaimHierarchyError(
            "parent and children must belong to the same session"
        )

    created: list[Claim] = []
    with write_txn(conn):
        for statement in statements:
            normalized = normalize_statement(statement)
            existing = conn.execute(
                "SELECT * FROM claims WHERE session_id = ? "
                "AND statement_normalized = ?",
                (session_id, normalized),
            ).fetchone()
            if existing is not None:
                child = Claim.from_row(existing)
                # Re-parenting rule: an existing child may not be moved to
                # a different parent.
                if (
                    child.parent_claim_id is not None
                    and child.parent_claim_id != parent_claim_id
                ):
                    raise ClaimHierarchyError(
                        f"claim {child.claim_id} already belongs to parent "
                        f"{child.parent_claim_id}; re-parenting is not allowed"
                    )
            else:
                child_id = store._next_id(  # noqa: SLF001 - shared package helper
                    conn, "claims", "C-", format_claim_id
                )
                conn.execute(
                    "INSERT INTO claims (id, session_id, statement, "
                    "statement_normalized, status, notes) "
                    "VALUES (?, ?, ?, ?, 'proposed', ?)",
                    (child_id, session_id, statement, normalized, notes),
                )
                conn.execute(
                    "INSERT INTO research_events (session_id, event_type, detail) "
                    "VALUES (?, 'claim_proposed', ?)",
                    (session_id, json.dumps({"claim_id": child_id})),
                )
                child = Claim.from_row(
                    conn.execute(
                        "SELECT * FROM claims WHERE id = ?", (child_id,)
                    ).fetchone()
                )

            # Hierarchy rules apply inside the transaction too.
            validate_parent_relation(
                conn, child.claim_id, parent_claim_id, child=child
            )
            conn.execute(
                "UPDATE claims SET parent_claim_id = ? WHERE id = ?",
                (parent_claim_id, child.claim_id),
            )
            for evidence_id in evidence_ids:
                existing_link = conn.execute(
                    "SELECT 1 FROM claim_evidence WHERE claim_id = ? "
                    "AND evidence_id = ? AND role = 'supporting'",
                    (child.claim_id, evidence_id),
                ).fetchone()
                if existing_link is None:
                    conn.execute(
                        "INSERT INTO claim_evidence (claim_id, evidence_id, role) "
                        "VALUES (?, ?, 'supporting')",
                        (child.claim_id, evidence_id),
                    )
            created.append(
                Claim.from_row(
                    conn.execute(
                        "SELECT * FROM claims WHERE id = ?", (child.claim_id,)
                    ).fetchone()
                )
            )
    return created


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
    # Link + audit event commit atomically inside store.link_evidence's
    # transaction (the event is passed in — no nested write_txn).
    link = store.link_evidence(
        conn,
        claim_id=claim_id,
        evidence_id=evidence_id,
        role=role,
        event=(
            claim.session_id,
            EVENT_EVIDENCE_LINKED,
            json.dumps({"claim_id": claim_id, "evidence_id": evidence_id,
                        "role": role}),
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
