"""C3 deterministic proposal verifier.

Pipeline::

    ReasonerPort
        -> proposal objects
        -> THIS VERIFIER (deterministic)
        -> validated proposals
        -> domain persistence (claims / contradictions / synthesis)

The reasoner never mutates authoritative state; every proposal is
checked against persisted objects and domain invariants before any
write. Proposals that reference fabricated IDs, invalid evidence, or
malformed scope are rejected with a deterministic reason.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Optional

from hermes_cli.sqlite_util import write_txn

from research import claims, store
import evidence.store as c1store
from research.reasoner import (
    CONTEXT_KEYS,
    COVERAGE_LABELS,
    ClaimContextProposal,
    ClaimDecompositionProposal,
    CoverageProposal,
    EvidenceRelationProposal,
    SemanticContradictionProposal,
)
from evidence import validation as c1validation

logger = logging.getLogger(__name__)


class VerifierError(Exception):
    """Base class for proposal verification failures."""


class ProposalRejectedError(VerifierError):
    """A proposal failed deterministic validation (reason attached)."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"proposal rejected: {reason}")


@dataclass(frozen=True)
class Verdict:
    """Deterministic validation outcome for one proposal."""

    accepted: bool
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Shared checks
# ---------------------------------------------------------------------------

def _claim_exists(conn, claim_id: str) -> bool:
    return store.get_claim(conn, claim_id) is not None


def _evidence_valid(conn, evidence_id: str) -> bool:
    return c1validation.validate_evidence(conn, evidence_id).valid


def _evidence_linked_to(conn, claim_id: str, evidence_id: str) -> bool:
    return any(
        link.evidence_id == evidence_id and link.role == "supporting"
        for link in store.get_links_for_claim(conn, claim_id)
    )


def _valid_context_dict(context) -> Optional[str]:
    """Validate a context mapping; returns an error reason or None.

    ``None`` is valid (no context recorded) — it is never interpreted as
    universal applicability.
    """
    if context is None:
        return None
    if not isinstance(context, dict):
        return "context must be a dict"
    for key, value in context.items():
        if key not in CONTEXT_KEYS:
            return f"unknown context key {key!r} (allowed: {CONTEXT_KEYS})"
        if value is not None and not isinstance(value, str):
            return f"context value for {key!r} must be a string or null"
    return None


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

def validate_decomposition(conn, proposal: ClaimDecompositionProposal) -> Verdict:
    """Children must reference evidence that exists, is valid, and is
    already linked to the source claim."""
    if not _claim_exists(conn, proposal.source_claim_id):
        return Verdict(False, f"source claim {proposal.source_claim_id} not found")
    if not proposal.child_statements:
        return Verdict(False, "child_statements must not be empty")
    for statement in proposal.child_statements:
        if not statement.strip():
            return Verdict(False, "child statement must not be empty")
    for evidence_id in proposal.child_evidence_ids:
        if c1store.get_evidence(conn, evidence_id) is None:
            return Verdict(False, f"fabricated evidence id {evidence_id}")
        if not _evidence_valid(conn, evidence_id):
            return Verdict(False, f"evidence {evidence_id} is invalid")
        if not _evidence_linked_to(conn, proposal.source_claim_id, evidence_id):
            return Verdict(
                False,
                f"evidence {evidence_id} is not linked to source claim "
                f"{proposal.source_claim_id}",
            )
    return Verdict(True)


def validate_evidence_relation(
    conn, proposal: EvidenceRelationProposal
) -> Verdict:
    if not _claim_exists(conn, proposal.claim_id):
        return Verdict(False, f"claim {proposal.claim_id} not found")
    if proposal.role not in ("supporting", "contradicting"):
        return Verdict(False, f"invalid role {proposal.role!r}")
    if c1store.get_evidence(conn, proposal.evidence_id) is None:
        return Verdict(False, f"fabricated evidence id {proposal.evidence_id}")
    if not _evidence_valid(conn, proposal.evidence_id):
        return Verdict(False, f"evidence {proposal.evidence_id} is invalid")
    return Verdict(True)


def validate_semantic_contradiction(
    conn, proposal: SemanticContradictionProposal
) -> Verdict:
    if proposal.claim_a_id == proposal.claim_b_id:
        return Verdict(False, "a claim cannot contradict itself")
    if not _claim_exists(conn, proposal.claim_a_id):
        return Verdict(False, f"claim {proposal.claim_a_id} not found")
    if not _claim_exists(conn, proposal.claim_b_id):
        return Verdict(False, f"claim {proposal.claim_b_id} not found")
    if not proposal.rationale.strip():
        return Verdict(False, "rationale must not be empty")
    if not proposal.evidence_a_ids:
        return Verdict(False, "evidence_a_ids must not be empty")
    if not proposal.evidence_b_ids:
        return Verdict(False, "evidence_b_ids must not be empty")
    if proposal.strength is not None and not 0.0 <= proposal.strength <= 1.0:
        return Verdict(False, "strength must be in [0, 1]")
    context_error = _valid_context_dict(proposal.context)
    if context_error:
        return Verdict(False, context_error)
    for evidence_id in proposal.evidence_a_ids:
        if c1store.get_evidence(conn, evidence_id) is None:
            return Verdict(False, f"fabricated evidence id {evidence_id}")
        if not _evidence_valid(conn, evidence_id):
            return Verdict(False, f"evidence {evidence_id} is invalid")
        if not _evidence_linked_to(conn, proposal.claim_a_id, evidence_id):
            return Verdict(
                False,
                f"evidence {evidence_id} is not linked to claim "
                f"{proposal.claim_a_id}",
            )
    for evidence_id in proposal.evidence_b_ids:
        if c1store.get_evidence(conn, evidence_id) is None:
            return Verdict(False, f"fabricated evidence id {evidence_id}")
        if not _evidence_valid(conn, evidence_id):
            return Verdict(False, f"evidence {evidence_id} is invalid")
        if not _evidence_linked_to(conn, proposal.claim_b_id, evidence_id):
            return Verdict(
                False,
                f"evidence {evidence_id} is not linked to claim "
                f"{proposal.claim_b_id}",
            )
    return Verdict(True)


def validate_context(conn, proposal: ClaimContextProposal) -> Verdict:
    if not _claim_exists(conn, proposal.claim_id):
        return Verdict(False, f"claim {proposal.claim_id} not found")
    context_error = _valid_context_dict(proposal.context)
    if context_error:
        return Verdict(False, context_error)
    return Verdict(True)


def validate_coverage(conn, proposal: CoverageProposal) -> Verdict:
    if not _claim_exists(conn, proposal.claim_id):
        return Verdict(False, f"claim {proposal.claim_id} not found")
    if proposal.coverage not in COVERAGE_LABELS:
        return Verdict(False, f"unknown coverage label {proposal.coverage!r}")
    derived = coverage_for_claim(conn, proposal.claim_id)
    # ``insufficient``/``irrelevant`` are reasoner-intent labels that are
    # never derived; allow them only when they do not contradict the
    # derived observation.
    if proposal.coverage not in (
        "insufficient", "irrelevant",
    ) and proposal.coverage != derived.coverage:
        return Verdict(
            False,
            f"proposed coverage {proposal.coverage!r} contradicts derived "
            f"coverage {derived.coverage!r}",
        )
    return Verdict(True)


# ---------------------------------------------------------------------------
# Coverage (derived, deterministic)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageObservation:
    """The authoritative, freshly-derived coverage of one claim."""

    claim_id: str
    coverage: str
    reasons: tuple[str, ...] = ()


def coverage_for_claim(conn, claim_id: str) -> CoverageObservation:
    """Derive a claim's evidence coverage from CURRENT persisted state.

    Labels produced here are exactly: directly_supported,
    partially_supported, unsupported, contradictory. The labels
    ``insufficient`` and ``irrelevant`` are accepted proposal intents but
    are NEVER derived — in particular, ``unsupported`` (no evidence
    found) is NOT a scientific research gap.
    """
    claim = store.get_claim(conn, claim_id)
    if claim is None:
        raise VerifierError(f"claim {claim_id} not found")

    fresh = claims.evaluate_claim(conn, claim_id)
    links = store.get_links_for_claim(conn, claim_id)
    supporting = [l for l in links if l.role == "supporting"]
    valid_support = sum(
        1 for l in supporting
        if c1validation.validate_evidence(conn, l.evidence_id).valid
    )
    valid_contra = sum(
        1 for l in links if l.role == "contradicting"
        and c1validation.validate_evidence(conn, l.evidence_id).valid
    )

    if fresh.status == "supported":
        return CoverageObservation(claim_id, "directly_supported",
                                   (f"{valid_support} valid supporting evidence",))
    if fresh.status == "contradicted":
        return CoverageObservation(
            claim_id, "contradictory",
            (f"{valid_contra} valid contradicting evidence",),
        )
    if supporting:
        return CoverageObservation(
            claim_id, "partially_supported",
            (f"{valid_support} valid / {len(supporting) - valid_support} "
             "invalid supporting evidence",),
        )
    return CoverageObservation(
        claim_id, "unsupported", ("no supporting evidence found",)
    )


# ---------------------------------------------------------------------------
# Apply validated proposals (persistence through the domain layer)
# ---------------------------------------------------------------------------

def _require_verdict(conn, validator, proposal) -> None:
    verdict = validator(conn, proposal)
    if not verdict.accepted:
        raise ProposalRejectedError(verdict.reason)


def apply_decomposition(conn, proposal: ClaimDecompositionProposal):
    """Create child claims for a validated decomposition — ATOMICALLY.

    Delegates to ``claims.create_decomposed_claims``, which performs the
    whole decomposition (children + parent assignment + evidence links +
    audit events) inside ONE write_txn. A failure rolls back everything;
    re-applying the same decomposition reuses existing children.
    """
    _require_verdict(conn, validate_decomposition, proposal)
    source = store.get_claim(conn, proposal.source_claim_id)
    return claims.create_decomposed_claims(
        conn,
        session_id=source.session_id,
        parent_claim_id=proposal.source_claim_id,
        statements=list(proposal.child_statements),
        evidence_ids=list(proposal.child_evidence_ids),
        notes=f"decomposition of {proposal.source_claim_id}",
    )


def apply_evidence_relation(conn, proposal: EvidenceRelationProposal):
    """Persist a validated claim<->evidence relationship via the domain
    linking function (idempotent)."""
    _require_verdict(conn, validate_evidence_relation, proposal)
    return claims.link_evidence(
        conn, claim_id=proposal.claim_id, evidence_id=proposal.evidence_id,
        role=proposal.role,
    )


def _contradiction_observation_key(proposal: SemanticContradictionProposal) -> str:
    """Deterministic identity of a semantic contradiction observation.

    Encodes the full relationship: claim pair (direction preserved),
    both evidence sides, context, rationale, strength. Two identical
    submissions produce the same key (idempotent); observations that
    differ in ANY of these are distinct records.
    """
    import hashlib

    canonical = {
        "claim_a": proposal.claim_a_id,
        "claim_b": proposal.claim_b_id,
        "evidence_a": sorted(proposal.evidence_a_ids),
        "evidence_b": sorted(proposal.evidence_b_ids),
        "context": proposal.context,
        "rationale": proposal.rationale,
        "strength": proposal.strength,
    }
    raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def apply_semantic_contradiction(
    conn, proposal: SemanticContradictionProposal
):
    """Persist a validated semantic contradiction OBSERVATION.

    Recorded as a ``claim_contradictions`` row with ``kind='semantic'``
    and a deterministic ``observation_key``. Neither claim's
    authoritative status is changed; both findings are preserved.

    Idempotency + concurrency: the UNIQUE index (claim_id, kind,
    observation_key) converges identical observations to one row — a
    concurrent duplicate INSERT raises IntegrityError, which is caught
    and resolved by returning the winning row. Distinct observations
    (different evidence/context/rationale/strength) each persist.
    """
    _require_verdict(conn, validate_semantic_contradiction, proposal)
    claim_a = store.get_claim(conn, proposal.claim_a_id)
    key = _contradiction_observation_key(proposal)
    existing = store.get_contradiction_by_key(
        conn, claim_id=proposal.claim_a_id, kind="semantic",
        observation_key=key,
    )
    if existing is not None:
        return existing
    detail = json.dumps(
        {
            "other_claim_id": proposal.claim_b_id,
            "evidence_a_ids": list(proposal.evidence_a_ids),
            "evidence_b_ids": list(proposal.evidence_b_ids),
            "rationale": proposal.rationale,
            "context": proposal.context,
            "strength": proposal.strength,
        },
        ensure_ascii=False,
    )
    try:
        with write_txn(conn):
            record = store.create_contradiction(
                conn, session_id=claim_a.session_id,
                claim_id=proposal.claim_a_id, kind="semantic",
                detail=detail, observation_key=key,
            )
            store.add_event(
                conn,
                session_id=claim_a.session_id,
                event_type="contradiction_detected",
                detail=json.dumps(
                    {"contradiction_id": record.id,
                     "claim_id": proposal.claim_a_id, "kind": "semantic",
                     "other_claim_id": proposal.claim_b_id},
                    ensure_ascii=False,
                ),
            )
    except sqlite3.IntegrityError:
        # A concurrent identical observation won the race: converge.
        record = store.get_contradiction_by_key(
            conn, claim_id=proposal.claim_a_id, kind="semantic",
            observation_key=key,
        )
        if record is None:
            raise
        return record
    return record


def apply_context(conn, proposal: ClaimContextProposal):
    """Persist a validated claim scope/context (JSON text), MERGING with
    any existing context: proposal keys win, unrelated existing keys are
    preserved. Repeated identical application is idempotent."""
    _require_verdict(conn, validate_context, proposal)
    claim = store.get_claim(conn, proposal.claim_id)
    existing = json.loads(claim.context) if claim.context else {}
    merged = {**existing, **proposal.context}
    store.set_claim_context(
        conn, proposal.claim_id, json.dumps(merged, ensure_ascii=False)
    )
    return store.get_claim(conn, proposal.claim_id)


def apply_coverage(conn, proposal: CoverageProposal) -> CoverageObservation:
    """Validate a coverage proposal and return the authoritative derived
    observation. Coverage is derivable state, so nothing is persisted."""
    _require_verdict(conn, validate_coverage, proposal)
    return coverage_for_claim(conn, proposal.claim_id)
