"""C2 contradictions: detection, adjudication/resolution, synthesis guard.

Detection is deterministic and honest about its limits:

* ``explicit`` contradictions come from ``claim_evidence`` links with
  role ``contradicting`` whose evidence is VALID right now (re-validated
  via ``evidence.validation.validate_evidence`` — an invalid
  contradicting evidence is NOT a contradiction).
* ``structural`` conflicts are a deterministic token-overlap heuristic
  (two claims in the same session with very similar statements but
  disjoint supporting evidence). This is explicitly NOT semantic
  understanding — it surfaces candidates for review.

Resolution uses :class:`ReasonerPort` only as a PROPOSAL source: the
deterministic mapping below decides what actually happens, and every
resolution is persisted in the ``claim_contradictions`` table (dedicated
schema, not stuffed into notes) plus a ``contradiction_resolved`` event.

Synthesis guard: ``synthesis_eligible_claims`` re-evaluates every claim
(so tampered evidence demotes claims BEFORE synthesis), keeps only
``supported`` claims, and additionally excludes claims with an OPEN
explicit contradiction. Structural conflicts are advisory and do not
block. Citations continue to derive from ``claim_evidence`` links.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Sequence

from research import claims, store
from research.model import Claim, Contradiction
from research.reasoner import (
    DECISION_CONTRADICTION_WINS,
    DECISION_SUPPORTING_STANDS,
    DECISION_UNDETERMINED,
    Conflict,
    DeterministicReasoner,
    ReasonerPort,
)
from evidence import validation as c1validation

logger = logging.getLogger(__name__)

#: Jaccard threshold on meaningful terms for a structural conflict.
STRUCTURAL_JACCARD_THRESHOLD = 0.7

EVENT_CONTRADICTION_DETECTED = "contradiction_detected"
EVENT_CONTRADICTION_RESOLVED = "contradiction_resolved"

VALID_DECISIONS = frozenset(
    {DECISION_SUPPORTING_STANDS, DECISION_CONTRADICTION_WINS, DECISION_UNDETERMINED}
)


class ContradictionError(Exception):
    """Base class for contradiction-domain failures."""


class ContradictionNotFoundError(ContradictionError):
    """No such contradiction record."""


class ContradictionClosedError(ContradictionError):
    """The record is already resolved/dismissed."""


class ContradictionInvalidDecisionError(ContradictionError):
    """Unknown resolution decision."""


class ContradictionStateConflictError(ContradictionError):
    """The resolution decision conflicts with the current evidence state."""


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _meaningful_terms(text: str) -> list[str]:
    from docpipe.search import STOPWORDS
    import re

    terms = []
    for raw in re.findall(r"\w+", text, re.UNICODE):
        term = raw.lower()
        if term in STOPWORDS or (len(term) == 1 and not term.isdigit()):
            continue
        terms.append(term)
    return terms


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    if not a and not b:
        return 1.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def _valid_evidence_ids(conn, links) -> tuple[list[str], list[str]]:
    """(valid supporting ids, valid contradicting ids) via C1 re-validation."""
    supporting: list[str] = []
    contradicting: list[str] = []
    for link in links:
        if not c1validation.validate_evidence(conn, link.evidence_id).valid:
            continue  # invalid evidence never counts (detection or status)
        (contradicting if link.role == "contradicting" else supporting).append(
            link.evidence_id
        )
    return supporting, contradicting


def detect_contradictions(conn, session_id: str) -> list[Contradiction]:
    """Detect and persist NEW open contradiction records.

    Idempotent: an existing OPEN record for (claim, kind) is never
    duplicated (no duplicate event either). Claim status is NOT changed
    here — ``claims.evaluate_claim`` owns status.
    """
    session = store.get_session(conn, session_id)
    if session is None:
        raise ValueError(f"session {session_id} not found")

    created: list[Contradiction] = []
    claims_by_id: dict[str, Claim] = {}

    for claim in store.list_claims(conn, session_id):
        claims_by_id[claim.claim_id] = claim
        links = store.get_links_for_claim(conn, claim.claim_id)
        _, valid_contradicting = _valid_evidence_ids(conn, links)

        # 1. Explicit contradiction: valid contradicting evidence.
        if valid_contradicting and store.find_open_contradiction(
            conn, claim.claim_id, "explicit"
        ) is None:
            record = store.create_contradiction(
                conn,
                session_id=session_id,
                claim_id=claim.claim_id,
                kind="explicit",
                detail=json.dumps(
                    {"contradicting_evidence_ids": valid_contradicting}
                ),
            )
            store.add_event(
                conn,
                session_id=session_id,
                event_type=EVENT_CONTRADICTION_DETECTED,
                detail=json.dumps(
                    {
                        "contradiction_id": record.id,
                        "claim_id": claim.claim_id,
                        "kind": "explicit",
                        "evidence_ids": valid_contradicting,
                    }
                ),
            )
            created.append(record)
            logger.info(
                "explicit contradiction detected: claim %s via %s",
                claim.claim_id, valid_contradicting,
            )

    # 2. Structural conflicts: similar statements, disjoint support.
    ordered = sorted(claims_by_id.values(), key=lambda c: c.claim_id)
    for i, claim_a in enumerate(ordered):
        terms_a = _meaningful_terms(claim_a.statement)
        if not terms_a:
            continue
        support_a, _ = _valid_evidence_ids(
            conn, store.get_links_for_claim(conn, claim_a.claim_id)
        )
        for claim_b in ordered[i + 1:]:
            terms_b = _meaningful_terms(claim_b.statement)
            if not terms_b:
                continue
            if _jaccard(terms_a, terms_b) < STRUCTURAL_JACCARD_THRESHOLD:
                continue
            support_b, _ = _valid_evidence_ids(
                conn, store.get_links_for_claim(conn, claim_b.claim_id)
            )
            if set(support_a) & set(support_b):
                continue  # same evidence supports both -> not a conflict
            if store.find_open_contradiction(conn, claim_a.claim_id, "structural") is None:
                record = store.create_contradiction(
                    conn,
                    session_id=session_id,
                    claim_id=claim_a.claim_id,
                    kind="structural",
                    detail=json.dumps(
                        {"other_claim_id": claim_b.claim_id,
                         "overlap": round(_jaccard(terms_a, terms_b), 3)}
                    ),
                )
                store.add_event(
                    conn,
                    session_id=session_id,
                    event_type=EVENT_CONTRADICTION_DETECTED,
                    detail=json.dumps(
                        {
                            "contradiction_id": record.id,
                            "claim_id": claim_a.claim_id,
                            "kind": "structural",
                            "other_claim_id": claim_b.claim_id,
                        }
                    ),
                )
                created.append(record)

    logger.info("contradiction detection: session %s -> %d new record(s)",
                session_id, len(created))
    return created


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _conflict_for_claim(conn, claim: Claim) -> Conflict:
    supporting, contradicting = _valid_evidence_ids(
        conn, store.get_links_for_claim(conn, claim.claim_id)
    )
    return Conflict(
        claim_id=claim.claim_id,
        statement=claim.statement,
        supporting_evidence_ids=tuple(supporting),
        contradicting_evidence_ids=tuple(contradicting),
    )


def resolve_contradiction(
    conn,
    contradiction_id: int,
    *,
    decision: Optional[str] = None,
    note: Optional[str] = None,
    reasoner: Optional[ReasonerPort] = None,
) -> Contradiction:
    """Resolve one OPEN contradiction record, deterministically and with
    full audit. ``decision`` may be supplied by the operator; otherwise the
    reasoner port's :meth:`adjudicate` proposal is used — but the state
    changes below are decided by deterministic rules, never by the
    reasoner directly.

    * ``contradiction_wins``  -> claim is withdrawn (after re-evaluation);
    * ``supporting_stands``   -> record dismissed, claim re-evaluated
      (may become supported when conditions are met);
    * ``undetermined``        -> record stays open, nothing changes.
    """
    record = store.get_contradiction(conn, contradiction_id)
    if record is None:
        raise ContradictionNotFoundError(
            f"contradiction {contradiction_id} not found"
        )
    if record.status != "open":
        raise ContradictionClosedError(
            f"contradiction {contradiction_id} already {record.status}"
        )

    claim = store.get_claim(conn, record.claim_id)
    if claim is None:  # pragma: no cover - FK prevents this
        raise ContradictionError(f"claim {record.claim_id} missing")

    if decision is None:
        reasoner = reasoner or DeterministicReasoner()
        decision = reasoner.adjudicate(_conflict_for_claim(conn, claim)).decision
    if decision not in VALID_DECISIONS:
        raise ContradictionInvalidDecisionError(
            f"unknown decision {decision!r}; expected one of {sorted(VALID_DECISIONS)}"
        )

    adjudicated_by = type(reasoner).__name__ if reasoner is not None else "operator"
    if decision == DECISION_CONTRADICTION_WINS:
        # The contradiction stands: re-evaluate to ground truth, then
        # withdraw. If reality no longer shows valid contradicting
        # evidence, this decision is stale and must not be applied.
        current = claims.evaluate_claim(conn, claim.claim_id)
        _, valid_contradicting = _valid_evidence_ids(
            conn, store.get_links_for_claim(conn, current.claim_id)
        )
        if not valid_contradicting:
            raise ContradictionStateConflictError(
                "decision contradiction_wins but no valid contradicting "
                "evidence remains; re-run detection"
            )
        claims.transition_claim(
            conn, current.claim_id, "withdrawn",
            reason="contradiction_resolution", notes=note,
        )
        store.close_contradiction(
            conn, contradiction_id, "resolved",
            json.dumps({"decision": decision, "note": note,
                        "adjudicated_by": adjudicated_by}),
        )
    elif decision == DECISION_SUPPORTING_STANDS:
        store.close_contradiction(
            conn, contradiction_id, "dismissed",
            json.dumps({"decision": decision, "note": note,
                        "adjudicated_by": adjudicated_by}),
        )
        claims.evaluate_claim(conn, claim.claim_id)  # may promote to supported
    else:  # undetermined — keep open, record the note for the audit trail
        store.add_event(
            conn,
            session_id=record.session_id,
            event_type=EVENT_CONTRADICTION_RESOLVED,
            detail=json.dumps(
                {"contradiction_id": contradiction_id,
                 "claim_id": claim.claim_id, "decision": decision,
                 "note": note, "applied": False}
            ),
        )
        return store.get_contradiction(conn, contradiction_id)

    store.add_event(
        conn,
        session_id=record.session_id,
        event_type=EVENT_CONTRADICTION_RESOLVED,
        detail=json.dumps(
            {"contradiction_id": contradiction_id,
             "claim_id": claim.claim_id, "decision": decision,
             "note": note, "applied": True}
        ),
    )
    logger.info("contradiction %s resolved: %s (claim %s)",
                contradiction_id, decision, claim.claim_id)
    return store.get_contradiction(conn, contradiction_id)


# ---------------------------------------------------------------------------
# Synthesis guard
# ---------------------------------------------------------------------------

def synthesis_eligible_claims(conn, session_id: str) -> list[Claim]:
    """The ONLY way claims may enter synthesis.

    1. every claim is re-evaluated first (tampered supporting evidence
       demotes the claim BEFORE synthesis — never after),
    2. only ``supported`` claims pass,
    3. a claim with an OPEN explicit contradiction is excluded even if it
       is currently marked supported (stale state cannot slip through).

    Structural conflicts are advisory and do not block. Citations derive
    from ``claim_evidence`` links at synthesis time.
    """
    eligible: list[Claim] = []
    for claim in store.list_claims(conn, session_id):
        fresh = claims.evaluate_claim(conn, claim.claim_id)
        if fresh.status != "supported":
            continue
        if store.find_open_contradiction(conn, fresh.claim_id, "explicit") is not None:
            logger.info("synthesis guard: %s blocked (open contradiction)",
                        fresh.claim_id)
            continue
        eligible.append(fresh)
    return eligible
