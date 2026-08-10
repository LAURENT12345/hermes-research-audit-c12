"""Claims tests: creation, transitions, linking, evaluation."""

from __future__ import annotations

import json
import sqlite3

import pytest

import docpipe.db as c0db
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, store
from research.model import ClaimEvidenceLink
from research.reasoner import ClaimProposal


@pytest.fixture
def seeded(tmp_path):
    """Session with two evidence objects (from a real C0 doc+chunks)."""
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    doc_id = c0db.insert_document(
        conn, title="Paper", file_path="/tmp/a.pdf", file_hash="ha",
        status="ready",
    )
    chunk_ids = []
    for i, text in enumerate(
        ["The dot product has variance dk.", "Softmax gradients get small."]
    ):
        chunk_ids.append(
            c0db.insert_chunk(
                conn, document_id=doc_id,
                chunk=Chunk(chunk_index=i, content=text,
                            normalized_content=text, page_number=i + 1,
                            section_path=None, token_count=5),
            )
        )
    ev1 = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_ids[0], evidence_type="chunk",
        source_type="local_pdf", source_title="Paper",
        content="The dot product has variance dk.", page_number=1,
        section_path=None, locator={"page": 1, "chunk_id": chunk_ids[0]},
    )
    ev2 = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_ids[1], evidence_type="chunk",
        source_type="local_pdf", source_title="Paper",
        content="Softmax gradients get small.", page_number=2,
        section_path=None, locator={"page": 2, "chunk_id": chunk_ids[1]},
    )
    session = store.create_session(conn, question="Why scale?")
    conn.close()
    return db_path, session.session_id, ev1.evidence_id, ev2.evidence_id


def _c(seeded):
    db_path, session_id, ev1, ev2 = seeded
    return db_path, session_id, ev1, ev2


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def test_create_claim_manual(seeded):
    db_path, session_id, _, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        claim = claims.create_claim(
            conn, session_id=session_id, statement="  Dot product   variance dk. "
        )
        assert claim.claim_id == "C-000001"
        assert claim.status == "proposed"
        assert claim.statement == "Dot product   variance dk."
        # event recorded
        events = store.list_events(conn, session_id)
        assert events[0].event_type == "claim_proposed"
        assert json.loads(events[0].detail) == {"claim_id": "C-000001"}
    finally:
        conn.close()


def test_create_claim_from_proposal(seeded):
    db_path, session_id, ev1, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        proposal = ClaimProposal(statement="Variance is dk.", evidence_ids=(ev1,))
        claim = claims.create_claim_from_proposal(
            conn, session_id=session_id, proposal=proposal
        )
        assert claim.statement == "Variance is dk."
        assert claim.status == "proposed"  # proposal evidence never auto-supports
        assert store.get_links_for_claim(conn, claim.claim_id) == []
    finally:
        conn.close()


def test_create_claim_empty_statement(seeded):
    db_path, session_id, _, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        with pytest.raises(claims.InvalidClaimError, match="empty"):
            claims.create_claim(conn, session_id=session_id, statement="   ")
    finally:
        conn.close()


def test_create_claim_too_long(seeded):
    db_path, session_id, _, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        with pytest.raises(claims.InvalidClaimError, match="exceeds"):
            claims.create_claim(conn, session_id=session_id, statement="x" * 401)
    finally:
        conn.close()


def test_duplicate_claim_same_id_no_event(seeded):
    """Invariant 6: duplicate claim -> same claim_id, no duplicate event."""
    db_path, session_id, _, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        c1 = claims.create_claim(conn, session_id=session_id, statement="Same claim")
        c2 = claims.create_claim(
            conn, session_id=session_id, statement="  same   claim "
        )
        assert c1.claim_id == c2.claim_id
        proposed_events = [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "claim_proposed"
        ]
        assert len(proposed_events) == 1
    finally:
        conn.close()


def test_same_statement_different_sessions(seeded):
    db_path, session_id, _, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        c1 = claims.create_claim(conn, session_id=session_id, statement="Shared")
        s2 = store.create_session(conn, question="Other?")
        c2 = claims.create_claim(conn, session_id=s2.session_id, statement="Shared")
        assert c1.claim_id != c2.claim_id
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def test_valid_transitions(seeded):
    db_path, session_id, _, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        assert claims.transition_claim(conn, c.claim_id, "supported",
                                       reason="test").status == "supported"
        assert claims.transition_claim(conn, c.claim_id, "insufficient",
                                       reason="test").status == "insufficient"
        assert claims.transition_claim(conn, c.claim_id, "supported",
                                       reason="test").status == "supported"
        assert claims.transition_claim(conn, c.claim_id, "contradicted",
                                       reason="test").status == "contradicted"
        assert claims.transition_claim(conn, c.claim_id, "withdrawn",
                                       reason="test").status == "withdrawn"
        assert claims.transition_claim(conn, c.claim_id, "invalidated",
                                       reason="test").status == "invalidated"
    finally:
        conn.close()


def test_invalid_transition_rejected(seeded):
    """Invalid transitions raise a clear domain error, never silent."""
    db_path, session_id, _, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        with pytest.raises(claims.ClaimTransitionError, match="not allowed"):
            claims.transition_claim(conn, c.claim_id, "withdrawn")  # proposed->withdrawn
        with pytest.raises(claims.ClaimTransitionError):
            claims.transition_claim(conn, c.claim_id, "proposed")  # self-loop
        assert store.get_claim(conn, c.claim_id).status == "proposed"
    finally:
        conn.close()


def test_transition_writes_event(seeded):
    """Invariant 5: every status change has a research_event."""
    db_path, session_id, _, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        claims.transition_claim(conn, c.claim_id, "insufficient",
                                reason="insufficient_evidence", notes="why")
        events = [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "claim_status_changed"
        ]
        assert len(events) == 1
        detail = json.loads(events[0].detail)
        assert detail == {
            "claim_id": c.claim_id, "from": "proposed", "to": "insufficient",
            "reason": "insufficient_evidence",
        }
        updated = store.get_claim(conn, c.claim_id)
        assert updated.notes == "why"
    finally:
        conn.close()


def test_transition_missing_claim(seeded):
    db_path, *_ = _c(seeded)
    conn = store.connect(db_path)
    try:
        with pytest.raises(claims.ClaimNotFoundError):
            claims.transition_claim(conn, "C-999999", "supported")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Evidence linking
# ---------------------------------------------------------------------------

def test_link_supporting_and_contradicting(seeded):
    db_path, session_id, ev1, ev2 = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        link1 = claims.link_evidence(conn, claim_id=c.claim_id,
                                     evidence_id=ev1, role="supporting")
        link2 = claims.link_evidence(conn, claim_id=c.claim_id,
                                     evidence_id=ev2, role="contradicting")
        assert isinstance(link1, ClaimEvidenceLink)
        assert link1.role == "supporting" and link2.role == "contradicting"
        assert len(store.get_links_for_claim(conn, c.claim_id)) == 2
        # link event with metadata only
        events = [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "evidence_linked"
        ]
        assert len(events) == 2
        assert "content" not in events[0].detail
    finally:
        conn.close()


def test_duplicate_link_idempotent(seeded):
    """Invariant 7: duplicate link -> one link only."""
    db_path, session_id, ev1, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        l1 = claims.link_evidence(conn, claim_id=c.claim_id,
                                  evidence_id=ev1, role="supporting")
        l2 = claims.link_evidence(conn, claim_id=c.claim_id,
                                  evidence_id=ev1, role="supporting")
        assert l1 == l2
        assert len(store.get_links_for_claim(conn, c.claim_id)) == 1
    finally:
        conn.close()


def test_link_errors(seeded):
    db_path, session_id, ev1, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        with pytest.raises(claims.ClaimNotFoundError):
            claims.link_evidence(conn, claim_id="C-999999", evidence_id=ev1)
        with pytest.raises(claims.EvidenceNotFoundError):
            claims.link_evidence(conn, claim_id=c.claim_id, evidence_id="E-999999")
        with pytest.raises(claims.InvalidRoleError):
            claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                                 role="ambiguous")
    finally:
        conn.close()


def test_link_does_not_change_status(seeded):
    db_path, session_id, ev1, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        assert store.get_claim(conn, c.claim_id).status == "proposed"
    finally:
        conn.close()


def test_many_to_many_links(seeded):
    db_path, session_id, ev1, ev2 = _c(seeded)
    conn = store.connect(db_path)
    try:
        c1 = claims.create_claim(conn, session_id=session_id, statement="One")
        c2 = claims.create_claim(conn, session_id=session_id, statement="Two")
        # multiple evidence per claim
        claims.link_evidence(conn, claim_id=c1.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c1.claim_id, evidence_id=ev2,
                             role="supporting")
        # one evidence supporting multiple claims
        claims.link_evidence(conn, claim_id=c2.claim_id, evidence_id=ev1,
                             role="supporting")
        assert len(store.get_links_for_claim(conn, c1.claim_id)) == 2
        assert len(store.get_links_for_evidence(conn, ev1)) == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_evaluate_no_evidence_insufficient(seeded):
    db_path, session_id, _, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        result = claims.evaluate_claim(conn, c.claim_id)
        assert result.status == "insufficient"
        event = [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "claim_status_changed"
        ][-1]
        assert json.loads(event.detail)["reason"] == "insufficient_evidence"
    finally:
        conn.close()


def test_evaluate_one_valid_support_supported(seeded):
    db_path, session_id, ev1, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        assert claims.evaluate_claim(conn, c.claim_id).status == "supported"
    finally:
        conn.close()


def test_evaluate_valid_contradiction_blocks(seeded):
    db_path, session_id, ev1, ev2 = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        result = claims.evaluate_claim(conn, c.claim_id)
        assert result.status == "contradicted"
        event = [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "claim_status_changed"
        ][-1]
        assert json.loads(event.detail)["reason"] == "contradicting_evidence"
    finally:
        conn.close()


def test_evaluate_supported_claim_then_contradiction_demotes(seeded):
    """A supported claim with a NEW valid contradicting evidence moves to
    contradicted (matrix extension, required by the evaluation rules)."""
    db_path, session_id, ev1, ev2 = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        assert claims.evaluate_claim(conn, c.claim_id).status == "supported"
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        assert claims.evaluate_claim(conn, c.claim_id).status == "contradicted"
    finally:
        conn.close()


def test_evaluate_invalid_support_does_not_count(seeded):
    """Invariant 3: invalid supporting evidence cannot count toward
    sufficiency. With min_supporting=1 one valid support still suffices."""
    db_path, session_id, ev1, ev2 = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="supporting")
        # Tamper chunk 2's content -> ev2 becomes invalid.
        conn.execute(
            "UPDATE chunks SET content = 'TAMPERED' WHERE "
            "id = (SELECT chunk_id FROM evidence WHERE id = ?)", (ev2,)
        )
        # ev1 still valid -> supported (min_supporting=1)
        assert claims.evaluate_claim(conn, c.claim_id).status == "supported"
    finally:
        conn.close()


def test_evaluate_supported_demoted_to_insufficient_on_tamper(seeded):
    db_path, session_id, ev1, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        assert claims.evaluate_claim(conn, c.claim_id).status == "supported"

        conn.execute(
            "UPDATE chunks SET content = 'TAMPERED' WHERE "
            "id = (SELECT chunk_id FROM evidence WHERE id = ?)", (ev1,)
        )
        result = claims.evaluate_claim(conn, c.claim_id)
        assert result.status == "insufficient"
        event = [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "claim_status_changed"
        ][-1]
        detail = json.loads(event.detail)
        assert detail["from"] == "supported"
        assert detail["to"] == "insufficient"
        assert detail["reason"] == "evidence_invalid"
        # Evidence itself is never repaired by the evaluator.
        assert c1store.get_evidence(conn, ev1).content == "The dot product has variance dk."
    finally:
        conn.close()


def test_evaluate_no_change_no_event(seeded):
    db_path, session_id, ev1, _ = _c(seeded)
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="T")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.evaluate_claim(conn, c.claim_id)  # -> supported
        before = len(store.list_events(conn, session_id))
        claims.evaluate_claim(conn, c.claim_id)  # still supported -> no-op
        assert len(store.list_events(conn, session_id)) == before
    finally:
        conn.close()


def test_evaluate_min_supporting_two(seeded):
    db_path, session_id, ev1, ev2 = _c(seeded)
    conn = store.connect(db_path)
    try:
        store.update_session_status(conn, session_id, "active")  # ensure exists
        # Create a fresh session with min_supporting=2.
        s2 = store.create_session(conn, question="Q2", min_supporting=2)
        c = claims.create_claim(conn, session_id=s2.session_id, statement="T")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        assert claims.evaluate_claim(conn, c.claim_id).status == "insufficient"
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="supporting")
        assert claims.evaluate_claim(conn, c.claim_id).status == "supported"
    finally:
        conn.close()
