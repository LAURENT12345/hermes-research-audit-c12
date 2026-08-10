"""Sufficiency tests: claim-level, session-level, plan inputs."""

from __future__ import annotations

import pytest

import docpipe.db as c0db
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, store
from research.sufficiency import (
    PlanInputs,
    build_plan_inputs,
    check_claim_sufficiency,
    check_session_sufficiency,
)


@pytest.fixture
def seeded(tmp_path):
    """Two evidence objects in one session (min_supporting=1)."""
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    doc_id = c0db.insert_document(
        conn, title="Paper", file_path="/tmp/a.pdf", file_hash="ha",
        status="ready",
    )
    chunk_ids = []
    for i, text in enumerate(
        ["First evidence content.", "Second evidence content."]
    ):
        chunk_ids.append(
            c0db.insert_chunk(
                conn, document_id=doc_id,
                chunk=Chunk(chunk_index=i, content=text,
                            normalized_content=text, page_number=i + 1,
                            section_path=None, token_count=3),
            )
        )
    ev1 = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_ids[0], evidence_type="chunk",
        source_type="local_pdf", source_title="Paper",
        content="First evidence content.", page_number=1, section_path=None,
        locator={"page": 1, "chunk_id": chunk_ids[0]},
    )
    ev2 = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_ids[1], evidence_type="chunk",
        source_type="local_pdf", source_title="Paper",
        content="Second evidence content.", page_number=2, section_path=None,
        locator={"page": 2, "chunk_id": chunk_ids[1]},
    )
    session = store.create_session(conn, question="Why scale?")
    conn.close()
    return db_path, session.session_id, ev1.evidence_id, ev2.evidence_id


def _make_claim(conn, session_id, statement):
    return claims.create_claim(conn, session_id=session_id, statement=statement)


# ---------------------------------------------------------------------------
# Claim-level
# ---------------------------------------------------------------------------

def test_claim_sufficiency_zero_evidence(seeded):
    db_path, session_id, _, _ = seeded
    conn = store.connect(db_path)
    try:
        c = _make_claim(conn, session_id, "Zero evidence claim")
        s = check_claim_sufficiency(conn, c.claim_id)
        assert s.sufficient is False
        assert s.supporting_valid_count == 0
        assert s.required_support == 1
        assert s.reasons
    finally:
        conn.close()


def test_claim_sufficiency_one_valid_support(seeded):
    db_path, session_id, ev1, _ = seeded
    conn = store.connect(db_path)
    try:
        c = _make_claim(conn, session_id, "Supported claim")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        s = check_claim_sufficiency(conn, c.claim_id)
        assert s.sufficient is True
        assert s.supporting_valid_count == 1
        assert s.supporting_invalid_count == 0
        assert s.contradicting_valid_count == 0
        assert s.reasons == []
    finally:
        conn.close()


def test_claim_sufficiency_requires_two(seeded):
    db_path, session_id, ev1, _ = seeded
    conn = store.connect(db_path)
    try:
        s2 = store.create_session(conn, question="Q2", min_supporting=2)
        c = _make_claim(conn, s2.session_id, "Needs two")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        s = check_claim_sufficiency(conn, c.claim_id)
        assert s.sufficient is False
        assert s.supporting_valid_count == 1
        assert "1 < required 2" in s.reasons[0]
    finally:
        conn.close()


def test_claim_sufficiency_invalid_evidence_not_counted(seeded):
    """Invariant 3: invalid supporting evidence cannot count."""
    db_path, session_id, ev1, ev2 = seeded
    conn = store.connect(db_path)
    try:
        c = _make_claim(conn, session_id, "Mixed evidence claim")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="supporting")
        conn.execute(
            "UPDATE chunks SET content = 'TAMPERED' WHERE "
            "id = (SELECT chunk_id FROM evidence WHERE id = ?)", (ev2,)
        )
        s = check_claim_sufficiency(conn, c.claim_id)
        assert s.supporting_valid_count == 1
        assert s.supporting_invalid_count == 1
        assert s.sufficient is True  # min_supporting=1 still met
    finally:
        conn.close()


def test_claim_sufficiency_contradiction_blocks(seeded):
    db_path, session_id, ev1, ev2 = seeded
    conn = store.connect(db_path)
    try:
        c = _make_claim(conn, session_id, "Contradicted claim")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        s = check_claim_sufficiency(conn, c.claim_id)
        assert s.sufficient is False
        assert s.contradicting_valid_count == 1
        assert any("contradicting" in r for r in s.reasons)
    finally:
        conn.close()


def test_claim_sufficiency_mixed_valid_invalid_contradiction(seeded):
    db_path, session_id, ev1, ev2 = seeded
    conn = store.connect(db_path)
    try:
        c = _make_claim(conn, session_id, "All mixed")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        conn.execute(
            "UPDATE chunks SET content = 'TAMPERED' WHERE "
            "id = (SELECT chunk_id FROM evidence WHERE id = ?)", (ev1,)
        )
        s = check_claim_sufficiency(conn, c.claim_id)
        # Invalid support doesn't count; valid contradiction still blocks.
        assert s.supporting_valid_count == 0
        assert s.supporting_invalid_count == 1
        assert s.contradicting_valid_count == 1
        assert s.sufficient is False
    finally:
        conn.close()


def test_claim_sufficiency_missing_claim(seeded):
    db_path, *_ = seeded
    conn = store.connect(db_path)
    try:
        with pytest.raises(ValueError, match="not found"):
            check_claim_sufficiency(conn, "C-999999")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Session-level
# ---------------------------------------------------------------------------

def _session_with_claims(conn, session_id, statuses):
    """Create claims and force statuses directly for state assessment."""
    claims_list = []
    for i, status in enumerate(statuses):
        c = _make_claim(conn, session_id, f"Claim {i}")
        if status != "proposed":
            store.update_claim_status(conn, c.claim_id, status)
        claims_list.append(c)
    return claims_list


def test_session_all_resolved_synthesis_ready(seeded):
    db_path, session_id, ev1, _ = seeded
    conn = store.connect(db_path)
    try:
        c = _make_claim(conn, session_id, "Real supported claim")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.evaluate_claim(conn, c.claim_id)  # -> supported (real path)
        s = check_session_sufficiency(conn, session_id)
        assert s.total_claims == 1
        assert s.supported_count == 1
        assert s.insufficient_count == 0
        assert s.contradicted_count == 0
        assert s.has_gaps is False
        assert s.synthesis_ready is True
        assert s.needs_more_retrieval is False
        assert s.gaps == []
    finally:
        conn.close()


def test_session_with_insufficient_claim_has_gaps(seeded):
    db_path, session_id, _, _ = seeded
    conn = store.connect(db_path)
    try:
        _session_with_claims(conn, session_id, ["insufficient"])
        s = check_session_sufficiency(conn, session_id)
        assert s.has_gaps is True
        assert s.needs_more_retrieval is True
        assert s.synthesis_ready is False
        assert len(s.gaps) == 1
        assert s.gaps[0]["status"] == "insufficient"
    finally:
        conn.close()


def test_session_with_contradicted_claim(seeded):
    db_path, session_id, _, _ = seeded
    conn = store.connect(db_path)
    try:
        _session_with_claims(conn, session_id, ["contradicted"])
        s = check_session_sufficiency(conn, session_id)
        assert s.has_gaps is True
        assert s.synthesis_ready is False
        assert s.gaps[0]["status"] == "contradicted"
        assert s.unresolved_count == 1
    finally:
        conn.close()


def test_session_zero_claims(seeded):
    db_path, session_id, _, _ = seeded
    conn = store.connect(db_path)
    try:
        s = check_session_sufficiency(conn, session_id)
        assert s.total_claims == 0
        assert s.has_gaps is False
        assert s.synthesis_ready is False  # no supported claim
        assert s.gaps == []
    finally:
        conn.close()


def test_session_mixed_statuses(seeded):
    db_path, session_id, ev1, _ = seeded
    conn = store.connect(db_path)
    try:
        supported = _make_claim(conn, session_id, "Supported one")
        claims.link_evidence(conn, claim_id=supported.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.evaluate_claim(conn, supported.claim_id)
        _session_with_claims(conn, session_id, ["insufficient"])
        s = check_session_sufficiency(conn, session_id)
        assert s.supported_count == 1
        assert s.insufficient_count == 1
        assert s.has_gaps is True
        assert s.synthesis_ready is False  # gaps block synthesis
        assert s.unresolved_count == 1
    finally:
        conn.close()


def test_session_missing(seeded):
    db_path, *_ = seeded
    conn = store.connect(db_path)
    try:
        with pytest.raises(ValueError, match="not found"):
            check_session_sufficiency(conn, "S-999999")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Plan inputs (deterministic, reasoner NOT called)
# ---------------------------------------------------------------------------

def test_build_plan_inputs_question_and_gaps(seeded):
    db_path, session_id, _, _ = seeded
    conn = store.connect(db_path)
    try:
        _session_with_claims(conn, session_id, ["insufficient", "contradicted"])
        inputs = build_plan_inputs(conn, session_id)
        assert isinstance(inputs, PlanInputs)
        assert inputs.question == "Why scale?"
        assert len(inputs.gaps) == 2
        gap_ids = {g.claim_id for g in inputs.gaps}
        assert len(gap_ids) == 2
        assert all(g.statement for g in inputs.gaps)
    finally:
        conn.close()


def test_build_plan_inputs_no_gaps(seeded):
    db_path, session_id, _, _ = seeded
    conn = store.connect(db_path)
    try:
        _session_with_claims(conn, session_id, ["supported"])
        inputs = build_plan_inputs(conn, session_id)
        assert inputs.question == "Why scale?"
        assert inputs.gaps == []
    finally:
        conn.close()
