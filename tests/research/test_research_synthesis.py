"""Synthesis tests: invariants, exclusion, citations, gaps, determinism,
persistence, idempotency, tamper/staleness."""

from __future__ import annotations

import json

import pytest

import docpipe.db as c0db
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, store
from research.synthesis import (
    GAP_REASON_CONTRADICTED,
    GAP_REASON_INSUFFICIENT,
    ResearchAnswer,
    SynthesisError,
    SynthesisNotAllowedError,
    build_answer,
    get_answer,
    load_answer,
    synthesize,
)

CHUNK_TEXTS = [
    "Scaled dot product attention divides by the square root of dk.",
    "Positional encoding adds order information to the model.",
    "The model has no recurrence and no convolution.",
]


@pytest.fixture
def env(tmp_path):
    """One document, three chunks, a draft session, and two evidence."""
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    doc_id = c0db.insert_document(
        conn, title="Paper", file_path="/tmp/paper.pdf", file_hash="ha",
        status="ready",
    )
    chunk_ids = []
    for i, text in enumerate(CHUNK_TEXTS):
        chunk_ids.append(
            c0db.insert_chunk(
                conn, document_id=doc_id,
                chunk=Chunk(chunk_index=i, content=text,
                            normalized_content=text, page_number=i + 1,
                            section_path=None, token_count=len(text.split())),
            )
        )
    ev1 = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_ids[0], evidence_type="chunk",
        source_type="local_pdf", source_title="Paper", content=CHUNK_TEXTS[0],
        page_number=1, section_path=None,
        locator={"page": 1, "chunk_id": chunk_ids[0]},
    )
    ev2 = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_ids[1], evidence_type="chunk",
        source_type="local_pdf", source_title="Paper", content=CHUNK_TEXTS[1],
        page_number=2, section_path=None,
        locator={"page": 2, "chunk_id": chunk_ids[1]},
    )
    session = store.create_session(conn, question="Why scale by sqrt dk?")
    store.update_session_status(conn, session.session_id, "active")
    conn.close()
    return db_path, doc_id, session.session_id, ev1.evidence_id, ev2.evidence_id


def _supported_claim(conn, session_id, statement, evidence_id):
    claim = claims.create_claim(conn, session_id=session_id, statement=statement)
    claims.link_evidence(conn, claim_id=claim.claim_id, evidence_id=evidence_id,
                         role="supporting")
    claims.evaluate_claim(conn, claim.claim_id)
    return claim


def _tamper_chunk_of(conn, evidence_id: str) -> None:
    conn.execute(
        "UPDATE chunks SET content = 'TAMPERED' WHERE id = "
        "(SELECT chunk_id FROM evidence WHERE id = ?)",
        (evidence_id,),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Basic
# ---------------------------------------------------------------------------

def test_supported_claim_becomes_statement_with_citation(env):
    db_path, _, session_id, ev1, _ = env
    conn = store.connect(db_path)
    claim = _supported_claim(conn, session_id, CHUNK_TEXTS[0], ev1)
    answer = build_answer(conn, session_id)
    assert len(answer.statements) == 1
    stmt = answer.statements[0]
    assert stmt.claim_id == claim.claim_id
    assert stmt.statement == CHUNK_TEXTS[0]
    assert len(stmt.citations) == 1
    cit = stmt.citations[0]
    assert cit.evidence_id == ev1
    assert cit.document_id == env[1]
    assert cit.page == 1
    assert cit.locator == {"page": 1, "chunk_id": cit.locator["chunk_id"]}
    conn.close()


def test_multiple_supported_claims_ordered(env):
    db_path, _, session_id, ev1, ev2 = env
    conn = store.connect(db_path)
    c2 = _supported_claim(conn, session_id, CHUNK_TEXTS[1], ev2)
    c1 = _supported_claim(conn, session_id, CHUNK_TEXTS[0], ev1)
    answer = build_answer(conn, session_id)
    assert [s.claim_id for s in answer.statements] == sorted([c1.claim_id, c2.claim_id])
    conn.close()


def test_answer_serialization_roundtrip(env):
    db_path, _, session_id, ev1, _ = env
    conn = store.connect(db_path)
    _supported_claim(conn, session_id, CHUNK_TEXTS[0], ev1)
    answer = build_answer(conn, session_id)
    restored = ResearchAnswer.from_json(answer.to_json())
    assert restored == answer
    assert restored.to_json() == answer.to_json()
    conn.close()


# ---------------------------------------------------------------------------
# Exclusion
# ---------------------------------------------------------------------------

def test_non_supported_statuses_excluded(env):
    db_path, _, session_id, ev1, ev2 = env
    conn = store.connect(db_path)
    # supported
    _supported_claim(conn, session_id, CHUNK_TEXTS[0], ev1)
    # insufficient (no evidence)
    insuff = claims.create_claim(conn, session_id=session_id,
                                 statement="Claim with no evidence.")
    # contradicted (valid contradicting evidence)
    contrad = claims.create_claim(conn, session_id=session_id,
                                  statement="Claim that is contradicted.")
    claims.link_evidence(conn, claim_id=contrad.claim_id, evidence_id=ev1,
                         role="supporting")
    claims.link_evidence(conn, claim_id=contrad.claim_id, evidence_id=ev2,
                         role="contradicting")
    # withdrawn (terminal) and invalidated (terminal) — set via the raw
    # store write; these statuses are normally reached through
    # contradiction resolution / invalidation flows.
    withdrawn = claims.create_claim(conn, session_id=session_id,
                                    statement="Claim withdrawn deliberately.")
    store.update_claim_status(conn, withdrawn.claim_id, "withdrawn",
                              notes="test fixture")
    invalidated = claims.create_claim(conn, session_id=session_id,
                                      statement="Claim invalidated.")
    store.update_claim_status(conn, invalidated.claim_id, "invalidated",
                              notes="test fixture")

    answer = build_answer(conn, session_id)
    statement_ids = {s.claim_id for s in answer.statements}
    assert insuff.claim_id not in statement_ids
    assert contrad.claim_id not in statement_ids
    assert withdrawn.claim_id not in statement_ids
    assert invalidated.claim_id not in statement_ids
    # insuff + contrad appear as gaps; withdrawn/invalidated nowhere.
    gap_ids = {g.claim_id for g in answer.gaps}
    assert insuff.claim_id in gap_ids
    assert contrad.claim_id in gap_ids
    assert withdrawn.claim_id not in gap_ids
    assert invalidated.claim_id not in gap_ids
    conn.close()


def test_invalidated_claim_evaluation_no_crash(env):
    """Terminal claims (withdrawn/invalidated) must not crash the
    evaluator that synthesis runs over every claim."""
    db_path, _, session_id, ev1, _ = env
    conn = store.connect(db_path)
    c = claims.create_claim(conn, session_id=session_id,
                            statement="Terminal claim with valid evidence.")
    claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                         role="supporting")
    store.update_claim_status(conn, c.claim_id, "withdrawn", notes="fixture")
    result = claims.evaluate_claim(conn, c.claim_id)  # must not raise
    assert result.status == "withdrawn"
    answer = build_answer(conn, session_id)
    assert answer.statements == ()
    conn.close()


# ---------------------------------------------------------------------------
# Evidence integrity
# ---------------------------------------------------------------------------

def test_tampered_evidence_demotes_and_excludes(env):
    db_path, _, session_id, ev1, _ = env
    conn = store.connect(db_path)
    claim = _supported_claim(conn, session_id, CHUNK_TEXTS[0], ev1)
    assert claims.evaluate_claim(conn, claim.claim_id).status == "supported"

    _tamper_chunk_of(conn, ev1)
    answer = build_answer(conn, session_id)  # re-validation is live
    assert answer.statements == ()
    gap_ids = {g.claim_id for g in answer.gaps}
    assert claim.claim_id in gap_ids
    assert all(
        g.reason == GAP_REASON_INSUFFICIENT for g in answer.gaps
        if g.claim_id == claim.claim_id
    )
    conn.close()


def test_invalid_evidence_never_cited(env):
    """Even a supported claim cannot cite evidence that fails validation."""
    db_path, _, session_id, ev1, ev2 = env
    conn = store.connect(db_path)
    claim = claims.create_claim(conn, session_id=session_id,
                                statement=CHUNK_TEXTS[0])
    claims.link_evidence(conn, claim_id=claim.claim_id, evidence_id=ev1,
                         role="supporting")
    claims.link_evidence(conn, claim_id=claim.claim_id, evidence_id=ev2,
                         role="supporting")
    claims.evaluate_claim(conn, claim.claim_id)  # supported (2 valid)
    _tamper_chunk_of(conn, ev2)  # one support becomes invalid
    answer = build_answer(conn, session_id)
    stmt = answer.statements[0]
    cited = [c.evidence_id for c in stmt.citations]
    assert ev1 in cited
    assert ev2 not in cited  # invalid evidence never appears as citation
    conn.close()


def test_valid_contradiction_blocks_synthesis(env):
    db_path, _, session_id, ev1, ev2 = env
    conn = store.connect(db_path)
    claim = claims.create_claim(conn, session_id=session_id,
                                statement="Disputed claim.")
    claims.link_evidence(conn, claim_id=claim.claim_id, evidence_id=ev1,
                         role="supporting")
    claims.link_evidence(conn, claim_id=claim.claim_id, evidence_id=ev2,
                         role="contradicting")
    claims.evaluate_claim(conn, claim.claim_id)
    answer = build_answer(conn, session_id)
    assert answer.statements == ()
    assert any(g.claim_id == claim.claim_id
               and g.reason == GAP_REASON_CONTRADICTED for g in answer.gaps)
    conn.close()


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------

def test_no_supported_claims_empty_statements(env):
    db_path, _, session_id, _, _ = env
    conn = store.connect(db_path)
    claims.create_claim(conn, session_id=session_id, statement="No evidence.")
    answer = build_answer(conn, session_id)
    assert answer.statements == ()
    assert len(answer.gaps) == 1
    conn.close()


def test_mixed_supported_and_insufficient(env):
    db_path, _, session_id, ev1, _ = env
    conn = store.connect(db_path)
    _supported_claim(conn, session_id, CHUNK_TEXTS[0], ev1)
    claims.create_claim(conn, session_id=session_id,
                        statement="Gap claim with no evidence.")
    answer = build_answer(conn, session_id)
    assert len(answer.statements) == 1
    assert len(answer.gaps) == 1
    assert answer.gaps[0].reason == GAP_REASON_INSUFFICIENT
    conn.close()


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------

def test_traceability_chain(env):
    """statement -> claim_id -> evidence_id -> document_id -> page -> locator."""
    db_path, doc_id, session_id, ev1, _ = env
    conn = store.connect(db_path)
    claim = _supported_claim(conn, session_id, CHUNK_TEXTS[0], ev1)
    answer = build_answer(conn, session_id)
    stmt = answer.statements[0]
    assert stmt.claim_id == claim.claim_id
    cit = stmt.citations[0]
    link = store.get_links_for_claim(conn, claim.claim_id)[0]
    assert cit.evidence_id == link.evidence_id
    evidence = c1store.get_evidence(conn, cit.evidence_id)
    assert cit.document_id == evidence.document_id == doc_id
    assert cit.page == evidence.page_number
    assert cit.locator == evidence.locator
    chunk = c0db.get_chunk(conn, evidence.chunk_id)
    assert chunk["document_id"] == doc_id
    conn.close()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_state_identical_answer(env):
    db_path, _, session_id, ev1, ev2 = env
    conn = store.connect(db_path)
    _supported_claim(conn, session_id, CHUNK_TEXTS[0], ev1)
    _supported_claim(conn, session_id, CHUNK_TEXTS[1], ev2)
    first = build_answer(conn, session_id).to_json()
    second = build_answer(conn, session_id).to_json()
    assert first == second
    conn.close()


# ---------------------------------------------------------------------------
# Persistence / idempotency / staleness
# ---------------------------------------------------------------------------

def test_synthesize_persists_and_loads_after_reconnect(env):
    db_path, _, session_id, ev1, _ = env
    conn = store.connect(db_path)
    _supported_claim(conn, session_id, CHUNK_TEXTS[0], ev1)
    store.update_session_status(conn, session_id, "complete")
    answer = synthesize(conn, session_id)
    conn.close()

    conn2 = store.connect(db_path)  # restart
    try:
        stored = load_answer(conn2, session_id)
        assert stored == answer
        assert stored.to_json() == answer.to_json()
        assert conn2.execute(
            "SELECT final_answer FROM research_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()["final_answer"] == answer.to_json()
    finally:
        conn2.close()


def test_synthesize_idempotent_no_duplicate_event(env):
    db_path, _, session_id, ev1, _ = env
    conn = store.connect(db_path)
    _supported_claim(conn, session_id, CHUNK_TEXTS[0], ev1)
    store.update_session_status(conn, session_id, "complete")

    first = synthesize(conn, session_id)
    second = synthesize(conn, session_id)
    assert first == second
    events = [
        e for e in store.list_events(conn, session_id)
        if e.event_type == "synthesized"
    ]
    assert len(events) == 1
    detail = json.loads(events[0].detail)
    assert detail == {"session_id": session_id, "claim_count": 1,
                      "citation_count": 1, "gap_count": 0}
    assert "content" not in events[0].detail
    conn.close()


def test_synthesize_rejects_draft(env):
    db_path, _, session_id, _, _ = env
    conn = store.connect(db_path)
    store.update_session_status(conn, session_id, "draft")
    with pytest.raises(SynthesisNotAllowedError):
        synthesize(conn, session_id)
    conn.close()


def test_tamper_makes_stale_answer_rewritten(env):
    """Old answer must not remain authoritative after evidence tampering."""
    db_path, _, session_id, ev1, _ = env
    conn = store.connect(db_path)
    claim = _supported_claim(conn, session_id, CHUNK_TEXTS[0], ev1)
    store.update_session_status(conn, session_id, "complete")
    first = synthesize(conn, session_id)
    assert first.statements

    _tamper_chunk_of(conn, ev1)
    fresh = get_answer(conn, session_id, refresh=True)  # stale -> re-synthesize
    assert fresh.statements == ()
    assert claim.claim_id in {g.claim_id for g in fresh.gaps}
    events = [
        e for e in store.list_events(conn, session_id)
        if e.event_type == "synthesized"
    ]
    assert len(events) == 2  # second write recorded
    conn.close()


def test_get_answer_no_refresh_returns_stored(env):
    db_path, _, session_id, ev1, _ = env
    conn = store.connect(db_path)
    _supported_claim(conn, session_id, CHUNK_TEXTS[0], ev1)
    store.update_session_status(conn, session_id, "complete")
    synthesize(conn, session_id)
    answer = get_answer(conn, session_id, refresh=False)
    assert len(answer.statements) == 1
    conn.close()


def test_get_answer_no_refresh_without_answer_raises(env):
    db_path, _, session_id, _, _ = env
    conn = store.connect(db_path)
    with pytest.raises(SynthesisError):
        get_answer(conn, session_id, refresh=False)
    conn.close()


def test_corrupt_stored_answer_raises(env):
    db_path, _, session_id, _, _ = env
    conn = store.connect(db_path)
    store.set_final_answer(conn, session_id, "{not json")
    with pytest.raises(SynthesisError):
        load_answer(conn, session_id)
    with pytest.raises(SynthesisError):
        get_answer(conn, session_id, refresh=False)
    conn.close()


def test_stored_answer_missing_field_raises(env):
    """A stored payload missing a required field (e.g. 'question') must
    raise SynthesisError deterministically — never a silent partial
    answer or a bare KeyError leaking to the caller."""
    db_path, _, session_id, _, _ = env
    conn = store.connect(db_path)
    store.set_final_answer(
        conn, session_id, json.dumps({"session_id": session_id})
    )
    with pytest.raises(SynthesisError):
        load_answer(conn, session_id)
    with pytest.raises(SynthesisError):
        get_answer(conn, session_id, refresh=False)
    conn.close()


def test_synthesize_preserves_complete_with_gaps_status(env):
    """Synthesis must persist the answer WITHOUT promoting the session
    to 'complete': a complete_with_gaps session stays complete_with_gaps."""
    db_path, _, session_id, ev1, _ = env
    conn = store.connect(db_path)
    _supported_claim(conn, session_id, CHUNK_TEXTS[0], ev1)
    store.update_session_status(conn, session_id, "complete_with_gaps")
    synthesize(conn, session_id)
    assert store.get_session(conn, session_id).status == "complete_with_gaps"
    conn.close()
