"""Synthesis tests: answer construction, citations, gaps, exclusion,
evidence tampering, determinism, persistence, idempotency, stale answers,
status gating, terminal claims, and from_json error handling.
"""

from __future__ import annotations

import json

import pytest

import docpipe.db as c0db
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, store
from research.synthesis import (
    EVENT_SYNTHESIZED,
    GAP_REASON_CONTRADICTED,
    GAP_REASON_INSUFFICIENT_EVIDENCE,
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

QUESTION = "Why does scaled dot product attention divides by the square root of dk?"


def _insert_doc(conn, title="Paper", texts=None):
    texts = texts or CHUNK_TEXTS
    doc_id = c0db.insert_document(
        conn, title=title, file_path=f"/tmp/{title}.pdf",
        file_hash=f"h-{title}", status="ready",
    )
    for i, text in enumerate(texts):
        c0db.insert_chunk(
            conn, document_id=doc_id,
            chunk=Chunk(chunk_index=i, content=text,
                        normalized_content=text, page_number=i + 1,
                        section_path=None, token_count=len(text.split())),
        )
    return doc_id


@pytest.fixture
def env(tmp_path):
    """One document with three chunks + a draft session bound to it."""
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    doc_id = _insert_doc(conn)
    session = store.create_session(conn, question=QUESTION)
    store.add_session_document(
        conn, session_id=session.session_id, document_id=doc_id
    )
    conn.close()
    return db_path, doc_id, session.session_id


def _make_evidence(conn, doc_id, chunk_index):
    """Create valid C1 evidence for one chunk (content/page from the chunk)."""
    row = conn.execute(
        "SELECT id, content, page_number FROM chunks "
        "WHERE document_id = ? AND chunk_index = ?",
        (doc_id, chunk_index),
    ).fetchone()
    ev = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=row["id"], evidence_type="chunk",
        source_type="local_pdf", source_title="Paper",
        content=row["content"], page_number=row["page_number"],
        section_path=None,
        locator={"page": row["page_number"], "chunk_id": row["id"]},
    )
    return ev.evidence_id


def _make_claim(conn, session_id, statement):
    return claims.create_claim(conn, session_id=session_id, statement=statement)


def _supported_claim(conn, session_id, statement, evidence_id):
    """Create a claim linked to one valid supporting evidence and evaluate."""
    claim = _make_claim(conn, session_id, statement)
    claims.link_evidence(conn, claim_id=claim.claim_id,
                         evidence_id=evidence_id, role="supporting")
    evaluated = claims.evaluate_claim(conn, claim.claim_id)
    assert evaluated.status == "supported"
    return claim


def _tamper_chunk_of(conn, evidence_id, content):
    conn.execute(
        "UPDATE chunks SET content = ? WHERE id = "
        "(SELECT chunk_id FROM evidence WHERE id = ?)",
        (content, evidence_id),
    )
    conn.commit()


def _mark_run(conn, session_id):
    store.update_session_status(conn, session_id, "active")


def _synthesized_events(conn, session_id):
    return [
        e for e in store.list_events(conn, session_id)
        if e.event_type == EVENT_SYNTHESIZED
    ]


def _statement_ids(answer):
    return [s.claim_id for s in answer.statements]


def _gap_ids(answer):
    return [g.claim_id for g in answer.gaps]


# ---------------------------------------------------------------------------
# A. Basic construction
# ---------------------------------------------------------------------------

def test_supported_claim_becomes_statement_with_citation(env):
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev = _make_evidence(conn, doc_id, 0)
        claim = _supported_claim(
            conn, session_id, "Attention divides by the square root of dk.", ev
        )
        answer = build_answer(conn, session_id)
        assert len(answer.statements) == 1
        st = answer.statements[0]
        assert st.claim_id == claim.claim_id
        assert st.statement == claim.statement
        assert len(st.citations) == 1
        assert st.citations[0].evidence_id == ev
        assert answer.gaps == ()
    finally:
        conn.close()


def test_citation_fields_trace_to_evidence(env):
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev = _make_evidence(conn, doc_id, 1)  # chunk index 1 -> page 2
        _supported_claim(conn, session_id, "Encoding adds order information.", ev)
        answer = build_answer(conn, session_id)
        citation = answer.statements[0].citations[0]
        evidence = c1store.get_evidence(conn, ev)
        assert citation.evidence_id == ev
        assert citation.document_id == doc_id == evidence.document_id
        assert citation.page == evidence.page_number == 2
        assert citation.locator == evidence.locator
        assert citation.locator["page"] == 2
    finally:
        conn.close()


def test_multiple_supported_claims_all_present(env):
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev1 = _make_evidence(conn, doc_id, 0)
        ev2 = _make_evidence(conn, doc_id, 1)
        c1 = _supported_claim(conn, session_id, "Alpha claim about attention.", ev1)
        c2 = _supported_claim(conn, session_id, "Beta claim about encoding.", ev2)
        answer = build_answer(conn, session_id)
        assert set(_statement_ids(answer)) == {c1.claim_id, c2.claim_id}
        by_id = {s.claim_id: s for s in answer.statements}
        assert by_id[c1.claim_id].statement == "Alpha claim about attention."
        assert by_id[c2.claim_id].statement == "Beta claim about encoding."
    finally:
        conn.close()


def test_statement_ordering_deterministic_by_claim_id(env):
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev1 = _make_evidence(conn, doc_id, 0)
        ev2 = _make_evidence(conn, doc_id, 1)
        # Created second -> higher claim_id, but ordering is by claim_id.
        _supported_claim(conn, session_id, "Zeta statement about encoding.", ev2)
        _supported_claim(conn, session_id, "Alpha statement about attention.", ev1)
        answer = build_answer(conn, session_id)
        ids = _statement_ids(answer)
        assert ids == sorted(ids)
        assert ids == [c.claim_id for c in store.list_claims(conn, session_id)]
    finally:
        conn.close()


def test_citation_ordering_deterministic_by_evidence_id(env):
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev1 = _make_evidence(conn, doc_id, 0)  # E-...1
        ev2 = _make_evidence(conn, doc_id, 1)  # E-...2
        claim = _make_claim(conn, session_id, "Two citations claim.")
        # Link in reverse id order; citations must still come out sorted.
        claims.link_evidence(conn, claim_id=claim.claim_id,
                             evidence_id=ev2, role="supporting")
        claims.link_evidence(conn, claim_id=claim.claim_id,
                             evidence_id=ev1, role="supporting")
        claims.evaluate_claim(conn, claim.claim_id)
        answer = build_answer(conn, session_id)
        cited = [c.evidence_id for c in answer.statements[0].citations]
        assert cited == [ev1, ev2]
    finally:
        conn.close()


def test_answer_json_round_trip(env):
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev = _make_evidence(conn, doc_id, 0)
        _supported_claim(conn, session_id, "Round-trip claim.", ev)
        answer = build_answer(conn, session_id)
        restored = ResearchAnswer.from_json(answer.to_json())
        assert restored.to_json() == answer.to_json()
        assert restored.to_dict() == answer.to_dict()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# B. Exclusion (non-supported statuses never enter statements)
# ---------------------------------------------------------------------------

def test_unsupported_claim_never_in_statements(env):
    """A claim without valid support (fresh 'proposed') is re-evaluated to
    insufficient and reported as a gap — never a statement."""
    db_path, _, session_id = env
    conn = store.connect(db_path)
    try:
        claim = _make_claim(conn, session_id, "Claim with no evidence at all.")
        answer = build_answer(conn, session_id)
        assert _statement_ids(answer) == []
        assert _gap_ids(answer) == [claim.claim_id]
        assert answer.gaps[0].reason == GAP_REASON_INSUFFICIENT_EVIDENCE
    finally:
        conn.close()


def test_insufficient_claim_goes_to_gap(env):
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev = _make_evidence(conn, doc_id, 0)
        claim = _make_claim(conn, session_id, "Insufficient claim.")
        claims.link_evidence(conn, claim_id=claim.claim_id,
                             evidence_id=ev, role="supporting")
        # Tamper the only support -> insufficient despite the link.
        _tamper_chunk_of(conn, ev, "TAMPERED CONTENT")
        answer = build_answer(conn, session_id)
        assert claim.claim_id not in _statement_ids(answer)
        gaps = [g for g in answer.gaps if g.claim_id == claim.claim_id]
        assert len(gaps) == 1
        assert gaps[0].reason == GAP_REASON_INSUFFICIENT_EVIDENCE
    finally:
        conn.close()


def test_contradicted_claim_goes_to_gap(env):
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev1 = _make_evidence(conn, doc_id, 0)
        ev2 = _make_evidence(conn, doc_id, 1)
        claim = _make_claim(conn, session_id, "Contradicted claim.")
        claims.link_evidence(conn, claim_id=claim.claim_id,
                             evidence_id=ev1, role="supporting")
        claims.link_evidence(conn, claim_id=claim.claim_id,
                             evidence_id=ev2, role="contradicting")
        answer = build_answer(conn, session_id)
        assert claim.claim_id not in _statement_ids(answer)
        gaps = [g for g in answer.gaps if g.claim_id == claim.claim_id]
        assert len(gaps) == 1
        assert gaps[0].reason == GAP_REASON_CONTRADICTED
    finally:
        conn.close()


def test_invalidated_claim_absent_from_answer(env):
    """Terminal 'invalidated' claims appear in neither statements nor gaps."""
    db_path, _, session_id = env
    conn = store.connect(db_path)
    try:
        claim = _make_claim(conn, session_id, "Invalidated claim.")
        claims.transition_claim(conn, claim.claim_id, "invalidated",
                                reason="test")
        answer = build_answer(conn, session_id)
        assert claim.claim_id not in _statement_ids(answer)
        assert claim.claim_id not in _gap_ids(answer)
    finally:
        conn.close()


def test_withdrawn_claim_absent_from_answer(env):
    """Terminal 'withdrawn' claims appear in neither statements nor gaps."""
    db_path, _, session_id = env
    conn = store.connect(db_path)
    try:
        claim = _make_claim(conn, session_id, "Withdrawn claim.")
        claims.transition_claim(conn, claim.claim_id, "contradicted",
                                reason="test")
        claims.transition_claim(conn, claim.claim_id, "withdrawn",
                                reason="test")
        answer = build_answer(conn, session_id)
        assert claim.claim_id not in _statement_ids(answer)
        assert claim.claim_id not in _gap_ids(answer)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# C. Evidence integrity (tampering, invalid citations, contradictions)
# ---------------------------------------------------------------------------

def test_tamper_supporting_evidence_demotes_claim(env):
    """Tampered chunk -> build_answer re-evaluates -> claim demoted to
    insufficient: out of statements, into gaps."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev = _make_evidence(conn, doc_id, 0)
        claim = _supported_claim(conn, session_id, "Tamper target claim.", ev)
        assert store.get_claim(conn, claim.claim_id).status == "supported"
        _tamper_chunk_of(conn, ev, "TAMPERED CONTENT")
        answer = build_answer(conn, session_id)
        assert claim.claim_id not in _statement_ids(answer)
        gaps = [g for g in answer.gaps if g.claim_id == claim.claim_id]
        assert len(gaps) == 1
        assert gaps[0].reason == GAP_REASON_INSUFFICIENT_EVIDENCE
        assert store.get_claim(conn, claim.claim_id).status == "insufficient"
    finally:
        conn.close()


def test_invalid_evidence_never_cited(env):
    """A citation is only derived from evidence that validates right now;
    invalid evidence never appears in a citation."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev1 = _make_evidence(conn, doc_id, 0)
        ev2 = _make_evidence(conn, doc_id, 1)
        claim = _make_claim(conn, session_id, "Cite only valid evidence.")
        claims.link_evidence(conn, claim_id=claim.claim_id,
                             evidence_id=ev1, role="supporting")
        claims.link_evidence(conn, claim_id=claim.claim_id,
                             evidence_id=ev2, role="supporting")
        claims.evaluate_claim(conn, claim.claim_id)  # supported via ev1
        _tamper_chunk_of(conn, ev2, "TAMPERED CONTENT")
        answer = build_answer(conn, session_id)
        assert len(answer.statements) == 1
        cited = [c.evidence_id for c in answer.statements[0].citations]
        assert ev2 not in cited
        assert cited == [ev1]
    finally:
        conn.close()


def test_valid_contradicting_evidence_blocks_claim(env):
    """A valid contradicting evidence link moves the claim to contradicted,
    which is reported as a gap, never a statement."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev1 = _make_evidence(conn, doc_id, 0)
        ev2 = _make_evidence(conn, doc_id, 1)
        claim = _make_claim(conn, session_id, "Blocked by contradiction.")
        claims.link_evidence(conn, claim_id=claim.claim_id,
                             evidence_id=ev1, role="supporting")
        claims.link_evidence(conn, claim_id=claim.claim_id,
                             evidence_id=ev2, role="contradicting")
        answer = build_answer(conn, session_id)
        assert claim.claim_id not in _statement_ids(answer)
        gaps = [g for g in answer.gaps if g.claim_id == claim.claim_id]
        assert len(gaps) == 1
        assert gaps[0].reason == GAP_REASON_CONTRADICTED
        assert store.get_claim(conn, claim.claim_id).status == "contradicted"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# D. Gaps
# ---------------------------------------------------------------------------

def test_no_supported_claims_empty_statements(env):
    db_path, _, session_id = env
    conn = store.connect(db_path)
    try:
        _make_claim(conn, session_id, "The only claim, unsupported.")
        answer = build_answer(conn, session_id)
        assert answer.statements == ()
        assert len(answer.gaps) == 1
        assert answer.summary == "0 supported statement(s), 1 gap(s)."
    finally:
        conn.close()


def test_mixed_supported_and_insufficient(env):
    """One supported claim -> statement; one insufficient claim -> gap."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev = _make_evidence(conn, doc_id, 0)
        supported = _supported_claim(conn, session_id, "Supported part.", ev)
        gap_claim = _make_claim(conn, session_id, "Unsupported part.")
        answer = build_answer(conn, session_id)
        assert _statement_ids(answer) == [supported.claim_id]
        assert _gap_ids(answer) == [gap_claim.claim_id]
        assert answer.gaps[0].reason == GAP_REASON_INSUFFICIENT_EVIDENCE
    finally:
        conn.close()


def test_supported_claim_blocked_by_open_contradiction(env):
    """A supported claim with an OPEN explicit contradiction record is
    excluded from statements and reported as a contradicted gap."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev = _make_evidence(conn, doc_id, 0)
        claim = _supported_claim(conn, session_id,
                                 "Blocked by open contradiction.", ev)
        store.create_contradiction(
            conn, session_id=session_id, claim_id=claim.claim_id,
            kind="explicit", detail=json.dumps({"contradicting_evidence_ids": []}),
        )
        answer = build_answer(conn, session_id)
        assert claim.claim_id not in _statement_ids(answer)
        gaps = [g for g in answer.gaps if g.claim_id == claim.claim_id]
        assert len(gaps) == 1
        assert gaps[0].reason == GAP_REASON_CONTRADICTED
        # The guard blocks at synthesis time; the claim status stays supported.
        assert store.get_claim(conn, claim.claim_id).status == "supported"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# E. Traceability
# ---------------------------------------------------------------------------

def test_statement_traceability_chain(env):
    """Every statement traces claim_id -> evidence_id -> document_id ->
    page -> locator."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev = _make_evidence(conn, doc_id, 2)  # page 3
        claim = _supported_claim(conn, session_id, "Fully traceable claim.", ev)
        answer = build_answer(conn, session_id)
        st = answer.statements[0]
        citation = st.citations[0]
        assert st.claim_id == claim.claim_id
        assert st.statement == claim.statement
        assert citation.evidence_id == ev
        assert citation.document_id == doc_id
        assert citation.page == 3
        assert citation.locator["page"] == 3
        evidence = c1store.get_evidence(conn, ev)
        assert citation.locator == evidence.locator
        assert citation.document_id == evidence.document_id
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# F. Determinism
# ---------------------------------------------------------------------------

def test_build_answer_deterministic_json(env):
    """Identical DB state -> identical answer JSON, every time."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        _supported_claim(conn, session_id, "Deterministic claim one.",
                         _make_evidence(conn, doc_id, 0))
        _supported_claim(conn, session_id, "Deterministic claim two.",
                         _make_evidence(conn, doc_id, 1))
        first = build_answer(conn, session_id)
        second = build_answer(conn, session_id)
        assert first.to_json() == second.to_json()
        assert first.to_dict() == second.to_dict()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# G. Persistence
# ---------------------------------------------------------------------------

def test_synthesize_persists_and_survives_reconnect(env):
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev = _make_evidence(conn, doc_id, 0)
        _supported_claim(conn, session_id, "Persistent claim.", ev)
        _mark_run(conn, session_id)
        answer = synthesize(conn, session_id)
    finally:
        conn.close()

    conn = store.connect(db_path)  # fresh connection, same file
    try:
        session = store.get_session(conn, session_id)
        assert session.final_answer is not None
        assert session.final_answer == answer.to_json()
        loaded = load_answer(conn, session_id)
        assert loaded is not None
        assert loaded.to_json() == answer.to_json()
        assert loaded.to_dict() == answer.to_dict()
        assert loaded.session_id == session_id
        assert loaded.question == QUESTION
    finally:
        conn.close()


def test_get_answer_no_refresh_requires_stored_answer(env):
    db_path, _, session_id = env
    conn = store.connect(db_path)
    try:
        with pytest.raises(SynthesisError):
            get_answer(conn, session_id, refresh=False)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# H. Idempotency
# ---------------------------------------------------------------------------

def test_synthesize_idempotent_single_event(env):
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev = _make_evidence(conn, doc_id, 0)
        _supported_claim(conn, session_id, "Idempotent claim.", ev)
        _mark_run(conn, session_id)
        first = synthesize(conn, session_id)
        second = synthesize(conn, session_id)
        assert first.to_json() == second.to_json()
        assert first.to_dict() == second.to_dict()
        synced = _synthesized_events(conn, session_id)
        assert len(synced) == 1
        detail = json.loads(synced[0].detail)
        assert detail["claim_count"] == 1
        assert detail["citation_count"] == 1
        assert detail["gap_count"] == 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# I. Tamper / stale answers
# ---------------------------------------------------------------------------

def test_tamper_refresh_overwrites_stale_answer(env):
    """After tampering, get_answer(refresh=True) re-synthesizes: the claim
    leaves the statements, final_answer is overwritten (never returned
    authoritative from the old payload), and a second 'synthesized' event
    is recorded."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev = _make_evidence(conn, doc_id, 0)
        claim = _supported_claim(conn, session_id, "Stale answer claim.", ev)
        _mark_run(conn, session_id)
        first = synthesize(conn, session_id)
        assert _statement_ids(first) == [claim.claim_id]
        assert len(_synthesized_events(conn, session_id)) == 1

        _tamper_chunk_of(conn, ev, "TAMPERED CONTENT")

        refreshed = get_answer(conn, session_id, refresh=True)
        assert claim.claim_id not in _statement_ids(refreshed)
        gaps = [g for g in refreshed.gaps if g.claim_id == claim.claim_id]
        assert len(gaps) == 1
        assert gaps[0].reason == GAP_REASON_INSUFFICIENT_EVIDENCE
        # The stored answer is overwritten — the old one is not authoritative.
        session = store.get_session(conn, session_id)
        assert session.final_answer == refreshed.to_json()
        assert session.final_answer != first.to_json()
        assert len(_synthesized_events(conn, session_id)) == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# J. Status gating
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["draft", "failed", "aborted"])
def test_synthesize_not_allowed_on_non_run_statuses(env, status):
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        _supported_claim(conn, session_id, "Gated claim.",
                         _make_evidence(conn, doc_id, 0))
        store.update_session_status(conn, session_id, status)
        with pytest.raises(SynthesisNotAllowedError):
            synthesize(conn, session_id)
    finally:
        conn.close()


def test_synthesize_unknown_session(env):
    db_path, _, _ = env
    conn = store.connect(db_path)
    try:
        with pytest.raises(SynthesisError):
            synthesize(conn, "S-999999")
    finally:
        conn.close()


def test_build_answer_unknown_session(env):
    db_path, _, _ = env
    conn = store.connect(db_path)
    try:
        with pytest.raises(SynthesisError):
            build_answer(conn, "S-999999")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# K. Terminal claim fix
# ---------------------------------------------------------------------------

def test_evaluate_claim_withdrawn_with_evidence_noop(env):
    """A withdrawn claim with live evidence is terminal: evaluate_claim
    returns it unchanged (no crash, no new event), and synthesis reports
    it nowhere."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    try:
        ev = _make_evidence(conn, doc_id, 0)
        claim = _make_claim(conn, session_id, "Withdrawn terminal claim.")
        claims.link_evidence(conn, claim_id=claim.claim_id,
                             evidence_id=ev, role="supporting")
        claims.transition_claim(conn, claim.claim_id, "contradicted",
                                reason="test")
        claims.transition_claim(conn, claim.claim_id, "withdrawn",
                                reason="test")
        before = len(store.list_events(conn, session_id))
        result = claims.evaluate_claim(conn, claim.claim_id)
        assert result.status == "withdrawn"
        assert result.claim_id == claim.claim_id
        assert len(store.list_events(conn, session_id)) == before  # no event
        answer = build_answer(conn, session_id)
        assert claim.claim_id not in _statement_ids(answer)
        assert claim.claim_id not in _gap_ids(answer)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# L. from_json error handling
# ---------------------------------------------------------------------------

def test_from_json_invalid_json_raises(env):
    db_path, _, _ = env
    conn = store.connect(db_path)
    try:
        with pytest.raises(SynthesisError):
            ResearchAnswer.from_json("{not json")
    finally:
        conn.close()


def test_from_json_missing_field_raises(env):
    db_path, _, _ = env
    conn = store.connect(db_path)
    try:
        with pytest.raises(SynthesisError, match="missing required fields"):
            ResearchAnswer.from_json('{"unrelated": 1}')
    finally:
        conn.close()
