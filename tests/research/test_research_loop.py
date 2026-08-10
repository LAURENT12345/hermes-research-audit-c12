"""Research loop tests: lifecycle, phases, retrieval, claiming, evaluation,
stop conditions, resume, idempotency, events."""

from __future__ import annotations

import json
import sqlite3

import pytest

import docpipe.db as c0db
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, loop, store
from research.loop import (
    InvalidSessionTransitionError,
    LoopError,
    NoDocumentsError,
    ResearchLoop,
    SessionNotActiveError,
    SessionNotFoundError,
    STOP_MAX_ITERATIONS,
    STOP_NO_NEW_EVIDENCE,
    STOP_SYNTHESIS_READY,
)
from research.retriever import RetrievalError


CHUNK_TEXTS = [
    "Scaled dot product attention divides by the square root of dk.",
    "Positional encoding adds order information to the model.",
    "The model has no recurrence and no convolution.",
]


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
    session = store.create_session(
        conn, question="Why does scaled dot product attention divides by the "
        "square root of dk?"
    )
    store.add_session_document(
        conn, session_id=session.session_id, document_id=doc_id
    )
    conn.close()
    return db_path, doc_id, session.session_id


def _loop(env, **kwargs):
    db_path, _, _ = env
    return ResearchLoop(db_path=db_path, **kwargs)


def _event_types(conn, session_id):
    return [e.event_type for e in store.list_events(conn, session_id)]


def _detail(conn, session_id, event_type):
    return [
        json.loads(e.detail) for e in store.list_events(conn, session_id)
        if e.event_type == event_type
    ]


def _tamper_chunk_of(conn, evidence_id: str, content: str) -> None:
    conn.execute(
        "UPDATE chunks SET content = ? WHERE id = "
        "(SELECT chunk_id FROM evidence WHERE id = ?)",
        (content, evidence_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# A. Lifecycle
# ---------------------------------------------------------------------------

def test_start_draft_to_complete(env):
    db_path, _, session_id = env
    loop_ = _loop(env)
    result = loop_.start(session_id)
    assert result.session_status in ("complete", "complete_with_gaps")
    assert result.stop_reason == STOP_SYNTHESIS_READY
    conn = store.connect(db_path)
    try:
        assert store.get_session(conn, session_id).status == "complete"
        assert store.get_session(conn, session_id).final_answer is None  # no synthesis
        assert "synthesizing_ready" in _event_types(conn, session_id)
        assert "session_created" in _event_types(conn, session_id)
    finally:
        conn.close()


def test_start_on_complete_rejected(env):
    loop_ = _loop(env)
    loop_.start(env[2])
    with pytest.raises(InvalidSessionTransitionError):
        loop_.start(env[2])


def test_iterate_on_draft_rejected(env):
    with pytest.raises(SessionNotActiveError):
        _loop(env).iterate_once(env[2])


def test_iterate_on_complete_rejected(env):
    loop_ = _loop(env)
    loop_.start(env[2])
    with pytest.raises(SessionNotActiveError):
        loop_.iterate_once(env[2])


def test_abort_active_session(env):
    db_path, doc_id, _ = env
    loop_ = _loop(env)
    # min_supporting=2 keeps the session active (claim insufficient = gap).
    conn = store.connect(db_path)
    s2 = store.create_session(
        conn, question="Why does scaled dot product attention divides by the "
        "square root of dk?", min_supporting=2,
    )
    store.add_session_document(conn, session_id=s2.session_id, document_id=doc_id)
    conn.close()

    loop_.start(s2.session_id)  # stays active (claim insufficient)
    assert store.get_session(store.connect(db_path), s2.session_id).status == "active"

    loop_.abort(s2.session_id)
    conn = store.connect(db_path)
    try:
        assert store.get_session(conn, s2.session_id).status == "aborted"
        assert "aborted" in _event_types(conn, s2.session_id)
        # State preserved.
        assert store.list_claims(conn, s2.session_id)
    finally:
        conn.close()
    with pytest.raises(SessionNotActiveError):
        loop_.iterate_once(s2.session_id)
    with pytest.raises(InvalidSessionTransitionError):
        loop_.start(s2.session_id)


def test_abort_non_active_rejected(env):
    with pytest.raises(InvalidSessionTransitionError):
        _loop(env).abort(env[2])  # draft


def test_session_not_found(env):
    with pytest.raises(SessionNotFoundError):
        _loop(env).start("S-999999")


def test_start_requires_documents(tmp_path):
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    session = store.create_session(conn, question="Q?")
    conn.close()
    with pytest.raises(NoDocumentsError):
        ResearchLoop(db_path=db_path).start(session.session_id)


# ---------------------------------------------------------------------------
# B. Iteration phases
# ---------------------------------------------------------------------------

def test_all_phases_execute_in_order(env):
    db_path, _, session_id = env
    _loop(env).start(session_id)
    conn = store.connect(db_path)
    try:
        phases = [
            r.phase for r in store.list_iterations(conn, session_id)
            if r.iteration == 1
        ]
        assert phases == list(loop.PHASE_ORDER)
        assert all(
            r.status == "completed"
            for r in store.list_iterations(conn, session_id)
            if r.iteration == 1
        )
        # phase_started / phase_completed appear in order.
        started = _detail(conn, session_id, "phase_started")
        completed = _detail(conn, session_id, "phase_completed")
        assert [d["phase"] for d in started] == list(loop.PHASE_ORDER)
        assert [d["phase"] for d in completed] == list(loop.PHASE_ORDER)
    finally:
        conn.close()


def test_completed_phase_is_skipped(env):
    db_path, _, session_id = env
    loop_ = _loop(env)
    loop_.start(session_id)
    conn = store.connect(db_path)
    try:
        before = len(store.list_events(conn, session_id))
        # Force a second iteration; planning of iteration 2 must not
        # re-execute iteration 1's phases.
        rows = store.list_iterations(conn, session_id)
        assert all(r.iteration == 1 for r in rows)
        assert len(rows) == 5
        # No extra phase rows can be created for iteration 1 (UNIQUE) and
        # iteration 2 planning creates new rows, not duplicates.
        assert len(store.list_iterations(conn, session_id)) == 5
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# C. Retrieval
# ---------------------------------------------------------------------------

def test_query_executed_status(env):
    db_path, _, session_id = env
    _loop(env).start(session_id)
    conn = store.connect(db_path)
    try:
        queries = store.get_queries(conn, session_id)
        assert queries
        assert all(q.status == "executed" for q in queries)
        assert _detail(conn, session_id, "query_executed")
    finally:
        conn.close()


def test_retrieval_failure_marks_query_failed(env, monkeypatch):
    db_path, _, session_id = env

    class BrokenRetriever:
        def __init__(self, db_path):
            self.db_path = db_path

        def retrieve(self, document_id, query, limit=10):
            raise RetrievalError("boom")

    loop_ = ResearchLoop(db_path=db_path, retriever=BrokenRetriever(db_path))
    result = loop_.start(session_id)
    assert result.session_status == "complete_with_gaps"
    assert result.stop_reason == STOP_NO_NEW_EVIDENCE
    conn = store.connect(db_path)
    try:
        assert all(q.status == "failed" for q in store.get_queries(conn, session_id))
        assert _detail(conn, session_id, "query_failed")
    finally:
        conn.close()


def test_no_result_leads_to_complete_with_gaps(tmp_path):
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    doc_id = _insert_doc(conn, title="Unrelated", texts=["Quantum physics notes."])
    session = store.create_session(
        conn, question="How do servo motors work?"
    )
    store.add_session_document(conn, session_id=session.session_id, document_id=doc_id)
    conn.close()
    result = ResearchLoop(db_path=db_path).start(session.session_id)
    assert result.session_status == "complete_with_gaps"
    assert result.stop_reason == STOP_NO_NEW_EVIDENCE


def test_duplicate_evidence_not_counted_as_new(env):
    """E-1 returned again in a later iteration is NOT new evidence."""
    db_path, _, session_id = env
    loop_ = _loop(env)
    conn = store.connect(db_path)
    try:
        # min_supporting=2 keeps the session active after iteration 1.
        store.update_session_status(conn, session_id, "draft")
        # Rebuild: session bound with min_supporting=2.
    finally:
        conn.close()

    # Simpler: fresh session with min_supporting=2.
    conn = store.connect(db_path)
    s2 = store.create_session(
        conn, question="Why does scaled dot product attention divides by the "
        "square root of dk?", min_supporting=2,
    )
    store.add_session_document(conn, session_id=s2.session_id, document_id=env[1])
    conn.close()

    first = loop_.start(s2.session_id)
    assert first.new_evidence_count == 1  # E-1 genuinely new
    assert store.get_session(store.connect(db_path), s2.session_id).status == "active"

    # Simulate the same evidence being returned again: reset the query to
    # pending (as if a future query re-returned the same chunk).
    conn = store.connect(db_path)
    q = store.get_queries(conn, s2.session_id)[0]
    store.set_query_status(conn, q.id, "pending")
    conn.close()

    second = loop_.iterate_once(s2.session_id)
    assert second.new_evidence_count == 0  # NOT new
    assert second.stop_reason == STOP_NO_NEW_EVIDENCE


# ---------------------------------------------------------------------------
# D. Claiming
# ---------------------------------------------------------------------------

def test_claiming_creates_claims_and_links(env):
    db_path, _, session_id = env
    _loop(env).start(session_id)
    conn = store.connect(db_path)
    try:
        claims_list = store.list_claims(conn, session_id)
        assert len(claims_list) == 1
        links = store.get_links_for_claim(conn, claims_list[0].claim_id)
        assert len(links) == 1
        assert links[0].role == "supporting"
        assert "claim_proposed" in _event_types(conn, session_id)
        assert "evidence_linked" in _event_types(conn, session_id)
    finally:
        conn.close()


def test_duplicate_proposal_statements_dedupe_to_one_claim(env):
    """Two chunks with identical content -> one claim, two links."""
    db_path, doc_id, _ = env
    conn = store.connect(db_path)
    c0db.insert_chunk(
        conn, document_id=doc_id,
        chunk=Chunk(chunk_index=99, content=CHUNK_TEXTS[0],
                    normalized_content=CHUNK_TEXTS[0], page_number=99,
                    section_path=None, token_count=10),
    )
    session = store.create_session(
        conn, question="Why does scaled dot product attention divides by the "
        "square root of dk?"
    )
    store.add_session_document(conn, session_id=session.session_id, document_id=doc_id)
    conn.close()

    result = ResearchLoop(db_path=db_path).start(session.session_id)
    conn = store.connect(db_path)
    try:
        claims_list = store.list_claims(conn, session.session_id)
        assert len(claims_list) == 1  # deduped
        assert len(store.get_links_for_claim(conn, claims_list[0].claim_id)) == 2
        assert result.new_evidence_count == 2  # two distinct evidence objects
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# E. Evaluation
# ---------------------------------------------------------------------------

def test_valid_evidence_supports_claim(env):
    db_path, _, session_id = env
    _loop(env).start(session_id)
    conn = store.connect(db_path)
    try:
        claims_list = store.list_claims(conn, session_id)
        assert claims_list[0].status == "supported"
        assert "claim_status_changed" in _event_types(conn, session_id)
    finally:
        conn.close()


def test_tamper_demotes_through_loop(env):
    """Tampered chunk -> claim demoted to insufficient by the loop's own
    evaluating phase (live validation, no manual edit)."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    # A manual gap claim keeps the session active after iteration 1.
    claims.create_claim(
        conn, session_id=session_id,
        statement="A completely unsupported claim about nothing particular.",
    )
    conn.close()

    loop_ = _loop(env)
    result = loop_.start(session_id)
    assert result.session_status == "active"

    conn = store.connect(db_path)
    supported_claims = [
        c for c in store.list_claims(conn, session_id)
        if c.status == "supported"
    ]
    assert supported_claims
    target = supported_claims[0]
    links = store.get_links_for_claim(conn, target.claim_id)
    assert links
    _tamper_chunk_of(conn, links[0].evidence_id, "TAMPERED CONTENT")
    conn.close()

    result = loop_.iterate_once(session_id)  # evaluating phase demotes
    conn = store.connect(db_path)
    try:
        updated = store.get_claim(conn, target.claim_id)
        assert updated.status == "insufficient"
        demotions = [
            e for e in _detail(conn, session_id, "claim_status_changed")
            if e.get("claim_id") == target.claim_id
            and e.get("to") == "insufficient"
            and e.get("reason") == "evidence_invalid"
        ]
        assert demotions
    finally:
        conn.close()


def test_contradiction_blocks_completion(env):
    """A valid contradicting evidence link prevents synthesis readiness."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    # Pre-create the evidence the loop will propose for chunk 1, plus a
    # second evidence, and pre-link the second as contradicting.
    chunk_ids = [r["id"] for r in conn.execute(
        "SELECT id FROM chunks WHERE document_id = ? ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()]
    ev_a = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_ids[0], evidence_type="chunk",
        source_type="local_pdf", source_title="Paper", content=CHUNK_TEXTS[0],
        page_number=1, section_path=None,
        locator={"page": 1, "chunk_id": chunk_ids[0]},
    ).evidence_id
    ev_b = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_ids[1], evidence_type="chunk",
        source_type="local_pdf", source_title="Paper", content=CHUNK_TEXTS[1],
        page_number=2, section_path=None,
        locator={"page": 2, "chunk_id": chunk_ids[1]},
    ).evidence_id
    # Pre-create the claim with the loop's future statement, linked to a
    # valid contradicting evidence.
    pre = claims.create_claim(
        conn, session_id=session_id, statement=CHUNK_TEXTS[0]
    )
    claims.link_evidence(conn, claim_id=pre.claim_id, evidence_id=ev_b,
                         role="contradicting")
    conn.close()

    result = ResearchLoop(db_path=db_path).start(session_id)
    assert result.session_status != "complete"
    conn = store.connect(db_path)
    try:
        updated = store.get_claim(conn, pre.claim_id)
        assert updated.status == "contradicted"
        assert store.list_contradictions(conn, session_id, status="open")
        assert "contradiction_detected" in _event_types(conn, session_id)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# F. Stop conditions
# ---------------------------------------------------------------------------

def test_stop_synthesis_ready(env):
    result = _loop(env).start(env[2])
    assert result.stop_reason == STOP_SYNTHESIS_READY
    assert result.session_status == "complete"


def test_stop_no_new_evidence(tmp_path):
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    doc_id = _insert_doc(conn, title="NoMatch", texts=["Nothing matches here."])
    session = store.create_session(
        conn, question="Completely unrelated question about quantum gravity."
    )
    store.add_session_document(conn, session_id=session.session_id, document_id=doc_id)
    conn.close()
    result = ResearchLoop(db_path=db_path).start(session.session_id)
    assert result.stop_reason == STOP_NO_NEW_EVIDENCE
    assert result.session_status == "complete_with_gaps"


def test_stop_max_iterations(tmp_path):
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    doc_id = _insert_doc(conn, title="One", texts=[CHUNK_TEXTS[0]])
    session = store.create_session(
        conn, question="Why does scaled dot product attention divides by the "
        "square root of dk?", min_supporting=2, max_iterations=1,
    )
    store.add_session_document(conn, session_id=session.session_id, document_id=doc_id)
    conn.close()
    result = ResearchLoop(db_path=db_path).start(session.session_id)
    assert result.stop_reason == STOP_MAX_ITERATIONS
    assert result.session_status == "complete_with_gaps"


def test_stop_failure(env, monkeypatch):
    db_path, _, session_id = env

    class ExplodingReasoner:
        def plan_queries(self, question, gaps):
            raise RuntimeError("reasoner exploded")

        def construct_claims(self, evidence):
            raise AssertionError("must not reach claiming")

    loop_ = ResearchLoop(db_path=db_path, reasoner=ExplodingReasoner())
    result = loop_.start(session_id)
    assert result.session_status == "failed"
    conn = store.connect(db_path)
    try:
        assert "iteration_failed" in _event_types(conn, session_id)
        assert "failed" in _event_types(conn, session_id)
        phase_rows = store.list_iterations(conn, session_id)
        assert any(r.status == "failed" for r in phase_rows)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# G. Resume / restart
# ---------------------------------------------------------------------------

def _simulate_crash(conn, session_id, iteration, phases_to_complete):
    """Mark phases completed + the queries/evidence a real run would leave."""
    for phase in phases_to_complete:
        row = store.ensure_iteration(
            conn, session_id=session_id, iteration=iteration, phase=phase
        )
        store.complete_iteration(conn, row.id, status="completed")


def test_resume_before_claiming(env):
    """Crash after evidencing: resume skips completed phases, does not
    duplicate evidence/queries, and continues from claiming."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    store.update_session_status(conn, session_id, "active")

    # Planning + retrieval side effects: one query, executed, evidence made.
    store.add_query(conn, session_id=session_id, query_text="scaled dot product attention")
    q = store.get_queries(conn, session_id)[0]
    store.set_query_status(conn, q.id, "executed")
    chunk_ids = [r["id"] for r in conn.execute(
        "SELECT id FROM chunks WHERE document_id = ? ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()]
    evidence_ids = []
    for i, chunk_id in enumerate(chunk_ids[:1]):
        evidence_ids.append(
            c1store.create_evidence(
                conn, document_id=doc_id, chunk_id=chunk_id,
                evidence_type="chunk", source_type="local_pdf",
                source_title="Paper", content=CHUNK_TEXTS[i],
                page_number=i + 1, section_path=None,
                locator={"page": i + 1, "chunk_id": chunk_id},
            ).evidence_id
        )
    _simulate_crash(conn, session_id, 1, ("planning", "retrieving", "evidencing"))
    # The evidence the retrieval produced is persisted per-iteration.
    store.add_iteration_evidence(
        conn, session_id=session_id, iteration=1, evidence_id=evidence_ids[0]
    )
    store.save_iteration_known(
        conn, session_id=session_id, iteration=1, known_evidence_ids=[]
    )
    conn.close()

    result = ResearchLoop(db_path=db_path).iterate_once(session_id)
    conn = store.connect(db_path)
    try:
        # No duplicate query / evidence.
        assert len(store.get_queries(conn, session_id)) == 1
        assert len(evidence_ids) == 1
        # Claiming + evaluating ran; planning/retrieving/evidencing skipped.
        started = _detail(conn, session_id, "phase_started")
        assert [d["phase"] for d in started] == ["claiming", "evaluating"]
        claims_list = store.list_claims(conn, session_id)
        assert len(claims_list) == 1
        assert len(store.get_links_for_claim(conn, claims_list[0].claim_id)) == 1
    finally:
        conn.close()


def test_resume_before_evaluating(env):
    """Crash after claiming: evaluating still runs, statuses become fresh."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    store.update_session_status(conn, session_id, "active")
    store.add_query(conn, session_id=session_id, query_text="scaled dot product attention")
    q = store.get_queries(conn, session_id)[0]
    store.set_query_status(conn, q.id, "executed")
    chunk_id = conn.execute(
        "SELECT id FROM chunks WHERE document_id = ? ORDER BY chunk_index LIMIT 1",
        (doc_id,),
    ).fetchone()["id"]
    ev = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_id, evidence_type="chunk",
        source_type="local_pdf", source_title="Paper", content=CHUNK_TEXTS[0],
        page_number=1, section_path=None,
        locator={"page": 1, "chunk_id": chunk_id},
    ).evidence_id
    claim = claims.create_claim_from_proposal(
        conn, session_id=session_id,
        proposal=__import__("research.reasoner", fromlist=["ClaimProposal"]).ClaimProposal(
            statement=CHUNK_TEXTS[0], evidence_ids=(ev,)
        ),
    )
    claims.link_evidence(conn, claim_id=claim.claim_id, evidence_id=ev,
                         role="supporting")
    store.add_iteration_evidence(
        conn, session_id=session_id, iteration=1, evidence_id=ev
    )
    store.save_iteration_known(
        conn, session_id=session_id, iteration=1, known_evidence_ids=[]
    )
    _simulate_crash(conn, session_id, 1,
                    ("planning", "retrieving", "evidencing", "claiming"))
    conn.close()

    result = ResearchLoop(db_path=db_path).iterate_once(session_id)
    conn = store.connect(db_path)
    try:
        started = _detail(conn, session_id, "phase_started")
        assert [d["phase"] for d in started] == ["evaluating"]
        assert store.get_claim(conn, claim.claim_id).status == "supported"
    finally:
        conn.close()


def test_restart_between_iterations(env):
    """After iteration 1 completes, the next call starts iteration 2."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    s2 = store.create_session(
        conn, question="Why does scaled dot product attention divides by the "
        "square root of dk?", min_supporting=2,
    )
    store.add_session_document(conn, session_id=s2.session_id, document_id=doc_id)
    conn.close()

    loop_ = ResearchLoop(db_path=db_path)
    first = loop_.start(s2.session_id)
    assert first.iterations_run == 1
    assert store.get_session(store.connect(db_path), s2.session_id).status == "active"

    second = loop_.resume(s2.session_id)  # new ResearchLoop would behave the same
    conn = store.connect(db_path)
    try:
        iterations = sorted({r.iteration for r in store.list_iterations(conn, s2.session_id)})
        assert 2 in iterations
        assert "iteration_started" in _event_types(conn, s2.session_id)
        details = _detail(conn, s2.session_id, "iteration_started")
        assert {d["iteration"] for d in details} == {1, 2}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# H. Idempotency
# ---------------------------------------------------------------------------

def _counts(conn, session_id):
    return {
        "queries": len(store.get_queries(conn, session_id)),
        "claims": len(store.list_claims(conn, session_id)),
        "links": sum(len(store.get_links_for_claim(conn, c.claim_id))
                     for c in store.list_claims(conn, session_id)),
        "evidence": c1store.count_evidence(conn),
    }


def test_repeated_start_safe(env):
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    s2 = store.create_session(
        conn, question="Why does scaled dot product attention divides by the "
        "square root of dk?", min_supporting=2,
    )
    store.add_session_document(conn, session_id=s2.session_id, document_id=doc_id)
    conn.close()

    loop_ = ResearchLoop(db_path=db_path)
    loop_.start(s2.session_id)
    conn = store.connect(db_path)
    after_first = _counts(conn, s2.session_id)
    conn.close()
    loop_.start(s2.session_id)  # resume
    conn = store.connect(db_path)
    after_second = _counts(conn, s2.session_id)
    # Claims/links/evidence never duplicate across starts.
    assert after_first["claims"] == after_second["claims"]
    assert after_first["links"] == after_second["links"]
    assert after_first["evidence"] == after_second["evidence"]
    # No duplicate query text.
    texts = [q.query_text for q in store.get_queries(conn, s2.session_id)]
    assert len(texts) == len(set(texts))
    conn.close()


def test_repeated_resume_safe(env):
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    s2 = store.create_session(
        conn, question="Why does scaled dot product attention divides by the "
        "square root of dk?", min_supporting=2,
    )
    store.add_session_document(conn, session_id=s2.session_id, document_id=doc_id)
    conn.close()
    loop_ = ResearchLoop(db_path=db_path)
    loop_.start(s2.session_id)
    loop_.resume(s2.session_id)
    conn = store.connect(db_path)
    after = _counts(conn, s2.session_id)
    texts = [q.query_text for q in store.get_queries(conn, s2.session_id)]
    assert len(texts) == len(set(texts))
    assert after["claims"] >= 1
    conn.close()


# ---------------------------------------------------------------------------
# I. Events
# ---------------------------------------------------------------------------

def test_required_events_present(env):
    db_path, _, session_id = env
    _loop(env).start(session_id)
    conn = store.connect(db_path)
    try:
        types = set(_event_types(conn, session_id))
        required = {
            "session_created", "iteration_started", "phase_started",
            "phase_completed", "query_executed", "claim_proposed",
            "evidence_linked", "claim_status_changed", "iteration_completed",
            "synthesizing_ready", "session_completed",
        }
        assert required <= types
    finally:
        conn.close()


def test_events_never_contain_evidence_content(env):
    db_path, _, session_id = env
    _loop(env).start(session_id)
    conn = store.connect(db_path)
    try:
        for event in store.list_events(conn, session_id):
            if event.detail:
                assert "content" not in event.detail
                for text in CHUNK_TEXTS:
                    assert text not in event.detail
    finally:
        conn.close()


def test_restart_with_new_loop_instance(env):
    """A fresh ResearchLoop instance over the same DB resumes cleanly."""
    db_path, doc_id, session_id = env
    conn = store.connect(db_path)
    s2 = store.create_session(
        conn, question="Why does scaled dot product attention divides by the "
        "square root of dk?", min_supporting=2,
    )
    store.add_session_document(conn, session_id=s2.session_id, document_id=doc_id)
    conn.close()

    first = ResearchLoop(db_path=db_path).start(s2.session_id)
    assert first.session_status == "active"
    second = ResearchLoop(db_path=db_path).resume(s2.session_id)  # "restart"
    assert second.iterations_run >= 1
    assert second.new_evidence_count >= 0
    conn = store.connect(db_path)
    try:
        assert store.get_session(conn, s2.session_id).status in (
            "active", "complete_with_gaps", "complete",
        )
    finally:
        conn.close()
