"""C2 answer/synthesis API tests (FastAPI TestClient).

Covers ``research.api_answer``:

* ``GET  /research/sessions/{session_id}/answer`` — default ``refresh=true``
  re-synthesizes and reports ``refreshed``/``stale_possible``; tampered
  evidence is re-detected live (statement disappears); ``refresh=false``
  reads the stored answer as-is and says so explicitly; no stored answer
  + ``refresh=false`` is a 404; a draft session is a 409.
* ``POST /research/sessions/{session_id}/synthesize`` — persisted flag is
  True on first write and False on an idempotent rebuild; draft/failed/
  aborted sessions are 409.
* ID format (400), missing session (404), no stack-trace leaks.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import docpipe.db as c0db
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, store
from research.api_answer import build_answer_router

QUESTION = (
    "Why does scaled dot product attention divide by the square root of dk?"
)

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


def _make_evidence(conn, doc_id, chunk_index):
    """Create valid C1 evidence for one chunk (content/page from the chunk)."""
    row = conn.execute(
        "SELECT id, content, page_number FROM chunks "
        "WHERE document_id = ? AND chunk_index = ?",
        (doc_id, chunk_index),
    ).fetchone()
    return c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=row["id"], evidence_type="chunk",
        source_type="local_pdf", source_title="Paper",
        content=row["content"], page_number=row["page_number"],
        section_path=None,
        locator={"page": row["page_number"], "chunk_id": row["id"]},
    )


def _supported_claim(conn, session_id, statement, evidence_id):
    claim = claims.create_claim(conn, session_id=session_id, statement=statement)
    claims.link_evidence(conn, claim_id=claim.claim_id,
                         evidence_id=evidence_id, role="supporting")
    evaluated = claims.evaluate_claim(conn, claim.claim_id)
    assert evaluated.status == "supported"
    return claim


def _mark_run(conn, session_id):
    """Simulate the loop having started: session is now 'active'."""
    store.update_session_status(conn, session_id, "active")


def _tamper_chunk_of(conn, evidence_id, content):
    conn.execute(
        "UPDATE chunks SET content = ? WHERE id = "
        "(SELECT chunk_id FROM evidence WHERE id = ?)",
        (content, evidence_id),
    )
    conn.commit()


@pytest.fixture
def client(tmp_path):
    """FastAPI app with ONLY the answer/synthesis router."""
    db_path = tmp_path / "docpipe.db"
    app = FastAPI()
    app.include_router(build_answer_router(db_path))
    with TestClient(app) as test_client:
        test_client.db_path = db_path  # type: ignore[attr-defined]
        yield test_client


def _seed_ready_session(db_path):
    """Doc + evidence + one supported claim; session marked 'active'.

    Returns ``(session_id, claim_id, evidence_id)``. No answer is
    synthesized yet — callers decide when to hit the endpoint.
    """
    conn = store.connect(db_path)
    try:
        doc_id = _insert_doc(conn)
        session = store.create_session(conn, question=QUESTION)
        store.add_session_document(
            conn, session_id=session.session_id, document_id=doc_id
        )
        ev = _make_evidence(conn, doc_id, 0)
        claim = _supported_claim(
            conn, session.session_id,
            "Attention divides by the square root of dk.", ev.evidence_id,
        )
        _mark_run(conn, session.session_id)
        return session.session_id, claim.claim_id, ev.evidence_id
    finally:
        conn.close()


def _seed_draft_session(db_path):
    """A bare draft session (no documents, no claims)."""
    conn = store.connect(db_path)
    try:
        session = store.create_session(conn, question=QUESTION)
        return session.session_id
    finally:
        conn.close()


def _assert_no_stack_trace(resp):
    assert "Traceback" not in resp.text
    assert "research\\" not in resp.text.replace("/", "\\")
    assert "File \"" not in resp.text


# ---------------------------------------------------------------------------
# GET /research/sessions/{session_id}/answer — refresh=true (default)
# ---------------------------------------------------------------------------

def test_answer_default_refresh_after_synthesis(client):
    """After loop+synth, the default refresh=true returns the answer with
    refreshed=true / stale_possible=false."""
    session_id, claim_id, _ = _seed_ready_session(client.db_path)

    synth = client.post(f"/research/sessions/{session_id}/synthesize")
    assert synth.status_code == 200, synth.text
    assert synth.json()["persisted"] is True

    resp = client.get(f"/research/sessions/{session_id}/answer")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["refreshed"] is True
    assert body["stale_possible"] is False
    answer = body["answer"]
    assert answer["session_id"] == session_id
    assert answer["question"] == QUESTION
    assert [s["claim_id"] for s in answer["statements"]] == [claim_id]
    assert answer["gaps"] == []


def test_answer_refresh_redetects_tampered_evidence(client):
    """Tampering the underlying chunk makes the statement disappear on the
    next refresh=true GET; the answer is re-synthesized (refreshed=true)."""
    session_id, claim_id, evidence_id = _seed_ready_session(client.db_path)
    first = client.post(f"/research/sessions/{session_id}/synthesize")
    assert first.status_code == 200
    assert [s["claim_id"] for s in first.json()["answer"]["statements"]] == [
        claim_id
    ]

    conn = store.connect(client.db_path)
    try:
        _tamper_chunk_of(conn, evidence_id, "TAMPERED CONTENT")
    finally:
        conn.close()

    resp = client.get(f"/research/sessions/{session_id}/answer")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["refreshed"] is True
    assert body["stale_possible"] is False
    answer = body["answer"]
    # The claim is demoted live: out of statements, into gaps.
    assert answer["statements"] == []
    assert [g["claim_id"] for g in answer["gaps"]] == [claim_id]
    assert answer["gaps"][0]["reason"] == "insufficient_evidence"
    assert answer != first.json()["answer"]


def test_answer_refresh_true_requires_runnable_session(client):
    """A draft session cannot be answered with refresh=true -> 409."""
    session_id = _seed_draft_session(client.db_path)
    resp = client.get(f"/research/sessions/{session_id}/answer")
    assert resp.status_code == 409
    assert resp.json() == {"error": "synthesis not allowed"}
    _assert_no_stack_trace(resp)


# ---------------------------------------------------------------------------
# GET /research/sessions/{session_id}/answer — refresh=false
# ---------------------------------------------------------------------------

def test_answer_refresh_false_stale_possible(client):
    """refresh=false reads the stored answer without re-checking: the
    response explicitly says stale_possible=true."""
    session_id, _, _ = _seed_ready_session(client.db_path)
    synth = client.post(f"/research/sessions/{session_id}/synthesize")
    assert synth.status_code == 200
    stored_answer = synth.json()["answer"]

    resp = client.get(
        f"/research/sessions/{session_id}/answer", params={"refresh": "false"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["refreshed"] is False
    assert body["stale_possible"] is True
    assert body["answer"] == stored_answer


def test_answer_refresh_false_no_stored_answer_404(client):
    """refresh=false with no stored answer -> 404 'no stored answer'."""
    session_id, _, _ = _seed_ready_session(client.db_path)  # never synthesized
    resp = client.get(
        f"/research/sessions/{session_id}/answer", params={"refresh": "false"}
    )
    assert resp.status_code == 404
    assert resp.json() == {"error": "no stored answer"}
    _assert_no_stack_trace(resp)


# ---------------------------------------------------------------------------
# POST /research/sessions/{session_id}/synthesize
# ---------------------------------------------------------------------------

def test_synthesize_persisted_true_then_false_idempotent(client):
    """First synthesize persists (persisted=true); the idempotent rebuild
    writes nothing (persisted=false) and returns the same answer."""
    session_id, _, _ = _seed_ready_session(client.db_path)

    first = client.post(f"/research/sessions/{session_id}/synthesize")
    assert first.status_code == 200, first.text
    assert first.json()["persisted"] is True
    assert first.json()["answer"]["session_id"] == session_id

    second = client.post(f"/research/sessions/{session_id}/synthesize")
    assert second.status_code == 200, second.text
    assert second.json()["persisted"] is False
    assert second.json()["answer"] == first.json()["answer"]


def test_synthesize_on_draft_session_409(client):
    session_id = _seed_draft_session(client.db_path)
    resp = client.post(f"/research/sessions/{session_id}/synthesize")
    assert resp.status_code == 409
    assert resp.json() == {"error": "synthesis not allowed"}
    _assert_no_stack_trace(resp)


# ---------------------------------------------------------------------------
# ID format / session existence
# ---------------------------------------------------------------------------

def test_answer_invalid_session_id_format(client):
    for bad in ("abc", "S-1", "S-00000a"):
        resp = client.get(f"/research/sessions/{bad}/answer")
        assert resp.status_code == 400, bad
        assert resp.json() == {"error": "invalid session ID"}
        resp = client.post(f"/research/sessions/{bad}/synthesize")
        assert resp.status_code == 400, bad
        assert resp.json() == {"error": "invalid session ID"}
        _assert_no_stack_trace(resp)


def test_answer_session_not_found(client):
    resp = client.get("/research/sessions/S-999999/answer")
    assert resp.status_code == 404
    assert resp.json() == {"error": "session not found"}
    _assert_no_stack_trace(resp)


def test_synthesize_session_not_found(client):
    resp = client.post("/research/sessions/S-999999/synthesize")
    assert resp.status_code == 404
    assert resp.json() == {"error": "session not found"}
    _assert_no_stack_trace(resp)


# ---------------------------------------------------------------------------
# refresh=false on a non-synthesizable status with a stored answer is fine
# ---------------------------------------------------------------------------

def test_answer_refresh_false_ignores_status_gate(client):
    """refresh=false is a pure read: even an 'aborted' session with a
    stored answer returns it (no status check, stale_possible=true)."""
    session_id, _, _ = _seed_ready_session(client.db_path)
    synth = client.post(f"/research/sessions/{session_id}/synthesize")
    assert synth.status_code == 200

    conn = store.connect(client.db_path)
    try:
        store.update_session_status(conn, session_id, "aborted")
    finally:
        conn.close()

    resp = client.get(
        f"/research/sessions/{session_id}/answer", params={"refresh": "false"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["stale_possible"] is True
    assert resp.json()["refreshed"] is False
