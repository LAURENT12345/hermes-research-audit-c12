"""C2 research session API tests (combined C0+C1+C2 app via TestClient).

Covers: create (valid/invalid question, missing docs, nonexistent
document, no auto-start), get (found/not found/invalid id format),
start (bind docs, terminal 409, no-documents 422), iterate (active,
draft 409), abort (active, draft 409), and the enriched GET payload
(current_iteration / answer_available / sufficiency).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import docpipe.db as c0db
from docpipe.chunker import Chunk
from research import store
from research.api_sessions import create_app

# Must share its content terms with CHUNK_TEXTS[0] verbatim: C0 search
# ANDs the question's terms with FTS5 (no stemming), so "divide" would
# never match "divides".
QUESTION = (
    "Why does scaled dot product attention divides by the square root of dk?"
)

CHUNK_TEXTS = [
    "Scaled dot product attention divides by the square root of dk.",
    "Positional encoding adds order information to the model.",
    "The model has no recurrence and no convolution.",
]


def _insert_doc(conn, title="Paper", texts=None):
    texts = texts or CHUNK_TEXTS
    doc_id = c0db.insert_document(
        conn,
        title=title,
        file_path=f"/tmp/{title}.pdf",
        file_hash=f"h-{title}",
        status="ready",
    )
    for i, text in enumerate(texts):
        c0db.insert_chunk(
            conn,
            document_id=doc_id,
            chunk=Chunk(
                chunk_index=i,
                content=text,
                normalized_content=text,
                page_number=i + 1,
                section_path=None,
                token_count=len(text.split()),
            ),
        )
    return doc_id


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "docpipe.db"
    app = create_app(db_path=db_path)
    with TestClient(app) as test_client:
        test_client.db_path = db_path  # type: ignore[attr-defined]
        yield test_client


@pytest.fixture
def env(client):
    """One ready C0 document with three chunks."""
    conn = store.connect(client.db_path)
    try:
        doc_id = _insert_doc(conn)
    finally:
        conn.close()
    return client, doc_id


def _create(client: TestClient, question=QUESTION, document_ids=None, **extra):
    payload = {"question": question, "document_ids": document_ids or []}
    payload.update(extra)
    return client.post("/research/sessions", json=payload)


def _create_session(client: TestClient, document_ids, **extra) -> str:
    resp = _create(client, document_ids=document_ids, **extra)
    assert resp.status_code == 200, resp.text
    return resp.json()["session"]["session_id"]


# ---------------------------------------------------------------------------
# POST /research/sessions
# ---------------------------------------------------------------------------


def test_create_session_draft_with_documents(client, env):
    _, doc_id = env
    resp = _create(client, document_ids=[doc_id])
    assert resp.status_code == 200, resp.text
    session = resp.json()["session"]
    assert session["session_id"] == "S-000001"
    assert session["status"] == "draft"  # never auto-started
    assert session["question"] == QUESTION
    assert session["min_supporting"] == 1
    assert session["max_iterations"] == 5
    assert session["final_answer"] is None
    assert session["document_ids"] == [doc_id]
    # Still a draft in the database (no auto-start).
    conn = store.connect(client.db_path)
    try:
        assert store.get_session(conn, "S-000001").status == "draft"
    finally:
        conn.close()


def test_create_session_strips_question_and_custom_params(client, env):
    _, doc_id = env
    resp = _create(
        client,
        question="  Trimmed question?  ",
        document_ids=[doc_id],
        min_supporting=2,
        max_iterations=7,
    )
    assert resp.status_code == 200, resp.text
    session = resp.json()["session"]
    assert session["question"] == "Trimmed question?"
    assert session["min_supporting"] == 2
    assert session["max_iterations"] == 7


def test_create_session_blank_question_422(client, env):
    _, doc_id = env
    for question in ("", "   ", "\t\n"):
        resp = _create(client, question=question, document_ids=[doc_id])
        assert resp.status_code == 422, repr(question)
        assert resp.json() == {"error": "question must not be empty"}


def test_create_session_empty_document_ids_422(client):
    resp = _create(client, question="Q?", document_ids=[])
    assert resp.status_code == 422
    assert resp.json() == {"error": "document_ids must not be empty"}


def test_create_session_missing_document_ids_key_422(client):
    resp = client.post("/research/sessions", json={"question": "Q?"})
    assert resp.status_code == 422
    assert resp.json() == {"error": "invalid request body"}


def test_create_session_nonexistent_document_404(client):
    resp = _create(client, question="Q?", document_ids=[999])
    assert resp.status_code == 404
    assert resp.json() == {"error": "document not found"}
    # Nothing is created when any referenced document is missing.
    conn = store.connect(client.db_path)
    try:
        assert store.list_sessions(conn) == []
    finally:
        conn.close()


def test_create_session_multiple_documents(client, env):
    _, doc_id = env
    conn = store.connect(client.db_path)
    try:
        doc2 = _insert_doc(conn, title="Second")
    finally:
        conn.close()
    resp = _create(client, document_ids=[doc_id, doc2])
    assert resp.status_code == 200, resp.text
    assert resp.json()["session"]["document_ids"] == sorted([doc_id, doc2])


# ---------------------------------------------------------------------------
# GET /research/sessions/{session_id}
# ---------------------------------------------------------------------------


def test_get_session(client, env):
    _, doc_id = env
    session_id = _create_session(client, [doc_id])
    resp = client.get(f"/research/sessions/{session_id}")
    assert resp.status_code == 200, resp.text
    session = resp.json()["session"]
    assert session["session_id"] == session_id
    assert session["status"] == "draft"
    assert session["document_ids"] == [doc_id]
    assert session["current_iteration"] == 0
    assert session["answer_available"] is False
    suff = session["sufficiency"]
    assert suff["total_claims"] == 0
    assert suff["supported_count"] == 0
    assert suff["insufficient_count"] == 0
    assert suff["contradicted_count"] == 0
    assert suff["unresolved_count"] == 0
    assert suff["has_gaps"] is False
    assert suff["synthesis_ready"] is False
    assert suff["needs_more_retrieval"] is False
    assert suff["gaps"] == []


def test_get_session_not_found_404(client):
    resp = client.get("/research/sessions/S-999999")
    assert resp.status_code == 404
    assert resp.json() == {"error": "session not found"}


def test_get_session_answer_available_flag(client, env):
    _, doc_id = env
    session_id = _create_session(client, [doc_id])
    conn = store.connect(client.db_path)
    try:
        store.set_final_answer(conn, session_id, "{}")
    finally:
        conn.close()
    session = client.get(f"/research/sessions/{session_id}").json()["session"]
    assert session["answer_available"] is True


def test_invalid_session_id_format_400(client):
    for bad in ("abc", "S-1", "S-00000a", "s-000001"):
        resp = client.get(f"/research/sessions/{bad}")
        assert resp.status_code == 400, bad
        assert resp.json() == {"error": "invalid session ID"}
        for action in ("start", "iterate", "abort"):
            resp = client.post(f"/research/sessions/{bad}/{action}", json={})
            assert resp.status_code == 400, (bad, action)


# ---------------------------------------------------------------------------
# POST /research/sessions/{session_id}/start
# ---------------------------------------------------------------------------


def test_start_session_completes(client, env):
    _, doc_id = env
    session_id = _create_session(client, [doc_id])
    resp = client.post(
        f"/research/sessions/{session_id}/start", json={"iterations": 1}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {
        "session_id",
        "session_status",
        "stop_reason",
        "iterations_run",
        "new_evidence_count",
    }
    assert body["session_id"] == session_id
    assert body["session_status"] == "complete"
    assert body["stop_reason"] == "synthesis_ready"
    assert body["iterations_run"] == 1
    assert body["new_evidence_count"] >= 1
    # GET reflects the new state.
    session = client.get(f"/research/sessions/{session_id}").json()["session"]
    assert session["status"] == "complete"
    assert session["current_iteration"] == 1


def test_start_session_binds_documents(client, env):
    _, doc_id = env
    conn = store.connect(client.db_path)
    try:
        doc2 = _insert_doc(conn, title="Second")
    finally:
        conn.close()
    session_id = _create_session(client, [doc_id])
    resp = client.post(
        f"/research/sessions/{session_id}/start",
        json={"document_ids": [doc2], "iterations": 1},
    )
    assert resp.status_code == 200, resp.text
    session = client.get(f"/research/sessions/{session_id}").json()["session"]
    assert session["document_ids"] == sorted([doc_id, doc2])


def test_start_session_nonexistent_document_404(client, env):
    _, doc_id = env
    session_id = _create_session(client, [doc_id])
    resp = client.post(
        f"/research/sessions/{session_id}/start",
        json={"document_ids": [999]},
    )
    assert resp.status_code == 404
    assert resp.json() == {"error": "document not found"}
    # Session is untouched (still draft).
    session = client.get(f"/research/sessions/{session_id}").json()["session"]
    assert session["status"] == "draft"


def test_start_session_after_complete_409(client, env):
    _, doc_id = env
    session_id = _create_session(client, [doc_id])
    first = client.post(
        f"/research/sessions/{session_id}/start", json={"iterations": 1}
    )
    assert first.status_code == 200
    second = client.post(
        f"/research/sessions/{session_id}/start", json={"iterations": 1}
    )
    assert second.status_code == 409
    assert second.json() == {"error": "invalid session transition"}


def test_start_session_without_documents_422(client):
    conn = store.connect(client.db_path)
    try:
        session = store.create_session(conn, question="No docs here?")
    finally:
        conn.close()
    resp = client.post(
        f"/research/sessions/{session.session_id}/start", json={}
    )
    assert resp.status_code == 422
    assert resp.json() == {"error": "session has no documents"}


# ---------------------------------------------------------------------------
# POST /research/sessions/{session_id}/iterate
# ---------------------------------------------------------------------------


def test_iterate_active_session(client, env):
    """min_supporting=2 keeps the session active after start, so a
    second iteration is legal and terminates on no_new_evidence."""
    _, doc_id = env
    session_id = _create_session(client, [doc_id], min_supporting=2)
    started = client.post(
        f"/research/sessions/{session_id}/start", json={"iterations": 1}
    )
    assert started.status_code == 200
    assert started.json()["session_status"] == "active"

    resp = client.post(f"/research/sessions/{session_id}/iterate")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {
        "session_id",
        "session_status",
        "stop_reason",
        "iterations_run",
        "new_evidence_count",
    }
    assert body["session_id"] == session_id
    assert body["session_status"] == "complete_with_gaps"
    assert body["stop_reason"] == "no_new_evidence"
    assert body["iterations_run"] == 1
    assert body["new_evidence_count"] == 0


def test_iterate_on_draft_409(client, env):
    _, doc_id = env
    session_id = _create_session(client, [doc_id])
    resp = client.post(f"/research/sessions/{session_id}/iterate")
    assert resp.status_code == 409
    assert resp.json() == {"error": "session not active"}


def test_iterate_on_complete_409(client, env):
    _, doc_id = env
    session_id = _create_session(client, [doc_id])
    client.post(
        f"/research/sessions/{session_id}/start", json={"iterations": 1}
    )
    resp = client.post(f"/research/sessions/{session_id}/iterate")
    assert resp.status_code == 409
    assert resp.json() == {"error": "session not active"}


# ---------------------------------------------------------------------------
# POST /research/sessions/{session_id}/abort
# ---------------------------------------------------------------------------


def test_abort_active_session(client, env):
    """min_supporting=2 keeps the session active so abort is legal."""
    _, doc_id = env
    session_id = _create_session(client, [doc_id], min_supporting=2)
    client.post(
        f"/research/sessions/{session_id}/start", json={"iterations": 1}
    )
    resp = client.post(f"/research/sessions/{session_id}/abort")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"session_id": session_id, "status": "aborted"}
    session = client.get(f"/research/sessions/{session_id}").json()["session"]
    assert session["status"] == "aborted"


def test_abort_on_draft_409(client, env):
    _, doc_id = env
    session_id = _create_session(client, [doc_id])
    resp = client.post(f"/research/sessions/{session_id}/abort")
    assert resp.status_code == 409
    assert resp.json() == {"error": "invalid session transition"}


def test_abort_on_complete_409(client, env):
    _, doc_id = env
    session_id = _create_session(client, [doc_id])
    client.post(
        f"/research/sessions/{session_id}/start", json={"iterations": 1}
    )
    resp = client.post(f"/research/sessions/{session_id}/abort")
    assert resp.status_code == 409
    assert resp.json() == {"error": "invalid session transition"}


# ---------------------------------------------------------------------------
# Lifecycle endpoints on nonexistent sessions
# ---------------------------------------------------------------------------


def test_lifecycle_endpoints_nonexistent_session_404(client):
    for action in ("start", "iterate", "abort"):
        resp = client.post(f"/research/sessions/S-999999/{action}", json={})
        assert resp.status_code == 404, action
        assert resp.json() == {"error": "session not found"}


# ---------------------------------------------------------------------------
# GET sufficiency after a completed run
# ---------------------------------------------------------------------------


def test_get_session_sufficiency_after_start(client, env):
    _, doc_id = env
    session_id = _create_session(client, [doc_id])
    client.post(
        f"/research/sessions/{session_id}/start", json={"iterations": 1}
    )
    session = client.get(f"/research/sessions/{session_id}").json()["session"]
    suff = session["sufficiency"]
    assert suff["total_claims"] == 1
    assert suff["supported_count"] == 1
    assert suff["insufficient_count"] == 0
    assert suff["has_gaps"] is False
    assert suff["synthesis_ready"] is True
    assert suff["needs_more_retrieval"] is False
    assert suff["gaps"] == []
