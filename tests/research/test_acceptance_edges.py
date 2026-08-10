"""Acceptance: edge cases + idempotency + consistency through public interfaces.

Drives the C2 research lifecycle exclusively through the two public
surfaces — the HTTP API (FastAPI TestClient) and the CLI handlers
(``research.cli``) — and asserts the error boundaries, idempotency and
consistency guarantees:

* error boundaries: no documents (422 / exit 1), nonexistent document
  (404 / exit 1), empty question (422 / exit 1), nonexistent session
  (404 / exit 1), terminal session (409 / exit 1), aborted session
  (409 / exit 1), failed session (phase failure -> status ``failed``,
  synthesis 409 / exit 1);
* Scenario F (idempotency): repeated start / iterate / synthesize never
  duplicate claims, links, evidence, queries or audit events;
* Scenario B (API): after tampering, ``refresh=false`` serves the stale
  stored answer and says ``stale_possible``, while ``refresh=true``
  re-validates and the statement disappears;
* JSON output is valid for every CLI command (success and error paths);
* deterministic answer serialization: two builds produce identical JSON;
* termination: multiple iterations, ``no_new_evidence`` termination and
  ``max_iterations`` termination.

No speculative behavior is asserted — every expectation here is an
existing, documented contract of the C2 domain.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import docpipe.ingest as c0ingest
from research import cli as research_cli
from research import store
from research.api import create_app

from tests.docpipe.pdf_factory import build_pdf

CHUNK_TEXTS = [
    "Scaled dot product attention divides by the square root of dk.",
]
QUESTION = "Why does scaled dot product attention divides by the square root of dk?"


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "docpipe.db"
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(build_pdf(CHUNK_TEXTS))
    ingested = c0ingest.ingest_pdf(pdf, db_path=db_path)
    return db_path, ingested.document_id


@pytest.fixture
def client(env):
    db_path, _ = env
    app = create_app(db_path=db_path)
    with TestClient(app) as c:
        yield c


def _cli_args(**kw):
    base = {"json": False, "db_path": None}
    base.update(kw)
    return argparse.Namespace(**base)


def _seed_session(env, *, min_supporting=1, max_iterations=5, with_documents=True):
    """A draft session bound to the env document (or bare when requested)."""
    db_path, doc_id = env
    conn = store.connect(db_path)
    try:
        session = store.create_session(
            conn, question=QUESTION,
            min_supporting=min_supporting, max_iterations=max_iterations,
        )
        if with_documents:
            store.add_session_document(
                conn, session_id=session.session_id, document_id=doc_id
            )
    finally:
        conn.close()
    return session.session_id


def _create_via_api(client, doc_id, **extra):
    payload = {"question": QUESTION, "document_ids": [doc_id]}
    payload.update(extra)
    resp = client.post("/research/sessions", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()["session"]["session_id"]


def _snapshot(db_path, session_id):
    """Idempotency-relevant counts: claims, links, evidence, queries, events."""
    conn = store.connect(db_path)
    try:
        claims = store.list_claims(conn, session_id)
        links = sum(
            len(store.get_links_for_claim(conn, c.claim_id)) for c in claims
        )
        events = store.list_events(conn, session_id)
        return {
            "claims": len(claims),
            "links": links,
            "evidence": conn.execute(
                "SELECT COUNT(*) AS n FROM evidence"
            ).fetchone()["n"],
            "queries": len(store.get_queries(conn, session_id)),
            "session_created": sum(
                1 for e in events if e.event_type == "session_created"
            ),
            "claim_proposed": sum(
                1 for e in events if e.event_type == "claim_proposed"
            ),
            "evidence_linked": sum(
                1 for e in events if e.event_type == "evidence_linked"
            ),
            "synthesized": sum(
                1 for e in events if e.event_type == "synthesized"
            ),
        }
    finally:
        conn.close()


_last_stdout_holder: list[str] = []


def _capture(fn, args):
    """Run a CLI handler with stdout redirected; returns its exit code."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(args)
    _last_stdout_holder.clear()
    _last_stdout_holder.append(buf.getvalue())
    return code


def _last_stdout() -> str:
    return _last_stdout_holder[0] if _last_stdout_holder else ""


class _ExplodingReasoner:
    """Reasoner whose planning phase always fails (used to force a failed
    session through the loop's retry-once phase policy)."""

    def plan_queries(self, question, gaps):
        raise RuntimeError("reasoner exploded")

    def construct_claims(self, evidence):
        raise AssertionError("must not reach claiming")


# ---------------------------------------------------------------------------
# Error boundaries: no documents
# ---------------------------------------------------------------------------


def test_api_start_no_documents_422(env, client):
    db_path, _ = env
    conn = store.connect(db_path)
    try:
        session = store.create_session(conn, question="Q?")  # no documents
    finally:
        conn.close()
    resp = client.post(f"/research/sessions/{session.session_id}/start")
    assert resp.status_code == 422
    assert resp.json() == {"error": "session has no documents"}


def test_cli_start_no_documents_exit_1(env, capsys):
    db_path, _ = env
    conn = store.connect(db_path)
    try:
        session = store.create_session(conn, question="Q?")  # no documents
    finally:
        conn.close()
    code = _capture(
        research_cli.cmd_start,
        _cli_args(session_id=session.session_id, db_path=str(db_path)),
    )
    assert code == 1
    assert "no documents" in capsys.readouterr().err
    assert "Traceback" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Error boundaries: nonexistent document / empty question
# ---------------------------------------------------------------------------


def test_api_new_nonexistent_document_404(env, client):
    resp = client.post(
        "/research/sessions", json={"question": "Q?", "document_ids": [999]}
    )
    assert resp.status_code == 404
    assert resp.json() == {"error": "document not found"}
    # Nothing was created.
    conn = store.connect(env[0])
    try:
        assert store.list_sessions(conn) == []
    finally:
        conn.close()


def test_cli_new_nonexistent_document_exit_1(env, capsys):
    db_path, _ = env
    code = _capture(
        research_cli.cmd_new,
        _cli_args(question="Q?", document_id=[999], db_path=str(db_path)),
    )
    assert code == 1
    assert "document 999 not found" in capsys.readouterr().err


def test_api_new_empty_question_422(env, client):
    resp = client.post(
        "/research/sessions", json={"question": "   ", "document_ids": [1]}
    )
    assert resp.status_code == 422
    assert resp.json() == {"error": "question must not be empty"}


def test_cli_new_empty_question_exit_1(env, capsys):
    db_path, doc_id = env
    code = _capture(
        research_cli.cmd_new,
        _cli_args(question="  ", document_id=[doc_id], db_path=str(db_path)),
    )
    assert code == 1
    assert "question must not be empty" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Error boundaries: nonexistent session
# ---------------------------------------------------------------------------


def test_api_nonexistent_session_404(env, client):
    assert client.get("/research/sessions/S-999999").status_code == 404
    assert client.get("/research/sessions/S-999999/claims").status_code == 404
    assert client.get("/research/sessions/S-999999/answer").status_code == 404
    assert client.post("/research/sessions/S-999999/synthesize").status_code == 404
    for action in ("start", "iterate", "abort"):
        resp = client.post(f"/research/sessions/S-999999/{action}")
        assert resp.status_code == 404, action
        assert resp.json() == {"error": "session not found"}


def test_cli_nonexistent_session_exit_1(env, capsys):
    db_path, _ = env
    for handler in (
        research_cli.cmd_show,
        research_cli.cmd_claims,
        research_cli.cmd_start,
        research_cli.cmd_iterate,
        research_cli.cmd_abort,
        research_cli.cmd_answer,
        research_cli.cmd_synthesize,
    ):
        code = _capture(
            handler,
            _cli_args(session_id="S-999999", db_path=str(db_path)),
        )
        assert code == 1, handler.__name__
        assert "not found" in capsys.readouterr().err, handler.__name__


# ---------------------------------------------------------------------------
# Error boundaries: terminal / aborted / failed sessions
# ---------------------------------------------------------------------------


def test_api_terminal_session_409(env, client):
    doc_id = env[1]
    session_id = _create_via_api(client, doc_id)  # min_supporting=1
    first = client.post(f"/research/sessions/{session_id}/start")
    assert first.status_code == 200
    assert first.json()["session_status"] == "complete"
    # start-after-complete and iterate-on-complete are both illegal.
    second = client.post(f"/research/sessions/{session_id}/start")
    assert second.status_code == 409
    assert second.json() == {"error": "invalid session transition"}
    assert client.post(f"/research/sessions/{session_id}/iterate").status_code == 409


def test_cli_terminal_session_exit_1(env, capsys):
    db_path, _ = env
    session_id = _seed_session(env)  # min_supporting=1
    assert _capture(
        research_cli.cmd_start,
        _cli_args(session_id=session_id, db_path=str(db_path)),
    ) == 0
    capsys.readouterr()  # discard the success output
    code = _capture(
        research_cli.cmd_start,
        _cli_args(session_id=session_id, db_path=str(db_path)),
    )
    assert code == 1
    assert "cannot start session" in capsys.readouterr().err
    code = _capture(
        research_cli.cmd_iterate,
        _cli_args(session_id=session_id, db_path=str(db_path)),
    )
    assert code == 1
    assert "only active sessions iterate" in capsys.readouterr().err


def test_api_aborted_session_not_resumable(env, client):
    doc_id = env[1]
    session_id = _create_via_api(client, doc_id, min_supporting=2)
    started = client.post(f"/research/sessions/{session_id}/start")
    assert started.status_code == 200
    assert started.json()["session_status"] == "active"
    assert client.post(f"/research/sessions/{session_id}/abort").status_code == 200

    # An aborted session is terminal: no resume, no iterate, no synthesis.
    assert client.post(f"/research/sessions/{session_id}/start").status_code == 409
    assert client.post(f"/research/sessions/{session_id}/iterate").status_code == 409
    assert client.post(
        f"/research/sessions/{session_id}/synthesize"
    ).status_code == 409
    session = client.get(f"/research/sessions/{session_id}").json()["session"]
    assert session["status"] == "aborted"


def test_cli_aborted_session_ops_exit_1(env, capsys):
    db_path, _ = env
    session_id = _seed_session(env, min_supporting=2)
    assert _capture(
        research_cli.cmd_start,
        _cli_args(session_id=session_id, db_path=str(db_path)),
    ) == 0
    capsys.readouterr()
    assert _capture(
        research_cli.cmd_abort,
        _cli_args(session_id=session_id, db_path=str(db_path)),
    ) == 0
    capsys.readouterr()
    for handler in (
        research_cli.cmd_start,
        research_cli.cmd_iterate,
        research_cli.cmd_synthesize,
    ):
        code = _capture(
            handler,
            _cli_args(session_id=session_id, db_path=str(db_path)),
        )
        assert code == 1, handler.__name__
        assert capsys.readouterr().err, handler.__name__


def test_api_phase_failure_marks_session_failed(env, client, monkeypatch):
    """A phase failing twice (retry-once exhausted) terminates the session
    as ``failed``; a failed session cannot synthesize/iterate/start."""
    doc_id = env[1]
    session_id = _create_via_api(client, doc_id)
    monkeypatch.setattr(
        "research.loop.DeterministicReasoner", _ExplodingReasoner
    )
    started = client.post(f"/research/sessions/{session_id}/start")
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["session_status"] == "failed"
    assert body["stop_reason"] == "phase_failure:planning"
    assert body["iterations_run"] == 1

    session = client.get(f"/research/sessions/{session_id}").json()["session"]
    assert session["status"] == "failed"

    # Synthesis is not allowed on a failed session (and no resume either).
    synth = client.post(f"/research/sessions/{session_id}/synthesize")
    assert synth.status_code == 409
    assert synth.json() == {"error": "synthesis not allowed"}
    assert client.post(f"/research/sessions/{session_id}/iterate").status_code == 409
    assert client.post(f"/research/sessions/{session_id}/start").status_code == 409


def test_cli_phase_failure_then_synthesize_exit_1(env, capsys, monkeypatch):
    db_path, _ = env
    session_id = _seed_session(env)
    monkeypatch.setattr(
        "research.loop.DeterministicReasoner", _ExplodingReasoner
    )
    # The loop absorbs the phase failure and reports the failed status.
    code = _capture(
        research_cli.cmd_start,
        _cli_args(json=True, session_id=session_id, db_path=str(db_path)),
    )
    assert code == 0
    assert json.loads(_last_stdout())["session_status"] == "failed"

    # The CLI surfaces the synthesis rejection as exit 1 (no traceback).
    code = _capture(
        research_cli.cmd_synthesize,
        _cli_args(json=True, session_id=session_id, db_path=str(db_path)),
    )
    assert code == 1
    body = json.loads(_last_stdout())
    assert "synthesize" in body["error"].lower()
    assert body["exit"] == 1
    assert capsys.readouterr().err == ""  # JSON error goes to stdout only


# ---------------------------------------------------------------------------
# Scenario F: repeated start / iterate / synthesize -> no duplicates
# ---------------------------------------------------------------------------


def test_api_repeated_start_resume_no_duplicates(env, client):
    db_path, doc_id = env
    session_id = _create_via_api(client, doc_id, min_supporting=2)
    first = client.post(
        f"/research/sessions/{session_id}/start", json={"iterations": 1}
    )
    assert first.status_code == 200
    assert first.json()["session_status"] == "active"

    before = _snapshot(db_path, session_id)
    # Repeat start = resume: runs one more iteration (no_new_evidence) and
    # must not duplicate claims/links/evidence/queries/events.
    second = client.post(
        f"/research/sessions/{session_id}/start", json={"iterations": 1}
    )
    assert second.status_code == 200
    assert second.json()["session_status"] == "complete_with_gaps"
    assert second.json()["stop_reason"] == "no_new_evidence"
    assert _snapshot(db_path, session_id) == before
    assert before["session_created"] == 1


def test_api_repeated_iterate_no_duplicates_then_409(env, client):
    db_path, doc_id = env
    session_id = _create_via_api(client, doc_id, min_supporting=2)
    started = client.post(
        f"/research/sessions/{session_id}/start", json={"iterations": 1}
    )
    assert started.json()["session_status"] == "active"

    before = _snapshot(db_path, session_id)
    first = client.post(f"/research/sessions/{session_id}/iterate")
    assert first.status_code == 200, first.text
    assert first.json()["session_status"] == "complete_with_gaps"
    assert _snapshot(db_path, session_id) == before  # no duplicates

    # Iterating a terminal session is rejected, never re-run.
    assert client.post(f"/research/sessions/{session_id}/iterate").status_code == 409
    assert _snapshot(db_path, session_id) == before


def test_api_repeated_synthesize_idempotent(env, client):
    db_path, doc_id = env
    session_id = _create_via_api(client, doc_id)
    started = client.post(f"/research/sessions/{session_id}/start")
    assert started.json()["session_status"] == "complete"

    first = client.post(f"/research/sessions/{session_id}/synthesize")
    assert first.status_code == 200, first.text
    assert first.json()["persisted"] is True
    first_json = json.dumps(first.json()["answer"], sort_keys=True)

    second = client.post(f"/research/sessions/{session_id}/synthesize")
    assert second.status_code == 200
    assert second.json()["persisted"] is False  # byte-identical -> no write
    assert json.dumps(second.json()["answer"], sort_keys=True) == first_json

    # Exactly one synthesized audit event survives the repetition.
    assert _snapshot(db_path, session_id)["synthesized"] == 1


def test_cli_repeated_start_no_duplicate_claims(env, capsys):
    db_path, _ = env
    session_id = _seed_session(env, min_supporting=2)
    code = _capture(
        research_cli.cmd_start,
        _cli_args(json=True, session_id=session_id, db_path=str(db_path)),
    )
    assert code == 0
    assert json.loads(_last_stdout())["session_status"] == "active"
    code = _capture(
        research_cli.cmd_start,
        _cli_args(json=True, session_id=session_id, db_path=str(db_path)),
    )
    assert code == 0
    assert json.loads(_last_stdout())["session_status"] == "complete_with_gaps"

    conn = store.connect(db_path)
    try:
        claims = store.list_claims(conn, session_id)
        assert len(claims) == 1  # resume did not duplicate the claim
        assert len(store.get_links_for_claim(conn, claims[0].claim_id)) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Scenario B (API): tamper -> refresh=false serves stale, refresh=true re-checks
# ---------------------------------------------------------------------------


def test_api_answer_after_tamper_refresh_false_stale(env, client):
    doc_id = env[1]
    session_id = _create_via_api(client, doc_id)
    started = client.post(f"/research/sessions/{session_id}/start")
    assert started.json()["session_status"] == "complete"

    first = client.get(f"/research/sessions/{session_id}/answer").json()
    assert first["refreshed"] is True
    assert len(first["answer"]["statements"]) == 1

    # Tamper the underlying chunk (never the evidence snapshot).
    conn = sqlite3.connect(str(env[0]))
    conn.execute("UPDATE chunks SET content = 'TAMPERED'")
    conn.commit()
    conn.close()

    # refresh=false is a pure read: the OLD stored answer is served
    # untouched and explicitly flagged as possibly stale.
    stale = client.get(
        f"/research/sessions/{session_id}/answer", params={"refresh": "false"}
    )
    assert stale.status_code == 200, stale.text
    body = stale.json()
    assert body["refreshed"] is False
    assert body["stale_possible"] is True
    assert body["answer"] == first["answer"]  # stale statement still present

    # refresh=true re-validates live: the tampered statement disappears
    # into a gap.
    fresh = client.get(f"/research/sessions/{session_id}/answer").json()
    assert fresh["refreshed"] is True
    assert fresh["stale_possible"] is False
    assert fresh["answer"]["statements"] == []
    assert [g["claim_id"] for g in fresh["answer"]["gaps"]] == [
        first["answer"]["statements"][0]["claim_id"]
    ]


# ---------------------------------------------------------------------------
# JSON purity: every CLI command emits valid JSON (success + error paths)
# ---------------------------------------------------------------------------


def test_cli_json_valid_every_command_success(env, capsys, monkeypatch):
    db_path, doc_id = env
    monkeypatch.setattr("docpipe.config.default_db_path", lambda: db_path)

    complete_id = _seed_session(env)  # min_supporting=1
    active_iter_id = _seed_session(env, min_supporting=2)
    active_abort_id = _seed_session(env, min_supporting=2)
    draft_id = _seed_session(env)

    # Prepare states through the CLI itself (each start is valid JSON too).
    started = _capture(
        research_cli.cmd_start,
        _cli_args(json=True, session_id=complete_id, db_path=str(db_path)),
    )
    assert started == 0
    assert json.loads(_last_stdout())["session_status"] == "complete"
    for sid in (active_iter_id, active_abort_id):
        code = _capture(
            research_cli.cmd_start,
            _cli_args(json=True, session_id=sid, db_path=str(db_path)),
        )
        assert code == 0
        assert json.loads(_last_stdout())["session_status"] == "active"

    cases = [
        (research_cli.cmd_new,
         {"question": QUESTION, "document_id": [doc_id]}),
        (research_cli.cmd_list, {}),
        (research_cli.cmd_show, {"session_id": complete_id}),
        (research_cli.cmd_start, {"session_id": draft_id}),
        (research_cli.cmd_iterate, {"session_id": active_iter_id}),
        (research_cli.cmd_abort, {"session_id": active_abort_id}),
        (research_cli.cmd_claims, {"session_id": complete_id}),
        (research_cli.cmd_answer, {"session_id": complete_id}),
        (research_cli.cmd_synthesize, {"session_id": complete_id}),
        (research_cli.cmd_db_path, {}),
    ]
    for handler, kw in cases:
        code = _capture(
            handler, _cli_args(json=True, db_path=str(db_path), **kw)
        )
        assert code == 0, handler.__name__
        body = json.loads(_last_stdout())  # pure JSON, no decoration
        assert isinstance(body, dict), handler.__name__
        assert capsys.readouterr().err == "", handler.__name__


def test_cli_json_valid_every_command_error(env, capsys):
    db_path, doc_id = env
    cases = [
        (research_cli.cmd_new, {"question": "  ", "document_id": [doc_id]}),
        (research_cli.cmd_new, {"question": "Q?", "document_id": [999]}),
        (research_cli.cmd_show, {"session_id": "S-999999"}),
        (research_cli.cmd_claims, {"session_id": "S-999999"}),
        (research_cli.cmd_start, {"session_id": "S-999999"}),
        (research_cli.cmd_iterate, {"session_id": "S-999999"}),
        (research_cli.cmd_abort, {"session_id": "S-999999"}),
        (research_cli.cmd_answer, {"session_id": "S-999999"}),
        (research_cli.cmd_synthesize, {"session_id": "S-999999"}),
    ]
    for handler, kw in cases:
        code = _capture(
            handler, _cli_args(json=True, db_path=str(db_path), **kw)
        )
        assert code == 1, handler.__name__
        body = json.loads(_last_stdout())  # error path is still pure JSON
        assert isinstance(body, dict) and "error" in body, handler.__name__
        assert body["exit"] == 1, handler.__name__
        assert capsys.readouterr().err == "", handler.__name__


# ---------------------------------------------------------------------------
# Deterministic answer serialization
# ---------------------------------------------------------------------------


def test_cli_deterministic_answer_serialization(env, capsys):
    """Two builds of the same answer produce byte-identical JSON."""
    db_path, _ = env
    session_id = _seed_session(env)
    code = _capture(
        research_cli.cmd_start,
        _cli_args(json=True, session_id=session_id, db_path=str(db_path)),
    )
    assert code == 0
    assert json.loads(_last_stdout())["session_status"] == "complete"

    code = _capture(
        research_cli.cmd_synthesize,
        _cli_args(json=True, session_id=session_id, db_path=str(db_path)),
    )
    assert code == 0
    first = json.loads(_last_stdout())
    assert first["persisted"] is True
    first_json = json.dumps(first["answer"], sort_keys=True)

    code = _capture(
        research_cli.cmd_synthesize,
        _cli_args(json=True, session_id=session_id, db_path=str(db_path)),
    )
    assert code == 0
    second = json.loads(_last_stdout())
    assert second["persisted"] is False  # identical -> nothing written
    assert json.dumps(second["answer"], sort_keys=True) == first_json


# ---------------------------------------------------------------------------
# Termination: multiple iterations / no_new_evidence / max_iterations
# ---------------------------------------------------------------------------


def test_api_multiple_iterations_no_new_evidence_termination(env, client):
    """A multi-iteration run ends on no_new_evidence with the exact count."""
    doc_id = env[1]
    session_id = _create_via_api(
        client, doc_id, min_supporting=2, max_iterations=5
    )
    body = client.post(
        f"/research/sessions/{session_id}/start", json={"iterations": 2}
    ).json()
    assert body["session_status"] == "complete_with_gaps"
    assert body["stop_reason"] == "no_new_evidence"
    assert body["iterations_run"] == 2
    session = client.get(f"/research/sessions/{session_id}").json()["session"]
    assert session["current_iteration"] == 2


def test_api_max_iterations_termination(env, client):
    """max_iterations caps the loop when sufficiency is not reached."""
    doc_id = env[1]
    session_id = _create_via_api(
        client, doc_id, min_supporting=2, max_iterations=1
    )
    body = client.post(f"/research/sessions/{session_id}/start").json()
    assert body["session_status"] == "complete_with_gaps"
    assert body["stop_reason"] == "max_iterations"
    assert body["iterations_run"] == 1
