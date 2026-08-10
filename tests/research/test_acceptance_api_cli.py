"""Acceptance: API and CLI expose the same underlying domain behavior.

The same research lifecycle (create -> start -> iterate -> claims ->
answer -> synthesize) is driven through BOTH public interfaces — the
FastAPI app (``research.api.create_app`` via TestClient) and the CLI
handlers (``research.cli.cmd_*`` with argparse Namespaces) — against the
SAME SQLite database, and the resulting state/results are compared for
semantic equivalence on the SAME session id.

Also verifies the Commit 7/8 safety conventions:

* API status-code mapping: 404 / 409 / 422 / 400.
* CLI errors: exit code 1, no traceback on stderr, ``--json`` errors
  are valid JSON on stdout.
* Neither interface bypasses domain rules: status transitions go through
  ``ResearchLoop`` / ``synthesis`` (no raw store writes) — verified via
  draft -> active -> terminal transitions and rejected illegal ones.
* Answer refresh semantics: default ``refresh=true`` re-validates live
  (a tampered chunk changes the answer); ``refresh=false`` serves the
  stored answer untouched and flags ``stale_possible`` — on both API
  and CLI.
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

TERMINAL_STATUSES = ("complete", "complete_with_gaps", "failed", "aborted")


@pytest.fixture
def env(tmp_path):
    """Ingested document + the shared SQLite database path."""
    db_path = tmp_path / "docpipe.db"
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(build_pdf(CHUNK_TEXTS))
    ingested = c0ingest.ingest_pdf(pdf, db_path=db_path)
    return db_path, ingested.document_id


@pytest.fixture
def client(env):
    """The combined C0+C1+C2 FastAPI app as a TestClient."""
    db_path, _ = env
    app = create_app(db_path=db_path)
    with TestClient(app) as c:
        yield c


def _cli_args(**kw):
    base = {"json": False, "db_path": None}
    base.update(kw)
    return argparse.Namespace(**base)


def _cli(handler, **kw) -> tuple[int, str]:
    """Run one CLI handler with a Namespace; returns (exit code, stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = handler(_cli_args(**kw))
    return int(code or 0), buf.getvalue()


def _cli_session_id(db_path, doc_id) -> str:
    code, out = _cli(
        research_cli.cmd_new,
        json=True, question=QUESTION, document_id=[doc_id],
        db_path=str(db_path),
    )
    assert code == 0, out
    return json.loads(out)["session_id"]


def _drive_to_terminal_cli(db_path, session_id) -> str:
    """CLI start, then iterate while active, until the session is terminal."""
    code, out = _cli(research_cli.cmd_start, json=True, session_id=session_id,
                     db_path=str(db_path))
    assert code == 0, out
    for _ in range(8):
        code, out = _cli(research_cli.cmd_show, json=True, session_id=session_id,
                         db_path=str(db_path))
        assert code == 0, out
        status = json.loads(out)["status"]
        if status in TERMINAL_STATUSES:
            return status
        code, out = _cli(research_cli.cmd_iterate, json=True,
                         session_id=session_id, db_path=str(db_path))
        assert code == 0, out
    raise AssertionError(f"CLI: session {session_id} never reached a terminal state")


def _drive_to_terminal_api(client, session_id) -> str:
    """API start, then iterate while active, until the session is terminal."""
    resp = client.post(f"/research/sessions/{session_id}/start")
    assert resp.status_code == 200, resp.text
    for _ in range(8):
        status = client.get(f"/research/sessions/{session_id}").json()["session"]["status"]
        if status in TERMINAL_STATUSES:
            return status
        resp = client.post(f"/research/sessions/{session_id}/iterate")
        assert resp.status_code == 200, resp.text
    raise AssertionError(f"API: session {session_id} never reached a terminal state")


# ---------------------------------------------------------------------------
# Same-session interoperability: the full lifecycle on ONE session id,
# created by one interface and driven by the other.
# ---------------------------------------------------------------------------

def test_same_session_api_created_then_cli_drives(env, client):
    db_path, doc_id = env

    # API creates the session (status draft — never auto-started).
    created = client.post(
        "/research/sessions",
        json={"question": QUESTION, "document_ids": [doc_id]},
    )
    assert created.status_code == 200, created.text
    session_id = created.json()["session"]["session_id"]
    assert created.json()["session"]["status"] == "draft"
    assert client.get(f"/research/sessions/{session_id}").json()["session"]["status"] == "draft"

    # CLI sees the SAME session id and the same draft state.
    code, out = _cli(research_cli.cmd_show, json=True, session_id=session_id,
                     db_path=str(db_path))
    assert code == 0, out
    show = json.loads(out)
    assert show["session_id"] == session_id
    assert show["status"] == "draft"
    assert show["document_ids"] == [doc_id]

    # CLI drives the loop to a terminal state on the API-created session.
    final_status = _drive_to_terminal_cli(db_path, session_id)
    assert final_status in ("complete", "complete_with_gaps")

    # API confirms the same terminal transition (loop-owned status).
    api_status = client.get(f"/research/sessions/{session_id}").json()["session"]["status"]
    assert api_status == final_status

    # CLI: claims -> answer -> synthesize.
    code, out = _cli(research_cli.cmd_claims, json=True, session_id=session_id,
                     db_path=str(db_path))
    assert code == 0, out
    cli_claims = json.loads(out)["claims"]
    assert cli_claims and cli_claims[0]["claim_id"].startswith("C-")

    code, out = _cli(research_cli.cmd_answer, json=True, session_id=session_id,
                     db_path=str(db_path))
    assert code == 0, out
    cli_answer = json.loads(out)["answer"]
    assert cli_answer["statements"], "expected at least one supported statement"
    assert cli_answer["statements"][0]["citations"]

    code, out = _cli(research_cli.cmd_synthesize, json=True, session_id=session_id,
                     db_path=str(db_path))
    assert code == 0, out
    synth = json.loads(out)
    assert synth["session_id"] == session_id
    assert synth["claim_count"] >= 1
    # The CLI answer call (default refresh) already persisted through the
    # synthesis domain, so the explicit synthesize is idempotent.
    assert synth["persisted"] is False

    # API returns the SAME claims + answer JSON on the same session id.
    api_claims = client.get(f"/research/sessions/{session_id}/claims").json()["claims"]
    assert api_claims == cli_claims
    api_answer = client.get(f"/research/sessions/{session_id}/answer").json()["answer"]
    assert api_answer["statements"] == cli_answer["statements"]
    assert api_answer["gaps"] == cli_answer["gaps"]


def test_same_session_cli_created_then_api_drives(env, client):
    db_path, doc_id = env

    # CLI creates the session.
    session_id = _cli_session_id(db_path, doc_id)

    # API can address the CLI-created session (same store, same id).
    api_session = client.get(f"/research/sessions/{session_id}").json()["session"]
    assert api_session["session_id"] == session_id
    assert api_session["status"] == "draft"
    assert api_session["document_ids"] == [doc_id]

    # API drives the loop to a terminal state.
    final_status = _drive_to_terminal_api(client, session_id)
    assert final_status in ("complete", "complete_with_gaps")

    # CLI confirms the same terminal state on the same session id.
    code, out = _cli(research_cli.cmd_show, json=True, session_id=session_id,
                     db_path=str(db_path))
    assert code == 0, out
    assert json.loads(out)["status"] == final_status

    # API: claims -> answer -> synthesize.
    api_claims = client.get(f"/research/sessions/{session_id}/claims").json()["claims"]
    assert api_claims and api_claims[0]["claim_id"].startswith("C-")
    api_answer = client.get(f"/research/sessions/{session_id}/answer").json()["answer"]
    assert api_answer["statements"] and api_answer["statements"][0]["citations"]
    synth = client.post(f"/research/sessions/{session_id}/synthesize").json()
    assert synth["persisted"] is False  # answer refresh already persisted it

    # CLI returns the SAME claims + answer JSON on the same session id.
    code, out = _cli(research_cli.cmd_claims, json=True, session_id=session_id,
                     db_path=str(db_path))
    assert code == 0, out
    assert json.loads(out)["claims"] == api_claims
    code, out = _cli(research_cli.cmd_answer, json=True, session_id=session_id,
                     db_path=str(db_path))
    assert code == 0, out
    cli_answer = json.loads(out)["answer"]
    assert cli_answer["statements"] == api_answer["statements"]
    assert cli_answer["gaps"] == api_answer["gaps"]


# ---------------------------------------------------------------------------
# Interface contract: creation payload shape
# ---------------------------------------------------------------------------

def test_api_and_cli_create_same_session(env, client):
    db_path, doc_id = env
    api_body = client.post(
        "/research/sessions",
        json={"question": QUESTION, "document_ids": [doc_id]},
    ).json()["session"]

    session_id = _cli_session_id(db_path, doc_id)
    code, out = _cli(research_cli.cmd_new, json=True, question=QUESTION,
                     document_id=[doc_id], db_path=str(db_path))
    assert code == 0, out
    cli_body = json.loads(out)

    # Same DB, distinct sessions: both created through the same domain
    # path. The CLI ``new`` returns only {"session_id": ...}; the API
    # returns the full session payload.
    assert api_body["session_id"].startswith("S-")
    assert set(cli_body) == {"session_id"}
    assert cli_body["session_id"].startswith("S-")
    assert api_body["document_ids"] == [doc_id]
    assert api_body["status"] == "draft"


# ---------------------------------------------------------------------------
# Domain rules: neither interface bypasses the loop / synthesis
# ---------------------------------------------------------------------------

def test_api_status_transitions_no_raw_writes(env, client):
    """Status transitions prove the API goes through ResearchLoop, not raw writes."""
    db_path, doc_id = env
    client.post("/research/sessions",
                json={"question": QUESTION, "document_ids": [doc_id]})
    session_id = "S-000001"

    # create -> draft (a session is never auto-started).
    assert client.get(f"/research/sessions/{session_id}").json()["session"]["status"] == "draft"

    # draft rejects iterate / synthesize / answer (domain state machine).
    assert client.post(f"/research/sessions/{session_id}/iterate").status_code == 409
    assert client.post(f"/research/sessions/{session_id}/synthesize").status_code == 409
    assert client.get(f"/research/sessions/{session_id}/answer").status_code == 409

    # start -> terminal through the loop.
    started = client.post(f"/research/sessions/{session_id}/start")
    assert started.status_code == 200, started.text
    status = client.get(f"/research/sessions/{session_id}").json()["session"]["status"]
    assert status in ("complete", "complete_with_gaps")

    # terminal sessions are never re-run / iterated.
    assert client.post(f"/research/sessions/{session_id}/start").status_code == 409
    assert client.post(f"/research/sessions/{session_id}/iterate").status_code == 409


def test_cli_domain_rules_no_bypass(env, capsys):
    """The CLI refuses the same illegal transitions the API rejects."""
    db_path, doc_id = env
    code, _ = _cli(research_cli.cmd_new, question="Q?", document_id=[doc_id],
                   db_path=str(db_path))
    assert code == 0
    capsys.readouterr()  # discard setup output

    # iterate on a draft -> exit 1, domain error, no traceback.
    code = research_cli.cmd_iterate(
        _cli_args(session_id="S-000001", db_path=str(db_path))
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "only active sessions iterate" in err
    assert "Traceback" not in err

    # synthesize on a draft -> exit 1, no traceback.
    code = research_cli.cmd_synthesize(
        _cli_args(session_id="S-000001", db_path=str(db_path))
    )
    assert code == 1
    assert "Traceback" not in capsys.readouterr().err

    # answer on a draft -> exit 1, no traceback.
    code = research_cli.cmd_answer(
        _cli_args(session_id="S-000001", db_path=str(db_path))
    )
    assert code == 1
    assert "Traceback" not in capsys.readouterr().err

    # start -> terminal; then start / iterate are rejected.
    code = research_cli.cmd_start(
        _cli_args(session_id="S-000001", db_path=str(db_path))
    )
    assert code == 0
    capsys.readouterr()
    code = research_cli.cmd_start(
        _cli_args(session_id="S-000001", db_path=str(db_path))
    )
    assert code == 1
    assert "Traceback" not in capsys.readouterr().err
    code = research_cli.cmd_iterate(
        _cli_args(session_id="S-000001", db_path=str(db_path))
    )
    assert code == 1
    assert "Traceback" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# API error mapping: 404 / 409 / 422 / 400
# ---------------------------------------------------------------------------

def test_api_error_mapping(env, client):
    db_path, doc_id = env
    assert client.get("/research/sessions/S-999999").status_code == 404
    assert client.post("/research/sessions/S-999999/iterate").status_code == 404
    assert client.post("/research/sessions/S-999999/start").status_code == 404

    client.post("/research/sessions",
                json={"question": QUESTION, "document_ids": [doc_id]})
    assert client.post("/research/sessions/S-000001/iterate").status_code == 409
    assert client.post("/research/sessions/S-000001/synthesize").status_code == 409
    assert client.get("/research/sessions/S-000001/answer").status_code == 409
    assert client.post(
        "/research/sessions", json={"question": "  ", "document_ids": [doc_id]}
    ).status_code == 422
    assert client.post(
        "/research/sessions", json={"question": "Q?", "document_ids": [999]}
    ).status_code == 404
    assert client.get("/research/sessions/abc").status_code == 400


# ---------------------------------------------------------------------------
# CLI error conventions: exit 1, no traceback, --json stays machine-readable
# ---------------------------------------------------------------------------

def test_cli_errors_no_traceback(env, capsys):
    db_path, _ = env
    code = research_cli.cmd_show(
        _cli_args(session_id="S-999999", db_path=str(db_path))
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "not found" in err


def test_cli_json_error_is_valid_json(env, capsys):
    db_path, _ = env
    code, out = _cli(research_cli.cmd_show, json=True, session_id="S-999999",
                     db_path=str(db_path))
    assert code == 1
    parsed = json.loads(out)
    assert parsed["error"] == "session S-999999 not found"
    assert parsed["exit"] == 1
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Answer refresh semantics: default refresh=true, refresh=false -> stale
# ---------------------------------------------------------------------------

def test_refresh_semantics_api(env, client):
    db_path, doc_id = env
    client.post("/research/sessions",
                json={"question": QUESTION, "document_ids": [doc_id]})
    _drive_to_terminal_api(client, "S-000001")

    first = client.get("/research/sessions/S-000001/answer").json()
    assert first["refreshed"] is True
    assert first["stale_possible"] is False

    stale = client.get(
        "/research/sessions/S-000001/answer", params={"refresh": "false"}
    ).json()
    assert stale["refreshed"] is False
    assert stale["stale_possible"] is True


def test_refresh_semantics_cli(env, capsys):
    db_path, doc_id = env
    session_id = _cli_session_id(db_path, doc_id)
    _drive_to_terminal_cli(db_path, session_id)

    code, out = _cli(research_cli.cmd_answer, json=True, session_id=session_id,
                     db_path=str(db_path))
    assert code == 0, out
    body = json.loads(out)
    assert body["refreshed"] is True
    assert body["stale_possible"] is False

    code, out = _cli(research_cli.cmd_answer, json=True, session_id=session_id,
                     no_refresh=True, db_path=str(db_path))
    assert code == 0, out
    body = json.loads(out)
    assert body["refreshed"] is False
    assert body["stale_possible"] is True


def test_api_tamper_then_refresh_excludes_claim(env, client):
    db_path, doc_id = env
    client.post("/research/sessions",
                json={"question": QUESTION, "document_ids": [doc_id]})
    _drive_to_terminal_api(client, "S-000001")
    before = client.get("/research/sessions/S-000001/answer").json()["answer"]
    assert len(before["statements"]) == 1

    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE chunks SET content = 'TAMPERED'")
    conn.commit()
    conn.close()

    after = client.get("/research/sessions/S-000001/answer").json()["answer"]
    assert after["statements"] == []
    assert after["gaps"], "demoted claim must surface as a gap"


def test_cli_tamper_default_refresh_changes_answer(env, capsys):
    db_path, doc_id = env
    session_id = _cli_session_id(db_path, doc_id)
    _drive_to_terminal_cli(db_path, session_id)

    # Default refresh=true persists the authoritative answer.
    code, out = _cli(research_cli.cmd_answer, json=True, session_id=session_id,
                     db_path=str(db_path))
    assert code == 0, out
    before = json.loads(out)
    assert before["refreshed"] is True
    assert len(before["answer"]["statements"]) == 1

    # Tamper the chunk -> refresh=false serves the stored answer untouched
    # but flags it as possibly stale (the CLI does NOT silently re-validate).
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE chunks SET content = 'TAMPERED'")
    conn.commit()
    conn.close()

    code, out = _cli(research_cli.cmd_answer, json=True, session_id=session_id,
                     no_refresh=True, db_path=str(db_path))
    assert code == 0, out
    stale = json.loads(out)
    assert stale["refreshed"] is False
    assert stale["stale_possible"] is True
    assert len(stale["answer"]["statements"]) == 1  # old, un-revalidated answer

    # Default refresh=true re-validates live -> the answer CHANGES.
    code, out = _cli(research_cli.cmd_answer, json=True, session_id=session_id,
                     db_path=str(db_path))
    assert code == 0, out
    after = json.loads(out)
    assert after["refreshed"] is True
    assert after["answer"]["statements"] == []
    assert after["answer"]["gaps"], "demoted claim must surface as a gap"
