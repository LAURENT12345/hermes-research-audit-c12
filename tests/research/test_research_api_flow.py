"""End-to-end API flow test: C0 -> C1 -> C2 through the combined app.

Upload a real PDF (C0) -> create session -> start -> iterate to
terminal -> inspect claims -> synthesize -> GET answer -> tamper the
chunk -> GET answer again and verify live re-validation.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from research.api import create_app

from tests.docpipe.pdf_factory import build_pdf

CHUNK_TEXTS = [
    "Scaled dot product attention divides by the square root of dk.",
    "Positional encoding adds order information to the model.",
]
QUESTION = "Why does scaled dot product attention divides by the square root of dk?"


@pytest.fixture
def client(tmp_path):
    app = create_app(db_path=tmp_path / "docpipe.db")
    with TestClient(app) as c:
        c.db_path = tmp_path / "docpipe.db"  # type: ignore[attr-defined]
        yield c


def _run_to_terminal(client, session_id, max_calls=8):
    """POST start/iterate until the session is terminal."""
    resp = client.post(f"/research/sessions/{session_id}/start")
    for _ in range(max_calls):
        status = client.get(f"/research/sessions/{session_id}").json()["session"]["status"]
        if status in ("complete", "complete_with_gaps", "failed", "aborted"):
            return status
        resp = client.post(f"/research/sessions/{session_id}/iterate")
        assert resp.status_code == 200, resp.text
    return client.get(f"/research/sessions/{session_id}").json()["session"]["status"]


def test_full_api_flow_c0_c1_c2(client):
    # 1. C0 upload via the API.
    upload = client.post(
        "/documents/upload",
        files={"file": ("paper.pdf", build_pdf(CHUNK_TEXTS), "application/pdf")},
    )
    assert upload.status_code == 200, upload.text
    doc_id = upload.json()["document_id"]

    # 2. Create session + bind document.
    created = client.post(
        "/research/sessions",
        json={"question": QUESTION, "document_ids": [doc_id]},
    )
    assert created.status_code == 200
    session_id = created.json()["session"]["session_id"]
    assert created.json()["session"]["document_ids"] == [doc_id]

    # 3. Run the loop to a terminal state.
    status = _run_to_terminal(client, session_id)
    assert status in ("complete", "complete_with_gaps")

    # 4. Claims exist with provenance.
    claims = client.get(f"/research/sessions/{session_id}/claims").json()["claims"]
    assert claims
    supported = [c for c in claims if c["status"] == "supported"]
    assert supported, "expected at least one supported claim"

    # 5. Synthesize and verify statements + derived citations.
    synth = client.post(f"/research/sessions/{session_id}/synthesize")
    assert synth.status_code == 200, synth.text
    body = synth.json()
    assert body["persisted"] is True
    answer = body["answer"]
    assert len(answer["statements"]) >= 1
    stmt = answer["statements"][0]
    assert stmt["claim_id"].startswith("C-")
    assert stmt["citations"]
    cit = stmt["citations"][0]
    assert cit["evidence_id"].startswith("E-")
    assert cit["document_id"] == doc_id
    assert cit["page"] == 1
    assert cit["locator"]["chunk_id"]

    # 6. GET answer (default refresh) returns the persisted answer.
    got = client.get(f"/research/sessions/{session_id}/answer")
    assert got.status_code == 200
    assert got.json()["refreshed"] is True
    assert got.json()["answer"]["statements"] == answer["statements"]

    # 7. Idempotent synthesize: nothing new persisted.
    again = client.post(f"/research/sessions/{session_id}/synthesize")
    assert again.json()["persisted"] is False

    # 8. Tamper the chunk -> GET answer re-validates (statement vanishes).
    conn = sqlite3.connect(str(client.db_path))
    conn.execute("UPDATE chunks SET content = 'TAMPERED CONTENT'")
    conn.commit()
    conn.close()

    after = client.get(f"/research/sessions/{session_id}/answer")
    assert after.status_code == 200
    refreshed = after.json()["answer"]
    assert refreshed["statements"] == []
    assert refreshed["gaps"], "demoted claim must surface as a gap"

    # 9. refresh=false still serves the old payload but marks it stale.
    stale = client.get(
        f"/research/sessions/{session_id}/answer", params={"refresh": "false"}
    )
    assert stale.json()["stale_possible"] is True
