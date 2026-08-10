"""C2 claims/evidence inspection API tests (FastAPI TestClient).

Covers the three read-only endpoints of ``research.api_claims``:

* ``GET /research/sessions/{session_id}/claims``
* ``GET /research/claims/{claim_id}``
* ``GET /research/claims/{claim_id}/evidence``

The ``valid`` flag is a LIVE C1 re-validation: tampering the underlying
chunk must flip it to ``false`` while the persisted evidence payload stays
untouched (read-only inspection).
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import docpipe.db as c0db
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, store
from research.api_claims import build_claims_router


@pytest.fixture
def seeded(tmp_path):
    """Session + C0 doc/chunks + two evidence objects + two claims.

    Claim C-000001 has one supporting and one contradicting link; claim
    C-000002 has no links.
    """
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
    claim = claims.create_claim(
        conn, session_id=session.session_id,
        statement="Dot product variance is dk.",
    )
    claims.link_evidence(
        conn, claim_id=claim.claim_id,
        evidence_id=ev1.evidence_id, role="supporting",
    )
    claims.link_evidence(
        conn, claim_id=claim.claim_id,
        evidence_id=ev2.evidence_id, role="contradicting",
    )
    second = claims.create_claim(
        conn, session_id=session.session_id, statement="Second claim here.",
    )
    conn.close()
    return db_path, session.session_id, claim.claim_id, second.claim_id


@pytest.fixture
def client(tmp_path, seeded):
    db_path = seeded[0]
    app = FastAPI()
    app.include_router(build_claims_router(db_path))
    with TestClient(app) as test_client:
        test_client.db_path = db_path  # type: ignore[attr-defined]
        yield test_client


# ---------------------------------------------------------------------------
# GET /research/sessions/{session_id}/claims
# ---------------------------------------------------------------------------

def test_list_claims(client, seeded):
    db_path, session_id, claim_id, second_id = seeded
    resp = client.get(f"/research/sessions/{session_id}/claims")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == session_id
    assert [c["claim_id"] for c in body["claims"]] == [claim_id, second_id]
    assert body["claims"][0]["statement"] == "Dot product variance is dk."
    assert body["claims"][0]["status"] == "proposed"

    conn = store.connect(db_path)
    try:
        expected = [c.to_dict() for c in store.list_claims(conn, session_id)]
    finally:
        conn.close()
    assert body["claims"] == expected


def test_list_claims_nonexistent_session(client):
    resp = client.get("/research/sessions/S-999999/claims")
    assert resp.status_code == 404
    assert resp.json() == {"error": "session not found"}


def test_list_claims_invalid_session_format(client):
    for bad in ("abc", "S-1", "S-00000a"):
        resp = client.get(f"/research/sessions/{bad}/claims")
        assert resp.status_code == 400, bad
        assert resp.json() == {"error": "invalid session ID"}


# ---------------------------------------------------------------------------
# GET /research/claims/{claim_id}
# ---------------------------------------------------------------------------

def test_get_claim(client, seeded):
    db_path, _, claim_id, _ = seeded
    resp = client.get(f"/research/claims/{claim_id}")
    assert resp.status_code == 200
    conn = store.connect(db_path)
    try:
        expected = store.get_claim(conn, claim_id).to_dict()
    finally:
        conn.close()
    assert resp.json() == expected


def test_get_claim_not_found(client):
    resp = client.get("/research/claims/C-999999")
    assert resp.status_code == 404
    assert resp.json() == {"error": "claim not found"}


def test_get_claim_invalid_format(client):
    for bad in ("abc", "C-1", "C-00000a"):
        resp = client.get(f"/research/claims/{bad}")
        assert resp.status_code == 400, bad
        assert resp.json() == {"error": "invalid claim ID"}


# ---------------------------------------------------------------------------
# GET /research/claims/{claim_id}/evidence
# ---------------------------------------------------------------------------

def test_claim_evidence_supporting_and_contradicting(client, seeded):
    _, _, claim_id, _ = seeded
    resp = client.get(f"/research/claims/{claim_id}/evidence")
    assert resp.status_code == 200
    body = resp.json()
    assert body["claim_id"] == claim_id
    links = body["links"]
    assert len(links) == 2
    by_role = {link["role"]: link for link in links}
    assert set(by_role) == {"supporting", "contradicting"}
    for role, link in by_role.items():
        assert link["claim_id"] == claim_id
        assert link["evidence_id"].startswith("E-")
        assert link["created_at"] is not None
        assert link["valid"] is True, role
        assert link["evidence"]["evidence_id"] == link["evidence_id"]
        assert link["evidence"]["content"]  # full persisted payload


def test_claim_evidence_valid_flag_after_tamper(client, seeded):
    """Tampering the underlying chunk flips valid -> false; payload intact."""
    db_path, _, claim_id, _ = seeded
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE chunks SET content = 'tampered content' "
        "WHERE id IN (SELECT chunk_id FROM evidence)"
    )
    conn.commit()
    conn.close()

    resp = client.get(f"/research/claims/{claim_id}/evidence")
    assert resp.status_code == 200
    links = resp.json()["links"]
    assert len(links) == 2
    for link in links:
        assert link["valid"] is False
        # read-only: the persisted evidence row is reported untouched
        assert link["evidence"]["content"] != "tampered content"


def test_claim_evidence_empty(client, seeded):
    _, _, _, second_id = seeded
    resp = client.get(f"/research/claims/{second_id}/evidence")
    assert resp.status_code == 200
    assert resp.json() == {"claim_id": second_id, "links": []}


def test_claim_evidence_not_found(client):
    resp = client.get("/research/claims/C-999999/evidence")
    assert resp.status_code == 404
    assert resp.json() == {"error": "claim not found"}


def test_claim_evidence_invalid_format(client):
    for bad in ("abc", "C-1"):
        resp = client.get(f"/research/claims/{bad}/evidence")
        assert resp.status_code == 400, bad
        assert resp.json() == {"error": "invalid claim ID"}
