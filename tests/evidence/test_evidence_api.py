"""C1 API tests (combined C0+C1 app via FastAPI TestClient).

The black-box acceptance flow (spec §24) is covered here end-to-end:
C0 upload -> C0 search -> POST /evidence -> GET -> validate -> tamper.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

import docpipe.db as c0db
from evidence import store
from evidence.api import create_app

from tests.docpipe.pdf_factory import build_pdf


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "docpipe.db"
    app = create_app(db_path=db_path)
    with TestClient(app) as test_client:
        test_client.db_path = db_path  # type: ignore[attr-defined]
        yield test_client


def _upload(client: TestClient, texts=None, name="doc.pdf", title=None) -> dict:
    texts = texts if texts is not None else [
        "Scaled dot product attention is the core mechanism.",
        "Positional encoding injects order information.",
    ]
    form = {"title": title} if title else {}
    resp = client.post(
        "/documents/upload",
        files={"file": (name, build_pdf(texts), "application/pdf")},
        data=form,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _search(client: TestClient, doc_id: int, q: str) -> list[dict]:
    resp = client.get(f"/documents/{doc_id}/search", params={"q": q})
    assert resp.status_code == 200
    return resp.json()["chunks"]


def test_health_on_combined_app(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_evidence_full_flow(client):
    """C0 upload -> search -> POST /evidence -> GET -> validate."""
    upload = _upload(client, title="Attention Is All You Need")
    doc_id = upload["document_id"]
    chunks = _search(client, doc_id, "scaled")
    assert chunks, "search must return at least one chunk"
    chunk = chunks[0]

    resp = client.post("/evidence", json={"chunk_id": chunk["chunk_id"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["evidence_id"] == "E-000001"
    assert body["document_id"] == doc_id
    assert body["chunk_id"] == chunk["chunk_id"]
    assert body["source_type"] == "local_pdf"
    assert body["source_title"] == "Attention Is All You Need"
    assert body["page_number"] == chunk["page_number"]
    assert body["content"] == chunk["content"]  # verbatim
    assert body["locator"] == {
        "page": chunk["page_number"],
        "chunk_id": chunk["chunk_id"],
    }

    got = client.get(f"/evidence/{body['evidence_id']}")
    assert got.status_code == 200
    assert got.json() == body

    val = client.get(f"/evidence/{body['evidence_id']}/validate")
    assert val.status_code == 200
    assert val.json() == {"evidence_id": "E-000001", "valid": True}


def test_duplicate_evidence_returns_existing(client):
    _upload(client)
    doc_id = 1
    chunk_id = _search(client, doc_id, "scaled")[0]["chunk_id"]

    first = client.post("/evidence", json={"chunk_id": chunk_id}).json()
    second = client.post("/evidence", json={"chunk_id": chunk_id}).json()
    assert second == first
    assert second["evidence_id"] == "E-000001"


def test_get_missing_evidence(client):
    resp = client.get("/evidence/E-999999")
    assert resp.status_code == 404
    assert resp.json() == {"error": "evidence not found"}


def test_invalid_evidence_id_format(client):
    for bad in ("abc", "E-1", "E-00000a"):
        resp = client.get(f"/evidence/{bad}")
        assert resp.status_code == 400, bad
        assert resp.json() == {"error": "invalid evidence ID"}
        resp = client.get(f"/evidence/{bad}/validate")
        assert resp.status_code == 400, bad


def test_validate_tampered_content(client):
    """Tamper test through the API: modify content, validate -> false."""
    _upload(client)
    chunk_id = _search(client, 1, "scaled")[0]["chunk_id"]
    ev = client.post("/evidence", json={"chunk_id": chunk_id}).json()

    conn = sqlite3.connect(str(client.db_path))
    conn.execute(
        "UPDATE evidence SET content = 'tampered content' WHERE id = ?",
        (ev["evidence_id"],),
    )
    conn.commit()
    conn.close()

    resp = client.get(f"/evidence/{ev['evidence_id']}/validate")
    assert resp.status_code == 200
    assert resp.json() == {
        "evidence_id": ev["evidence_id"],
        "valid": False,
        "reason": "content_mismatch",
    }


def test_create_evidence_missing_chunk(client):
    _upload(client)
    resp = client.post("/evidence", json={"chunk_id": 9999})
    assert resp.status_code == 404
    assert resp.json() == {"error": "chunk not found"}


def test_create_evidence_malformed_body(client):
    _upload(client)
    for payload in ({}, {"chunk_id": "abc"}, {"chunk_id": None}, [], "x"):
        resp = client.post("/evidence", json=payload)
        assert resp.status_code == 422, payload
        assert resp.json() == {"error": "invalid request body"}


def test_list_document_evidence(client):
    _upload(client)
    _upload(client, name="second.pdf", texts=["Different document words."])
    chunk_a = _search(client, 1, "scaled")[0]["chunk_id"]
    chunk_b = _search(client, 2, "different")[0]["chunk_id"]
    client.post("/evidence", json={"chunk_id": chunk_a})
    client.post("/evidence", json={"chunk_id": chunk_b})

    resp = client.get("/documents/1/evidence")
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == 1
    assert len(body["evidence"]) == 1
    assert body["evidence"][0]["chunk_id"] == chunk_a

    resp2 = client.get("/documents/2/evidence")
    assert len(resp2.json()["evidence"]) == 1
    assert resp2.json()["evidence"][0]["chunk_id"] == chunk_b


def test_list_document_evidence_missing_document(client):
    resp = client.get("/documents/999/evidence")
    assert resp.status_code == 404
    assert resp.json() == {"error": "document not found"}


def test_list_document_evidence_empty(client):
    _upload(client)
    resp = client.get("/documents/1/evidence")
    assert resp.status_code == 200
    assert resp.json() == {"document_id": 1, "evidence": []}


def test_c0_endpoints_unaffected_on_combined_app(client):
    """C0 regression on the combined app: upload, duplicate, search."""
    first = _upload(client)
    doc_id = first["document_id"]
    dup = _upload(client)
    assert dup["document_id"] == doc_id
    assert dup["status"] == "duplicate"
    listing = client.get("/documents").json()
    assert len(listing["documents"]) == 1
    assert listing["documents"][0]["chunk_count"] == 2
    assert len(_search(client, doc_id, "positional")) == 1


def test_evidence_persists_across_restart(client, tmp_path):
    """Server-restart simulation: evidence survives and validates."""
    _upload(client)
    chunk_id = _search(client, 1, "scaled")[0]["chunk_id"]
    ev = client.post("/evidence", json={"chunk_id": chunk_id}).json()

    # "Restart": fresh app over the same DB file.
    app2 = create_app(db_path=tmp_path / "docpipe.db")
    with TestClient(app2) as client2:
        got = client2.get(f"/evidence/{ev['evidence_id']}")
        assert got.status_code == 200
        assert got.json()["evidence_id"] == ev["evidence_id"]
        val = client2.get(f"/evidence/{ev['evidence_id']}/validate")
        assert val.json() == {"evidence_id": ev["evidence_id"], "valid": True}
