"""HTTP API tests (FastAPI TestClient): health, upload, dedupe, list, search."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from docpipe import config
from docpipe.api import create_app

from tests.docpipe.pdf_factory import build_pdf


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "api.db"
    app = create_app(db_path=db_path)
    with TestClient(app) as test_client:
        yield test_client


def _upload(client: TestClient, name: str = "doc.pdf", texts=None, title=None):
    texts = texts if texts is not None else ["Servo motor control via PWM."]
    data = build_pdf(texts)
    files = {"file": (name, data, "application/pdf")}
    form = {"title": title} if title else {}
    return client.post("/documents/upload", files=files, data=form)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_upload_ready(client):
    resp = _upload(client, title="Robot Doc")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["chunks_created"] == 1
    assert isinstance(body["document_id"], int)


def test_duplicate_upload(client):
    first = _upload(client).json()
    second = _upload(client).json()
    assert second["document_id"] == first["document_id"]
    assert second["status"] == "duplicate"
    assert "chunks_created" not in second


def test_invalid_file_type(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )
    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid file type"}


def test_upload_invalid_pdf(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("broken.pdf", b"not a pdf", "application/pdf")},
    )
    assert resp.status_code == 422
    assert resp.json() == {"error": "invalid PDF"}


def test_upload_empty_pdf(client):
    resp = _upload(client, texts=[""])
    assert resp.status_code == 422
    assert resp.json() == {"error": "empty PDF"}


def test_upload_file_too_large(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    resp = _upload(client, texts=["x" * 500])
    assert resp.status_code == 413
    assert resp.json() == {"error": "file too large"}


def test_upload_missing_file_field(client):
    resp = client.post("/documents/upload")
    assert resp.status_code == 422  # FastAPI validation rejects missing file


def test_list_documents(client):
    assert client.get("/documents").json() == {"documents": []}
    first = _upload(client, title="Alpha").json()
    _upload(client, name="other.pdf", texts=["Different words."], title="Beta")
    body = client.get("/documents").json()
    docs = {d["title"]: d for d in body["documents"]}
    assert set(docs) == {"Alpha", "Beta"}
    assert docs["Alpha"]["document_id"] == first["document_id"]
    assert docs["Alpha"]["status"] == "ready"
    assert docs["Alpha"]["chunk_count"] == 1
    assert docs["Alpha"]["file_hash"]


def test_search_returns_relevant_chunks(client):
    doc_id = _upload(
        client,
        texts=["Servo motors are controlled by PWM pulses.", "ESP32 wifi module setup."],
    ).json()["document_id"]
    # Note: FTS5 unicode61 does not stem, so queries must use exact token
    # forms ("servo" matches, "motor" would not match "motors").
    resp = client.get(f"/documents/{doc_id}/search", params={"q": "servo"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == doc_id
    assert body["query"] == "servo"
    assert len(body["chunks"]) == 1
    chunk = body["chunks"][0]
    assert "servo" in chunk["content"].lower()
    assert chunk["page_number"] == 1
    assert chunk["section_path"] is None
    assert isinstance(chunk["chunk_id"], int)


def test_search_document_not_found(client):
    resp = client.get("/documents/999/search", params={"q": "anything"})
    assert resp.status_code == 404
    assert resp.json() == {"error": "document not found"}


def test_search_invalid_query(client):
    doc_id = _upload(client).json()["document_id"]
    resp = client.get(f"/documents/{doc_id}/search", params={"q": ""})
    assert resp.status_code == 400
    assert resp.json() == {"error": "invalid search query"}
    resp = client.get(f"/documents/{doc_id}/search", params={"q": "the and of"})
    assert resp.status_code == 400


def test_search_limit_clamped(client):
    doc_id = _upload(
        client, texts=[f"matching keyword in chunk {i}" for i in range(25)]
    ).json()["document_id"]
    resp = client.get(
        f"/documents/{doc_id}/search", params={"q": "matching", "limit": 1000}
    )
    assert resp.status_code == 200
    assert len(resp.json()["chunks"]) <= config.SEARCH_MAX_LIMIT


def test_search_special_characters_safe(client):
    doc_id = _upload(client, texts=["servo AND motor testing"]).json()["document_id"]
    resp = client.get(
        f"/documents/{doc_id}/search", params={"q": 'servo "quoted" AND (x)'}
    )
    assert resp.status_code in (200, 400)  # never 500


def test_full_acceptance_flow(client):
    """PDF -> upload -> sha256 -> sqlite -> parse -> chunk -> fts -> search."""
    texts = [
        "Chapter one: the robot dog platform.",
        "The servo motor gives the tail its wag.",
        "ESP32 provides wifi connectivity and OTA updates.",
    ]
    doc_id = _upload(client, name="acceptance.pdf", texts=texts).json()["document_id"]

    assert client.get(f"/documents/{doc_id}/search", params={"q": "wag"}).json()[
        "chunks"
    ][0]["page_number"] == 2
    assert client.get(f"/documents/{doc_id}/search", params={"q": "wifi"}).json()[
        "chunks"
    ][0]["page_number"] == 3

    # Same PDF twice -> same document, no re-ingestion.
    dup = _upload(client, name="acceptance.pdf", texts=texts).json()
    assert dup["document_id"] == doc_id
    assert dup["status"] == "duplicate"
