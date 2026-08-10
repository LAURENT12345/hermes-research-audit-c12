"""End-to-end ingestion tests: dedupe, rollback, provenance, errors."""

from __future__ import annotations

import pytest

from docpipe import config, db
from docpipe.chunker import Chunk
from docpipe.ingest import (
    EmptyPDFError,
    FileTooLargeError,
    IngestError,
    InvalidPDFError,
    MissingFileError,
    ingest_pdf,
)
from docpipe.search import search

from tests.docpipe.pdf_factory import build_pdf


@pytest.fixture
def store(tmp_path):
    """An isolated HERMES_HOME-style DB at a known path."""
    return tmp_path / "docpipe.db"


def _write_pdf(tmp_path, name: str, texts: list[str]) -> str:
    path = tmp_path / name
    path.write_bytes(build_pdf(texts))
    return str(path)


def test_ingest_creates_document_and_searchable_chunks(tmp_path, store):
    pdf = _write_pdf(tmp_path, "robot.pdf", ["Servo motors are controlled by PWM."])
    result = ingest_pdf(pdf, db_path=store)

    assert result.status == "ready"
    assert result.chunks_created == 1
    assert not result.duplicate

    conn = db.connect(store)
    try:
        doc = db.get_document(conn, result.document_id)
        assert doc["status"] == "ready"
        assert doc["title"] == "robot"
        assert doc["file_hash"]
        hits = search(conn, result.document_id, "servo")
        assert len(hits) == 1
        assert hits[0].page_number == 1
        assert "PWM" in hits[0].content
    finally:
        conn.close()


def test_duplicate_upload_returns_existing_document(tmp_path, store):
    pdf = _write_pdf(tmp_path, "dup.pdf", ["Identical content here."])
    first = ingest_pdf(pdf, db_path=store)
    second = ingest_pdf(pdf, db_path=store)

    assert second.document_id == first.document_id
    assert second.status == "duplicate"
    assert second.duplicate
    assert second.chunks_created == 0

    conn = db.connect(store)
    try:
        assert db.chunk_count(conn, first.document_id) == first.chunks_created
    finally:
        conn.close()


def test_different_files_become_different_documents(tmp_path, store):
    pdf_a = _write_pdf(tmp_path, "a.pdf", ["Content of A."])
    pdf_b = _write_pdf(tmp_path, "b.pdf", ["Content of B."])
    a = ingest_pdf(pdf_a, db_path=store)
    b = ingest_pdf(pdf_b, db_path=store)
    assert a.document_id != b.document_id


def test_same_content_different_filenames_is_duplicate(tmp_path, store):
    pdf_a = _write_pdf(tmp_path, "copy1.pdf", ["Same bytes."])
    pdf_b = _write_pdf(tmp_path, "copy2.pdf", ["Same bytes."])
    a = ingest_pdf(pdf_a, db_path=store)
    b = ingest_pdf(pdf_b, db_path=store)
    assert a.document_id == b.document_id
    assert b.status == "duplicate"


def test_title_override(tmp_path, store):
    pdf = _write_pdf(tmp_path, "file.pdf", ["Some text."])
    result = ingest_pdf(pdf, title="Custom Title", db_path=store)
    conn = db.connect(store)
    try:
        assert db.get_document(conn, result.document_id)["title"] == "Custom Title"
    finally:
        conn.close()


def test_invalid_pdf_leaves_database_untouched(tmp_path, store):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(InvalidPDFError):
        ingest_pdf(bad, db_path=store)
    conn = db.connect(store)
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] == 0
    finally:
        conn.close()


def test_empty_pdf_rejected(tmp_path, store):
    empty = _write_pdf(tmp_path, "empty.pdf", [""])
    with pytest.raises(EmptyPDFError):
        ingest_pdf(empty, db_path=store)
    conn = db.connect(store)
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"] == 0
    finally:
        conn.close()


def test_missing_file(tmp_path, store):
    with pytest.raises(MissingFileError):
        ingest_pdf(tmp_path / "ghost.pdf", db_path=store)


def test_file_too_large(tmp_path, store, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)
    big = tmp_path / "big.pdf"
    big.write_bytes(build_pdf(["x" * 500]))
    with pytest.raises(FileTooLargeError):
        ingest_pdf(big, db_path=store)


def test_provenance_fields_present(tmp_path, store):
    pdf = _write_pdf(tmp_path, "prov.pdf", ["First page text.", "Second page text."])
    result = ingest_pdf(pdf, db_path=store)
    conn = db.connect(store)
    try:
        chunks = db.list_chunks(conn, result.document_id)
        assert len(chunks) == 2
        for row in chunks:
            assert row["document_id"] == result.document_id
            assert row["content"]
            assert row["normalized_content"]
            assert row["page_number"] in (1, 2)
            assert row["section_path"] is None
            assert row["token_count"] > 0
            assert row["chunk_index"] in (0, 1)
    finally:
        conn.close()


def test_multipage_chunks_keep_page_numbers(tmp_path, store):
    pdf = _write_pdf(
        tmp_path, "pages.pdf",
        ["Servo on page one.", "Motor on page two.", "Wifi on page three."],
    )
    result = ingest_pdf(pdf, db_path=store)
    conn = db.connect(store)
    try:
        hits = search(conn, result.document_id, "motor")
        assert hits[0].page_number == 2
    finally:
        conn.close()


def test_ingest_error_types_are_ingest_errors():
    assert issubclass(InvalidPDFError, IngestError)
    assert issubclass(EmptyPDFError, IngestError)
    assert issubclass(FileTooLargeError, IngestError)
    assert issubclass(MissingFileError, IngestError)
