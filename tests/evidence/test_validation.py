"""Validation tests: integrity, tamper detection, and failure reasons."""

from __future__ import annotations

import pytest

import docpipe.db as c0db
from docpipe.chunker import Chunk
from evidence import builder, store, validation


@pytest.fixture
def seeded(tmp_path):
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    doc_a = c0db.insert_document(
        conn, title="Paper A", file_path="/tmp/a.pdf",
        file_hash="ha", status="ready",
    )
    doc_b = c0db.insert_document(
        conn, title="Paper B", file_path="/tmp/b.pdf",
        file_hash="hb", status="ready",
    )
    chunk_a = c0db.insert_chunk(
        conn, document_id=doc_a,
        chunk=Chunk(chunk_index=0, content="original chunk text",
                    normalized_content="original chunk text",
                    page_number=4, section_path=None, token_count=3),
    )
    c0db.insert_chunk(
        conn, document_id=doc_b,
        chunk=Chunk(chunk_index=0, content="other doc text",
                    normalized_content="other doc text",
                    page_number=1, section_path=None, token_count=3),
    )
    conn.close()
    return db_path, doc_a, doc_b, chunk_a


@pytest.fixture
def evidence_id(seeded):
    db_path, _, _, chunk_a = seeded
    return builder.create_evidence(chunk_a, db_path=db_path).evidence_id


def _conn(db_path):
    return store.connect(db_path)


def test_valid_evidence_passes(seeded, evidence_id):
    db_path, _, _, _ = seeded
    conn = _conn(db_path)
    try:
        result = validation.validate_evidence(conn, evidence_id)
        assert result.valid is True
        assert result.reason is None
        assert result.to_dict() == {"evidence_id": evidence_id, "valid": True}
    finally:
        conn.close()


def test_tampered_content_detected(seeded, evidence_id):
    """The spec's tamper test: modified content -> content_mismatch."""
    db_path, _, _, _ = seeded
    conn = _conn(db_path)
    try:
        conn.execute(
            "UPDATE evidence SET content = ? WHERE id = ?",
            ("modified text", evidence_id),
        )
        result = validation.validate_evidence(conn, evidence_id)
        assert result.valid is False
        assert result.reason == "content_mismatch"
        # Validation must NOT repair the row.
        row = conn.execute(
            "SELECT content FROM evidence WHERE id = ?", (evidence_id,)
        ).fetchone()
        assert row["content"] == "modified text"
    finally:
        conn.close()


def test_wrong_page_detected(seeded, evidence_id):
    db_path, _, _, _ = seeded
    conn = _conn(db_path)
    try:
        conn.execute(
            "UPDATE evidence SET page_number = 99 WHERE id = ?", (evidence_id,)
        )
        result = validation.validate_evidence(conn, evidence_id)
        assert result.valid is False
        assert result.reason == "page_mismatch"
    finally:
        conn.close()


def test_wrong_document_relationship_detected(seeded, evidence_id):
    """evidence.document_id points at another EXISTING document."""
    db_path, _, doc_b, _ = seeded
    conn = _conn(db_path)
    try:
        conn.execute(
            "UPDATE evidence SET document_id = ? WHERE id = ?",
            (doc_b, evidence_id),
        )
        result = validation.validate_evidence(conn, evidence_id)
        assert result.valid is False
        assert result.reason == "document_chunk_mismatch"
    finally:
        conn.close()


def test_missing_document_detected(seeded, evidence_id):
    db_path, _, _, _ = seeded
    conn = _conn(db_path)
    try:
        # FK constraint blocks pointing at a nonexistent document; simulate
        # the corrupt state with FK enforcement off (as with chunks).
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "UPDATE evidence SET document_id = 999 WHERE id = ?", (evidence_id,)
        )
        conn.execute("PRAGMA foreign_keys=ON")
        result = validation.validate_evidence(conn, evidence_id)
        assert result.valid is False
        assert result.reason == "document_not_found"
    finally:
        conn.close()


def test_missing_chunk_detected(seeded, evidence_id):
    db_path, _, _, chunk_a = seeded
    conn = _conn(db_path)
    try:
        # The FK constraint normally prevents deleting a chunk that has
        # evidence. Simulate a dangling reference (e.g. data imported
        # without FK enforcement) and verify validation reports it.
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_a,))
        conn.execute("PRAGMA foreign_keys=ON")
        result = validation.validate_evidence(conn, evidence_id)
        assert result.valid is False
        assert result.reason == "chunk_not_found"
    finally:
        conn.close()


def test_wrong_source_type_detected(seeded, evidence_id):
    db_path, _, _, _ = seeded
    conn = _conn(db_path)
    try:
        conn.execute(
            "UPDATE evidence SET source_type = 'web' WHERE id = ?",
            (evidence_id,),
        )
        result = validation.validate_evidence(conn, evidence_id)
        assert result.valid is False
        assert result.reason == "source_type_mismatch"
    finally:
        conn.close()


def test_missing_evidence_detected(seeded):
    db_path, _, _, _ = seeded
    conn = _conn(db_path)
    try:
        result = validation.validate_evidence(conn, "E-999999")
        assert result.valid is False
        assert result.reason == "evidence_not_found"
    finally:
        conn.close()
