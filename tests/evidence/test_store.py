"""Evidence store tests: CRUD, ID stability, persistence, FK integrity."""

from __future__ import annotations

import sqlite3

import pytest

import docpipe.db as c0db
from evidence import store


@pytest.fixture
def seeded(tmp_path):
    """Fresh store with one document and two chunks."""
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    doc_id = c0db.insert_document(
        conn, title="Attention Is All You Need", file_path="/tmp/a.pdf",
        file_hash="h1", status="ready",
    )
    from docpipe.chunker import Chunk

    c1 = c0db.insert_chunk(
        conn, document_id=doc_id,
        chunk=Chunk(chunk_index=0, content="scaled dot product attention",
                    normalized_content="scaled dot product attention",
                    page_number=4, section_path=None, token_count=4),
    )
    c2 = c0db.insert_chunk(
        conn, document_id=doc_id,
        chunk=Chunk(chunk_index=1, content="multi head attention",
                    normalized_content="multi head attention",
                    page_number=5, section_path=None, token_count=3),
    )
    conn.close()
    return db_path, doc_id, c1, c2


def _make(db_path, doc_id: int, chunk_id: int) -> Evidence:
    conn = store.connect(db_path)
    try:
        return store.create_evidence(
            conn,
            document_id=doc_id, chunk_id=chunk_id, evidence_type="chunk",
            source_type="local_pdf", source_title="Attention Is All You Need",
            content="scaled dot product attention", page_number=4,
            section_path=None, locator={"page": 4, "chunk_id": chunk_id},
        )
    finally:
        conn.close()


def test_create_and_get(seeded):
    db_path, doc_id, c1, _ = seeded
    ev = _make(db_path, doc_id, c1)
    assert ev.evidence_id == "E-000001"
    assert ev.document_id == doc_id
    assert ev.chunk_id == c1
    assert ev.locator == {"page": 4, "chunk_id": c1}
    assert ev.created_at  # populated by DB default

    conn = store.connect(db_path)
    try:
        got = store.get_evidence(conn, "E-000001")
        assert got == ev
        assert got.content == "scaled dot product attention"
        assert got.source_type == "local_pdf"
        assert got.source_title == "Attention Is All You Need"
    finally:
        conn.close()


def test_sequential_ids(seeded):
    db_path, doc_id, c1, c2 = seeded
    conn = store.connect(db_path)
    try:
        for chunk_id in (c1, c2):
            store.create_evidence(
                conn, document_id=doc_id, chunk_id=chunk_id,
                evidence_type="chunk", source_type="local_pdf",
                source_title="t", content="x", page_number=1,
                section_path=None, locator={"page": 1, "chunk_id": chunk_id},
            )
        assert store.get_evidence_by_chunk(conn, c1).evidence_id == "E-000001"
        assert store.get_evidence_by_chunk(conn, c2).evidence_id == "E-000002"
        assert store.count_evidence(conn) == 2
    finally:
        conn.close()


def test_ids_stable_after_restart(seeded):
    """Restart simulation: new connection continues the sequence."""
    db_path, doc_id, c1, c2 = seeded
    _make(db_path, doc_id, c1)  # E-000001

    conn = store.connect(db_path)  # "restart"
    try:
        ev2 = store.create_evidence(
            conn, document_id=doc_id, chunk_id=c2, evidence_type="chunk",
            source_type="local_pdf", source_title="t", content="y",
            page_number=5, section_path=None,
            locator={"page": 5, "chunk_id": c2},
        )
        assert ev2.evidence_id == "E-000002"  # continues, not resets
        assert store.get_evidence(conn, "E-000001") is not None
    finally:
        conn.close()


def test_list_evidence(seeded):
    db_path, doc_id, c1, c2 = seeded
    conn = store.connect(db_path)
    try:
        store.create_evidence(
            conn, document_id=doc_id, chunk_id=c1, evidence_type="chunk",
            source_type="local_pdf", source_title="t", content="a",
            page_number=4, section_path=None,
            locator={"page": 4, "chunk_id": c1},
        )
        store.create_evidence(
            conn, document_id=doc_id, chunk_id=c2, evidence_type="chunk",
            source_type="local_pdf", source_title="t", content="b",
            page_number=5, section_path=None,
            locator={"page": 5, "chunk_id": c2},
        )
        rows = store.list_evidence(conn, doc_id)
        assert [e.evidence_id for e in rows] == ["E-000001", "E-000002"]
    finally:
        conn.close()


def test_foreign_key_integrity(seeded):
    db_path, doc_id, c1, _ = seeded
    conn = store.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            store.create_evidence(
                conn, document_id=doc_id, chunk_id=9999, evidence_type="chunk",
                source_type="local_pdf", source_title="t", content="x",
                page_number=1, section_path=None,
                locator={"page": 1, "chunk_id": 9999},
            )
        with pytest.raises(sqlite3.IntegrityError):
            store.create_evidence(
                conn, document_id=9999, chunk_id=c1, evidence_type="chunk",
                source_type="local_pdf", source_title="t", content="x",
                page_number=1, section_path=None,
                locator={"page": 1, "chunk_id": c1},
            )
    finally:
        conn.close()


def test_unique_chunk_constraint(seeded):
    db_path, doc_id, c1, _ = seeded
    conn = store.connect(db_path)
    try:
        store.create_evidence(
            conn, document_id=doc_id, chunk_id=c1, evidence_type="chunk",
            source_type="local_pdf", source_title="t", content="x",
            page_number=1, section_path=None,
            locator={"page": 1, "chunk_id": c1},
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.create_evidence(
                conn, document_id=doc_id, chunk_id=c1, evidence_type="chunk",
                source_type="local_pdf", source_title="t", content="y",
                page_number=1, section_path=None,
                locator={"page": 1, "chunk_id": c1},
            )
    finally:
        conn.close()


def test_schema_idempotent(seeded):
    db_path, _, _, _ = seeded
    conn = store.connect(db_path)
    try:
        store.init_schema(conn)  # second application must be a no-op
        store.init_schema(conn)
    finally:
        conn.close()
