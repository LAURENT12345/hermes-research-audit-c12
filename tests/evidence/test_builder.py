"""Evidence Builder tests: creation, dedupe, and failure modes."""

from __future__ import annotations

import pytest

import docpipe.db as c0db
from docpipe.chunker import Chunk
from evidence import builder, store
from evidence.model import LOCAL_PDF_SOURCE_TYPE


@pytest.fixture
def seeded(tmp_path):
    """Two documents, two chunks each, at a fresh store."""
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)

    def make_doc(title: str, hash_: str, texts: list[str]) -> tuple[int, list[int]]:
        doc_id = c0db.insert_document(
            conn, title=title, file_path=f"/tmp/{hash_}.pdf",
            file_hash=hash_, status="ready",
        )
        chunk_ids = []
        for i, text in enumerate(texts):
            chunk_ids.append(
                c0db.insert_chunk(
                    conn, document_id=doc_id,
                    chunk=Chunk(
                        chunk_index=i, content=text,
                        normalized_content=text, page_number=i + 1,
                        section_path=None, token_count=len(text.split()),
                    ),
                )
            )
        return doc_id, chunk_ids

    doc_a, chunks_a = make_doc(
        "Attention Is All You Need", "ha",
        ["scaled dot product attention formula", "positional encoding section"],
    )
    doc_b, chunks_b = make_doc("Second Paper", "hb", ["unrelated content here"])
    conn.close()
    return tmp_path / "docpipe.db", doc_a, chunks_a, doc_b, chunks_b


def test_valid_chunk_creates_evidence(seeded):
    db_path, doc_a, chunks_a, _, _ = seeded
    chunk_id = chunks_a[0]

    ev = builder.create_evidence(chunk_id, db_path=db_path)

    conn = store.connect(db_path)
    try:
        chunk = c0db.get_chunk(conn, chunk_id)
    finally:
        conn.close()

    assert ev.evidence_id == "E-000001"
    assert ev.document_id == doc_a
    assert ev.chunk_id == chunk_id
    assert ev.page_number == chunk["page_number"]
    assert ev.section_path == chunk["section_path"]
    assert ev.content == chunk["content"]  # verbatim, byte-identical
    assert ev.source_type == LOCAL_PDF_SOURCE_TYPE
    assert ev.source_title == "Attention Is All You Need"
    assert ev.evidence_type == "chunk"
    assert ev.locator == {"page": chunk["page_number"], "chunk_id": chunk_id}


def test_provenance_fields_match_chunk(seeded):
    """The full provenance contract from the spec."""
    db_path, doc_a, chunks_a, _, _ = seeded
    chunk_id = chunks_a[1]
    ev = builder.create_evidence(chunk_id, db_path=db_path)

    conn = store.connect(db_path)
    try:
        chunk = c0db.get_chunk(conn, chunk_id)
        assert ev.document_id == chunk["document_id"]
        assert ev.chunk_id == chunk["id"]
        assert ev.page_number == chunk["page_number"]
        assert ev.content == chunk["content"]
    finally:
        conn.close()


def test_missing_chunk_fails(seeded):
    db_path, _, _, _, _ = seeded
    with pytest.raises(builder.ChunkNotFoundError):
        builder.create_evidence(9999, db_path=db_path)


def test_missing_document_fails(seeded, monkeypatch):
    """Corrupt C0 state: chunk exists but its document does not."""
    db_path, _, chunks_a, _, _ = seeded
    monkeypatch.setattr(
        c0db, "get_document", lambda conn, document_id: None
    )
    with pytest.raises(builder.DocumentNotFoundError):
        builder.create_evidence(chunks_a[0], db_path=db_path)


def test_duplicate_creation_returns_existing(seeded):
    db_path, _, chunks_a, _, _ = seeded
    chunk_id = chunks_a[0]
    first = builder.create_evidence(chunk_id, db_path=db_path)
    second = builder.create_evidence(chunk_id, db_path=db_path)

    assert second.evidence_id == first.evidence_id
    assert second == first
    conn = store.connect(db_path)
    try:
        assert store.count_evidence(conn) == 1
    finally:
        conn.close()


def test_builder_uses_chunk_document(seeded):
    """The document_id always comes from the chunk (never caller-supplied)."""
    db_path, _, _, doc_b, chunks_b = seeded
    ev = builder.create_evidence(chunks_b[0], db_path=db_path)
    assert ev.document_id == doc_b
    assert ev.source_title == "Second Paper"


def test_failed_creation_leaves_no_rows(seeded):
    """A failure never persists partial evidence."""
    db_path, _, _, _, _ = seeded
    with pytest.raises(builder.ChunkNotFoundError):
        builder.create_evidence(424242, db_path=db_path)
    conn = store.connect(db_path)
    try:
        assert store.count_evidence(conn) == 0
    finally:
        conn.close()
