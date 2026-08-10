"""Database tests: schema, FTS trigger synchronization, persistence.

Covers the spec's database requirements: create, insert document,
insert chunks, FTS search, update chunk, delete chunk, reopen/restart.
"""

from __future__ import annotations

import sqlite3

import pytest

from docpipe import db
from docpipe.chunker import Chunk


def _chunk(index: int, text: str, page: int = 1) -> Chunk:
    return Chunk(
        chunk_index=index,
        content=text,
        normalized_content=text,
        page_number=page,
        section_path=None,
        token_count=len(text.split()),
    )


@pytest.fixture
def conn(tmp_path):
    """Fresh C0 store at an isolated path."""
    handle = db.connect(tmp_path / "docpipe.db")
    yield handle
    handle.close()


def test_init_creates_expected_tables(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    names = {r["name"] for r in rows}
    assert {"documents", "chunks", "chunks_fts"} <= names


def test_init_db_is_idempotent(conn):
    db.init_db(conn)
    db.init_db(conn)
    rows = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
    assert rows["n"] == 0


def test_insert_document_and_chunks_then_fts_search(conn):
    doc_id = db.insert_document(
        conn, title="manual", file_path="/tmp/a.pdf", file_hash="h1"
    )
    with db.write_txn(conn):  # noqa: SIM117 - explicit single transaction
        db.insert_chunk(conn, document_id=doc_id, chunk=_chunk(0, "servo motor control"))
        db.insert_chunk(conn, document_id=doc_id, chunk=_chunk(1, "esp32 wifi setup"))

    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'servo'"
    ).fetchall()
    assert [r["rowid"] for r in rows] == [1]
    assert db.chunk_count(conn, doc_id) == 2


def test_update_chunk_syncs_fts(conn):
    doc_id = db.insert_document(
        conn, title="t", file_path="/tmp/a.pdf", file_hash="h1"
    )
    chunk_id = db.insert_chunk(
        conn, document_id=doc_id, chunk=_chunk(0, "old keyword here")
    )
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'old'"
    ).fetchone()["n"] == 1

    assert db.update_chunk_content(
        conn, chunk_id, "brand new phrase", "brand new phrase"
    )
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'old'"
    ).fetchone()["n"] == 0
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'phrase'"
    ).fetchone()["n"] == 1


def test_delete_chunk_syncs_fts(conn):
    doc_id = db.insert_document(
        conn, title="t", file_path="/tmp/a.pdf", file_hash="h1"
    )
    chunk_id = db.insert_chunk(
        conn, document_id=doc_id, chunk=_chunk(0, "doomed text")
    )
    assert db.delete_chunk(conn, chunk_id)
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'doomed'"
    ).fetchone()["n"] == 0
    assert db.get_chunk(conn, chunk_id) is None


def test_delete_document_cascades_to_chunks_and_fts(conn):
    doc_id = db.insert_document(
        conn, title="t", file_path="/tmp/a.pdf", file_hash="h1"
    )
    db.insert_chunk(conn, document_id=doc_id, chunk=_chunk(0, "cascade me"))
    assert db.delete_document(conn, doc_id)
    assert db.chunk_count(conn, doc_id) == 0
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'cascade'"
    ).fetchone()["n"] == 0


def test_reopen_database_persists(tmp_path):
    """Application-restart simulation: close, reconnect, search still works."""
    db_path = tmp_path / "docpipe.db"
    conn1 = db.connect(db_path)
    doc_id = db.insert_document(
        conn1, title="persisted", file_path="/tmp/a.pdf", file_hash="h1"
    )
    db.insert_chunk(conn1, document_id=doc_id, chunk=_chunk(0, "persistent content"))
    conn1.close()

    conn2 = db.connect(db_path)  # restart
    try:
        assert db.get_document(conn2, doc_id) is not None
        rows = conn2.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'persistent'"
        ).fetchall()
        assert len(rows) == 1
    finally:
        conn2.close()


def test_rebuild_fts_repairs_index(conn):
    doc_id = db.insert_document(
        conn, title="t", file_path="/tmp/a.pdf", file_hash="h1"
    )
    chunk_id = db.insert_chunk(
        conn, document_id=doc_id, chunk=_chunk(0, "rebuild target")
    )
    # Simulate a desynced index by deleting the FTS row directly.
    conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'rebuild'"
    ).fetchone()["n"] == 0
    db.rebuild_fts(conn)
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'rebuild'"
    ).fetchone()["n"] == 1


def test_document_hash_is_unique(conn):
    db.insert_document(conn, title="a", file_path="/tmp/a.pdf", file_hash="same")
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_document(conn, title="b", file_path="/tmp/b.pdf", file_hash="same")


def test_fts_rank_is_bm25(conn):
    """BM25 ranking: a chunk containing the term twice ranks above one
    containing it once (all else equal)."""
    doc_id = db.insert_document(
        conn, title="t", file_path="/tmp/a.pdf", file_hash="h1"
    )
    db.insert_chunk(
        conn, document_id=doc_id, chunk=_chunk(0, "method method method")
    )
    db.insert_chunk(conn, document_id=doc_id, chunk=_chunk(1, "method"))
    rows = conn.execute(
        "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts "
        "WHERE chunks_fts MATCH 'method' ORDER BY score"
    ).fetchall()
    assert [r["rowid"] for r in rows] == [1, 2]
