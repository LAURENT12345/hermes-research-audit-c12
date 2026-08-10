"""SQLite persistence for C0: ``documents``, ``chunks``, ``chunks_fts``.

Conventions follow the existing Hermes stores (kanban_db / projects_db):

* stdlib ``sqlite3``, WAL journal mode, busy timeout,
* ``BEGIN IMMEDIATE`` write transactions via
  ``hermes_cli.sqlite_util.write_txn``,
* DB file at ``<HERMES_HOME>/docpipe.db``.

FTS5 index
----------
``chunks_fts`` is an external-content FTS5 table over ``chunks``
(``content='chunks'``, ``content_rowid='id'``), kept in sync by AFTER
INSERT / UPDATE / DELETE triggers. Synchronization is verified by tests
after INSERT, UPDATE, DELETE and after reopening the database (the
triggers make restart-desync impossible; ``rebuild_fts`` exists as a
deliberate repair path and is also tested).

Cascades: deleting a ``documents`` row cascades to ``chunks``, and the
DELETE triggers clean the FTS index as a side effect.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from hermes_cli.sqlite_util import write_txn

from docpipe import config
from docpipe.chunker import Chunk

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'uploaded',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    page_number INTEGER,
    section_path TEXT,
    token_count INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    normalized_content,
    content='chunks',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, normalized_content)
    VALUES (new.id, new.content, new.normalized_content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, normalized_content)
    VALUES ('delete', old.id, old.content, old.normalized_content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, normalized_content)
    VALUES ('delete', old.id, old.content, old.normalized_content);
    INSERT INTO chunks_fts(rowid, content, normalized_content)
    VALUES (new.id, new.content, new.normalized_content);
END;
"""


class DatabaseError(Exception):
    """C0-level database failure (wraps sqlite3 errors at boundaries)."""


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the C0 store with schema applied.

    Returns a connection with ``sqlite3.Row`` factories, WAL mode and
    foreign keys enabled. Caller owns the connection (``close()``).
    """
    path = Path(db_path) if db_path else config.default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # isolation_level=None -> autocommit; every write transaction is an
    # explicit BEGIN IMMEDIATE via hermes_cli.sqlite_util.write_txn (the
    # same convention as kanban_db / projects_db).
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Apply the schema idempotently. Safe to call on every connect."""
    conn.executescript(SCHEMA_SQL)


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------

def insert_document(
    conn: sqlite3.Connection,
    *,
    title: str,
    file_path: str,
    file_hash: str,
    status: str = "uploaded",
) -> int:
    """Insert a document row; returns its id. Raises on hash collision."""
    cur = conn.execute(
        "INSERT INTO documents (title, file_path, file_hash, status) "
        "VALUES (?, ?, ?, ?)",
        (title, file_path, file_hash, status),
    )
    return int(cur.lastrowid)


def get_document(conn: sqlite3.Connection, document_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM documents WHERE id = ?", (document_id,)
    ).fetchone()


def get_document_by_hash(
    conn: sqlite3.Connection, file_hash: str
) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
    ).fetchone()


def list_documents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """All documents (newest first) with chunk counts."""
    return conn.execute(
        """
        SELECT d.*, COUNT(c.id) AS chunk_count
        FROM documents d
        LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.id
        ORDER BY d.id DESC
        """
    ).fetchall()


def delete_document(conn: sqlite3.Connection, document_id: int) -> bool:
    """Delete a document and its chunks (cascade) + FTS entries (triggers)."""
    cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# chunks
# ---------------------------------------------------------------------------

def insert_chunk(
    conn: sqlite3.Connection,
    *,
    document_id: int,
    chunk: Chunk,
) -> int:
    cur = conn.execute(
        "INSERT INTO chunks (document_id, chunk_index, content, "
        "normalized_content, page_number, section_path, token_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            document_id,
            chunk.chunk_index,
            chunk.content,
            chunk.normalized_content,
            chunk.page_number,
            chunk.section_path,
            chunk.token_count,
        ),
    )
    return int(cur.lastrowid)


def insert_chunks(
    conn: sqlite3.Connection, document_id: int, chunks: Iterable[Chunk]
) -> int:
    """Insert many chunks inside one transaction; returns the count."""
    count = 0
    with write_txn(conn):
        for chunk in chunks:
            insert_chunk(conn, document_id=document_id, chunk=chunk)
            count += 1
    return count


def get_chunk(conn: sqlite3.Connection, chunk_id: int) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()


def list_chunks(
    conn: sqlite3.Connection, document_id: int
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
        (document_id,),
    ).fetchall()


def update_chunk_content(
    conn: sqlite3.Connection, chunk_id: int, content: str, normalized_content: str
) -> bool:
    """Update a chunk's text; the AFTER UPDATE trigger re-syncs FTS."""
    cur = conn.execute(
        "UPDATE chunks SET content = ?, normalized_content = ? WHERE id = ?",
        (content, normalized_content, chunk_id),
    )
    return cur.rowcount > 0


def delete_chunk(conn: sqlite3.Connection, chunk_id: int) -> bool:
    """Delete a chunk; the AFTER DELETE trigger removes its FTS entry."""
    cur = conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    return cur.rowcount > 0


def chunk_count(conn: sqlite3.Connection, document_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE document_id = ?", (document_id,)
    ).fetchone()
    return int(row["n"])


# ---------------------------------------------------------------------------
# FTS maintenance
# ---------------------------------------------------------------------------

def rebuild_fts(conn: sqlite3.Connection) -> None:
    """Full rebuild of the FTS index from ``chunks`` (repair path)."""
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
