"""Evidence storage: the ``evidence`` table in the shared C0 SQLite database.

Same conventions as C0 (``docpipe.db``): stdlib sqlite3, WAL,
``BEGIN IMMEDIATE`` write transactions via ``hermes_cli.sqlite_util``,
idempotent ``CREATE TABLE IF NOT EXISTS`` applied on every connect.

Schema follows the C1 specification; ``locator`` is stored as JSON text.
A UNIQUE index on ``chunk_id`` enforces the deterministic
one-evidence-per-chunk behavior at the database level.

Foreign keys point at C0's ``documents`` / ``chunks`` tables. C0's
``chunks`` cascade-delete with their document; evidence rows reference
chunks with NO ACTION, so deleting a document that has evidence attached
fails loudly instead of silently destroying provenance. Evidence rows are
removed explicitly (future C2+ concern), never implicitly.

``created_at`` is populated by the database default (CURRENT_TIMESTAMP).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from hermes_cli.sqlite_util import write_txn

import docpipe.db as c0db

from evidence.model import Evidence

logger = logging.getLogger(__name__)

EVIDENCE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    document_id INTEGER NOT NULL,
    chunk_id INTEGER NOT NULL,
    evidence_type TEXT NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER,
    section_path TEXT,
    source_title TEXT,
    source_type TEXT NOT NULL,
    locator TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_evidence_chunk ON evidence(chunk_id);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    """Apply the evidence schema idempotently (runs on every connect)."""
    conn.executescript(EVIDENCE_SCHEMA_SQL)


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open the shared C0 store with the evidence schema applied.

    Delegates to ``docpipe.db.connect`` (WAL, autocommit, foreign keys,
    C0 schema) so C0 and C1 always share one consistent database file.
    """
    conn = c0db.connect(db_path)
    init_schema(conn)
    return conn


def _next_evidence_id(conn: sqlite3.Connection) -> str:
    """Deterministic sequential ID, stable across restarts.

    Reads MAX(sequence) from persisted rows inside the caller's write
    transaction, so concurrent writers cannot collide (SQLite serializes
    IMMEDIATE writers).
    """
    row = conn.execute(
        "SELECT COALESCE(MAX(CAST(SUBSTR(id, 3) AS INTEGER)), 0) + 1 AS n "
        "FROM evidence"
    ).fetchone()
    from evidence.model import format_evidence_id

    return format_evidence_id(row["n"])


def create_evidence(
    conn: sqlite3.Connection,
    *,
    document_id: int,
    chunk_id: int,
    evidence_type: str,
    source_type: str,
    source_title: Optional[str],
    content: str,
    page_number: Optional[int],
    section_path: Optional[str],
    locator: dict,
) -> Evidence:
    """Insert one evidence row (call inside ``write_txn``); returns it."""
    evidence_id = _next_evidence_id(conn)
    conn.execute(
        "INSERT INTO evidence (id, document_id, chunk_id, evidence_type, "
        "content, page_number, section_path, source_title, source_type, "
        "locator) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            evidence_id,
            document_id,
            chunk_id,
            evidence_type,
            content,
            page_number,
            section_path,
            source_title,
            source_type,
            json.dumps(locator, ensure_ascii=False),
        ),
    )
    created = get_evidence(conn, evidence_id)
    assert created is not None  # just inserted
    logger.info("evidence created: %s (document %s, chunk %s)", evidence_id, document_id, chunk_id)
    return created


def get_evidence(
    conn: sqlite3.Connection, evidence_id: str
) -> Optional[Evidence]:
    row = conn.execute(
        "SELECT * FROM evidence WHERE id = ?", (evidence_id,)
    ).fetchone()
    return Evidence.from_row(row) if row is not None else None


def get_evidence_by_chunk(
    conn: sqlite3.Connection, chunk_id: int
) -> Optional[Evidence]:
    row = conn.execute(
        "SELECT * FROM evidence WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    return Evidence.from_row(row) if row is not None else None


def list_evidence(
    conn: sqlite3.Connection, document_id: int
) -> list[Evidence]:
    rows = conn.execute(
        "SELECT * FROM evidence WHERE document_id = ? ORDER BY id",
        (document_id,),
    ).fetchall()
    return [Evidence.from_row(row) for row in rows]


def count_evidence(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS n FROM evidence").fetchone()
    return int(row["n"])
