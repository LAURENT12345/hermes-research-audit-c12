"""C2 storage: ``research_*`` tables in the shared C0/C1 SQLite database.

Same conventions as C0/C1: stdlib sqlite3, WAL, autocommit +
``BEGIN IMMEDIATE`` via ``hermes_cli.sqlite_util.write_txn``, idempotent
``CREATE TABLE IF NOT EXISTS`` applied on every connect. No separate
database, no changes to C0/C1 tables.

Integrity notes:

* ``claims.statement_normalized`` is NOT NULL and participates in
  ``UNIQUE (session_id, statement_normalized)`` — the deterministic
  deduplication key for claims (approved design correction).
* ``claim_evidence`` PK ``(claim_id, evidence_id, role)`` makes links
  idempotent at the database level.
* ``claim_evidence.evidence_id`` references the C1 ``evidence`` table
  WITHOUT cascade: deleting linked evidence fails loudly instead of
  silently destroying claim provenance (same policy as C1).
* Session children (queries/claims/iterations/events) cascade on session
  deletion.
* All writes go through idempotent helpers: repeat calls return the
  existing row rather than raising or duplicating.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from hermes_cli.sqlite_util import write_txn

import evidence.store as c1store

from research import config
from research.model import (
    Claim,
    ClaimEvidenceLink,
    LoopIteration,
    ResearchEvent,
    ResearchQuery,
    ResearchSession,
    format_claim_id,
    format_session_id,
    normalize_statement,
)

logger = logging.getLogger(__name__)

RESEARCH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS research_sessions (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    min_supporting INTEGER NOT NULL DEFAULT 1,
    max_iterations INTEGER NOT NULL DEFAULT 5,
    final_answer TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS research_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    origin TEXT NOT NULL DEFAULT 'user',
    status TEXT NOT NULL DEFAULT 'pending',
    iteration INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (session_id, query_text)
);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    statement TEXT NOT NULL,
    statement_normalized TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed',
    confidence REAL,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (session_id, statement_normalized)
);

CREATE TABLE IF NOT EXISTS claim_evidence (
    claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL REFERENCES evidence(id),
    role TEXT NOT NULL CHECK (role IN ('supporting', 'contradicting')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (claim_id, evidence_id, role)
);

CREATE TABLE IF NOT EXISTS research_iterations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    iteration INTEGER NOT NULL,
    phase TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    notes TEXT,
    UNIQUE (session_id, iteration, phase)
);

CREATE TABLE IF NOT EXISTS research_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    detail TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    """Apply the research schema idempotently (runs on every connect)."""
    conn.executescript(RESEARCH_SCHEMA_SQL)


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open the shared store with C0 + C1 + C2 schemas applied."""
    conn = c1store.connect(db_path)
    init_schema(conn)
    return conn


def _next_id(conn: sqlite3.Connection, table: str, prefix: str, formatter) -> str:
    """Sequential ``PREFIX-######`` id from persisted MAX (caller's txn)."""
    row = conn.execute(
        f"SELECT COALESCE(MAX(CAST(SUBSTR(id, {len(prefix) + 1}) AS INTEGER)), 0) + 1 "
        f"AS n FROM {table}"
    ).fetchone()
    return formatter(row["n"])


# ---------------------------------------------------------------------------
# research_sessions
# ---------------------------------------------------------------------------

def create_session(
    conn: sqlite3.Connection,
    *,
    question: str,
    min_supporting: int = config.DEFAULT_MIN_SUPPORTING,
    max_iterations: int = config.DEFAULT_MAX_ITERATIONS,
) -> ResearchSession:
    """Create a draft session; returns it (with its persisted id)."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    with write_txn(conn):
        session_id = _next_id(
            conn, "research_sessions", "S-", format_session_id
        )
        conn.execute(
            "INSERT INTO research_sessions "
            "(id, question, status, min_supporting, max_iterations) "
            "VALUES (?, ?, 'draft', ?, ?)",
            (session_id, question, min_supporting, max_iterations),
        )
    created = get_session(conn, session_id)
    assert created is not None  # just inserted
    logger.info("research session created: %s", session_id)
    return created


def get_session(
    conn: sqlite3.Connection, session_id: str
) -> Optional[ResearchSession]:
    row = conn.execute(
        "SELECT * FROM research_sessions WHERE id = ?", (session_id,)
    ).fetchone()
    return ResearchSession.from_row(row) if row is not None else None


def list_sessions(conn: sqlite3.Connection) -> list[ResearchSession]:
    rows = conn.execute(
        "SELECT * FROM research_sessions ORDER BY id"
    ).fetchall()
    return [ResearchSession.from_row(row) for row in rows]


def update_session_status(
    conn: sqlite3.Connection, session_id: str, status: str
) -> bool:
    """Set session status and bump ``updated_at``; returns whether it hit."""
    cur = conn.execute(
        "UPDATE research_sessions SET status = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ?",
        (status, session_id),
    )
    return cur.rowcount > 0


def set_session_answer(
    conn: sqlite3.Connection, session_id: str, final_answer: str
) -> bool:
    """Persist the synthesized answer (JSON text) and mark complete."""
    cur = conn.execute(
        "UPDATE research_sessions SET final_answer = ?, status = 'complete', "
        "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (final_answer, session_id),
    )
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# research_queries
# ---------------------------------------------------------------------------

def add_query(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    query_text: str,
    origin: str = "user",
    iteration: Optional[int] = None,
) -> ResearchQuery:
    """Add a retrieval query (idempotent: same session+text returns existing)."""
    query_text = query_text.strip()
    if not query_text:
        raise ValueError("query_text must not be empty")
    with write_txn(conn):
        existing = conn.execute(
            "SELECT * FROM research_queries WHERE session_id = ? AND query_text = ?",
            (session_id, query_text),
        ).fetchone()
        if existing is not None:
            return ResearchQuery.from_row(existing)
        conn.execute(
            "INSERT INTO research_queries (session_id, query_text, origin, "
            "iteration, status) VALUES (?, ?, ?, ?, 'pending')",
            (session_id, query_text, origin, iteration),
        )
    row = conn.execute(
        "SELECT * FROM research_queries WHERE session_id = ? AND query_text = ?",
        (session_id, query_text),
    ).fetchone()
    return ResearchQuery.from_row(row)


def get_queries(
    conn: sqlite3.Connection, session_id: str
) -> list[ResearchQuery]:
    rows = conn.execute(
        "SELECT * FROM research_queries WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [ResearchQuery.from_row(row) for row in rows]


def set_query_status(
    conn: sqlite3.Connection, query_id: int, status: str
) -> bool:
    cur = conn.execute(
        "UPDATE research_queries SET status = ? WHERE id = ?",
        (status, query_id),
    )
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------

def create_claim(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    statement: str,
    notes: Optional[str] = None,
) -> Claim:
    """Create a proposed claim (idempotent on normalized statement).

    ``statement_normalized`` is computed deterministically from the
    statement; the UNIQUE(session_id, statement_normalized) constraint
    guarantees one claim per folded statement within a session. A repeat
    call returns the existing claim.
    """
    statement = statement.strip()
    if not statement:
        raise ValueError("statement must not be empty")
    if len(statement) > config.MAX_STATEMENT_CHARS:
        raise ValueError(
            f"statement exceeds {config.MAX_STATEMENT_CHARS} characters"
        )
    normalized = normalize_statement(statement)
    with write_txn(conn):
        existing = conn.execute(
            "SELECT * FROM claims WHERE session_id = ? AND statement_normalized = ?",
            (session_id, normalized),
        ).fetchone()
        if existing is not None:
            return Claim.from_row(existing)
        claim_id = _next_id(conn, "claims", "C-", format_claim_id)
        conn.execute(
            "INSERT INTO claims (id, session_id, statement, "
            "statement_normalized, status, notes) "
            "VALUES (?, ?, ?, ?, 'proposed', ?)",
            (claim_id, session_id, statement, normalized, notes),
        )
    row = conn.execute(
        "SELECT * FROM claims WHERE id = ?", (claim_id,)
    ).fetchone()
    return Claim.from_row(row)


def get_claim(conn: sqlite3.Connection, claim_id: str) -> Optional[Claim]:
    row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
    return Claim.from_row(row) if row is not None else None


def list_claims(
    conn: sqlite3.Connection, session_id: str
) -> list[Claim]:
    rows = conn.execute(
        "SELECT * FROM claims WHERE session_id = ? ORDER BY id", (session_id,)
    ).fetchall()
    return [Claim.from_row(row) for row in rows]


def update_claim_status(
    conn: sqlite3.Connection,
    claim_id: str,
    status: str,
    notes: Optional[str] = None,
) -> bool:
    """Raw status write (transition rules live in the claims layer)."""
    cur = conn.execute(
        "UPDATE claims SET status = ?, notes = COALESCE(?, notes), "
        "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (status, notes, claim_id),
    )
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# claim_evidence
# ---------------------------------------------------------------------------

def link_evidence(
    conn: sqlite3.Connection,
    *,
    claim_id: str,
    evidence_id: str,
    role: str,
) -> ClaimEvidenceLink:
    """Link evidence to a claim (idempotent on the 3-column PK).

    Raises sqlite3.IntegrityError for unknown claim/evidence or invalid
    role — FK/CHECK violations are never silently ignored.
    """
    with write_txn(conn):
        existing = conn.execute(
            "SELECT * FROM claim_evidence WHERE claim_id = ? AND evidence_id = ? "
            "AND role = ?",
            (claim_id, evidence_id, role),
        ).fetchone()
        if existing is not None:
            return ClaimEvidenceLink.from_row(existing)
        conn.execute(
            "INSERT INTO claim_evidence (claim_id, evidence_id, role) "
            "VALUES (?, ?, ?)",
            (claim_id, evidence_id, role),
        )
    row = conn.execute(
        "SELECT * FROM claim_evidence WHERE claim_id = ? AND evidence_id = ? "
        "AND role = ?",
        (claim_id, evidence_id, role),
    ).fetchone()
    return ClaimEvidenceLink.from_row(row)


def get_links_for_claim(
    conn: sqlite3.Connection, claim_id: str
) -> list[ClaimEvidenceLink]:
    rows = conn.execute(
        "SELECT * FROM claim_evidence WHERE claim_id = ? ORDER BY role, evidence_id",
        (claim_id,),
    ).fetchall()
    return [ClaimEvidenceLink.from_row(row) for row in rows]


def get_links_for_evidence(
    conn: sqlite3.Connection, evidence_id: str
) -> list[ClaimEvidenceLink]:
    rows = conn.execute(
        "SELECT * FROM claim_evidence WHERE evidence_id = ? ORDER BY claim_id",
        (evidence_id,),
    ).fetchall()
    return [ClaimEvidenceLink.from_row(row) for row in rows]


# ---------------------------------------------------------------------------
# research_iterations
# ---------------------------------------------------------------------------

def ensure_iteration(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    iteration: int,
    phase: str,
) -> LoopIteration:
    """Record a phase execution; idempotent per (session, iteration, phase)."""
    with write_txn(conn):
        existing = conn.execute(
            "SELECT * FROM research_iterations WHERE session_id = ? "
            "AND iteration = ? AND phase = ?",
            (session_id, iteration, phase),
        ).fetchone()
        if existing is not None:
            return LoopIteration.from_row(existing)
        conn.execute(
            "INSERT INTO research_iterations (session_id, iteration, phase) "
            "VALUES (?, ?, ?)",
            (session_id, iteration, phase),
        )
    row = conn.execute(
        "SELECT * FROM research_iterations WHERE session_id = ? "
        "AND iteration = ? AND phase = ?",
        (session_id, iteration, phase),
    ).fetchone()
    return LoopIteration.from_row(row)


def complete_iteration(
    conn: sqlite3.Connection,
    iteration_id: int,
    status: str = "completed",
    notes: Optional[str] = None,
) -> bool:
    cur = conn.execute(
        "UPDATE research_iterations SET status = ?, completed_at = "
        "CURRENT_TIMESTAMP, notes = COALESCE(?, notes) WHERE id = ?",
        (status, notes, iteration_id),
    )
    return cur.rowcount > 0


def list_iterations(
    conn: sqlite3.Connection, session_id: str
) -> list[LoopIteration]:
    rows = conn.execute(
        "SELECT * FROM research_iterations WHERE session_id = ? "
        "ORDER BY iteration, id",
        (session_id,),
    ).fetchall()
    return [LoopIteration.from_row(row) for row in rows]


# ---------------------------------------------------------------------------
# research_events (append-only)
# ---------------------------------------------------------------------------

def add_event(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    event_type: str,
    detail: Optional[str] = None,
) -> ResearchEvent:
    """Append an audit event; returns it with its persisted timestamp."""
    conn.execute(
        "INSERT INTO research_events (session_id, event_type, detail) "
        "VALUES (?, ?, ?)",
        (session_id, event_type, detail),
    )
    row = conn.execute(
        "SELECT * FROM research_events WHERE id = last_insert_rowid()"
    ).fetchone()
    return ResearchEvent.from_row(row)


def list_events(
    conn: sqlite3.Connection, session_id: str
) -> list[ResearchEvent]:
    rows = conn.execute(
        "SELECT * FROM research_events WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [ResearchEvent.from_row(row) for row in rows]
