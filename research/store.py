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

from hermes_cli.sqlite_util import add_column_if_missing, write_txn

import evidence.store as c1store

from research import config
from research.model import (
    Claim,
    ClaimEvidenceLink,
    Contradiction,
    LoopIteration,
    ResearchEvent,
    ResearchQuery,
    ResearchSession,
    format_claim_id,
    format_session_id,
    normalize_statement,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version / migration foundation
#
# The shared SQLite file's ``PRAGMA user_version`` is the version anchor
# for the research (C2) schema. Version 1 = the current research_* schema.
# C0/C1 layers still use their own idempotent CREATE TABLE IF NOT EXISTS
# (they cannot be modified); the research layer owns the version counter.
#
# Migrations are ordered dicts {version: SQL}; each is applied inside its
# own write_txn together with the version bump, so a crash cannot leave a
# half-applied migration. A database with a HIGHER version than this
# build understands is rejected deterministically.
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2


class SchemaVersionError(Exception):
    """The database schema version is newer than this build supports."""


def _exec_sql_statements(conn: sqlite3.Connection, sql: str) -> None:
    """Execute a multi-statement SQL script statement-by-statement.

    ``executescript`` would implicitly COMMIT an open transaction, which
    would break the atomic migration + version bump; splitting on ``;``
    keeps every statement inside the caller's write_txn.
    """
    for statement in sql.split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    """C3: claims gain decomposition provenance + scope context.

    Uses the repo's idempotent ``add_column_if_missing`` so a database
    that somehow already carries the columns (e.g. version reset on a
    current DB) upgrades cleanly instead of failing on duplicate column.
    """
    add_column_if_missing(
        conn, "claims", "parent_claim_id",
        "parent_claim_id TEXT REFERENCES claims(id)",
    )
    add_column_if_missing(
        conn, "claims", "context", "context TEXT"
    )


def _apply_migrations(conn: sqlite3.Connection) -> None:
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if current > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"database schema version {current} is newer than supported "
            f"version {SCHEMA_VERSION}; upgrade Hermes first"
        )
    for version in range(current + 1, SCHEMA_VERSION + 1):
        migration = _MIGRATIONS.get(version)
        if migration is None:  # pragma: no cover - guarded by SCHEMA_VERSION
            raise SchemaVersionError(f"no migration defined for version {version}")
        with write_txn(conn):
            if callable(migration):
                migration(conn)
            else:
                _exec_sql_statements(conn, migration)
            conn.execute(f"PRAGMA user_version = {version}")
        logger.info("research schema migrated to version %d", version)


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

CREATE TABLE IF NOT EXISTS research_iteration_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    iteration INTEGER NOT NULL,
    evidence_id TEXT NOT NULL REFERENCES evidence(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (session_id, iteration, evidence_id)
);

CREATE TABLE IF NOT EXISTS research_iteration_state (
    session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    iteration INTEGER NOT NULL,
    known_evidence_ids TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, iteration)
);

CREATE TABLE IF NOT EXISTS research_session_documents (
    session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (session_id, document_id)
);

CREATE TABLE IF NOT EXISTS claim_contradictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    claim_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    detail TEXT,
    resolution TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved_at DATETIME
);

CREATE TABLE IF NOT EXISTS research_session_locks (
    session_id TEXT PRIMARY KEY REFERENCES research_sessions(id) ON DELETE CASCADE,
    token TEXT NOT NULL,
    acquired_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL
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
    """Apply the research schema idempotently, driving version migrations.

    ``PRAGMA user_version`` anchors the schema version: 0/1 databases are
    migrated to ``SCHEMA_VERSION`` (migration 1 is the current schema);
    a newer version is rejected deterministically.
    """
    _apply_migrations(conn)


#: Ordered migrations {version: SQL or callable}; migration 1 is the
#: current schema. Callables receive the connection and run inside the
#: migration's write_txn (used for idempotent ALTERs).
_MIGRATIONS: dict[int, object] = {
    1: RESEARCH_SCHEMA_SQL,
    2: _migrate_to_v2,
}


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
    """Set session status and bump ``updated_at``; returns whether it hit.

    INTERNAL raw write — do NOT call outside the loop layer. Lifecycle
    transitions belong to ``ResearchLoop.start/abort/_apply_stop``.
    """
    cur = conn.execute(
        "UPDATE research_sessions SET status = ?, updated_at = CURRENT_TIMESTAMP "
        "WHERE id = ?",
        (status, session_id),
    )
    return cur.rowcount > 0


def set_final_answer(
    conn: sqlite3.Connection, session_id: str, final_answer: str
) -> bool:
    """Persist the synthesized answer WITHOUT touching session status
    (a complete_with_gaps session must stay complete_with_gaps)."""
    cur = conn.execute(
        "UPDATE research_sessions SET final_answer = ?, updated_at = "
        "CURRENT_TIMESTAMP WHERE id = ?",
        (final_answer, session_id),
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
    """INTERNAL raw status write — do NOT call outside the claims layer.

    Bypasses the claim transition matrix; callers MUST go through
    ``claims.transition_claim`` / ``claims.evaluate_claim`` (domain rules
    live there, including the atomic audit event).
    """
    cur = conn.execute(
        "UPDATE claims SET status = ?, notes = COALESCE(?, notes), "
        "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (status, notes, claim_id),
    )
    return cur.rowcount > 0


def set_claim_parent(
    conn: sqlite3.Connection, claim_id: str, parent_claim_id: Optional[str]
) -> bool:
    """Record decomposition provenance (C3): the claim's parent claim."""
    cur = conn.execute(
        "UPDATE claims SET parent_claim_id = ? WHERE id = ?",
        (parent_claim_id, claim_id),
    )
    return cur.rowcount > 0


def set_claim_context(
    conn: sqlite3.Connection, claim_id: str, context_json: Optional[str]
) -> bool:
    """Store a claim's scope context as JSON text (C3); None clears it."""
    cur = conn.execute(
        "UPDATE claims SET context = ? WHERE id = ?", (context_json, claim_id)
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
    event: Optional[tuple[str, str, Optional[str]]] = None,
) -> ClaimEvidenceLink:
    """Link evidence to a claim (idempotent on the 3-column PK).

    Raises sqlite3.IntegrityError for unknown claim/evidence or invalid
    role — FK/CHECK violations are never silently ignored.

    ``event`` optionally carries ``(session_id, event_type, detail)`` for
    an audit event inserted INSIDE the same transaction, so the link and
    its audit event commit atomically (no nested write_txn).
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
        if event is not None:
            session_id, event_type, detail = event
            conn.execute(
                "INSERT INTO research_events (session_id, event_type, detail) "
                "VALUES (?, ?, ?)",
                (session_id, event_type, detail),
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
# research_session_documents
# ---------------------------------------------------------------------------

def add_session_document(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    document_id: int,
) -> None:
    """Bind a C0 document to a session (idempotent on the PK).

    The document must exist — the FK raises on unknown ids (no
    INSERT OR IGNORE, so integrity errors are never swallowed).
    """
    existing = conn.execute(
        "SELECT 1 FROM research_session_documents "
        "WHERE session_id = ? AND document_id = ?",
        (session_id, document_id),
    ).fetchone()
    if existing is not None:
        return
    conn.execute(
        "INSERT INTO research_session_documents (session_id, document_id) "
        "VALUES (?, ?)",
        (session_id, document_id),
    )


def list_session_documents(
    conn: sqlite3.Connection, session_id: str
) -> list[int]:
    rows = conn.execute(
        "SELECT document_id FROM research_session_documents "
        "WHERE session_id = ? ORDER BY document_id",
        (session_id,),
    ).fetchall()
    return [int(r["document_id"]) for r in rows]


# ---------------------------------------------------------------------------
# research_session_locks — per-session execution lease
# ---------------------------------------------------------------------------

def _sqlite_now(conn: sqlite3.Connection) -> str:
    return str(conn.execute("SELECT datetime('now')").fetchone()[0])


def acquire_session_lock(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    token: str,
    ttl_seconds: int = config.SESSION_LOCK_TTL_SECONDS,
) -> bool:
    """Try to take the per-session execution lease.

    Returns True when ownership was acquired (fresh or after the previous
    lease expired — expiry makes the lease crash-safe across processes).
    Returns False when another live lease holds the session.

    The compare-and-set happens inside ``BEGIN IMMEDIATE``, so concurrent
    callers cannot both win.
    """
    with write_txn(conn):
        row = conn.execute(
            "SELECT expires_at FROM research_session_locks "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is not None and row["expires_at"] > _sqlite_now(conn):
            return False  # held by a live lease
        conn.execute(
            "INSERT INTO research_session_locks (session_id, token, expires_at) "
            "VALUES (?, ?, datetime('now', ?)) "
            "ON CONFLICT(session_id) DO UPDATE SET token = excluded.token, "
            "expires_at = excluded.expires_at, acquired_at = CURRENT_TIMESTAMP",
            (session_id, token, f"+{int(ttl_seconds)} seconds"),
        )
        return True


def release_session_lock(
    conn: sqlite3.Connection, *, session_id: str, token: str
) -> bool:
    """Release the lease; only the current token holder may release."""
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM research_session_locks WHERE session_id = ? AND token = ?",
            (session_id, token),
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# research_iteration_evidence / research_iteration_state
# ---------------------------------------------------------------------------

def add_iteration_evidence(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    iteration: int,
    evidence_id: str,
) -> None:
    """Record an evidence id returned during an iteration (idempotent)."""
    existing = conn.execute(
        "SELECT 1 FROM research_iteration_evidence "
        "WHERE session_id = ? AND iteration = ? AND evidence_id = ?",
        (session_id, iteration, evidence_id),
    ).fetchone()
    if existing is not None:
        return
    conn.execute(
        "INSERT INTO research_iteration_evidence (session_id, iteration, "
        "evidence_id) VALUES (?, ?, ?)",
        (session_id, iteration, evidence_id),
    )


def list_iteration_evidence(
    conn: sqlite3.Connection, session_id: str, iteration: int
) -> list[str]:
    rows = conn.execute(
        "SELECT evidence_id FROM research_iteration_evidence "
        "WHERE session_id = ? AND iteration = ? ORDER BY evidence_id",
        (session_id, iteration),
    ).fetchall()
    return [r["evidence_id"] for r in rows]


def save_iteration_known(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    iteration: int,
    known_evidence_ids: list[str],
) -> None:
    """Persist the session's known-evidence set at iteration start."""
    import json as _json

    conn.execute(
        "INSERT INTO research_iteration_state (session_id, iteration, "
        "known_evidence_ids) VALUES (?, ?, ?) "
        "ON CONFLICT (session_id, iteration) DO UPDATE SET known_evidence_ids = "
        "excluded.known_evidence_ids",
        (session_id, iteration, _json.dumps(sorted(known_evidence_ids))),
    )


def load_iteration_known(
    conn: sqlite3.Connection, session_id: str, iteration: int
) -> set[str] | None:
    """The persisted known-evidence set for an iteration, if recorded."""
    import json as _json

    row = conn.execute(
        "SELECT known_evidence_ids FROM research_iteration_state "
        "WHERE session_id = ? AND iteration = ?",
        (session_id, iteration),
    ).fetchone()
    if row is None:
        return None
    return set(_json.loads(row["known_evidence_ids"]))


# ---------------------------------------------------------------------------
# claim_contradictions
# ---------------------------------------------------------------------------

def create_contradiction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    claim_id: str,
    kind: str,
    detail: Optional[str] = None,
) -> Contradiction:
    """Insert an open contradiction record; returns it."""
    conn.execute(
        "INSERT INTO claim_contradictions (session_id, claim_id, kind, status, detail) "
        "VALUES (?, ?, ?, 'open', ?)",
        (session_id, claim_id, kind, detail),
    )
    row = conn.execute(
        "SELECT * FROM claim_contradictions WHERE id = last_insert_rowid()"
    ).fetchone()
    return Contradiction.from_row(row)


def get_contradiction(
    conn: sqlite3.Connection, contradiction_id: int
) -> Optional[Contradiction]:
    row = conn.execute(
        "SELECT * FROM claim_contradictions WHERE id = ?", (contradiction_id,)
    ).fetchone()
    return Contradiction.from_row(row) if row is not None else None


def list_contradictions(
    conn: sqlite3.Connection,
    session_id: str,
    status: Optional[str] = None,
) -> list[Contradiction]:
    if status is None:
        rows = conn.execute(
            "SELECT * FROM claim_contradictions WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM claim_contradictions WHERE session_id = ? AND status = ? "
            "ORDER BY id",
            (session_id, status),
        ).fetchall()
    return [Contradiction.from_row(row) for row in rows]


def find_open_contradiction(
    conn: sqlite3.Connection, claim_id: str, kind: str
) -> Optional[Contradiction]:
    """The open record for (claim, kind), if any (idempotency anchor)."""
    row = conn.execute(
        "SELECT * FROM claim_contradictions WHERE claim_id = ? AND kind = ? "
        "AND status = 'open' ORDER BY id LIMIT 1",
        (claim_id, kind),
    ).fetchone()
    return Contradiction.from_row(row) if row is not None else None


def close_contradiction(
    conn: sqlite3.Connection,
    contradiction_id: int,
    status: str,
    resolution: Optional[str] = None,
) -> bool:
    """Close a record ('resolved' | 'dismissed') with its resolution JSON."""
    cur = conn.execute(
        "UPDATE claim_contradictions SET status = ?, resolution = ?, "
        "resolved_at = CURRENT_TIMESTAMP WHERE id = ?",
        (status, resolution, contradiction_id),
    )
    return cur.rowcount > 0


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
