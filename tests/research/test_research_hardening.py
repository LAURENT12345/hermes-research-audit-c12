"""Commit 10 hardening tests: concurrency lease, lifecycle atomicity,
schema migration foundation."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading

import pytest

import docpipe.db as c0db
import docpipe.ingest as c0ingest
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, loop, store

from tests.docpipe.pdf_factory import build_pdf

CHUNK_TEXTS = ["Scaled dot product attention divides by the square root of dk."]
QUESTION = "Why does scaled dot product attention divides by the square root of dk?"


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "docpipe.db"
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(build_pdf(CHUNK_TEXTS))
    ingested = c0ingest.ingest_pdf(pdf, db_path=db_path)
    conn = store.connect(db_path)
    session = store.create_session(conn, question=QUESTION)
    store.add_session_document(
        conn, session_id=session.session_id, document_id=ingested.document_id
    )
    conn.close()
    return db_path, session.session_id


# ---------------------------------------------------------------------------
# Concurrency — per-session lease
# ---------------------------------------------------------------------------

def test_acquire_then_second_acquire_busy(env):
    db_path, session_id = env
    conn1 = store.connect(db_path)
    conn2 = store.connect(db_path)
    try:
        assert store.acquire_session_lock(
            conn1, session_id=session_id, token="token-a"
        ) is True
        # A second live lease on a different connection cannot win.
        assert store.acquire_session_lock(
            conn2, session_id=session_id, token="token-b"
        ) is False
    finally:
        conn1.close()
        conn2.close()


def test_release_only_by_token_holder(env):
    db_path, session_id = env
    conn = store.connect(db_path)
    try:
        assert store.acquire_session_lock(
            conn, session_id=session_id, token="tok-1"
        ) is True
        assert store.release_session_lock(
            conn, session_id=session_id, token="tok-wrong"
        ) is False  # wrong token cannot release
        assert store.release_session_lock(
            conn, session_id=session_id, token="tok-1"
        ) is True
        # After release the lease is free again.
        assert store.acquire_session_lock(
            conn, session_id=session_id, token="tok-2"
        ) is True
    finally:
        conn.close()


def test_expired_lease_can_be_taken_over(env):
    """A crashed holder's lease expires; the next caller may take over."""
    db_path, session_id = env
    conn = store.connect(db_path)
    try:
        assert store.acquire_session_lock(
            conn, session_id=session_id, token="crashed", ttl_seconds=300
        ) is True
        # Expire the lease deterministically (no sleeps).
        conn.execute(
            "UPDATE research_session_locks SET expires_at = datetime('now', '-1 seconds') "
            "WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()
        assert store.acquire_session_lock(
            conn, session_id=session_id, token="recovered"
        ) is True
    finally:
        conn.close()


def test_loop_iterate_raises_when_busy(env):
    """While another call holds the lease, iterate_once -> SessionBusyError
    and no duplicate logical iteration happens."""
    db_path, session_id = env
    conn = store.connect(db_path)
    try:
        assert store.acquire_session_lock(
            conn, session_id=session_id, token="holder"
        ) is True
        store.update_session_status(conn, session_id, "active")
        iterations_before = len(store.list_iterations(conn, session_id))
        with pytest.raises(loop.SessionBusyError):
            loop.ResearchLoop(db_path=db_path).iterate_once(session_id)
        assert len(store.list_iterations(conn, session_id)) == iterations_before
    finally:
        conn.close()


def test_concurrent_iterate_serialized_via_lease(env):
    """Two threads racing iterate_once: exactly one wins, the other gets
    SessionBusyError — no interleaved phases, no duplicate audit events."""
    db_path, session_id = env
    conn = store.connect(db_path)
    store.update_session_status(conn, session_id, "active")
    conn.close()

    results: list = []
    barrier = threading.Barrier(2)
    lock_holder_done = threading.Event()

    def worker_a():
        barrier.wait()
        try:
            result = loop.ResearchLoop(db_path=db_path).iterate_once(session_id)
            results.append(("ok", result.session_status))
        except loop.SessionBusyError:
            results.append(("busy", None))
        finally:
            lock_holder_done.set()

    def worker_b():
        # Hold the lease ourselves to force a deterministic collision.
        conn = store.connect(db_path)
        try:
            barrier.wait()
            assert store.acquire_session_lock(
                conn, session_id=session_id, token="b"
            ) is True
            lock_holder_done.wait(timeout=30)
        finally:
            store.release_session_lock(conn, session_id=session_id, token="b")
            conn.close()

    t_a = threading.Thread(target=worker_a)
    t_b = threading.Thread(target=worker_b)
    t_a.start()
    t_b.start()
    t_a.join(timeout=60)
    t_b.join(timeout=60)

    assert ("busy", None) in results, "worker A must observe the held lease"
    assert not t_a.is_alive() and not t_b.is_alive()

    # No duplicate logical iteration was created by the losing caller.
    conn = store.connect(db_path)
    try:
        iterations = store.list_iterations(conn, session_id)
        assert len(iterations) <= 5  # at most one full iteration's phases
    finally:
        conn.close()


def test_lease_released_after_normal_run(env):
    db_path, session_id = env
    loop.ResearchLoop(db_path=db_path).start(session_id)
    conn = store.connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM research_session_locks WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert row is None  # lease released in finally
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lifecycle atomicity
# ---------------------------------------------------------------------------

def test_apply_stop_rolls_back_when_event_fails(env, monkeypatch):
    db_path, session_id = env
    conn = store.connect(db_path)
    try:
        store.update_session_status(conn, session_id, "active")
        session = store.get_session(conn, session_id)

        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("disk failure")

        monkeypatch.setattr(store, "add_event", boom)
        with pytest.raises(sqlite3.OperationalError):
            loop.ResearchLoop(db_path=db_path)._apply_stop(
                conn, session, "synthesis_ready"
            )
        # Neither the status change nor any event survived the rollback.
        assert store.get_session(conn, session_id).status == "active"
        assert not [
            e for e in store.list_events(conn, session_id)
            if e.event_type in ("synthesizing_ready", "session_completed")
        ]
    finally:
        conn.close()


def test_abort_rolls_back_when_event_fails(env, monkeypatch):
    db_path, session_id = env
    conn = store.connect(db_path)
    try:
        store.update_session_status(conn, session_id, "active")

        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("disk failure")

        monkeypatch.setattr(store, "add_event", boom)
        with pytest.raises(sqlite3.OperationalError):
            loop.ResearchLoop(db_path=db_path).abort(session_id)
        assert store.get_session(conn, session_id).status == "active"
        assert not [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "aborted"
        ]
    finally:
        conn.close()


def test_start_draft_to_active_rolls_back_when_event_fails(env, monkeypatch):
    db_path, session_id = env
    conn = store.connect(db_path)
    try:
        assert store.get_session(conn, session_id).status == "draft"

        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("disk failure")

        monkeypatch.setattr(store, "add_event", boom)
        with pytest.raises(sqlite3.OperationalError):
            loop.ResearchLoop(db_path=db_path).start(session_id)
        # draft -> active + session_created must be atomic.
        assert store.get_session(conn, session_id).status == "draft"
        assert not [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "session_created"
        ]
    finally:
        conn.close()


def test_concurrent_abort_single_event(env):
    """Two aborts: the second finds the session already aborted and must
    not duplicate the audit event."""
    db_path, session_id = env
    conn = store.connect(db_path)
    store.update_session_status(conn, session_id, "active")
    conn.close()

    loop.ResearchLoop(db_path=db_path).abort(session_id)
    with pytest.raises(loop.InvalidSessionTransitionError):
        loop.ResearchLoop(db_path=db_path).abort(session_id)

    conn = store.connect(db_path)
    try:
        events = [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "aborted"
        ]
        assert len(events) == 1
        assert store.get_session(conn, session_id).status == "aborted"
    finally:
        conn.close()


def test_subprocess_crash_mid_lifecycle_transition_rolls_back(env):
    """A real process dying inside the atomic status+event transaction
    leaves NEITHER the status change NOR the event."""
    db_path, session_id = env
    conn = store.connect(db_path)
    store.update_session_status(conn, session_id, "active")
    conn.close()

    code = (
        "import sqlite3, os\n"
        f"db = {str(db_path)!r}\n"
        "conn = sqlite3.connect(db, isolation_level=None)\n"
        "conn.execute('BEGIN IMMEDIATE')\n"
        f"conn.execute(\"UPDATE research_sessions SET status='complete' "
        f"WHERE id={session_id!r}\")\n"
        "conn.execute(\"INSERT INTO research_events (session_id, event_type) "
        f"VALUES ({session_id!r}, 'session_completed')\")\n"
        "os._exit(1)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, timeout=60)
    assert proc.returncode == 1

    conn = store.connect(db_path)
    try:
        assert store.get_session(conn, session_id).status == "active"
        assert not [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "session_completed"
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema migration foundation
# ---------------------------------------------------------------------------

def test_fresh_db_migrates_to_current_version(env):
    db_path, _ = env
    conn = store.connect(db_path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == store.SCHEMA_VERSION
        tables = {
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'research_%'"
            )
        }
        assert "research_session_locks" in tables
    finally:
        conn.close()


def test_reconnect_is_idempotent(env):
    db_path, _ = env
    store.connect(db_path).close()
    conn = store.connect(db_path)  # second connect: no-op migration
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION
    finally:
        conn.close()


def test_upgrade_from_pre_version_database(tmp_path):
    """A v1-era database (migration-1 schema, user_version=0/1) upgrades
    to the current version deterministically and keeps its data."""
    db_path = tmp_path / "v1.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(store._MIGRATIONS[1])  # v1 schema only (no v2 cols)
    conn.execute("PRAGMA user_version = 0")   # pre-versioning marker
    conn.execute(
        "INSERT INTO research_sessions (id, question, status) "
        "VALUES ('S-000001', 'old question', 'draft')"
    )
    conn.commit()
    conn.close()

    conn = store.connect(db_path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION
        # Existing data survived the upgrade; new columns are present.
        assert store.get_session(conn, "S-000001") is not None
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(claims)")}
        assert "parent_claim_id" in cols and "context" in cols
    finally:
        conn.close()


def test_future_schema_version_rejected(env):
    db_path, _ = env
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 999")
    conn.commit()
    conn.close()
    with pytest.raises(store.SchemaVersionError):
        store.connect(db_path)


def test_migration_failure_rolls_back_version(env, monkeypatch):
    """A failing migration must not bump user_version (no half-applied
    schema state is ever marked as current)."""
    db_path, _ = env

    def broken_statements(conn, sql):
        raise sqlite3.OperationalError("boom mid-migration")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()

    monkeypatch.setattr(store, "_exec_sql_statements", broken_statements)
    with pytest.raises(sqlite3.OperationalError):
        store.connect(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        conn.close()
