"""Atomicity regression tests for the Commit 7 hardening.

Covers the two Kimi K3 findings:

* P0-A: claim transition / evidence link / phase completion must commit
  their state AND audit event atomically — a failure mid-way rolls back
  BOTH, so resume can never produce duplicate or missing audit events.
* P0-B: synthesis must commit ``final_answer`` AND the ``synthesized``
  event atomically.

Two techniques:
1. deterministic: monkeypatch ``store.add_event`` to raise inside the
   transaction — the surrounding ``write_txn`` must roll the state back;
2. subprocess crash: a child process dies (``os._exit``) mid-transaction —
   SQLite's rollback-on-close must leave no partial write.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys

import pytest

import docpipe.db as c0db
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, loop, store
from research.synthesis import synthesize


@pytest.fixture
def seeded(tmp_path):
    """One document, one chunk, one evidence, one session."""
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    doc_id = c0db.insert_document(
        conn, title="Paper", file_path="/tmp/p.pdf", file_hash="ha",
        status="ready",
    )
    chunk_id = c0db.insert_chunk(
        conn, document_id=doc_id,
        chunk=Chunk(chunk_index=0,
                    content="Scaled dot product attention divides by the square root of dk.",
                    normalized_content="Scaled dot product attention divides by the square root of dk.",
                    page_number=1, section_path=None, token_count=12),
    )
    ev = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_id, evidence_type="chunk",
        source_type="local_pdf", source_title="Paper",
        content="Scaled dot product attention divides by the square root of dk.",
        page_number=1, section_path=None,
        locator={"page": 1, "chunk_id": chunk_id},
    )
    session = store.create_session(conn, question="Why scale by sqrt dk?")
    store.update_session_status(conn, session.session_id, "active")
    conn.close()
    return db_path, doc_id, session.session_id, ev.evidence_id


# ---------------------------------------------------------------------------
# P0-A: transition_claim atomicity
# ---------------------------------------------------------------------------

def test_transition_claim_rolls_back_status_when_event_fails(seeded, monkeypatch):
    db_path, _, session_id, _ = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Atomic claim")
        assert c.status == "proposed"

        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("disk failure")

        monkeypatch.setattr(store, "add_event", boom)
        with pytest.raises(sqlite3.OperationalError):
            claims.transition_claim(conn, c.claim_id, "supported")

        # Both the status change AND the event must be rolled back.
        assert store.get_claim(conn, c.claim_id).status == "proposed"
        assert not [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "claim_status_changed"
        ]
    finally:
        conn.close()


def test_link_evidence_event_commits_with_link(seeded):
    """The evidence_linked event is inserted INSIDE the same transaction
    as the link row (store.link_evidence 'event' parameter)."""
    db_path, _, session_id, ev_id = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Linked claim")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev_id,
                             role="supporting")

        # Both the link and its audit event exist together.
        assert len(store.get_links_for_claim(conn, c.claim_id)) == 1
        events = [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "evidence_linked"
        ]
        assert len(events) == 1

        # Idempotent duplicate link: no second link, no second event.
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev_id,
                             role="supporting")
        assert len(store.get_links_for_claim(conn, c.claim_id)) == 1
        assert len([
            e for e in store.list_events(conn, session_id)
            if e.event_type == "evidence_linked"
        ]) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# P0-A: phase completion atomicity
# ---------------------------------------------------------------------------

def test_phase_completion_rolls_back_when_event_fails(seeded, monkeypatch):
    db_path, _, session_id, _ = seeded
    conn = store.connect(db_path)
    try:
        row = store.ensure_iteration(
            conn, session_id=session_id, iteration=1, phase="planning"
        )
        assert row.status == "running"

        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("disk failure")

        monkeypatch.setattr(store, "add_event", boom)
        loop_ = loop.ResearchLoop(db_path=db_path)
        # The completion write fails atomically: the phase stays 'running'
        # and no phase_completed event is recorded.
        with pytest.raises(sqlite3.OperationalError):
            loop_._run_phase(conn, store.get_session(conn, session_id), 1,
                             "planning", lambda: None)

        rows = store.list_iterations(conn, session_id)
        assert rows[0].status == "running"
        assert not [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "phase_completed"
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# P0-B: synthesis atomicity
# ---------------------------------------------------------------------------

def test_synthesize_rolls_back_answer_when_event_fails(seeded, monkeypatch):
    db_path, _, session_id, ev_id = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id,
                                statement="Supported statement text.")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev_id,
                             role="supporting")
        claims.evaluate_claim(conn, c.claim_id)
        store.update_session_status(conn, session_id, "complete")

        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("disk failure")

        monkeypatch.setattr(store, "add_event", boom)
        with pytest.raises(sqlite3.OperationalError):
            synthesize(conn, session_id)

        # final_answer must NOT be persisted without its event.
        assert store.get_session(conn, session_id).final_answer is None
        assert not [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "synthesized"
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Subprocess crash test (real os._exit mid-transaction)
# ---------------------------------------------------------------------------

def _child_script(db_path: str) -> str:
    return (
        "import sqlite3, os, sys\n"
        f"db = {db_path!r}\n"
        "conn = sqlite3.connect(db, isolation_level=None)\n"
        "conn.execute('BEGIN IMMEDIATE')\n"
        "conn.execute(\"UPDATE claims SET status='supported' WHERE id='C-000001'\")\n"
        "conn.execute(\"INSERT INTO research_events (session_id, event_type) \"\n"
        "             \"VALUES ('S-000001', 'crash_marker')\")\n"
        "os._exit(1)  # die WITHOUT commit/rollback\n"
    )


def test_subprocess_crash_mid_transaction_rolls_back(seeded):
    """A process dying mid-transaction leaves NO partial write."""
    db_path, _, session_id, _ = seeded
    conn = store.connect(db_path)
    c = claims.create_claim(conn, session_id=session_id, statement="Crash claim")
    conn.close()
    assert c.claim_id == "C-000001"

    code = _child_script(str(db_path))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 1  # os._exit(1)

    conn = store.connect(db_path)
    try:
        # No partial write survived the crash.
        assert store.get_claim(conn, "C-000001").status == "proposed"
        assert not [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "crash_marker"
        ]
    finally:
        conn.close()
