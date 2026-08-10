"""C2 store tests: schema, idempotency, FK integrity, persistence.

Seeds real C0 document/chunk rows and a real C1 evidence row so every
research table is exercised against the actual shared database.
"""

from __future__ import annotations

import sqlite3

import pytest

import docpipe.db as c0db
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import store


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "docpipe.db"


@pytest.fixture
def seeded(db_path):
    """Store with one C0 doc+chunk, one C1 evidence, one research session."""
    conn = store.connect(db_path)
    doc_id = c0db.insert_document(
        conn, title="Attention Is All You Need", file_path="/tmp/a.pdf",
        file_hash="ha", status="ready",
    )
    chunk_id = c0db.insert_chunk(
        conn, document_id=doc_id,
        chunk=Chunk(chunk_index=0, content="The dot product has variance d_k.",
                    normalized_content="The dot product has variance d_k.",
                    page_number=4, section_path=None, token_count=7),
    )
    evidence = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_id, evidence_type="chunk",
        source_type="local_pdf", source_title="Attention Is All You Need",
        content="The dot product has variance d_k.", page_number=4,
        section_path=None, locator={"page": 4, "chunk_id": chunk_id},
    )
    session = store.create_session(conn, question="Why scale by sqrt(d_k)?")
    conn.close()
    return db_path, doc_id, chunk_id, evidence.evidence_id, session.session_id


def _connect(db_path):
    return store.connect(db_path)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_init_creates_expected_tables(db_path):
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('research_sessions','research_queries','claims','claim_evidence',"
            " 'research_iterations','research_events') ORDER BY name"
        ).fetchall()
        assert {r["name"] for r in rows} == {
            "research_sessions", "research_queries", "claims",
            "claim_evidence", "research_iterations", "research_events",
        }
        # The shared C0/C1 tables must still exist.
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('documents','chunks','chunks_fts','evidence')"
        ).fetchall()
        assert len(rows) == 4
    finally:
        conn.close()


def test_init_schema_idempotent(db_path):
    conn = _connect(db_path)
    try:
        store.init_schema(conn)
        store.init_schema(conn)
    finally:
        conn.close()


def test_claims_has_statement_normalized_not_null(db_path):
    """The approved schema correction: UNIQUE references this column."""
    conn = _connect(db_path)
    try:
        cols = conn.execute("PRAGMA table_info(claims)").fetchall()
        col = next(c for c in cols if c["name"] == "statement_normalized")
        assert col["notnull"] == 1
        # The UNIQUE constraint covers statement_normalized.
        indexes = conn.execute("PRAGMA index_list(claims)").fetchall()
        covered = False
        for idx in indexes:
            info = conn.execute(
                f"PRAGMA index_info({idx['name']!r})"
            ).fetchall()
            if any(r["name"] == "statement_normalized" for r in info):
                covered = True
        assert covered
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def test_create_session_defaults(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        s = store.create_session(conn, question="Second question?")
        assert s.session_id == "S-000002"
        assert s.status == "draft"
        assert s.min_supporting == 1
        assert s.max_iterations == 5
        assert s.final_answer is None
        assert s.created_at
    finally:
        conn.close()


def test_create_session_custom_limits(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        s = store.create_session(
            conn, question="Q?", min_supporting=2, max_iterations=9
        )
        assert s.min_supporting == 2
        assert s.max_iterations == 9
    finally:
        conn.close()


def test_create_session_empty_question_rejected(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        with pytest.raises(ValueError, match="empty"):
            store.create_session(conn, question="   ")
    finally:
        conn.close()


def test_get_and_list_sessions(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        s1 = store.get_session(conn, "S-000001")
        assert s1 is not None and s1.question == "Why scale by sqrt(d_k)?"
        assert store.get_session(conn, "S-999999") is None
        store.create_session(conn, question="Another?")
        assert len(store.list_sessions(conn)) == 2
    finally:
        conn.close()


def test_update_session_status(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        assert store.update_session_status(conn, "S-000001", "active")
        assert store.get_session(conn, "S-000001").status == "active"
        assert not store.update_session_status(conn, "S-999999", "active")
    finally:
        conn.close()


def test_set_session_answer(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        store.set_session_answer(conn, "S-000001", '{"statements": []}')
        s = store.get_session(conn, "S-000001")
        assert s.status == "complete"
        assert s.final_answer == '{"statements": []}'
    finally:
        conn.close()


def test_set_final_answer_preserves_status(seeded):
    """set_final_answer persists the payload WITHOUT touching session
    status — a complete_with_gaps session must stay complete_with_gaps
    (synthesis must never force 'complete')."""
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        store.update_session_status(conn, "S-000001", "complete_with_gaps")
        assert store.set_final_answer(conn, "S-000001", '{"statements": []}')
        s = store.get_session(conn, "S-000001")
        assert s.status == "complete_with_gaps"
        assert s.final_answer == '{"statements": []}'
        # Unknown session -> False, no row touched.
        assert not store.set_final_answer(conn, "S-999999", "{}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def test_add_query_and_idempotent(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        q1 = store.add_query(conn, session_id="S-000001", query_text="scaled dot product")
        q2 = store.add_query(conn, session_id="S-000001", query_text="scaled dot product")
        assert q1.id == q2.id
        assert q1.origin == "user" and q1.status == "pending"
        assert len(store.get_queries(conn, "S-000001")) == 1
    finally:
        conn.close()


def test_add_query_origin_and_iteration(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        q = store.add_query(
            conn, session_id="S-000001", query_text="softmax gradients",
            origin="derived", iteration=2,
        )
        assert q.origin == "derived"
        assert q.iteration == 2
    finally:
        conn.close()


def test_add_query_fk_enforced(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            store.add_query(conn, session_id="S-999999", query_text="x")
    finally:
        conn.close()


def test_set_query_status(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        q = store.add_query(conn, session_id="S-000001", query_text="servo")
        assert store.set_query_status(conn, q.id, "executed")
        assert store.get_queries(conn, "S-000001")[0].status == "executed"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

def test_create_claim_normalized(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        c = store.create_claim(
            conn, session_id="S-000001",
            statement="  The dot product   has variance d_k.  ",
        )
        assert c.claim_id == "C-000001"
        assert c.status == "proposed"
        assert c.statement == "The dot product   has variance d_k."
        assert c.statement_normalized == "the dot product has variance d_k."
        assert c.confidence is None
    finally:
        conn.close()


def test_claim_dedupe_on_normalized_statement(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        c1 = store.create_claim(
            conn, session_id="S-000001", statement="Variance is d_k."
        )
        c2 = store.create_claim(
            conn, session_id="S-000001", statement="  variance   is d_k. "
        )
        assert c1.claim_id == c2.claim_id
        assert len(store.list_claims(conn, "S-000001")) == 1
    finally:
        conn.close()


def test_claim_same_statement_different_session_allowed(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        c1 = store.create_claim(conn, session_id="S-000001", statement="Same claim")
        s2 = store.create_session(conn, question="Another session?")
        c2 = store.create_claim(conn, session_id=s2.session_id, statement="Same claim")
        assert c1.claim_id != c2.claim_id
    finally:
        conn.close()


def test_claim_empty_and_too_long_rejected(seeded, monkeypatch):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        with pytest.raises(ValueError, match="empty"):
            store.create_claim(conn, session_id="S-000001", statement="  ")
        with pytest.raises(ValueError, match="exceeds"):
            store.create_claim(conn, session_id="S-000001", statement="x" * 401)
    finally:
        conn.close()


def test_claim_fk_enforced(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            store.create_claim(conn, session_id="S-999999", statement="orphan")
    finally:
        conn.close()


def test_update_claim_status(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        c = store.create_claim(conn, session_id="S-000001", statement="Test claim")
        assert store.update_claim_status(conn, c.claim_id, "supported", notes="ok")
        updated = store.get_claim(conn, c.claim_id)
        assert updated.status == "supported"
        assert updated.notes == "ok"
        assert not store.update_claim_status(conn, "C-999999", "supported")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Claim-evidence links
# ---------------------------------------------------------------------------

def test_link_evidence_and_idempotent(seeded):
    db_path, _, _, evidence_id, _ = seeded
    conn = _connect(db_path)
    try:
        c = store.create_claim(conn, session_id="S-000001", statement="Claim A")
        link1 = store.link_evidence(
            conn, claim_id=c.claim_id, evidence_id=evidence_id, role="supporting"
        )
        link2 = store.link_evidence(
            conn, claim_id=c.claim_id, evidence_id=evidence_id, role="supporting"
        )
        assert link1 == link2
        assert len(store.get_links_for_claim(conn, c.claim_id)) == 1
    finally:
        conn.close()


def test_link_roles_supporting_and_contradicting(seeded):
    db_path, _, _, evidence_id, _ = seeded
    conn = _connect(db_path)
    try:
        c = store.create_claim(conn, session_id="S-000001", statement="Claim B")
        store.link_evidence(
            conn, claim_id=c.claim_id, evidence_id=evidence_id, role="supporting"
        )
        store.link_evidence(
            conn, claim_id=c.claim_id, evidence_id=evidence_id, role="contradicting"
        )
        links = store.get_links_for_claim(conn, c.claim_id)
        assert {l.role for l in links} == {"supporting", "contradicting"}
    finally:
        conn.close()


def test_link_fk_and_check_constraints(seeded):
    db_path, _, _, evidence_id, _ = seeded
    conn = _connect(db_path)
    try:
        c = store.create_claim(conn, session_id="S-000001", statement="Claim C")
        with pytest.raises(sqlite3.IntegrityError):
            store.link_evidence(
                conn, claim_id=c.claim_id, evidence_id="E-999999", role="supporting"
            )
        with pytest.raises(sqlite3.IntegrityError):
            store.link_evidence(
                conn, claim_id="C-999999", evidence_id=evidence_id, role="supporting"
            )
        with pytest.raises(sqlite3.IntegrityError):
            store.link_evidence(
                conn, claim_id=c.claim_id, evidence_id=evidence_id, role="ambiguous"
            )
    finally:
        conn.close()


def test_get_links_for_evidence(seeded):
    db_path, _, _, evidence_id, _ = seeded
    conn = _connect(db_path)
    try:
        c1 = store.create_claim(conn, session_id="S-000001", statement="First")
        c2 = store.create_claim(conn, session_id="S-000001", statement="Second")
        store.link_evidence(conn, claim_id=c1.claim_id, evidence_id=evidence_id,
                            role="supporting")
        store.link_evidence(conn, claim_id=c2.claim_id, evidence_id=evidence_id,
                            role="supporting")
        links = store.get_links_for_evidence(conn, evidence_id)
        assert {l.claim_id for l in links} == {c1.claim_id, c2.claim_id}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Iterations & events
# ---------------------------------------------------------------------------

def test_ensure_iteration_idempotent(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        i1 = store.ensure_iteration(
            conn, session_id="S-000001", iteration=1, phase="planning"
        )
        i2 = store.ensure_iteration(
            conn, session_id="S-000001", iteration=1, phase="planning"
        )
        assert i1.id == i2.id
        assert len(store.list_iterations(conn, "S-000001")) == 1
    finally:
        conn.close()


def test_complete_iteration(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        it = store.ensure_iteration(
            conn, session_id="S-000001", iteration=1, phase="retrieving"
        )
        assert store.complete_iteration(conn, it.id, notes="3 chunks")
        done = store.list_iterations(conn, "S-000001")[0]
        assert done.status == "completed"
        assert done.completed_at is not None
        assert done.notes == "3 chunks"
    finally:
        conn.close()


def test_events_append_and_list(seeded):
    db_path, *_ = seeded
    conn = _connect(db_path)
    try:
        e1 = store.add_event(conn, session_id="S-000001",
                             event_type="session_created")
        e2 = store.add_event(conn, session_id="S-000001",
                             event_type="claim_proposed",
                             detail='{"claim_id": "C-000001"}')
        events = store.list_events(conn, "S-000001")
        assert [e.id for e in events] == [e1.id, e2.id]
        assert events[1].detail == '{"claim_id": "C-000001"}'
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Persistence & integrity
# ---------------------------------------------------------------------------

def test_restart_persistence(seeded):
    """Close and reopen: everything survives; ID sequences continue."""
    db_path, _, _, evidence_id, session_id = seeded
    conn1 = _connect(db_path)
    c = store.create_claim(conn1, session_id=session_id, statement="Persist me")
    store.link_evidence(conn1, claim_id=c.claim_id, evidence_id=evidence_id,
                        role="supporting")
    store.add_event(conn1, session_id=session_id, event_type="iteration_completed")
    conn1.close()

    conn2 = _connect(db_path)  # "restart"
    try:
        assert store.get_session(conn2, session_id) is not None
        assert store.get_claim(conn2, c.claim_id).statement == "Persist me"
        assert len(store.get_links_for_claim(conn2, c.claim_id)) == 1
        assert len(store.list_events(conn2, session_id)) == 1
        s2 = store.create_session(conn2, question="After restart?")
        assert s2.session_id == "S-000002"  # sequence continues
        c2 = store.create_claim(conn2, session_id=session_id, statement="New")
        assert c2.claim_id == "C-000002"
    finally:
        conn2.close()


def test_session_delete_cascades_children(seeded):
    db_path, _, _, evidence_id, session_id = seeded
    conn = _connect(db_path)
    try:
        c = store.create_claim(conn, session_id=session_id, statement="Cascade")
        store.link_evidence(conn, claim_id=c.claim_id, evidence_id=evidence_id,
                            role="supporting")
        store.add_query(conn, session_id=session_id, query_text="q")
        store.ensure_iteration(conn, session_id=session_id, iteration=1, phase="done")
        store.add_event(conn, session_id=session_id, event_type="x")

        conn.execute("DELETE FROM research_sessions WHERE id = ?", (session_id,))
        for table, where in (
            ("research_queries", "session_id = ?"),
            ("claims", "session_id = ?"),
            ("claim_evidence", "claim_id IN (SELECT id FROM claims WHERE session_id = ?)"),
            ("research_iterations", "session_id = ?"),
            ("research_events", "session_id = ?"),
        ):
            n = conn.execute(
                f"SELECT COUNT(*) AS n FROM {table} WHERE {where}",
                (session_id,),
            ).fetchone()["n"]
            assert n == 0, table
        # Evidence is global: must still exist after session deletion.
        assert c1store.get_evidence(conn, evidence_id) is not None
    finally:
        conn.close()


def test_linked_evidence_delete_blocked(seeded):
    """Deleting evidence that is linked to a claim fails loudly."""
    db_path, _, _, evidence_id, _ = seeded
    conn = _connect(db_path)
    try:
        c = store.create_claim(conn, session_id="S-000001", statement="Linked")
        store.link_evidence(conn, claim_id=c.claim_id, evidence_id=evidence_id,
                            role="supporting")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM evidence WHERE id = ?", (evidence_id,))
    finally:
        conn.close()
