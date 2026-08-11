"""C3 hardening tests — part 2: context integrity, evidence tamper /
contradiction consistency, verifier/domain boundary, migration v3."""

from __future__ import annotations

import json
import sqlite3

import pytest

import docpipe.db as c0db
import docpipe.ingest as c0ingest
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, store, verifier
from research.reasoner import (
    ClaimContextProposal,
    SemanticContradictionProposal,
)

from tests.docpipe.pdf_factory import build_pdf

CHUNK_TEXTS = [
    "Scaled dot product attention divides by the square root of dk.",
    "Multi head attention runs several attention layers in parallel.",
]


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "docpipe.db"
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(build_pdf(CHUNK_TEXTS))
    ingested = c0ingest.ingest_pdf(pdf, db_path=db_path)
    conn = store.connect(db_path)
    session = store.create_session(conn, question="attention")
    store.update_session_status(conn, session.session_id, "active")
    store.add_session_document(
        conn, session_id=session.session_id, document_id=ingested.document_id
    )
    conn.close()
    from research.loop import ResearchLoop

    ResearchLoop(db_path=db_path).start(session.session_id)
    conn = store.connect(db_path)
    supported = [c for c in store.list_claims(conn, session.session_id)
                 if c.status == "supported"]
    ev_by_claim = {
        c.claim_id: [
            l.evidence_id for l in store.get_links_for_claim(conn, c.claim_id)
            if l.role == "supporting"
        ]
        for c in supported
    }
    conn.close()
    return db_path, session.session_id, supported, ev_by_claim


def _semantic_proposal(env, rationale="observed"):
    db_path, _, supported, ev_by_claim = env
    claim_a, claim_b = supported[0], supported[1]
    return SemanticContradictionProposal(
        claim_a_id=claim_a.claim_id, claim_b_id=claim_b.claim_id,
        evidence_a_ids=(ev_by_claim[claim_a.claim_id][0],),
        evidence_b_ids=(ev_by_claim[claim_b.claim_id][0],),
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# P6 — context integrity
# ---------------------------------------------------------------------------

def test_context_merge_preserves_unrelated_keys(env):
    db_path, _, supported, _ = env
    claim_id = supported[0].claim_id
    conn = store.connect(db_path)
    try:
        verifier.apply_context(
            conn, ClaimContextProposal(claim_id, {"dataset": "WMT14"})
        )
        verifier.apply_context(
            conn, ClaimContextProposal(claim_id, {"task": "translation"})
        )
        claim = store.get_claim(conn, claim_id)
        assert json.loads(claim.context) == {
            "dataset": "WMT14", "task": "translation",
        }
    finally:
        conn.close()


def test_context_repeated_application_idempotent(env):
    db_path, _, supported, _ = env
    claim_id = supported[0].claim_id
    conn = store.connect(db_path)
    try:
        proposal = ClaimContextProposal(claim_id, {"dataset": "WMT14"})
        verifier.apply_context(conn, proposal)
        verifier.apply_context(conn, proposal)
        claim = store.get_claim(conn, claim_id)
        assert json.loads(claim.context) == {"dataset": "WMT14"}
    finally:
        conn.close()


def test_context_update_overwrites_same_key(env):
    db_path, _, supported, _ = env
    claim_id = supported[0].claim_id
    conn = store.connect(db_path)
    try:
        verifier.apply_context(conn, ClaimContextProposal(claim_id, {"task": "a"}))
        verifier.apply_context(conn, ClaimContextProposal(claim_id, {"task": "b"}))
        claim = store.get_claim(conn, claim_id)
        assert json.loads(claim.context) == {"task": "b"}
    finally:
        conn.close()


def test_context_null_value_distinct_from_absent(env):
    db_path, _, supported, _ = env
    claim_id = supported[0].claim_id
    conn = store.connect(db_path)
    try:
        verifier.apply_context(
            conn, ClaimContextProposal(claim_id, {"population": None})
        )
        claim = store.get_claim(conn, claim_id)
        assert json.loads(claim.context) == {"population": None}
        assert "population" in json.loads(claim.context)  # explicit, not absent
    finally:
        conn.close()


def test_context_malformed_still_rejected(env):
    db_path, _, supported, _ = env
    claim_id = supported[0].claim_id
    conn = store.connect(db_path)
    try:
        with pytest.raises(verifier.ProposalRejectedError):
            verifier.apply_context(
                conn, ClaimContextProposal(claim_id, {"bogus_key": "x"})
            )
        with pytest.raises(verifier.ProposalRejectedError):
            verifier.apply_context(
                conn, ClaimContextProposal(claim_id, {"task": 42})
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# P5 — evidence tamper / contradiction consistency
# ---------------------------------------------------------------------------

def test_tamper_synthesis_and_observation_consistency(env):
    """Tampering support evidence: synthesis excludes it, the semantic
    OBSERVATION stays auditable, and repeated application does not
    duplicate records. Restoring evidence is deterministic."""
    db_path, session_id, supported, ev_by_claim = env
    claim_a = supported[0]
    ev_a = ev_by_claim[claim_a.claim_id][0]
    conn = store.connect(db_path)
    try:
        # 1. valid evidence -> observation exists.
        record = verifier.apply_semantic_contradiction(
            conn, _semantic_proposal(env)
        )
        assert record.id > 0

        # 2. tamper the supporting chunk -> evidence invalid.
        conn.execute(
            "UPDATE chunks SET content = 'TAMPERED' WHERE id = "
            "(SELECT chunk_id FROM evidence WHERE id = ?)", (ev_a,)
        )
        conn.commit()

        # 3. synthesis excludes the invalid support.
        claims.evaluate_claim(conn, claim_a.claim_id)
        assert store.get_claim(conn, claim_a.claim_id).status == "insufficient"

        # 4. the observation remains auditable (never auto-invalidated).
        session = store.get_session(conn, session_id)
        open_records = store.list_contradictions(conn, session_id, status="open")
        assert any(r.id == record.id and r.kind == "semantic"
                   for r in open_records)

        # 5. re-application while evidence is INVALID is fail-closed: the
        #    verifier rejects the proposal and no duplicate record appears.
        with pytest.raises(verifier.ProposalRejectedError):
            verifier.apply_semantic_contradiction(
                conn, _semantic_proposal(env)
            )
        semantic = [r for r in store.list_contradictions(conn, session_id)
                    if r.kind == "semantic"]
        assert len(semantic) == 1

        # 6. restore the chunk -> deterministic recovery.
        original = c1store.get_evidence(conn, ev_a).content
        conn.execute(
            "UPDATE chunks SET content = ? WHERE id = "
            "(SELECT chunk_id FROM evidence WHERE id = ?)",
            (original, ev_a),
        )
        conn.commit()
        claims.evaluate_claim(conn, claim_a.claim_id)
        assert store.get_claim(conn, claim_a.claim_id).status == "supported"

        # 7. re-application after restore converges to the SAME record.
        again = verifier.apply_semantic_contradiction(
            conn, _semantic_proposal(env)
        )
        assert again.id == record.id
        semantic = [r for r in store.list_contradictions(conn, session_id)
                    if r.kind == "semantic"]
        assert len(semantic) == 1
    finally:
        conn.close()


def test_tamper_does_not_recreate_observation(env):
    """Restoring evidence must not create a SECOND observation record."""
    db_path, session_id, supported, ev_by_claim = env
    claim_a = supported[0]
    ev_a = ev_by_claim[claim_a.claim_id][0]
    conn = store.connect(db_path)
    try:
        record = verifier.apply_semantic_contradiction(
            conn, _semantic_proposal(env)
        )
        original = c1store.get_evidence(conn, ev_a).content

        conn.execute(
            "UPDATE chunks SET content = 'TAMPERED' WHERE id = "
            "(SELECT chunk_id FROM evidence WHERE id = ?)", (ev_a,)
        )
        conn.commit()
        conn.execute(
            "UPDATE chunks SET content = ? WHERE id = "
            "(SELECT chunk_id FROM evidence WHERE id = ?)",
            (original, ev_a),
        )
        conn.commit()

        # Re-applying after restore still converges to the SAME record.
        again = verifier.apply_semantic_contradiction(
            conn, _semantic_proposal(env)
        )
        assert again.id == record.id
        semantic = [r for r in store.list_contradictions(conn, session_id)
                    if r.kind == "semantic"]
        assert len(semantic) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# P7 — verifier/domain boundary
# ---------------------------------------------------------------------------

def test_verifier_never_calls_raw_status_writers(env):
    """The verifier persists through domain functions; it must never call
    the raw status writers."""
    import ast
    from pathlib import Path

    source = Path(
        r"C:\Users\ACER\AppData\Local\hermes\hermes-agent\research\verifier.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for writer in ("update_claim_status", "update_session_status",
                   "set_query_status", "set_session_answer"):
        assert writer not in calls


def test_reasoner_still_proposal_only(env):
    """The hardening must not give the reasoner DB access."""
    import ast
    from pathlib import Path

    source = Path(
        r"C:\Users\ACER\AppData\Local\hermes\hermes-agent\research\reasoner.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "store" in node.module:
            raise AssertionError(f"reasoner imports {node.module}")


# ---------------------------------------------------------------------------
# P8 — migration v3
# ---------------------------------------------------------------------------

def test_fresh_db_v3_has_observation_key(env):
    db_path, *_ = env
    conn = store.connect(db_path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION == 3
        cols = {r["name"] for r in conn.execute(
            "PRAGMA table_info(claim_contradictions)")}
        assert "observation_key" in cols
    finally:
        conn.close()


def test_v2_db_upgrades_to_v3_preserving_data(tmp_path):
    """A v2 database upgrades to v3; existing data and NULL keys survive."""
    db_path = tmp_path / "v2.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(store._MIGRATIONS[1])
    conn.execute("PRAGMA user_version = 1")
    conn.executescript(
        "INSERT INTO research_sessions (id, question, status) "
        "VALUES ('S-000001', 'q', 'draft');"
    )
    conn.commit()
    conn.close()
    # Apply v2 then set version to 2 (simulate a v2-era DB).
    conn = sqlite3.connect(str(db_path))
    store._exec_sql_statements(conn, "ALTER TABLE claims ADD COLUMN parent_claim_id TEXT;"
                                     "ALTER TABLE claims ADD COLUMN context TEXT;")
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()

    conn = store.connect(db_path)  # upgrades 2 -> 3
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
        assert store.get_session(conn, "S-000001") is not None
        cols = {r["name"] for r in conn.execute(
            "PRAGMA table_info(claim_contradictions)")}
        assert "observation_key" in cols
    finally:
        conn.close()


def test_repeated_migration_v3_idempotent(env):
    db_path, *_ = env
    conn = store.connect(db_path)
    conn.close()
    conn = store.connect(db_path)  # second connect: no-op
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
    finally:
        conn.close()


def test_migration_failure_rolls_back_version(env, monkeypatch):
    """A failing migration must not bump user_version."""
    db_path, *_ = env
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()

    def broken(conn):
        raise sqlite3.OperationalError("boom during migration 3")

    # The migration table holds the function object; patch the dict entry
    # (patching store._migrate_to_v3 would not affect _MIGRATIONS).
    monkeypatch.setitem(store._MIGRATIONS, 3, broken)
    with pytest.raises(sqlite3.OperationalError):
        store.connect(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        conn.close()


def test_future_version_still_rejected(env):
    db_path, *_ = env
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(store.SchemaVersionError):
        store.connect(db_path)
