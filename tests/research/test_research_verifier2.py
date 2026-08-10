"""C3 verifier tests: semantic contradiction, coverage, synthesis safety,
API/CLI inspection, schema migration v2."""

from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import docpipe.db as c0db
import docpipe.ingest as c0ingest
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, cli as research_cli, store, verifier
from research.api import create_app
from research.loop import ResearchLoop
from research.reasoner import (
    SemanticContradictionProposal,
    CoverageProposal,
)

from tests.docpipe.pdf_factory import build_pdf

CHUNK_TEXTS = [
    "Scaled dot product attention divides by the square root of dk.",
    "Multi head attention runs several attention layers in parallel.",
    "The model has no recurrence and no convolution.",
]


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "docpipe.db"
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(build_pdf(CHUNK_TEXTS))
    ingested = c0ingest.ingest_pdf(pdf, db_path=db_path)
    conn = store.connect(db_path)
    # "attention" matches BOTH chunks -> two supported claims after the loop.
    session = store.create_session(conn, question="attention")
    store.update_session_status(conn, session.session_id, "active")
    store.add_session_document(
        conn, session_id=session.session_id, document_id=ingested.document_id
    )
    conn.close()
    return db_path, session.session_id


def _two_supported_claims(env):
    """Run the loop to completion -> two supported claims."""
    db_path, session_id = env
    loop = ResearchLoop(db_path=db_path)
    loop.start(session_id)
    conn = store.connect(db_path)
    claims_list = store.list_claims(conn, session_id)
    supported = [c for c in claims_list if c.status == "supported"]
    conn.close()
    return supported


def _claim_evidence(db_path, claim_id):
    conn = store.connect(db_path)
    ids = [
        l.evidence_id
        for l in store.get_links_for_claim(conn, claim_id)
        if l.role == "supporting"
    ]
    conn.close()
    return ids


# ---------------------------------------------------------------------------
# C. Semantic contradiction
# ---------------------------------------------------------------------------

def test_semantic_contradiction_valid_accepted(env):
    db_path, _ = env
    claims_list = _two_supported_claims(env)
    assert len(claims_list) >= 2
    claim_a, claim_b = claims_list[0], claims_list[1]
    ev_a = _claim_evidence(db_path, claim_a.claim_id)
    ev_b = _claim_evidence(db_path, claim_b.claim_id)

    conn = store.connect(db_path)
    try:
        proposal = SemanticContradictionProposal(
            claim_a_id=claim_a.claim_id, claim_b_id=claim_b.claim_id,
            evidence_a_ids=(ev_a[0],), evidence_b_ids=(ev_b[0],),
            rationale="semantic conflict observed",
        )
        verdict = verifier.validate_semantic_contradiction(conn, proposal)
        assert verdict.accepted, verdict.reason
        record = verifier.apply_semantic_contradiction(conn, proposal)
        assert record.kind == "semantic"
        assert record.status == "open"
        detail = json.loads(record.detail)
        assert detail["other_claim_id"] == claim_b.claim_id
    finally:
        conn.close()


def test_semantic_contradiction_evidence_must_belong_to_claim(env):
    db_path, _ = env
    claims_list = _two_supported_claims(env)
    claim_a, claim_b = claims_list[0], claims_list[1]
    ev_a = _claim_evidence(db_path, claim_a.claim_id)

    conn = store.connect(db_path)
    try:
        # evidence from claim A proposed as supporting claim B: rejected.
        proposal = SemanticContradictionProposal(
            claim_a_id=claim_a.claim_id, claim_b_id=claim_b.claim_id,
            evidence_a_ids=(ev_a[0],), evidence_b_ids=(ev_a[0],),
            rationale="wrong-side evidence",
        )
        verdict = verifier.validate_semantic_contradiction(conn, proposal)
        assert not verdict.accepted
        assert "not linked" in verdict.reason
    finally:
        conn.close()


def test_semantic_contradiction_does_not_change_statuses(env):
    db_path, _ = env
    claims_list = _two_supported_claims(env)
    claim_a, claim_b = claims_list[0], claims_list[1]
    ev_a = _claim_evidence(db_path, claim_a.claim_id)
    ev_b = _claim_evidence(db_path, claim_b.claim_id)

    conn = store.connect(db_path)
    try:
        verifier.apply_semantic_contradiction(
            conn,
            SemanticContradictionProposal(
                claim_a_id=claim_a.claim_id, claim_b_id=claim_b.claim_id,
                evidence_a_ids=(ev_a[0],), evidence_b_ids=(ev_b[0],),
                rationale="observation only",
            ),
        )
        # Neither claim is invalidated; both stay supported.
        assert store.get_claim(conn, claim_a.claim_id).status == "supported"
        assert store.get_claim(conn, claim_b.claim_id).status == "supported"
    finally:
        conn.close()


def test_semantic_contradiction_observations_distinguishable(env):
    db_path, _ = env
    claims_list = _two_supported_claims(env)
    claim_a, claim_b = claims_list[0], claims_list[1]
    ev_a = _claim_evidence(db_path, claim_a.claim_id)
    ev_b = _claim_evidence(db_path, claim_b.claim_id)

    conn = store.connect(db_path)
    try:
        record1 = verifier.apply_semantic_contradiction(
            conn,
            SemanticContradictionProposal(
                claim_a_id=claim_a.claim_id, claim_b_id=claim_b.claim_id,
                evidence_a_ids=(ev_a[0],), evidence_b_ids=(ev_b[0],),
                rationale="first observation",
            ),
        )
        # Idempotent: the same observation returns the same record.
        record2 = verifier.apply_semantic_contradiction(
            conn,
            SemanticContradictionProposal(
                claim_a_id=claim_a.claim_id, claim_b_id=claim_b.claim_id,
                evidence_a_ids=(ev_a[0],), evidence_b_ids=(ev_b[0],),
                rationale="first observation",
            ),
        )
        assert record1.id == record2.id
    finally:
        conn.close()


def test_semantic_contradiction_fabricated_evidence_rejected(env):
    db_path, _ = env
    claims_list = _two_supported_claims(env)
    claim_a, claim_b = claims_list[0], claims_list[1]
    ev_b = _claim_evidence(db_path, claim_b.claim_id)

    conn = store.connect(db_path)
    try:
        proposal = SemanticContradictionProposal(
            claim_a_id=claim_a.claim_id, claim_b_id=claim_b.claim_id,
            evidence_a_ids=("E-999999",), evidence_b_ids=(ev_b[0],),
            rationale="fabricated",
        )
        verdict = verifier.validate_semantic_contradiction(conn, proposal)
        assert not verdict.accepted
        assert "fabricated" in verdict.reason
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# E. Coverage (derived, deterministic)
# ---------------------------------------------------------------------------

def test_coverage_directly_supported(env):
    db_path, _ = env
    claim = _two_supported_claims(env)[0]
    conn = store.connect(db_path)
    try:
        obs = verifier.coverage_for_claim(conn, claim.claim_id)
        assert obs.coverage == "directly_supported"
    finally:
        conn.close()


def test_coverage_unsupported_when_no_evidence(env):
    db_path, session_id = env
    conn = store.connect(db_path)
    claim = claims.create_claim(conn, session_id=session_id,
                                statement="A claim with no evidence at all.")
    claims.evaluate_claim(conn, claim.claim_id)
    obs = verifier.coverage_for_claim(conn, claim.claim_id)
    assert obs.coverage == "unsupported"
    conn.close()


def test_coverage_contradictory(env):
    db_path, _ = env
    claims_list = _two_supported_claims(env)
    target = claims_list[0]
    conn = store.connect(db_path)
    try:
        # Add a valid contradicting link from the other claim's evidence.
        other_ev = _claim_evidence(db_path, claims_list[1].claim_id)[0]
        claims.link_evidence(conn, claim_id=target.claim_id,
                             evidence_id=other_ev, role="contradicting")
        claims.evaluate_claim(conn, target.claim_id)
        obs = verifier.coverage_for_claim(conn, target.claim_id)
        assert obs.coverage == "contradictory"
    finally:
        conn.close()


def test_coverage_no_evidence_is_not_research_gap(env):
    """'unsupported' is a retrieval observation, never a research-gap
    claim — the label set has no research-gap concept."""
    from research.reasoner import COVERAGE_LABELS

    assert "research_gap" not in COVERAGE_LABELS
    db_path, session_id = env
    conn = store.connect(db_path)
    claim = claims.create_claim(conn, session_id=session_id,
                                statement="No evidence anywhere.")
    obs = verifier.coverage_for_claim(conn, claim.claim_id)
    assert obs.coverage == "unsupported"
    conn.close()


def test_coverage_proposal_contradicting_derived_rejected(env):
    db_path, _ = env
    claim = _two_supported_claims(env)[0]
    conn = store.connect(db_path)
    try:
        proposal = CoverageProposal(claim.claim_id, "unsupported")
        verdict = verifier.validate_coverage(conn, proposal)
        assert not verdict.accepted
        assert "contradicts derived" in verdict.reason
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# F. Synthesis safety
# ---------------------------------------------------------------------------

def test_synthesis_keeps_both_sides_of_semantic_observation(env):
    db_path, _ = env
    claims_list = _two_supported_claims(env)
    claim_a, claim_b = claims_list[0], claims_list[1]
    ev_a = _claim_evidence(db_path, claim_a.claim_id)
    ev_b = _claim_evidence(db_path, claim_b.claim_id)

    conn = store.connect(db_path)
    try:
        verifier.apply_semantic_contradiction(
            conn,
            SemanticContradictionProposal(
                claim_a_id=claim_a.claim_id, claim_b_id=claim_b.claim_id,
                evidence_a_ids=(ev_a[0],), evidence_b_ids=(ev_b[0],),
                rationale="observed",
            ),
        )
        from research.synthesis import synthesize

        answer = synthesize(conn, store.get_claim(conn, claim_a.claim_id).session_id)
        statement_ids = {s.claim_id for s in answer.statements}
        assert claim_a.claim_id in statement_ids  # neither side erased
        assert claim_b.claim_id in statement_ids
        for stmt in answer.statements:
            for cit in stmt.citations:
                assert cit.evidence_id.startswith("E-")  # persisted IDs only
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API / CLI inspection (read-only)
# ---------------------------------------------------------------------------

@pytest.fixture
def client(env):
    db_path, _ = env
    app = create_app(db_path=db_path)
    with TestClient(app) as c:
        yield c


def test_api_coverage_endpoint(env, client):
    claim = _two_supported_claims(env)[0]
    resp = client.get(f"/research/claims/{claim.claim_id}/coverage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["claim_id"] == claim.claim_id
    assert body["coverage"] == "directly_supported"


def test_api_contradictions_endpoint(env, client):
    db_path, _ = env
    claims_list = _two_supported_claims(env)
    claim_a, claim_b = claims_list[0], claims_list[1]
    ev_a = _claim_evidence(db_path, claim_a.claim_id)
    ev_b = _claim_evidence(db_path, claim_b.claim_id)
    conn = store.connect(db_path)
    verifier.apply_semantic_contradiction(
        conn,
        SemanticContradictionProposal(
            claim_a_id=claim_a.claim_id, claim_b_id=claim_b.claim_id,
            evidence_a_ids=(ev_a[0],), evidence_b_ids=(ev_b[0],),
            rationale="observed",
        ),
    )
    conn.close()

    resp = client.get(f"/research/claims/{claim_a.claim_id}/contradictions")
    assert resp.status_code == 200
    observations = resp.json()["observations"]
    assert any(o["kind"] == "semantic" for o in observations)


def test_cli_coverage_command(env, capsys):
    import argparse

    db_path, _ = env
    claim = _two_supported_claims(env)[0]
    args = argparse.Namespace(json=True, db_path=str(db_path),
                              claim_id=claim.claim_id)
    code = research_cli.cmd_coverage(args)
    assert code == 0
    out = capsys.readouterr().out
    body = json.loads(out)
    assert body["coverage"] == "directly_supported"


# ---------------------------------------------------------------------------
# G. Migration v2 (fresh + upgrade + idempotent)
# ---------------------------------------------------------------------------

def test_fresh_db_has_v2_columns(env):
    db_path, _ = env
    conn = store.connect(db_path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION == 2
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(claims)")}
        assert "parent_claim_id" in cols and "context" in cols
    finally:
        conn.close()


def test_upgrade_v1_to_v2_preserves_data(tmp_path):
    db_path = tmp_path / "v1.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(store._MIGRATIONS[1])
    conn.execute("PRAGMA user_version = 1")
    conn.execute(
        "INSERT INTO research_sessions (id, question, status) "
        "VALUES ('S-000001', 'v1 question', 'draft')"
    )
    conn.commit()
    conn.close()

    conn = store.connect(db_path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert store.get_session(conn, "S-000001") is not None
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(claims)")}
        assert "parent_claim_id" in cols and "context" in cols
        # The migration is idempotent on reconnect.
        conn.close()
        conn = store.connect(db_path)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        conn.close()
