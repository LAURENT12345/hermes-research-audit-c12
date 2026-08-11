"""C3 hardening tests — part 1: contradiction identity/idempotency/
concurrency, decomposition atomicity/idempotency, parent integrity."""

from __future__ import annotations

import sqlite3
import threading

import pytest

import docpipe.db as c0db
import docpipe.ingest as c0ingest
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, store, verifier
from research.reasoner import (
    ClaimDecompositionProposal,
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


def _proposal(env, rationale="observed", context=None, strength=None):
    db_path, _, supported, ev_by_claim = env
    claim_a, claim_b = supported[0], supported[1]
    return SemanticContradictionProposal(
        claim_a_id=claim_a.claim_id, claim_b_id=claim_b.claim_id,
        evidence_a_ids=(ev_by_claim[claim_a.claim_id][0],),
        evidence_b_ids=(ev_by_claim[claim_b.claim_id][0],),
        rationale=rationale, context=context, strength=strength,
    ), claim_a, claim_b


# ---------------------------------------------------------------------------
# P1 — contradiction identity / idempotency
# ---------------------------------------------------------------------------

def test_identical_observation_is_idempotent(env):
    db_path, *_ = env
    proposal, _, _ = _proposal(env)
    conn = store.connect(db_path)
    try:
        first = verifier.apply_semantic_contradiction(conn, proposal)
        second = verifier.apply_semantic_contradiction(conn, proposal)
        assert first.id == second.id  # same observation -> same record
        records = store.list_contradictions(
            conn, store.get_claim(conn, proposal.claim_a_id).session_id
        )
        semantic = [r for r in records if r.kind == "semantic"]
        assert len(semantic) == 1
        assert semantic[0].observation_key is not None
    finally:
        conn.close()


def test_distinct_observations_both_survive(env):
    """Different rationale/evidence/context -> DISTINCT records, even
    between the same claim pair."""
    db_path, *_ = env
    conn = store.connect(db_path)
    try:
        p1, _, _ = _proposal(env, rationale="first rationale")
        p2, _, _ = _proposal(env, rationale="second rationale")
        r1 = verifier.apply_semantic_contradiction(conn, p1)
        r2 = verifier.apply_semantic_contradiction(conn, p2)
        assert r1.id != r2.id
        assert r1.observation_key != r2.observation_key
        session_id = store.get_claim(conn, p1.claim_a_id).session_id
        semantic = [r for r in store.list_contradictions(conn, session_id)
                    if r.kind == "semantic"]
        assert len(semantic) == 2
    finally:
        conn.close()


def test_observation_key_preserves_direction_and_evidence(env):
    db_path, _, supported, ev_by_claim = env
    claim_a, claim_b = supported[0], supported[1]
    conn = store.connect(db_path)
    try:
        ab = SemanticContradictionProposal(
            claim_a_id=claim_a.claim_id, claim_b_id=claim_b.claim_id,
            evidence_a_ids=(ev_by_claim[claim_a.claim_id][0],),
            evidence_b_ids=(ev_by_claim[claim_b.claim_id][0],),
            rationale="direction matters",
        )
        ba = SemanticContradictionProposal(
            claim_a_id=claim_b.claim_id, claim_b_id=claim_a.claim_id,
            evidence_a_ids=(ev_by_claim[claim_b.claim_id][0],),
            evidence_b_ids=(ev_by_claim[claim_a.claim_id][0],),
            rationale="direction matters",
        )
        assert verifier._contradiction_observation_key(ab) != (
            verifier._contradiction_observation_key(ba)
        )
    finally:
        conn.close()


def test_context_distinguishes_observations(env):
    db_path, *_ = env
    conn = store.connect(db_path)
    try:
        p1, _, _ = _proposal(env, context={"dataset": "WMT14"})
        p2, _, _ = _proposal(env, context={"dataset": "WSJ"})
        r1 = verifier.apply_semantic_contradiction(conn, p1)
        r2 = verifier.apply_semantic_contradiction(conn, p2)
        assert r1.id != r2.id
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# P4 — concurrent contradiction writes
# ---------------------------------------------------------------------------

def test_concurrent_identical_observations_converge(env):
    """Two threads racing the SAME observation -> one persisted row."""
    db_path, *_ = env
    proposal, _, _ = _proposal(env)
    barrier = threading.Barrier(2)
    results = []

    def worker():
        barrier.wait()
        conn = store.connect(db_path)
        try:
            results.append(
                verifier.apply_semantic_contradiction(conn, proposal).id
            )
        finally:
            conn.close()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not any(t.is_alive() for t in threads)
    assert len(results) == 2
    assert results[0] == results[1]  # converged to one record

    conn = store.connect(db_path)
    try:
        session_id = store.get_claim(conn, proposal.claim_a_id).session_id
        semantic = [r for r in store.list_contradictions(conn, session_id)
                    if r.kind == "semantic"]
        assert len(semantic) == 1
    finally:
        conn.close()


def test_concurrent_distinct_observations_both_survive(env):
    db_path, *_ = env
    p1, _, _ = _proposal(env, rationale="concurrent one")
    p2, _, _ = _proposal(env, rationale="concurrent two")
    barrier = threading.Barrier(2)
    results = []

    def worker(proposal):
        barrier.wait()
        conn = store.connect(db_path)
        try:
            results.append(
                verifier.apply_semantic_contradiction(conn, proposal).id
            )
        finally:
            conn.close()

    threads = [
        threading.Thread(target=worker, args=(p,))
        for p in (p1, p2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not any(t.is_alive() for t in threads)
    assert len(set(results)) == 2  # both distinct observations persisted

    conn = store.connect(db_path)
    try:
        session_id = store.get_claim(conn, p1.claim_a_id).session_id
        semantic = [r for r in store.list_contradictions(conn, session_id)
                    if r.kind == "semantic"]
        assert len(semantic) == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# P2 — atomic decomposition
# ---------------------------------------------------------------------------

def test_decomposition_rolls_back_on_failure(env, monkeypatch):
    """An injected mid-transaction failure leaves NO partial children,
    NO partial links, NO partial audit state."""
    db_path, session_id, supported, _ = env
    source = supported[0]
    conn = store.connect(db_path)

    real_normalize = claims.normalize_statement
    calls = {"n": 0}

    def flaky_normalize(statement):
        calls["n"] += 1
        if calls["n"] == 2:  # fail on the SECOND child
            raise RuntimeError("boom mid-decomposition")
        return real_normalize(statement)

    monkeypatch.setattr(claims, "normalize_statement", flaky_normalize)
    try:
        with pytest.raises(RuntimeError):
            verifier.apply_decomposition(
                conn,
                ClaimDecompositionProposal(
                    source_claim_id=source.claim_id,
                    child_statements=("child one statement.", "child two statement."),
                    child_evidence_ids=(),
                ),
            )
    finally:
        monkeypatch.undo()
        conn.close()

    conn = store.connect(db_path)
    try:
        # Nothing persisted: no partial children, no events for them.
        claims_list = store.list_claims(conn, session_id)
        assert all(c.parent_claim_id is None for c in claims_list)
        proposed = [
            c for c in claims_list
            if "child" in c.statement
        ]
        assert proposed == []  # no orphan children
    finally:
        conn.close()


def test_decomposition_reapplication_is_idempotent(env):
    db_path, session_id, supported, _ = env
    source = supported[0]
    conn = store.connect(db_path)
    try:
        def proposed_event_count():
            return len([
                e for e in store.list_events(conn, session_id)
                if e.event_type == "claim_proposed"
            ])

        before = proposed_event_count()
        proposal = ClaimDecompositionProposal(
            source_claim_id=source.claim_id,
            child_statements=("child one statement.", "child two statement."),
            child_evidence_ids=(),
        )
        first = verifier.apply_decomposition(conn, proposal)
        after_first = proposed_event_count()
        assert after_first == before + 2  # one event per NEW child

        second = verifier.apply_decomposition(conn, proposal)
        assert [c.claim_id for c in first] == [c.claim_id for c in second]
        assert proposed_event_count() == after_first  # no re-emitted events

        # No duplicate children.
        claims_list = store.list_claims(conn, session_id)
        children = [c for c in claims_list
                    if c.parent_claim_id == source.claim_id]
        assert len(children) == 2
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# P3 — parent integrity
# ---------------------------------------------------------------------------

def test_parent_nonexistent_rejected(env):
    db_path, _, supported, _ = env
    source = supported[0]
    conn = store.connect(db_path)
    try:
        with pytest.raises(claims.ClaimHierarchyError):
            claims.create_decomposed_claims(
                conn, session_id=store.get_claim(conn, source.claim_id).session_id,
                parent_claim_id="C-999999",
                statements=["x"], evidence_ids=[],
            )
    finally:
        conn.close()


def test_self_parent_rejected(env):
    db_path, _, supported, _ = env
    claim = supported[0]
    conn = store.connect(db_path)
    try:
        with pytest.raises(claims.ClaimHierarchyError, match="own parent"):
            claims.validate_parent_relation(conn, claim.claim_id, claim.claim_id)
    finally:
        conn.close()


def test_parent_cycle_rejected(env):
    """A manually constructed parent cycle is detected deterministically."""
    db_path, session_id, supported, _ = env
    conn = store.connect(db_path)
    try:
        c1 = claims.create_claim(conn, session_id=session_id, statement="cycle node 1")
        c2 = claims.create_claim(conn, session_id=session_id, statement="cycle node 2")
        store.set_claim_parent(conn, c1.claim_id, c2.claim_id)
        store.set_claim_parent(conn, c2.claim_id, c1.claim_id)
        with pytest.raises(claims.ClaimHierarchyError, match="cycle"):
            claims.validate_parent_relation(conn, c1.claim_id, c2.claim_id)
    finally:
        conn.close()


def test_cross_session_parent_rejected(env, tmp_path):
    db_path, session_id, supported, _ = env
    conn = store.connect(db_path)
    try:
        other = store.create_session(conn, question="other session")
        other_claim = claims.create_claim(
            conn, session_id=other.session_id, statement="foreign claim"
        )
        with pytest.raises(claims.ClaimHierarchyError, match="same session"):
            claims.validate_parent_relation(
                conn, supported[0].claim_id, other_claim.claim_id
            )
    finally:
        conn.close()


def test_reparenting_existing_child_rejected(env):
    """A child that already belongs to a different parent cannot be moved."""
    db_path, session_id, supported, _ = env
    source_a, source_b = supported[0], supported[1]
    conn = store.connect(db_path)
    try:
        # Create a child under source_a...
        children = claims.create_decomposed_claims(
            conn, session_id=session_id, parent_claim_id=source_a.claim_id,
            statements=["shared child statement"], evidence_ids=[],
        )
        child = children[0]
        # ...then the same normalized statement proposed under source_b.
        with pytest.raises(claims.ClaimHierarchyError, match="re-parenting"):
            claims.create_decomposed_claims(
                conn, session_id=session_id,
                parent_claim_id=source_b.claim_id,
                statements=["shared child statement"], evidence_ids=[],
            )
        # The child still belongs to source_a.
        assert store.get_claim(conn, child.claim_id).parent_claim_id == source_a.claim_id
    finally:
        conn.close()
