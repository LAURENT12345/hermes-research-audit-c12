"""C3 verifier tests: proposal validation, decomposition, context."""

from __future__ import annotations

import json

import pytest

import docpipe.db as c0db
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, store, verifier
from research.reasoner import (
    ClaimContextProposal,
    ClaimDecompositionProposal,
    EvidenceRelationProposal,
)


@pytest.fixture
def env(tmp_path):
    """Document, two evidence, one supported claim."""
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    doc_id = c0db.insert_document(
        conn, title="Paper", file_path="/tmp/p.pdf", file_hash="ha",
        status="ready",
    )
    chunk_ids = []
    for i, text in enumerate(
        ["Scaled dot product attention divides by the square root of dk.",
         "Multi head attention runs several attention layers in parallel."]
    ):
        chunk_ids.append(
            c0db.insert_chunk(
                conn, document_id=doc_id,
                chunk=Chunk(chunk_index=i, content=text,
                            normalized_content=text, page_number=i + 1,
                            section_path=None, token_count=10),
            )
        )
    ev1 = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_ids[0], evidence_type="chunk",
        source_type="local_pdf", source_title="Paper",
        content="Scaled dot product attention divides by the square root of dk.",
        page_number=1, section_path=None,
        locator={"page": 1, "chunk_id": chunk_ids[0]},
    ).evidence_id
    ev2 = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_ids[1], evidence_type="chunk",
        source_type="local_pdf", source_title="Paper",
        content="Multi head attention runs several attention layers in parallel.",
        page_number=2, section_path=None,
        locator={"page": 2, "chunk_id": chunk_ids[1]},
    ).evidence_id
    session = store.create_session(conn, question="Why scale by sqrt dk?")
    store.update_session_status(conn, session.session_id, "active")
    claim = claims.create_claim(
        conn, session_id=session.session_id,
        statement="Scaled dot product attention divides by the square root of dk.",
    )
    claims.link_evidence(conn, claim_id=claim.claim_id, evidence_id=ev1,
                         role="supporting")
    claims.evaluate_claim(conn, claim.claim_id)
    conn.close()
    return db_path, ev1, ev2, claim.claim_id


def _connect(env):
    return store.connect(env[0])


# ---------------------------------------------------------------------------
# A. Proposal validation
# ---------------------------------------------------------------------------

def test_valid_decomposition_accepted(env):
    db_path, ev1, _, claim_id = env
    conn = _connect(env)
    try:
        proposal = ClaimDecompositionProposal(
            source_claim_id=claim_id,
            child_statements=("Dot product attention divides by sqrt dk.",),
            child_evidence_ids=(ev1,),
        )
        assert verifier.validate_decomposition(conn, proposal).accepted
    finally:
        conn.close()


def test_decomposition_nonexistent_source_rejected(env):
    conn = _connect(env)
    try:
        proposal = ClaimDecompositionProposal(
            source_claim_id="C-999999",
            child_statements=("x",), child_evidence_ids=(),
        )
        verdict = verifier.validate_decomposition(conn, proposal)
        assert not verdict.accepted
        assert "not found" in verdict.reason
    finally:
        conn.close()


def test_decomposition_fabricated_evidence_rejected(env):
    db_path, ev1, _, claim_id = env
    conn = _connect(env)
    try:
        proposal = ClaimDecompositionProposal(
            source_claim_id=claim_id,
            child_statements=("x",),
            child_evidence_ids=("E-999999",),
        )
        verdict = verifier.validate_decomposition(conn, proposal)
        assert not verdict.accepted
        assert "fabricated" in verdict.reason
    finally:
        conn.close()


def test_decomposition_evidence_not_linked_to_source_rejected(env):
    db_path, _, ev2, claim_id = env
    conn = _connect(env)
    try:
        proposal = ClaimDecompositionProposal(
            source_claim_id=claim_id,
            child_statements=("x",),
            child_evidence_ids=(ev2,),  # linked to NO claim yet
        )
        verdict = verifier.validate_decomposition(conn, proposal)
        assert not verdict.accepted
        assert "not linked" in verdict.reason
    finally:
        conn.close()


def test_decomposition_invalid_evidence_rejected(env):
    import sqlite3

    db_path, ev1, _, claim_id = env
    conn = _connect(env)
    try:
        conn.execute(
            "UPDATE chunks SET content = 'TAMPERED' WHERE id = "
            "(SELECT chunk_id FROM evidence WHERE id = ?)", (ev1,)
        )
        conn.commit()
        proposal = ClaimDecompositionProposal(
            source_claim_id=claim_id,
            child_statements=("x",), child_evidence_ids=(ev1,),
        )
        verdict = verifier.validate_decomposition(conn, proposal)
        assert not verdict.accepted
        assert "invalid" in verdict.reason
    finally:
        conn.close()


def test_evidence_relation_validation(env):
    db_path, ev1, ev2, claim_id = env
    conn = _connect(env)
    try:
        assert verifier.validate_evidence_relation(
            conn, EvidenceRelationProposal(claim_id, ev2, "supporting")
        ).accepted
        assert not verifier.validate_evidence_relation(
            conn, EvidenceRelationProposal("C-999999", ev1, "supporting")
        ).accepted
        assert not verifier.validate_evidence_relation(
            conn, EvidenceRelationProposal(claim_id, "E-999999", "supporting")
        ).accepted
        assert not verifier.validate_evidence_relation(
            conn, EvidenceRelationProposal(claim_id, ev1, "ambiguous")
        ).accepted
    finally:
        conn.close()


def test_context_validation(env):
    db_path, _, _, claim_id = env
    conn = _connect(env)
    try:
        good = ClaimContextProposal(claim_id, {"dataset": "WMT14", "task": "translation"})
        assert verifier.validate_context(conn, good).accepted
        malformed = ClaimContextProposal(claim_id, {"bogus_key": "x"})
        verdict = verifier.validate_context(conn, malformed)
        assert not verdict.accepted
        assert "unknown context key" in verdict.reason
        assert not verifier.validate_context(
            conn, ClaimContextProposal("C-999999", {"task": "x"})
        ).accepted
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# B. Claim decomposition
# ---------------------------------------------------------------------------

def test_decomposition_apply_creates_children_with_provenance(env):
    db_path, ev1, _, claim_id = env
    conn = _connect(env)
    try:
        children = verifier.apply_decomposition(
            conn,
            ClaimDecompositionProposal(
                source_claim_id=claim_id,
                child_statements=(
                    "Dot product attention divides by sqrt dk.",
                    "The division counteracts large dot products.",
                ),
                child_evidence_ids=(ev1,),
            ),
        )
        assert len(children) == 2
        for child in children:
            row = store.get_claim(conn, child.claim_id)
            assert row.parent_claim_id == claim_id  # provenance preserved
            assert len(store.get_links_for_claim(conn, child.claim_id)) == 1
    finally:
        conn.close()


def test_decomposition_unsupported_child_never_enters_synthesis(env):
    db_path, ev1, _, claim_id = env
    conn = _connect(env)
    try:
        children = verifier.apply_decomposition(
            conn,
            ClaimDecompositionProposal(
                source_claim_id=claim_id,
                child_statements=("An unsupported child proposition.",),
                child_evidence_ids=(),  # no evidence -> child unsupported
            ),
        )
        child = children[0]
        claims.evaluate_claim(conn, child.claim_id)
        assert store.get_claim(conn, child.claim_id).status == "insufficient"

        from research.synthesis import build_answer

        session_id = store.get_claim(conn, claim_id).session_id
        answer = build_answer(conn, session_id)
        assert child.claim_id not in {s.claim_id for s in answer.statements}
        assert any(g.claim_id == child.claim_id for g in answer.gaps)
    finally:
        conn.close()


def test_decomposition_apply_rejects_fabricated(env):
    db_path, _, _, claim_id = env
    conn = _connect(env)
    try:
        with pytest.raises(verifier.ProposalRejectedError):
            verifier.apply_decomposition(
                conn,
                ClaimDecompositionProposal(
                    source_claim_id=claim_id,
                    child_statements=("x",),
                    child_evidence_ids=("E-424242",),
                ),
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# D. Context persistence / backward compatibility
# ---------------------------------------------------------------------------

def test_context_persists_and_roundtrips(env):
    db_path, _, _, claim_id = env
    conn = _connect(env)
    try:
        proposal = ClaimContextProposal(
            claim_id, {"dataset": "WMT14", "task": "translation",
                       "population": None}
        )
        updated = verifier.apply_context(conn, proposal)
        assert updated.context is not None
        assert json.loads(updated.context) == {
            "dataset": "WMT14", "task": "translation", "population": None,
        }
    finally:
        conn.close()


def test_old_claims_have_null_context(env):
    """Backward compatibility: claims created before C3 have NULL context,
    which is DISTINCT from an explicit empty/universal scope."""
    db_path, _, _, claim_id = env
    conn = _connect(env)
    try:
        claim = store.get_claim(conn, claim_id)
        assert claim.context is None  # not {}- not 'universal'
    finally:
        conn.close()


def test_context_malformed_rejected_on_apply(env):
    db_path, _, _, claim_id = env
    conn = _connect(env)
    try:
        with pytest.raises(verifier.ProposalRejectedError):
            verifier.apply_context(
                conn, ClaimContextProposal(claim_id, {"bogus": 1})
            )
    finally:
        conn.close()
