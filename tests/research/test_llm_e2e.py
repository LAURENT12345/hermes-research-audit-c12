"""Commit 12 — end-to-end black-box test (Phase 10).

Full deterministic pipeline with a FAKE LLM transport:

    research input -> LLM adapter -> proposals -> deterministic verifier
        -> domain-layer apply -> synthesis

The fake transport serves canned structured output built from REAL
persisted ids (claims/evidence created by the deterministic loop), so the
only thing the LLM contributes is proposal CONTENT — every acceptance
decision and every write happens in the deterministic layers.

No live external API is required for this test (the default test suite
must never touch the network). A real-provider smoke test would be
opt-in and environment-dependent; none is enabled here.
"""

from __future__ import annotations

import json

import pytest

import docpipe.db as c0db
import docpipe.ingest as c0ingest
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, store, verifier
from research.llm_reasoner import LLMReasoner
from research.reasoner import (
    ClaimContextProposal,
    ClaimDecompositionProposal,
    CoverageProposal,
    EvidenceRelationProposal,
    SemanticContradictionProposal,
)
from research.synthesis import synthesize

from tests.docpipe.pdf_factory import build_pdf

CHUNK_TEXTS = [
    "Scaled dot product attention divides by the square root of dk.",
    "Multi head attention runs several attention layers in parallel.",
]


@pytest.fixture
def env(tmp_path):
    """PDF -> ingest -> deterministic loop -> supported claims + evidence."""
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
    assert len(supported) >= 2, "loop must produce at least 2 supported claims"
    ev_by_claim = {
        c.claim_id: [
            l.evidence_id for l in store.get_links_for_claim(conn, c.claim_id)
            if l.role == "supporting"
        ]
        for c in supported
    }
    conn.close()
    return db_path, session.session_id, supported, ev_by_claim


class FakeLLMTransport:
    """Canned-response transport; the responder receives the messages and
    returns raw text (built from real ids by the test)."""

    def __init__(self, responder):
        self._responder = responder
        self.calls = 0

    def complete(self, messages, *, timeout):
        self.calls += 1
        return self._responder(messages)


def _decomposition_response(claim_id, evidence_ids, statements):
    return json.dumps(
        {
            "proposals": [
                {
                    "type": "claim_decomposition",
                    "source_claim_id": claim_id,
                    "child_statements": statements,
                    "child_evidence_ids": evidence_ids,
                }
            ]
        }
    )


def _contradiction_response(claim_a, claim_b, ev_a, ev_b):
    return json.dumps(
        {
            "proposals": [
                {
                    "type": "semantic_contradiction",
                    "claim_a_id": claim_a,
                    "claim_b_id": claim_b,
                    "evidence_a_ids": [ev_a],
                    "evidence_b_ids": [ev_b],
                    "rationale": "one divides by sqrt dk, the other runs "
                                 "several layers in parallel",
                    "strength": 0.8,
                }
            ]
        }
    )


def test_full_pipeline_llm_proposals_apply_and_synthesize(env):
    """research input -> LLM adapter -> proposals -> verifier -> domain
    apply -> synthesis, all with deterministic fake model output."""
    db_path, session_id, supported, ev_by_claim = env
    claim_a, claim_b = supported[0], supported[1]
    ev_a = ev_by_claim[claim_a.claim_id][0]
    ev_b = ev_by_claim[claim_b.claim_id][0]

    conn = store.connect(db_path)
    try:
        claim_a_obj = store.get_claim(conn, claim_a.claim_id)
        claim_b_obj = store.get_claim(conn, claim_b.claim_id)
        evidence_a = c1store.get_evidence(conn, ev_a)
        evidence_b = c1store.get_evidence(conn, ev_b)
    finally:
        conn.close()

    # Scripted model: one response per proposal method, using REAL ids.
    def responder(messages):
        task = json.loads(messages[1]["content"])["task"]
        if "atomic child propositions" in task:
            return _decomposition_response(
                claim_a.claim_id, [ev_a],
                ["Scaled dot product attention divides by the square root "
                 "of the key dimension."],
            )
        if "semantically contradictory" in task:
            return _contradiction_response(
                claim_a.claim_id, claim_b.claim_id, ev_a, ev_b
            )
        if "scope/context qualifiers" in task:
            return json.dumps(
                {
                    "proposals": [
                        {
                            "type": "claim_context",
                            "claim_id": claim_a.claim_id,
                            "context": {"task": "machine translation"},
                        }
                    ]
                }
            )
        if "evidence coverage" in task:
            return json.dumps(
                {
                    "proposals": [
                        {
                            "type": "coverage",
                            "claim_id": claim_a.claim_id,
                            "coverage": "directly_supported",
                        }
                    ]
                }
            )
        return '{"proposals": []}'

    transport = FakeLLMTransport(responder)
    reasoner = LLMReasoner(transport)
    conn = store.connect(db_path)
    try:
        # 1. decomposition proposal -> verifier -> domain apply.
        proposals = reasoner.propose_decomposition(
            claim_a_obj, evidence=[evidence_a]
        )
        assert len(proposals) == 1
        assert isinstance(proposals[0], ClaimDecompositionProposal)
        children = verifier.apply_decomposition(conn, proposals[0])
        assert len(children) == 1
        child = children[0]
        assert child.parent_claim_id == claim_a.claim_id
        # The child inherits the validated support.
        assert claims.evaluate_claim(conn, child.claim_id).status == "supported"

        # 2. semantic contradiction proposal -> observation only.
        contra = reasoner.propose_semantic_contradictions(
            claim_a_obj, claim_b_obj,
            evidence_a=[evidence_a], evidence_b=[evidence_b],
        )
        assert len(contra) == 1
        assert isinstance(contra[0], SemanticContradictionProposal)
        verifier.apply_semantic_contradiction(conn, contra[0])
        # Authoritative claim status untouched by the observation.
        assert store.get_claim(conn, claim_a.claim_id).status == "supported"
        assert store.get_claim(conn, claim_b.claim_id).status == "supported"

        # 3. context proposal -> merged into the claim record.
        ctx = reasoner.propose_context(claim_a_obj)
        assert isinstance(ctx, ClaimContextProposal)
        verifier.apply_context(conn, ctx)
        stored = store.get_claim(conn, claim_a.claim_id)
        assert json.loads(stored.context) == {"task": "machine translation"}

        # 4. coverage proposal matching derived state -> accepted.
        cov = reasoner.propose_coverage(claim_a_obj)
        assert isinstance(cov, CoverageProposal)
        observation = verifier.apply_coverage(conn, cov)
        assert observation.coverage == "directly_supported"

        # 5. synthesis includes the LLM-proposed child as a statement.
        answer = synthesize(conn, session_id)
        statements = [s.statement for s in answer.statements]
        assert any(child.statement == s for s in statements)

        # 6. the model never wrote anything itself: no events originate
        #    from the adapter (events come from domain applies only).
        event_types = [
            e.event_type
            for e in store.list_events(conn, session_id)
            if "llm" in e.event_type.lower()
        ]
        assert event_types == []
    finally:
        conn.close()


def test_malicious_model_output_cannot_mutate_state(env):
    """A hallucinating model proposing fabricated ids is fully neutralized
    by the deterministic verifier: nothing is written."""
    db_path, session_id, supported, ev_by_claim = env
    claim_a = supported[0]
    ev_a = ev_by_claim[claim_a.claim_id][0]

    def malicious_decomposition(messages):
        return json.dumps(
            {
                "proposals": [
                    {
                        "type": "claim_decomposition",
                        "source_claim_id": claim_a.claim_id,
                        "child_statements": ["Fabricated child."],
                        # Invented evidence id.
                        "child_evidence_ids": ["E-999999"],
                    }
                ]
            }
        )

    def malicious_relation(messages):
        return json.dumps(
            {
                "proposals": [
                    {
                        "type": "evidence_relation",
                        # Invented claim id + invented evidence id.
                        "claim_id": "C-999999",
                        "evidence_id": "E-999999",
                        "role": "supporting",
                    }
                ]
            }
        )

    conn = store.connect(db_path)
    try:
        claim_a_obj = store.get_claim(conn, claim_a.claim_id)
        evidence_a = c1store.get_evidence(conn, ev_a)
        before = len(store.list_claims(conn, session_id))

        # Fabricated evidence id: parsed as a proposal, rejected by the
        # verifier, and NOTHING is written.
        reasoner = LLMReasoner(FakeLLMTransport(malicious_decomposition))
        proposals = reasoner.propose_decomposition(claim_a_obj)
        assert proposals[0].child_evidence_ids == ("E-999999",)
        with pytest.raises(verifier.ProposalRejectedError):
            verifier.apply_decomposition(conn, proposals[0])
        assert len(store.list_claims(conn, session_id)) == before

        # Fabricated claim id in an evidence relation: same path.
        reasoner2 = LLMReasoner(FakeLLMTransport(malicious_relation))
        relations = reasoner2.propose_evidence_relations(
            claim_a_obj, [evidence_a]
        )
        assert relations[0].claim_id == "C-999999"
        with pytest.raises(verifier.ProposalRejectedError):
            verifier.apply_evidence_relation(conn, relations[0])

        # No claim status changed, no events emitted by the adapter.
        assert store.get_claim(conn, claim_a.claim_id).status == "supported"
        links = store.get_links_for_claim(conn, claim_a.claim_id)
        assert all(l.evidence_id == ev_a for l in links)
    finally:
        conn.close()


def test_adapter_short_circuits_without_evidence(env):
    """Without evidence the adapter makes NO model call (deterministic
    baseline behavior) — an LLM failure can never masquerade as a
    research-gap or 'no evidence' finding."""
    db_path, session_id, supported, ev_by_claim = env
    conn = store.connect(db_path)
    try:
        claim_a_obj = store.get_claim(conn, supported[0].claim_id)
        claim_b_obj = store.get_claim(conn, supported[1].claim_id)
    finally:
        conn.close()

    class ExplodingTransport:
        def complete(self, messages, *, timeout):
            raise AssertionError("transport must not be called")

    reasoner = LLMReasoner(ExplodingTransport())
    assert reasoner.propose_evidence_relations(claim_a_obj, []) == []
    assert reasoner.propose_semantic_contradictions(
        claim_a_obj, claim_b_obj
    ) == []
