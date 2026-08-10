"""DeterministicReasoner tests: determinism, proposals, status neutrality."""

from __future__ import annotations

import pytest

from evidence.model import Evidence
from research import config
from research.reasoner import (
    DECISION_CONTRADICTION_WINS,
    DECISION_SUPPORTING_STANDS,
    ClaimProposal,
    Conflict,
    DeterministicReasoner,
    Gap,
    ReasonerPort,
    SupportedClaim,
)

REASONER = DeterministicReasoner()


def _evidence(eid: str, content: str) -> Evidence:
    return Evidence(
        evidence_id=eid, document_id=1, chunk_id=int(eid[-1]),
        evidence_type="chunk", source_type="local_pdf",
        source_title="Attention Is All You Need", content=content,
        page_number=4, section_path=None,
        locator={"page": 4, "chunk_id": int(eid[-1])},
        created_at="t",
    )


def test_reasoner_port_protocol():
    """DeterministicReasoner satisfies the ReasonerPort protocol."""
    assert isinstance(REASONER, ReasonerPort)


def test_deterministic_output_same_input():
    question = "Why does scaled dot product attention divide by sqrt dk?"
    gaps = [Gap(claim_id="C-000001", statement="The softmax gradients shrink.")]
    evidence = [_evidence("E-000001", "Dot products grow large in magnitude.")]

    for _ in range(2):
        q1 = REASONER.plan_queries(question, gaps)
        q2 = REASONER.plan_queries(question, gaps)
        assert q1 == q2
        c1 = REASONER.construct_claims(evidence)
        c2 = REASONER.construct_claims(evidence)
        assert c1 == c2


def test_plan_queries_from_question():
    proposals = REASONER.plan_queries(
        "Why does scaled dot product attention divide by the square root of dk?",
        [],
    )
    assert proposals
    query = proposals[0].query_text
    # Stopwords removed, content terms present, deterministic order.
    assert "scaled" in query and "attention" in query
    assert "the" not in query.split()
    assert all(p.origin == "derived" for p in proposals)


def test_plan_queries_includes_gaps():
    base = REASONER.plan_queries("Question about transformers?", [])
    with_gaps = REASONER.plan_queries(
        "Question about transformers?",
        [Gap(claim_id="C-000001", statement="softmax gradients vanishing")],
    )
    assert len(with_gaps) > len(base)
    assert any("softmax" in p.query_text for p in with_gaps)


def test_plan_queries_empty_inputs():
    assert REASONER.plan_queries("", []) == []
    assert REASONER.plan_queries("???", []) == []
    assert REASONER.plan_queries("the and of", [Gap(claim_id="C-1", statement="")]) == []


def test_plan_queries_capped():
    gaps = [Gap(claim_id=f"C-{i:06d}", statement=f"unique term number {i}") for i in range(20)]
    proposals = REASONER.plan_queries("base question", gaps)
    assert len(proposals) <= config.MAX_QUERIES_PER_ITERATION


def test_plan_queries_deduplicates():
    proposals = REASONER.plan_queries(
        "scaled dot product attention",
        [Gap(claim_id="C-000001", statement="scaled dot product attention")],
    )
    assert len(proposals) == 1


def test_construct_claims_one_per_evidence():
    evidence = [
        _evidence("E-000001", "First content."),
        _evidence("E-000002", "Second content."),
    ]
    proposals = REASONER.construct_claims(evidence)
    assert len(proposals) == 2
    assert proposals[0].statement == "First content."
    assert proposals[0].evidence_ids == ("E-000001",)
    assert proposals[1].evidence_ids == ("E-000002",)


def test_construct_claims_empty():
    assert REASONER.construct_claims([]) == []


def test_construct_claims_truncates_long_content():
    long = "Word. " * 200  # ~1200 chars > 400 limit
    proposals = REASONER.construct_claims([_evidence("E-000001", long)])
    assert len(proposals[0].statement) <= config.MAX_STATEMENT_CHARS
    # Truncation prefers a sentence boundary.
    assert proposals[0].statement.endswith(".")


def test_claim_proposal_has_no_status():
    """The reasoner must not decide claim status."""
    proposal = REASONER.construct_claims([_evidence("E-000001", "Text.")])[0]
    assert not hasattr(proposal, "status")
    assert isinstance(proposal, ClaimProposal)


def test_adjudicate_no_contradiction():
    result = REASONER.adjudicate(
        Conflict(
            claim_id="C-000001", statement="X",
            supporting_evidence_ids=("E-000001",),
            contradicting_evidence_ids=(),
        )
    )
    assert result.decision == DECISION_SUPPORTING_STANDS
    assert result.claim_id == "C-000001"


def test_adjudicate_with_contradiction():
    result = REASONER.adjudicate(
        Conflict(
            claim_id="C-000001", statement="X",
            supporting_evidence_ids=("E-000001",),
            contradicting_evidence_ids=("E-000002",),
        )
    )
    assert result.decision == DECISION_CONTRADICTION_WINS
    assert result.note


def test_synthesize_interface_and_ordering():
    draft = REASONER.synthesize(
        [
            SupportedClaim(claim_id="C-000002", statement="Second", evidence_ids=("E-1",)),
            SupportedClaim(claim_id="C-000001", statement="First", evidence_ids=("E-2",)),
        ]
    )
    assert draft.summary == "Deterministic synthesis of 2 supported claim(s)"
    assert [s["claim_id"] for s in draft.statements] == ["C-000001", "C-000002"]
    assert draft.statements[0]["evidence_ids"] == ["E-2"]


def test_synthesize_empty():
    draft = REASONER.synthesize([])
    assert draft.statements == ()
    assert "0 supported claim" in draft.summary


def test_reasoner_does_not_touch_claim_status(tmp_path):
    """Running the reasoner never changes persisted claim status."""
    from research import store

    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    try:
        session = store.create_session(conn, question="Q?")
        claim = store.create_claim(conn, session_id=session.session_id,
                                   statement="Proposed claim text")
        assert claim.status == "proposed"

        REASONER.plan_queries("some question", [Gap(claim_id=claim.claim_id,
                                                    statement=claim.statement)])
        REASONER.construct_claims(
            [_evidence("E-000001", "Some evidence content.")]
        )

        after = store.get_claim(conn, claim.claim_id)
        assert after.status == "proposed"
        assert after.statement == "Proposed claim text"
    finally:
        conn.close()


def test_reasoner_module_has_no_store_reference():
    """Structural guarantee: reasoner.py must not import research.store."""
    import research.reasoner as reasoner_module

    assert "store" not in dir(reasoner_module)
