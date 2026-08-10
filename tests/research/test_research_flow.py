"""C2 integration flow: C0 ingest -> C0 search -> C1 evidence ->
LocalRetriever -> DeterministicReasoner proposals.

This proves C2 really uses C0+C1 (no duplicated logic) and that the
reasoner only proposes — it never decides claim status.
"""

from __future__ import annotations

import docpipe.ingest as c0ingest
from research import store
from research.reasoner import DeterministicReasoner
from research.retriever import LocalRetriever

from tests.docpipe.pdf_factory import build_pdf


def test_full_retrieval_to_proposal_flow(tmp_path):
    db_path = tmp_path / "docpipe.db"

    # C0 ingest (real pipeline, not direct inserts).
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(
        build_pdf(
            [
                "Scaled dot product attention divides by the square root of dk.",
                "Positional encoding injects order information into the model.",
                "Multi head attention runs several attention layers in parallel.",
            ]
        )
    )
    ingested = c0ingest.ingest_pdf(pdf, title="Manual", db_path=db_path)
    assert ingested.status == "ready"
    assert ingested.chunks_created == 3

    # C2 retrieval: C0 search + C1 evidence.
    retriever = LocalRetriever(db_path=db_path)
    result = retriever.retrieve(ingested.document_id, "scaled dot product")
    assert result.evidence, "expected at least one evidence"
    assert result.skipped == []

    evidence = result.evidence[0]
    # Provenance: document -> chunk -> evidence.
    assert evidence.document_id == ingested.document_id
    assert evidence.source_title == "Manual"
    assert evidence.source_type == "local_pdf"
    assert evidence.page_number == 1
    assert "scaled dot product attention" in evidence.content.lower()
    assert evidence.locator["chunk_id"] == evidence.chunk_id

    # Reasoner proposes claims from evidence — proposals only.
    reasoner = DeterministicReasoner()
    proposals = reasoner.construct_claims(result.evidence)
    assert proposals
    proposal = proposals[0]
    assert proposal.evidence_ids == (evidence.evidence_id,)
    assert proposal.statement == evidence.content
    assert not hasattr(proposal, "status")

    # plan_queries derives follow-up queries from the question.
    queries = reasoner.plan_queries(
        "Why does scaled dot product attention divide by sqrt dk?", []
    )
    assert queries
    assert all(q.query_text for q in queries)

    # Reasoner never changes persisted claim status.
    conn = store.connect(db_path)
    try:
        session = store.create_session(conn, question="Why scale by sqrt dk?")
        claim = store.create_claim(
            conn, session_id=session.session_id,
            statement="The dot products grow large in magnitude.",
        )
        assert claim.status == "proposed"
        reasoner.construct_claims(result.evidence)
        reasoner.plan_queries(session.question, [])
        after = store.get_claim(conn, claim.claim_id)
        assert after.status == "proposed"
    finally:
        conn.close()


def test_retriever_reasoner_consistency_over_multiple_queries(tmp_path):
    """Two queries over the same document yield distinct, consistent evidence."""
    db_path = tmp_path / "docpipe.db"
    pdf = tmp_path / "multi.pdf"
    pdf.write_bytes(
        build_pdf(
            [
                "The servo motor is controlled by PWM pulses.",
                "The ESP32 connects to wifi for telemetry.",
            ]
        )
    )
    ingested = c0ingest.ingest_pdf(pdf, db_path=db_path)

    retriever = LocalRetriever(db_path=db_path)
    results = retriever.retrieve_many(
        ingested.document_id, ["servo motor", "wifi telemetry"]
    )
    assert [r.query for r in results] == ["servo motor", "wifi telemetry"]
    evidence_ids = {e.evidence_id for r in results for e in r.evidence}
    assert len(evidence_ids) == 2

    reasoner = DeterministicReasoner()
    all_evidence = [e for r in results for e in r.evidence]
    proposals = reasoner.construct_claims(all_evidence)
    assert {p.evidence_ids[0] for p in proposals} == evidence_ids
    # Every proposal traces to exactly its own evidence content.
    by_id = {e.evidence_id: e for e in all_evidence}
    for p in proposals:
        assert p.statement == by_id[p.evidence_ids[0]].content
