"""Commit 3 acceptance: the tamper flow, end-to-end with real C0/C1/C2.

C0 ingest -> C1 evidence -> claim -> link -> evaluate -> supported
-> tamper chunk -> evaluate -> insufficient (event evidence_invalid)
-> restore chunk -> evaluate -> supported again.

Uses the real pipelines (docpipe.ingest, docpipe.search,
evidence.builder/validation) — nothing here mocks validation.
"""

from __future__ import annotations

import json

import docpipe.ingest as c0ingest
from research import claims, store
from research.reasoner import DeterministicReasoner
from research.retriever import LocalRetriever

from tests.docpipe.pdf_factory import build_pdf


def _tamper_chunk(conn, chunk_id: int, content: str) -> None:
    conn.execute("UPDATE chunks SET content = ? WHERE id = ?", (content, chunk_id))
    conn.commit()


def _last_event(conn, session_id: str, event_type: str):
    events = [e for e in store.list_events(conn, session_id)
              if e.event_type == event_type]
    return events[-1] if events else None


def test_tamper_flow_demotes_and_restores(tmp_path):
    db_path = tmp_path / "docpipe.db"

    # C0 ingest (real pipeline).
    pdf = tmp_path / "manual.pdf"
    original = "The dot product has variance dk."
    pdf.write_bytes(build_pdf([original, "Positional encoding adds order."]))
    ingested = c0ingest.ingest_pdf(pdf, title="Manual", db_path=db_path)

    # C2 retrieval -> C1 evidence (real pipeline).
    result = LocalRetriever(db_path=db_path).retrieve(
        ingested.document_id, "dot product variance"
    )
    assert result.evidence and not result.skipped
    evidence = result.evidence[0]

    conn = store.connect(db_path)
    try:
        session = store.create_session(conn, question="Why scale by sqrt dk?")

        # Claim from a real reasoner proposal.
        proposal = DeterministicReasoner().construct_claims([evidence])[0]
        claim = claims.create_claim_from_proposal(
            conn, session_id=session.session_id, proposal=proposal
        )
        claims.link_evidence(conn, claim_id=claim.claim_id,
                             evidence_id=evidence.evidence_id, role="supporting")

        # Evaluate -> supported.
        supported = claims.evaluate_claim(conn, claim.claim_id)
        assert supported.status == "supported"

        # TAMPER: change the CHUNK (never the evidence).
        _tamper_chunk(conn, evidence.chunk_id, "TAMPERED CONTENT")
        demoted = claims.evaluate_claim(conn, claim.claim_id)
        assert demoted.status == "insufficient"

        event = _last_event(conn, session.session_id, "claim_status_changed")
        assert event is not None
        detail = json.loads(event.detail)
        assert detail["from"] == "supported"
        assert detail["to"] == "insufficient"
        assert detail["reason"] == "evidence_invalid"

        # Evidence snapshot itself is untouched.
        import evidence.store as c1store

        assert c1store.get_evidence(conn, evidence.evidence_id).content == original

        # RESTORE the chunk -> supported again.
        _tamper_chunk(conn, evidence.chunk_id, original)
        restored = claims.evaluate_claim(conn, claim.claim_id)
        assert restored.status == "supported"
        event = _last_event(conn, session.session_id, "claim_status_changed")
        assert json.loads(event.detail)["reason"] == "sufficient_evidence"
    finally:
        conn.close()


def test_tamper_flow_with_two_supporting_one_invalid(tmp_path):
    """With min_supporting=1, one valid + one invalid support still holds."""
    db_path = tmp_path / "docpipe.db"
    pdf = tmp_path / "two.pdf"
    pdf.write_bytes(
        build_pdf(["First supporting fact.", "Second supporting fact."])
    )
    ingested = c0ingest.ingest_pdf(pdf, db_path=db_path)

    retriever = LocalRetriever(db_path=db_path)
    results = retriever.retrieve_many(ingested.document_id, ["supporting fact"])
    evidence = [e for r in results for e in r.evidence]
    assert len(evidence) == 2

    conn = store.connect(db_path)
    try:
        session = store.create_session(conn, question="Q?")
        claim = claims.create_claim(
            conn, session_id=session.session_id, statement="Claim with two supports"
        )
        for ev in evidence:
            claims.link_evidence(conn, claim_id=claim.claim_id,
                                 evidence_id=ev.evidence_id, role="supporting")
        assert claims.evaluate_claim(conn, claim.claim_id).status == "supported"

        # Invalidate one support.
        _tamper_chunk(conn, evidence[0].chunk_id, "TAMPERED")
        still = claims.evaluate_claim(conn, claim.claim_id)
        assert still.status == "supported"  # one valid support remains

        # Invalidate both -> insufficient.
        _tamper_chunk(conn, evidence[1].chunk_id, "TAMPERED TOO")
        demoted = claims.evaluate_claim(conn, claim.claim_id)
        assert demoted.status == "insufficient"
    finally:
        conn.close()
