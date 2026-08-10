"""Contradiction tests: detection, resolution, synthesis guard."""

from __future__ import annotations

import json

import pytest

import docpipe.db as c0db
import evidence.store as c1store
from docpipe.chunker import Chunk
from research import claims, contradictions, store
from research.contradictions import (
    ContradictionClosedError,
    ContradictionInvalidDecisionError,
    ContradictionNotFoundError,
    ContradictionStateConflictError,
    detect_contradictions,
    resolve_contradiction,
    synthesis_eligible_claims,
)
from research.reasoner import (
    DECISION_CONTRADICTION_WINS,
    DECISION_SUPPORTING_STANDS,
)


@pytest.fixture
def seeded(tmp_path):
    """Session with three evidence objects."""
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    doc_id = c0db.insert_document(
        conn, title="Paper", file_path="/tmp/a.pdf", file_hash="ha",
        status="ready",
    )
    chunk_ids = []
    for i, text in enumerate(
        [
            "The dot product has variance dk.",
            "Softmax gradients get extremely small.",
            "Positional encoding adds order information.",
        ]
    ):
        chunk_ids.append(
            c0db.insert_chunk(
                conn, document_id=doc_id,
                chunk=Chunk(chunk_index=i, content=text,
                            normalized_content=text, page_number=i + 1,
                            section_path=None, token_count=5),
            )
        )
    evs = []
    for i, (chunk_id, content) in enumerate(
        zip(chunk_ids, ["The dot product has variance dk.",
                        "Softmax gradients get extremely small.",
                        "Positional encoding adds order information."])
    ):
        evs.append(
            c1store.create_evidence(
                conn, document_id=doc_id, chunk_id=chunk_id,
                evidence_type="chunk", source_type="local_pdf",
                source_title="Paper", content=content,
                page_number=i + 1, section_path=None,
                locator={"page": i + 1, "chunk_id": chunk_id},
            ).evidence_id
        )
    session = store.create_session(conn, question="Why scale?")
    conn.close()
    return db_path, session.session_id, evs


def _tamper_chunk_of(conn, evidence_id: str, content: str) -> None:
    conn.execute(
        "UPDATE chunks SET content = ? WHERE id = "
        "(SELECT chunk_id FROM evidence WHERE id = ?)",
        (content, evidence_id),
    )
    conn.commit()


def _events(conn, session_id, event_type):
    return [e for e in store.list_events(conn, session_id)
            if e.event_type == event_type]


def _open_records(conn, session_id):
    return store.list_contradictions(conn, session_id, status="open")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_explicit_contradiction_detected(seeded):
    db_path, session_id, (ev1, ev2, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")

        records = detect_contradictions(conn, session_id)
        assert len(records) == 1
        assert records[0].kind == "explicit"
        assert records[0].status == "open"
        assert json.loads(records[0].detail)["contradicting_evidence_ids"] == [ev2]

        events = _events(conn, session_id, "contradiction_detected")
        assert len(events) == 1
        assert json.loads(events[0].detail)["claim_id"] == c.claim_id
        assert "content" not in events[0].detail  # metadata only
    finally:
        conn.close()


def test_invalid_contradicting_evidence_not_a_contradiction(seeded):
    """An INVALID contradicting evidence is not a valid contradiction."""
    db_path, session_id, (_, ev2, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        _tamper_chunk_of(conn, ev2, "TAMPERED")

        assert detect_contradictions(conn, session_id) == []
        assert _open_records(conn, session_id) == []
    finally:
        conn.close()


def test_structural_conflict_detected(seeded):
    db_path, session_id, (ev1, _, ev3) = seeded
    conn = store.connect(db_path)
    try:
        c1 = claims.create_claim(
            conn, session_id=session_id,
            statement="Scaled dot product attention divides by the square root of dk",
        )
        c2 = claims.create_claim(
            conn, session_id=session_id,
            statement="scaled dot product attention divides by the square root of "
            "the key dimension",
        )
        assert c1.claim_id != c2.claim_id  # distinct statements, high overlap
        claims.link_evidence(conn, claim_id=c1.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c2.claim_id, evidence_id=ev3,
                             role="supporting")

        records = detect_contradictions(conn, session_id)
        structural = [r for r in records if r.kind == "structural"]
        assert len(structural) == 1
        detail = json.loads(structural[0].detail)
        assert detail["other_claim_id"] in (c1.claim_id, c2.claim_id)
    finally:
        conn.close()


def test_shared_evidence_prevents_structural_conflict(seeded):
    """Two similar claims supported by the SAME evidence are not a conflict."""
    db_path, session_id, (ev1, _, _) = seeded
    conn = store.connect(db_path)
    try:
        c1 = claims.create_claim(
            conn, session_id=session_id,
            statement="The dot product variance is dk in the attention mechanism",
        )
        c2 = claims.create_claim(
            conn, session_id=session_id,
            statement="the dot product variance is equal to dk in the attention "
            "mechanism",
        )
        assert c1.claim_id != c2.claim_id
        claims.link_evidence(conn, claim_id=c1.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c2.claim_id, evidence_id=ev1,
                             role="supporting")
        assert detect_contradictions(conn, session_id) == []
    finally:
        conn.close()


def test_detection_idempotent(seeded):
    db_path, session_id, (ev1, ev2, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")

        detect_contradictions(conn, session_id)
        again = detect_contradictions(conn, session_id)
        assert again == []  # no duplicate records
        assert len(_open_records(conn, session_id)) == 1
        assert len(_events(conn, session_id, "contradiction_detected")) == 1
    finally:
        conn.close()


def test_detection_does_not_change_claim_status(seeded):
    db_path, session_id, (ev1, ev2, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        detect_contradictions(conn, session_id)
        assert store.get_claim(conn, c.claim_id).status == "proposed"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_resolution_withdraws_claim(seeded):
    db_path, session_id, (ev1, ev2, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        record = detect_contradictions(conn, session_id)[0]

        resolved = resolve_contradiction(conn, record.id,
                                         decision=DECISION_CONTRADICTION_WINS,
                                         note="contradiction confirmed")
        assert resolved.status == "resolved"
        assert json.loads(resolved.resolution)["decision"] == "contradiction_wins"
        assert store.get_claim(conn, c.claim_id).status == "withdrawn"

        events = _events(conn, session_id, "contradiction_resolved")
        assert len(events) == 1
        detail = json.loads(events[0].detail)
        assert detail["decision"] == "contradiction_wins"
        assert detail["applied"] is True
    finally:
        conn.close()


def test_resolution_supporting_stands_promotes_claim(seeded):
    """Contradicting evidence becomes invalid -> supporting stands ->
    record dismissed -> claim re-evaluated to supported."""
    db_path, session_id, (ev1, ev2, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        record = detect_contradictions(conn, session_id)[0]

        # The contradicting evidence is no longer valid.
        _tamper_chunk_of(conn, ev2, "TAMPERED")

        resolved = resolve_contradiction(conn, record.id,
                                         decision=DECISION_SUPPORTING_STANDS,
                                         note="contradiction gone")
        assert resolved.status == "dismissed"
        assert store.get_claim(conn, c.claim_id).status == "supported"
    finally:
        conn.close()


def test_resolution_undetermined_keeps_open(seeded):
    db_path, session_id, (ev1, ev2, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        record = detect_contradictions(conn, session_id)[0]

        resolve_contradiction(conn, record.id, decision="undetermined",
                              note="needs human review")
        still_open = store.get_contradiction(conn, record.id)
        assert still_open.status == "open"
        # An event records the non-application explicitly.
        detail = json.loads(
            _events(conn, session_id, "contradiction_resolved")[0].detail
        )
        assert detail["applied"] is False
    finally:
        conn.close()


def test_resolution_errors(seeded):
    db_path, session_id, (ev1, ev2, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        record = detect_contradictions(conn, session_id)[0]

        with pytest.raises(ContradictionNotFoundError):
            resolve_contradiction(conn, 99999)
        with pytest.raises(ContradictionInvalidDecisionError):
            resolve_contradiction(conn, record.id, decision="maybe")

        resolve_contradiction(conn, record.id, decision=DECISION_CONTRADICTION_WINS)
        with pytest.raises(ContradictionClosedError):
            resolve_contradiction(conn, record.id, decision=DECISION_CONTRADICTION_WINS)
    finally:
        conn.close()


def test_resolution_stale_decision_rejected(seeded):
    """contradiction_wins with no valid contradicting evidence left is a
    state conflict and must not silently withdraw the claim."""
    db_path, session_id, (ev1, ev2, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        record = detect_contradictions(conn, session_id)[0]

        _tamper_chunk_of(conn, ev2, "TAMPERED")  # contradiction gone
        with pytest.raises(ContradictionStateConflictError):
            resolve_contradiction(conn, record.id,
                                  decision=DECISION_CONTRADICTION_WINS)
        # Claim was re-evaluated to supported (not withdrawn).
        assert store.get_claim(conn, c.claim_id).status == "supported"
    finally:
        conn.close()


def test_resolution_uses_reasoner_port_by_default(seeded):
    """With decision=None the DeterministicReasoner adjudicates."""
    db_path, session_id, (ev1, ev2, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        record = detect_contradictions(conn, session_id)[0]
        resolved = resolve_contradiction(conn, record.id)  # no decision
        assert resolved.status == "resolved"  # contradiction_wins (MVP rule)
        assert store.get_claim(conn, c.claim_id).status == "withdrawn"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Synthesis guard
# ---------------------------------------------------------------------------

def test_contradicted_claim_blocked_from_synthesis(seeded):
    db_path, session_id, (ev1, ev2, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        assert claims.evaluate_claim(conn, c.claim_id).status == "contradicted"
        assert synthesis_eligible_claims(conn, session_id) == []
    finally:
        conn.close()


def test_supported_claim_without_conflict_eligible(seeded):
    db_path, session_id, (ev1, _, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        eligible = synthesis_eligible_claims(conn, session_id)
        assert [cl.claim_id for cl in eligible] == [c.claim_id]
    finally:
        conn.close()


def test_tampered_support_demotes_before_synthesis(seeded):
    """Tampered supporting evidence -> claim demoted BEFORE the guard."""
    db_path, session_id, (ev1, _, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        assert claims.evaluate_claim(conn, c.claim_id).status == "supported"

        _tamper_chunk_of(conn, ev1, "TAMPERED")
        assert synthesis_eligible_claims(conn, session_id) == []
        assert store.get_claim(conn, c.claim_id).status == "insufficient"
    finally:
        conn.close()


def test_open_contradiction_blocks_even_stale_supported(seeded):
    """A claim marked supported with an OPEN explicit contradiction record
    is excluded by the guard regardless."""
    db_path, session_id, (ev1, ev2, _) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev2,
                             role="contradicting")
        detect_contradictions(conn, session_id)  # open record persisted
        # Force a stale status: evaluate while contra still valid would set
        # contradicted; simulate the stale case by direct status write.
        store.update_claim_status(conn, c.claim_id, "supported")
        assert synthesis_eligible_claims(conn, session_id) == []
        # The guard's evaluation moved it back to contradicted.
        assert store.get_claim(conn, c.claim_id).status == "contradicted"
    finally:
        conn.close()


def test_structural_conflict_advisory_does_not_block(seeded):
    db_path, session_id, (ev1, _, _) = seeded
    conn = store.connect(db_path)
    try:
        c1 = claims.create_claim(
            conn, session_id=session_id,
            statement="Scaled dot product attention divides by the square root of dk",
        )
        claims.create_claim(
            conn, session_id=session_id,
            statement="scaled dot product attention divides by the square root of "
            "the key dimension",
        )
        claims.link_evidence(conn, claim_id=c1.claim_id, evidence_id=ev1,
                             role="supporting")
        detect_contradictions(conn, session_id)  # structural record created
        eligible = synthesis_eligible_claims(conn, session_id)
        assert [cl.claim_id for cl in eligible] == [c1.claim_id]
    finally:
        conn.close()


def test_citations_derive_from_links(seeded):
    """Guard keeps claims whose citations (supporting links) are intact."""
    db_path, session_id, (ev1, _, ev3) = seeded
    conn = store.connect(db_path)
    try:
        c = claims.create_claim(conn, session_id=session_id, statement="Scaling helps")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev1,
                             role="supporting")
        claims.link_evidence(conn, claim_id=c.claim_id, evidence_id=ev3,
                             role="supporting")
        eligible = synthesis_eligible_claims(conn, session_id)
        assert len(eligible) == 1
        links = store.get_links_for_claim(conn, c.claim_id)
        assert {l.evidence_id for l in links} == {ev1, ev3}
    finally:
        conn.close()
