"""Acceptance: full pipeline — real PDF -> C0 ingest -> C2 loop ->
supported claim -> synthesis -> exact citation -> persisted answer.

Scenario A (happy path): deterministic and reproducible. The fixture is a
REAL PDF (valid bytes produced by ``pdf_factory.build_pdf``, parsed by
pypdf) whose content is fixed — it does not depend on machine state. The
same input runs twice and must produce the same answer JSON.

Additional end-to-end coverage: when the real "Attention Is All You Need"
paper happens to be present on this machine, one extra test exercises the
full pipeline over it (skipped elsewhere — never fails on machines
without the file).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import docpipe.ingest as c0ingest
import evidence.store as c1store
import evidence.validation as c1validation
from research import store
from research.loop import ResearchLoop
from research.synthesis import load_answer, synthesize

from tests.docpipe.pdf_factory import build_pdf

CHUNK_TEXTS = [
    "Scaled dot product attention divides by the square root of dk.",
    "Positional encoding adds order information to the model.",
]
QUESTION = "Why does scaled dot product attention divides by the square root of dk?"

REAL_PDF = Path(r"C:\Users\ACER\dnse-sdk\Downloads\1706.03762v7.pdf")
REAL_QUESTION = "Why does scaled dot product attention divide by dk?"

#: Session statuses that can no longer be started/iterated.
_TERMINAL_STATUSES = ("complete", "complete_with_gaps", "failed", "aborted")


def _run_to_terminal(loop, session_id, max_calls=8):
    """Drive ``start`` + ``iterate_once`` until a terminal status.

    Uses only the public loop API (``LoopResult.session_status``), never
    private internals. ``max_calls`` is generous: the session defaults to
    ``max_iterations=5``, so 8 calls can always reach a terminal state;
    anything else fails the caller's terminal assertion loudly.
    """
    result = loop.start(session_id)
    for _ in range(max_calls):
        if result.session_status in _TERMINAL_STATUSES:
            return result.session_status
        result = loop.iterate_once(session_id)
    return result.session_status


def _assert_answer_fully_supported(conn, answer):
    """Every answer statement is a supported claim; every citation's
    evidence exists and validates right now — the answer contains NO
    unsupported claim and NO invented provenance."""
    assert answer.statements, "answer must contain at least one statement"
    for stmt in answer.statements:
        assert stmt.claim_id, "statement must carry a claim_id"
        claim = store.get_claim(conn, stmt.claim_id)
        assert claim is not None, f"statement {stmt.claim_id} has no claim row"
        assert claim.status == "supported", (
            f"answer contains non-supported claim {stmt.claim_id} "
            f"(status={claim.status!r})"
        )
        assert stmt.citations, f"statement {stmt.claim_id} has no citation"
        for cit in stmt.citations:
            evidence = c1store.get_evidence(conn, cit.evidence_id)
            assert evidence is not None, (
                f"citation {cit.evidence_id} has no evidence row"
            )
            result = c1validation.validate_evidence(conn, cit.evidence_id)
            assert result.valid, (
                f"citation {cit.evidence_id} is invalid: {result.reason}"
            )


def _ingest_and_run(pdf_bytes: bytes, db_path: Path) -> tuple[str, str]:
    """Ingest a PDF and drive a session over it to a terminal status.

    Returns ``(session_id, terminal_status)``. The session binds the
    ingested document and runs the real C2 loop (deterministic reasoner).
    """
    pdf = db_path.with_name(db_path.stem + ".pdf")
    pdf.write_bytes(pdf_bytes)
    ingested = c0ingest.ingest_pdf(pdf, title="Paper", db_path=db_path)
    assert ingested.status == "ready"
    assert ingested.chunks_created > 0

    conn = store.connect(db_path)
    try:
        session = store.create_session(conn, question=QUESTION)
        store.add_session_document(
            conn, session_id=session.session_id, document_id=ingested.document_id
        )
    finally:
        conn.close()

    loop = ResearchLoop(db_path=db_path)
    terminal = _run_to_terminal(loop, session.session_id)
    return session.session_id, terminal


def test_scenario_a_happy_path_deterministic(tmp_path):
    db_path = tmp_path / "docpipe.db"
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(build_pdf(CHUNK_TEXTS))

    # C0 ingest: document ready, exactly the two fixture chunks.
    ingested = c0ingest.ingest_pdf(pdf, title="Paper", db_path=db_path)
    assert ingested.status == "ready"
    assert ingested.chunks_created == 2

    # Research session + bind document.
    conn = store.connect(db_path)
    try:
        session = store.create_session(conn, question=QUESTION)
        store.add_session_document(
            conn, session_id=session.session_id, document_id=ingested.document_id
        )
    finally:
        conn.close()

    # Loop start -> iterate -> terminal (stop condition A: synthesis ready).
    loop = ResearchLoop(db_path=db_path)
    terminal = _run_to_terminal(loop, session.session_id)
    assert terminal == "complete"

    # Claim supported with real, validated evidence provenance.
    conn = store.connect(db_path)
    try:
        claims_list = store.list_claims(conn, session.session_id)
        supported = [c for c in claims_list if c.status == "supported"]
        assert supported, "expected at least one supported claim"
        claim = supported[0]
        links = store.get_links_for_claim(conn, claim.claim_id)
        assert links and links[0].role == "supporting"

        # Synthesis -> answer; citation points at the REAL C1 evidence.
        answer = synthesize(conn, session.session_id)
        assert len(answer.statements) == 1
        stmt = answer.statements[0]
        assert stmt.claim_id == claim.claim_id
        assert stmt.citations
        cit = stmt.citations[0]
        assert cit.evidence_id == links[0].evidence_id
        evidence = c1store.get_evidence(conn, cit.evidence_id)
        assert evidence is not None
        assert cit.document_id == evidence.document_id == ingested.document_id
        assert cit.page == evidence.page_number
        assert cit.locator == evidence.locator
        assert cit.locator["chunk_id"] == evidence.chunk_id

        # No unsupported claim in the answer.
        _assert_answer_fully_supported(conn, answer)

        # Persisted; re-synthesis is idempotent (same JSON, no rewrite).
        answer_json = answer.to_json()
        second = synthesize(conn, session.session_id)
        assert second.to_json() == answer_json
    finally:
        conn.close()

    # final_answer is durable in SQLite: reload via a brand-new
    # connection returns the identical answer.
    conn = store.connect(db_path)
    try:
        reloaded = store.get_session(conn, session.session_id)
        assert reloaded is not None
        assert reloaded.final_answer == answer_json
        assert load_answer(conn, session.session_id).to_json() == answer_json
    finally:
        conn.close()


def test_scenario_a_reproducible_same_input_same_answer(tmp_path):
    """Two independent runs over the same PDF bytes produce the same
    answer JSON (deterministic pipeline end to end)."""
    pdf_bytes = build_pdf(CHUNK_TEXTS)
    answers = []
    for run in range(2):
        db_path = tmp_path / f"run{run}.db"
        session_id, terminal = _ingest_and_run(pdf_bytes, db_path)
        assert terminal == "complete"
        conn = store.connect(db_path)
        try:
            answers.append(synthesize(conn, session_id).to_json())
        finally:
            conn.close()
    assert answers[0] == answers[1]


@pytest.mark.skipif(
    not REAL_PDF.is_file(),
    reason="real paper not present on this machine (user-specific path)",
)
def test_scenario_a_real_paper(tmp_path):
    """End-to-end over the actual 'Attention Is All You Need' PDF.

    Only runs when the paper file exists on this machine; the guard makes
    the suite machine-independent (skips elsewhere, never fails).
    """
    db_path = tmp_path / "docpipe.db"
    ingested = c0ingest.ingest_pdf(
        REAL_PDF, title="Attention Is All You Need", db_path=db_path
    )
    assert ingested.status == "ready"
    assert ingested.chunks_created > 0

    conn = store.connect(db_path)
    try:
        session = store.create_session(conn, question=REAL_QUESTION)
        store.add_session_document(
            conn, session_id=session.session_id, document_id=ingested.document_id
        )
    finally:
        conn.close()

    loop = ResearchLoop(db_path=db_path)
    terminal = _run_to_terminal(loop, session.session_id)
    assert terminal in ("complete", "complete_with_gaps")

    conn = store.connect(db_path)
    try:
        answer = synthesize(conn, session.session_id)
        assert len(answer.statements) >= 1
        for stmt in answer.statements:
            assert stmt.claim_id
            for cit in stmt.citations:
                assert cit.evidence_id.startswith("E-")
                assert cit.document_id == ingested.document_id
        # Same no-unsupported-claim guarantee as the deterministic path.
        _assert_answer_fully_supported(conn, answer)
    finally:
        conn.close()
