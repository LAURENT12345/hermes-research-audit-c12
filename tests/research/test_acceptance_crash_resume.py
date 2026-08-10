"""Acceptance: crash/resume recovery (Scenario C) and tamper/re-validation
(Scenario B) at system level, using the real C0/C1/C2 pipelines.

Scenario C — restart/resume:

* a REAL subprocess dies (``os._exit(1)``) inside ``BEGIN IMMEDIATE`` —
  SQLite's rollback-on-close leaves no partial write, and the session
  resumes through the actual ``ResearchLoop``;
* phases already marked ``completed`` are never re-executed (phase rows +
  ``phase_started`` events prove only the remaining phases run);
* a phase row left ``running`` by a crash is re-executed, and the retry is
  idempotent — no duplicate claim / link / evidence;
* audit events are not duplicated across a crash/retry: the counts of
  ``claim_status_changed`` / ``evidence_linked`` before and after resume
  prove it;
* identical crash+resume input produces identical session state and a
  byte-identical answer.

Scenario B — tamper: a supported claim whose chunk is modified in SQLite
is demoted to ``insufficient`` by the loop's own evaluating phase
(``evidence_invalid``), the statement disappears from synthesis; restoring
the chunk and re-evaluating restores the claim and the statement, with the
answer byte-identical to the pre-tamper answer.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import pytest

import docpipe.ingest as c0ingest
import evidence.store as c1store
from research import claims, store
from research.loop import PHASE_ORDER, ResearchLoop
from research.synthesis import synthesize

from tests.docpipe.pdf_factory import build_pdf

CHUNK_TEXTS = [
    "Scaled dot product attention divides by the square root of dk.",
]
QUESTION = "Why does scaled dot product attention divides by the square root of dk?"
TERMINAL_STATUSES = ("complete", "complete_with_gaps", "failed", "aborted")


@pytest.fixture
def env(tmp_path):
    """One real PDF ingested through C0 (no mocks anywhere)."""
    db_path = tmp_path / "docpipe.db"
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(build_pdf(CHUNK_TEXTS))
    ingested = c0ingest.ingest_pdf(pdf, db_path=db_path)
    return db_path, ingested.document_id


def _session(env, question=QUESTION):
    """A draft session bound to the ingested document."""
    db_path, doc_id = env
    conn = store.connect(db_path)
    session = store.create_session(conn, question=question)
    store.add_session_document(
        conn, session_id=session.session_id, document_id=doc_id
    )
    conn.close()
    return db_path, session.session_id


def _event_types(conn, session_id):
    return [e.event_type for e in store.list_events(conn, session_id)]


def _details(conn, session_id, event_type):
    return [
        json.loads(e.detail) if e.detail else {}
        for e in store.list_events(conn, session_id)
        if e.event_type == event_type
    ]


def _run_to_terminal(db_path, session_id, max_calls=10):
    """iterate_once until the session reaches a terminal status."""
    for _ in range(max_calls):
        conn = store.connect(db_path)
        try:
            status = store.get_session(conn, session_id).status
        finally:
            conn.close()
        if status in TERMINAL_STATUSES:
            return status
        ResearchLoop(db_path=db_path).iterate_once(session_id)
    return "not-terminal"


def _first_chunk(conn, doc_id):
    return conn.execute(
        "SELECT id, content, page_number FROM chunks "
        "WHERE document_id = ? ORDER BY chunk_index LIMIT 1",
        (doc_id,),
    ).fetchone()


def _simulate_crash_state(
    conn,
    session_id,
    doc_id,
    *,
    crash_in_claiming: bool,
    partial_side_effects: bool,
):
    """Persist exactly what a real crash mid-iteration-1 would leave.

    planning/retrieving/evidencing are ``completed`` with their side
    effects (query executed, evidence created and recorded in the
    iteration's persisted evidence set). When ``crash_in_claiming`` the
    claiming row exists with status ``running``; when
    ``partial_side_effects`` the claim + link that claiming had already
    written are present too (the loop's claiming is multi-transaction, so
    a crash can leave a partially-claimed iteration).

    Returns ``(evidence_id, claim_or_None)``.
    """
    chunk = _first_chunk(conn, doc_id)
    store.add_query(
        conn, session_id=session_id, query_text="scaled dot product attention"
    )
    query = store.get_queries(conn, session_id)[0]
    store.set_query_status(conn, query.id, "executed")
    evidence_id = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk["id"], evidence_type="chunk",
        source_type="local_pdf", source_title="Paper", content=chunk["content"],
        page_number=chunk["page_number"], section_path=None,
        locator={"page": chunk["page_number"], "chunk_id": chunk["id"]},
    ).evidence_id
    store.add_iteration_evidence(
        conn, session_id=session_id, iteration=1, evidence_id=evidence_id
    )
    store.save_iteration_known(
        conn, session_id=session_id, iteration=1, known_evidence_ids=[]
    )
    for phase in ("planning", "retrieving", "evidencing"):
        row = store.ensure_iteration(
            conn, session_id=session_id, iteration=1, phase=phase
        )
        store.complete_iteration(conn, row.id, status="completed")
    claim = None
    if crash_in_claiming:
        row = store.ensure_iteration(
            conn, session_id=session_id, iteration=1, phase="claiming"
        )
        assert row.status == "running"  # the crash point
    if partial_side_effects:
        claim = claims.create_claim(
            conn, session_id=session_id, statement=chunk["content"]
        )
        claims.link_evidence(
            conn, claim_id=claim.claim_id, evidence_id=evidence_id,
            role="supporting",
        )
    return evidence_id, claim


def _tamper_chunk(db_path, chunk_id, content):
    """Tamper the C0 chunk DIRECTLY in SQLite (outside the domain layer)."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("UPDATE chunks SET content = ? WHERE id = ?", (content, chunk_id))
        conn.commit()
    finally:
        conn.close()


def _counts(conn, session_id):
    return {
        "queries": len(store.get_queries(conn, session_id)),
        "claims": len(store.list_claims(conn, session_id)),
        "links": sum(
            len(store.get_links_for_claim(conn, c.claim_id))
            for c in store.list_claims(conn, session_id)
        ),
        "evidence": c1store.count_evidence(conn),
    }


def _event_counts(conn, session_id):
    return {
        "claim_proposed": _event_types(conn, session_id).count("claim_proposed"),
        "evidence_linked": _event_types(conn, session_id).count("evidence_linked"),
        "claim_status_changed": _event_types(conn, session_id).count(
            "claim_status_changed"
        ),
    }


# ---------------------------------------------------------------------------
# Scenario C: crash / resume
# ---------------------------------------------------------------------------

def test_resume_after_real_crash_mid_transaction(env):
    """A child dying inside BEGIN IMMEDIATE leaves NO partial write, and
    the session then resumes through the real loop to a terminal state."""
    db_path, _ = env
    db_path, session_id = _session(env)
    conn = store.connect(db_path)
    store.update_session_status(conn, session_id, "active")
    claims.create_claim(conn, session_id=session_id, statement="Crash target")
    conn.close()

    code = (
        "import sqlite3, os\n"
        f"db = {str(db_path)!r}\n"
        "conn = sqlite3.connect(db, isolation_level=None)\n"
        "conn.execute('BEGIN IMMEDIATE')\n"
        "conn.execute(\"UPDATE claims SET status='supported' "
        "WHERE id='C-000001'\")\n"
        "conn.execute(\"INSERT INTO research_events (session_id, event_type) "
        f"VALUES ({session_id!r}, 'crash_marker')\")\n"
        "os._exit(1)  # die WITHOUT commit/rollback\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, timeout=60
    )
    assert proc.returncode == 1  # os._exit(1)

    conn = store.connect(db_path)
    try:
        # No partial write survived the crash.
        assert store.get_claim(conn, "C-000001").status == "proposed"
        assert not [
            e for e in store.list_events(conn, session_id)
            if e.event_type == "crash_marker"
        ]
    finally:
        conn.close()

    # Resume through the REAL loop: iteration 1 runs, then the loop keeps
    # going until the terminal state (the orphan "Crash target" claim is a
    # gap, so the session ends complete_with_gaps — never hangs).
    terminal = _run_to_terminal(db_path, session_id)
    assert terminal == "complete_with_gaps"
    conn = store.connect(db_path)
    try:
        # The crashed claim is still there and was evaluated, not lost.
        assert store.get_claim(conn, "C-000001").status == "insufficient"
        assert len(store.list_claims(conn, session_id)) == 2
    finally:
        conn.close()


def test_completed_phases_not_rerun_on_resume(env):
    """Pre-marked planning/retrieving/evidencing completed stay completed;
    resume runs ONLY claiming/evaluating — phase rows and events prove the
    completed phases never re-executed, and nothing duplicated."""
    db_path, doc_id = env
    db_path, session_id = _session(env)
    conn = store.connect(db_path)
    store.update_session_status(conn, session_id, "active")
    _simulate_crash_state(
        conn, session_id, doc_id,
        crash_in_claiming=False, partial_side_effects=False,
    )
    conn.close()

    result = ResearchLoop(db_path=db_path).iterate_once(session_id)
    assert result.session_status == "complete"

    conn = store.connect(db_path)
    try:
        # Events: only claiming + evaluating ever STARTED.
        started = [
            d["phase"] for d in _details(conn, session_id, "phase_started")
        ]
        assert sorted(started) == ["claiming", "evaluating"]

        # Phase rows: exactly one row per phase (UNIQUE held), the
        # pre-marked ones still completed, the retried ones completed now.
        rows = store.list_iterations(conn, session_id)
        assert len(rows) == len(PHASE_ORDER)
        by_phase = {r.phase: r.status for r in rows}
        assert by_phase == {phase: "completed" for phase in PHASE_ORDER}

        # No duplicate side effects: one query, one evidence, one claim,
        # one link.
        assert _counts(conn, session_id) == {
            "queries": 1, "claims": 1, "links": 1, "evidence": 1,
        }
    finally:
        conn.close()


def test_running_phase_retry_is_idempotent(env):
    """A claiming row left 'running' (crash before any side effect) is
    re-executed on resume — exactly one claim/link/evidence results."""
    db_path, doc_id = env
    db_path, session_id = _session(env)
    conn = store.connect(db_path)
    store.update_session_status(conn, session_id, "active")
    _simulate_crash_state(
        conn, session_id, doc_id,
        crash_in_claiming=True, partial_side_effects=False,
    )
    conn.close()

    result = ResearchLoop(db_path=db_path).iterate_once(session_id)
    assert result.session_status == "complete"

    conn = store.connect(db_path)
    try:
        # The running phase was re-run and completed; the retry created
        # exactly the right rows and no duplicates.
        rows = store.list_iterations(conn, session_id)
        by_phase = {r.phase: r.status for r in rows}
        assert by_phase == {phase: "completed" for phase in PHASE_ORDER}
        assert _counts(conn, session_id) == {
            "queries": 1, "claims": 1, "links": 1, "evidence": 1,
        }
        claim = store.list_claims(conn, session_id)[0]
        assert claim.status == "supported"
        # One of each audit event — the retry did not double-claim.
        assert _event_counts(conn, session_id) == {
            "claim_proposed": 1,
            "evidence_linked": 1,
            "claim_status_changed": 1,
        }
    finally:
        conn.close()


def test_no_duplicate_audit_events_after_crash_retry(env):
    """Crash mid-claiming (claim + link already written) -> resume. The
    claim/evidence_linked counts do NOT grow (idempotent retry) and only
    the genuine proposed->supported transition adds one event."""
    db_path, doc_id = env
    db_path, session_id = _session(env)
    conn = store.connect(db_path)
    store.update_session_status(conn, session_id, "active")
    _, claim = _simulate_crash_state(
        conn, session_id, doc_id,
        crash_in_claiming=True, partial_side_effects=True,
    )
    before = _event_counts(conn, session_id)
    conn.close()
    assert before == {
        "claim_proposed": 1, "evidence_linked": 1, "claim_status_changed": 0,
    }

    result = ResearchLoop(db_path=db_path).iterate_once(session_id)
    assert result.session_status == "complete"

    conn = store.connect(db_path)
    try:
        after = _event_counts(conn, session_id)
        # No duplicate claim/link/audit events from the retried claiming.
        assert after["claim_proposed"] == before["claim_proposed"] == 1
        assert after["evidence_linked"] == before["evidence_linked"] == 1
        # Exactly ONE status transition happened (proposed -> supported);
        # a second resume would add none (terminal session).
        assert after["claim_status_changed"] == before["claim_status_changed"] + 1
        assert len(store.get_links_for_claim(conn, claim.claim_id)) == 1
        assert store.get_claim(conn, claim.claim_id).status == "supported"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Scenario B: tamper / re-validation
# ---------------------------------------------------------------------------

def test_tamper_demotes_excludes_restores_through_loop(env):
    """supported claim -> SQLite chunk tampered -> loop's evaluating phase
    demotes to insufficient (evidence_invalid) -> synthesis excludes the
    statement -> chunk restored -> re-evaluate -> supported again -> answer
    byte-identical to the pre-tamper answer."""
    db_path, _ = env
    db_path, session_id = _session(env)
    conn = store.connect(db_path)
    # A gap claim keeps the session ACTIVE after iteration 1, so the loop
    # can re-run its evaluating phase after the tamper.
    claims.create_claim(
        conn, session_id=session_id,
        statement="A completely unsupported claim about nothing particular.",
    )
    conn.close()

    loop = ResearchLoop(db_path=db_path)
    first = loop.start(session_id)
    assert first.session_status == "active"

    conn = store.connect(db_path)
    supported = [
        c for c in store.list_claims(conn, session_id)
        if c.status == "supported"
    ]
    assert len(supported) == 1
    target = supported[0]
    ev_id = store.get_links_for_claim(conn, target.claim_id)[0].evidence_id
    chunk_id = conn.execute(
        "SELECT chunk_id FROM evidence WHERE id = ?", (ev_id,)
    ).fetchone()["chunk_id"]
    original = conn.execute(
        "SELECT content FROM chunks WHERE id = ?", (chunk_id,)
    ).fetchone()["content"]
    # The answer BEFORE the tamper is the deterministic baseline.
    answer_before = synthesize(conn, session_id)
    assert any(s.claim_id == target.claim_id for s in answer_before.statements)
    conn.close()

    # TAMPER the chunk directly in SQLite.
    _tamper_chunk(db_path, chunk_id, "TAMPERED CONTENT")

    # The loop's evaluating phase re-evaluates: claim demoted.
    loop.iterate_once(session_id)
    conn = store.connect(db_path)
    try:
        demoted = store.get_claim(conn, target.claim_id)
        assert demoted.status == "insufficient"
        demotion = [
            d for d in _details(conn, session_id, "claim_status_changed")
            if d.get("claim_id") == target.claim_id
            and d.get("from") == "supported"
            and d.get("to") == "insufficient"
        ]
        assert demotion, "missing supported -> insufficient evidence_invalid event"
        assert demotion[-1]["reason"] == "evidence_invalid"

        # Synthesis excludes the statement: it moved into gaps.
        answer_tampered = synthesize(conn, session_id)
        assert not any(
            s.claim_id == target.claim_id for s in answer_tampered.statements
        )
        assert any(
            g.claim_id == target.claim_id for g in answer_tampered.gaps
        )
        # The evidence snapshot itself is untouched (C1 invariant).
        assert c1store.get_evidence(conn, ev_id).content == original
    finally:
        conn.close()

    # RESTORE the chunk -> re-evaluate -> supported again.
    _tamper_chunk(db_path, chunk_id, original)
    conn = store.connect(db_path)
    try:
        restored = claims.evaluate_claim(conn, target.claim_id)
        assert restored.status == "supported"
        answer_restored = synthesize(conn, session_id)
        assert any(
            s.claim_id == target.claim_id for s in answer_restored.statements
        )
        # Deterministic: byte-identical to the pre-tamper answer.
        assert answer_restored.to_json() == answer_before.to_json()
        # Session state is fully deterministic after the whole dance.
        session = store.get_session(conn, session_id)
        assert session.status == "complete_with_gaps"
        assert session.final_answer == answer_restored.to_json()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Determinism across identical crash+resume runs
# ---------------------------------------------------------------------------

def test_crash_resume_deterministic_across_runs(tmp_path):
    """Two independent DBs taken through the same crash state, the same
    resume and the same synthesis produce identical session state and a
    byte-identical answer."""
    pdf_bytes = build_pdf(CHUNK_TEXTS)
    answers = []
    states = []
    for run in range(2):
        db_path = tmp_path / f"run{run}.db"
        pdf = tmp_path / f"paper{run}.pdf"
        pdf.write_bytes(pdf_bytes)
        ingested = c0ingest.ingest_pdf(pdf, db_path=db_path)

        conn = store.connect(db_path)
        session = store.create_session(conn, question=QUESTION)
        store.add_session_document(
            conn, session_id=session.session_id,
            document_id=ingested.document_id,
        )
        store.update_session_status(conn, session.session_id, "active")
        _simulate_crash_state(
            conn, session.session_id, ingested.document_id,
            crash_in_claiming=True, partial_side_effects=True,
        )
        conn.close()

        loop = ResearchLoop(db_path=db_path)
        resume = loop.iterate_once(session.session_id)
        assert resume.session_status == "complete"
        _run_to_terminal(db_path, session.session_id)

        conn = store.connect(db_path)
        answer = synthesize(conn, session.session_id)
        answers.append(answer.to_json())
        states.append(
            {
                "status": store.get_session(conn, session.session_id).status,
                "claims": sorted(
                    (c.claim_id, c.statement, c.status)
                    for c in store.list_claims(conn, session.session_id)
                ),
                "iterations": sorted(
                    (r.iteration, r.phase, r.status)
                    for r in store.list_iterations(conn, session.session_id)
                ),
                "queries": sorted(
                    (q.query_text, q.status)
                    for q in store.get_queries(conn, session.session_id)
                ),
                "events": _event_types(conn, session.session_id),
            }
        )
        conn.close()

    assert answers[0] == answers[1]
    assert states[0] == states[1]
