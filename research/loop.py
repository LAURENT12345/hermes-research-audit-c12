"""C2 research loop — deterministic, resumable state machine.

The loop ORCHESTRATES the existing C2 components; it never reimplements
search, evidence construction/validation, claim evaluation, or semantic
reasoning:

* planning    -> ``sufficiency.build_plan_inputs`` + ``ReasonerPort.plan_queries``
* retrieving  -> ``LocalRetriever`` (C0 search + C1 evidence creation)
* evidencing  -> consolidate returned evidence ids, count NEW evidence
* claiming    -> ``ReasonerPort.construct_claims`` + ``claims`` (dedupe/link)
* evaluating  -> ``claims.evaluate_claim`` (live C1 validation) +
                 ``contradictions.detect_contradictions``

Phase persistence uses the existing ``research_iterations`` table with
``UNIQUE (session_id, iteration, phase)`` as the idempotency boundary: a
phase whose row has ``status='completed'`` is never executed again. A row
still ``running`` (crash mid-phase) is re-executed — every side effect in
the loop is idempotent (query/claim/link/evidence dedupe), so re-running
a phase cannot duplicate work.

Stop conditions (evaluated in this order, all deterministic):

A ``synthesis_ready``   — >=1 supported claim, no gaps, no open explicit
                          contradiction -> session ``complete`` (NO answer
                          is generated in this commit; Commit 6 owns
                          synthesis)
B ``no_new_evidence``   — the iteration produced zero genuinely NEW
                          evidence (evidence already known to the session
                          via earlier links does not count)
C ``max_iterations``    — current iteration >= session.max_iterations and
                          not sufficient -> ``complete_with_gaps``

Abort and unrecoverable phase failure terminate the session as
``aborted`` / ``failed``. Terminal sessions are never silently re-run.

No LLM is called anywhere in this module; the reasoner is used only as a
proposal source and never writes state.
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass
from typing import Optional, Sequence

from hermes_cli.sqlite_util import write_txn

import docpipe.db as c0db
import evidence.store as c1store

from research import claims, config, contradictions, store, sufficiency
from research.retriever import LocalRetriever, RetrievalError
from research.reasoner import DeterministicReasoner, ReasonerPort

logger = logging.getLogger(__name__)

#: Execution order of the phases inside one iteration.
PHASE_ORDER = (
    "planning",
    "retrieving",
    "evidencing",
    "claiming",
    "evaluating",
)

STOP_SYNTHESIS_READY = "synthesis_ready"
STOP_NO_NEW_EVIDENCE = "no_new_evidence"
STOP_MAX_ITERATIONS = "max_iterations"

#: Session states that can no longer be started/iterated.
_TERMINAL_SESSION_STATUSES = (
    "complete",
    "complete_with_gaps",
    "failed",
    "aborted",
)


class LoopError(Exception):
    """Base class for research-loop failures."""


class SessionNotFoundError(LoopError):
    """The session does not exist."""


class InvalidSessionTransitionError(LoopError):
    """A session lifecycle transition is not allowed."""


class SessionNotActiveError(LoopError):
    """The session must be 'active' for this operation."""


class SessionBusyError(LoopError):
    """Another caller holds the per-session execution lease."""


class NoDocumentsError(LoopError):
    """The session has no bound C0 documents to search."""


class LoopPhaseError(LoopError):
    """A phase failed twice (retry-once exhausted)."""

    def __init__(self, phase: str, error_type: str):
        self.phase = phase
        self.error_type = error_type
        super().__init__(f"phase {phase} failed twice ({error_type})")


@dataclass(frozen=True)
class LoopResult:
    """Deterministic outcome of start/iterate/resume."""

    session_id: str
    session_status: str
    stop_reason: Optional[str]
    iterations_run: int
    new_evidence_count: int


class ResearchLoop:
    """Deterministic, resumable research loop over C0 + C1 + C2."""

    def __init__(
        self,
        db_path=None,
        retriever: Optional[LocalRetriever] = None,
        reasoner: Optional[ReasonerPort] = None,
    ):
        self._db_path = db_path
        self._retriever = retriever or LocalRetriever(db_path)
        self._reasoner = reasoner or DeterministicReasoner()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        session_id: str,
        *,
        document_ids: Optional[Sequence[int]] = None,
        iterations: int = config.ITERATIONS_PER_REQUEST,
    ) -> LoopResult:
        """Start a draft session (or resume an active one).

        ``document_ids`` binds C0 documents to the session on first
        start; a session with no documents cannot search.

        The whole logical run holds the per-session execution lease, so
        concurrent callers on the same session get ``SessionBusyError``
        instead of interleaving phases.
        """
        conn = self._connect()
        token = secrets.token_hex(16)
        try:
            session = self._require_session(conn, session_id)
            if not store.acquire_session_lock(
                conn, session_id=session_id, token=token
            ):
                raise SessionBusyError(
                    f"session {session_id} is busy; another call is running"
                )
            try:
                if session.status == "draft":
                    if document_ids:
                        for doc_id in document_ids:
                            if c0db.get_document(conn, int(doc_id)) is None:
                                raise LoopError(f"document {doc_id} not found in C0")
                            store.add_session_document(
                                conn, session_id=session_id, document_id=int(doc_id)
                            )
                    if not store.list_session_documents(conn, session_id):
                        raise NoDocumentsError(
                            f"session {session_id} has no documents; pass document_ids"
                        )
                    # draft -> active + audit event commit atomically.
                    with write_txn(conn):
                        store.update_session_status(conn, session_id, "active")
                        store.add_event(
                            conn,
                            session_id=session_id,
                            event_type="session_created",
                            detail=json.dumps({"session_id": session_id}),
                        )
                elif session.status == "active":
                    pass  # resume
                else:
                    raise InvalidSessionTransitionError(
                        f"cannot start session {session_id} in state {session.status}"
                    )
                return self._run_iterations(conn, session_id, iterations)
            finally:
                store.release_session_lock(
                    conn, session_id=session_id, token=token
                )
        finally:
            conn.close()

    def iterate_once(self, session_id: str) -> LoopResult:
        """Run exactly one iteration on an active session."""
        return self._iterate(session_id, 1)

    def resume(
        self, session_id: str, *, iterations: int = config.ITERATIONS_PER_REQUEST
    ) -> LoopResult:
        """Continue an active session (alias of iterate with a budget)."""
        return self._iterate(session_id, iterations)

    def abort(self, session_id: str) -> None:
        """Abort an active session; state is preserved, no synthesis.

        Abort is a poison pill: it does not take the execution lease, so
        it can stop a session while another call is mid-run; the running
        loop observes the terminal status and stops at the next
        iteration boundary. The status change and its audit event commit
        atomically; a second concurrent abort cannot duplicate the event.
        """
        conn = self._connect()
        try:
            session = self._require_session(conn, session_id)
            if session.status != "active":
                raise InvalidSessionTransitionError(
                    f"only active sessions can be aborted; session {session_id} "
                    f"is {session.status}"
                )
            with write_txn(conn):
                changed = store.update_session_status(conn, session_id, "aborted")
                if changed:
                    store.add_event(
                        conn,
                        session_id=session_id,
                        event_type="aborted",
                        detail=json.dumps({"session_id": session_id}),
                    )
            logger.info("research session aborted: %s", session_id)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Iteration driver
    # ------------------------------------------------------------------

    def _iterate(self, session_id: str, iterations: int) -> LoopResult:
        conn = self._connect()
        token = secrets.token_hex(16)
        try:
            session = self._require_session(conn, session_id)
            if session.status != "active":
                raise SessionNotActiveError(
                    f"session {session_id} is {session.status}; "
                    "only active sessions iterate"
                )
            if not store.acquire_session_lock(
                conn, session_id=session_id, token=token
            ):
                raise SessionBusyError(
                    f"session {session_id} is busy; another call is running"
                )
            try:
                return self._run_iterations(conn, session_id, iterations)
            finally:
                store.release_session_lock(
                    conn, session_id=session_id, token=token
                )
        finally:
            conn.close()

    def _run_iterations(
        self, conn, session_id: str, count: int
    ) -> LoopResult:
        session = self._require_session(conn, session_id)
        iterations_run = 0
        new_total = 0
        stop_reason: Optional[str] = None

        for _ in range(max(0, int(count))):
            if session.status != "active":
                break
            iteration, _ = self._current_iteration(conn, session_id)
            self._event(
                conn, session_id, "iteration_started",
                {"session_id": session_id, "iteration": iteration},
            )
            try:
                new_count = self._run_iteration(conn, session, iteration)
            except LoopPhaseError as exc:
                self._event(
                    conn, session_id, "iteration_failed",
                    {"session_id": session_id, "iteration": iteration,
                     "phase": exc.phase, "error": exc.error_type},
                )
                # active -> failed + audit event commit atomically.
                with write_txn(conn):
                    store.update_session_status(conn, session_id, "failed")
                    self._event(
                        conn, session_id, "failed",
                        {"session_id": session_id,
                         "reason": f"phase failure in {exc.phase} ({exc.error_type})"},
                    )
                iterations_run += 1
                stop_reason = f"phase_failure:{exc.phase}"
                break

            new_total += new_count
            iterations_run += 1
            self._event(
                conn, session_id, "iteration_completed",
                {"session_id": session_id, "iteration": iteration,
                 "new_evidence_count": new_count},
            )
            session = self._require_session(conn, session_id)
            stop_reason = self._evaluate_stop(conn, session, iteration, new_count)
            if stop_reason is not None:
                self._apply_stop(conn, session, stop_reason)
                break

        final = self._require_session(conn, session_id)
        return LoopResult(
            session_id=session_id,
            session_status=final.status,
            stop_reason=stop_reason,
            iterations_run=iterations_run,
            new_evidence_count=new_total,
        )

    def _run_iteration(self, conn, session, iteration: int) -> int:
        """Run the remaining phases of one iteration; returns NEW evidence.

        Evidence returned by retrieval and the session's known-evidence
        set are PERSISTED (``research_iteration_evidence`` /
        ``research_iteration_state``), so a resume after a crash keeps the
        exact state of the interrupted iteration: claiming and the new-
        evidence count work from the database, never from memory.
        """
        known_before = store.load_iteration_known(
            conn, session.session_id, iteration
        )
        if known_before is None:
            known_before = self._known_evidence_ids(conn, session.session_id)
            store.save_iteration_known(
                conn,
                session_id=session.session_id,
                iteration=iteration,
                known_evidence_ids=sorted(known_before),
            )
        retrieved: list = []

        def _planning() -> None:
            self._planning(conn, session, iteration)

        def _retrieving() -> None:
            self._retrieving(conn, session, iteration, retrieved)

        def _evidencing() -> None:
            # Audit only: record skipped chunks (counting is side-effect
            # free and reads persisted state, see _count_new_evidence).
            self._evidencing_events(conn, session, iteration, retrieved)

        def _claiming() -> None:
            self._claiming(conn, session, iteration)

        def _evaluating() -> None:
            self._evaluating(conn, session, iteration)

        self._run_phase(conn, session, iteration, "planning", _planning)
        self._run_phase(conn, session, iteration, "retrieving", _retrieving)
        self._run_phase(conn, session, iteration, "evidencing", _evidencing)
        new_count = self._count_new_evidence(conn, session, iteration, known_before)
        self._run_phase(conn, session, iteration, "claiming", _claiming)
        self._run_phase(conn, session, iteration, "evaluating", _evaluating)
        return new_count

    # ------------------------------------------------------------------
    # Phase execution with idempotent completion + retry-once
    # ------------------------------------------------------------------

    def _run_phase(self, conn, session, iteration: int, phase: str, fn):
        row = store.ensure_iteration(
            conn, session_id=session.session_id, iteration=iteration, phase=phase
        )
        if row.status == "completed":
            logger.info(
                "phase already completed: session=%s iteration=%s phase=%s",
                session.session_id, iteration, phase,
            )
            return None

        self._event(
            conn, session.session_id, "phase_started",
            {"session_id": session.session_id, "iteration": iteration, "phase": phase},
        )
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - retry-once policy
            try:
                result = fn()
            except Exception as exc2:  # noqa: BLE001 - surfaced as LoopPhaseError
                # Phase failure state + audit event commit atomically.
                with write_txn(conn):
                    store.complete_iteration(
                        conn, row.id, status="failed", notes=type(exc2).__name__
                    )
                    self._event(
                        conn, session.session_id, "phase_failed",
                        {"session_id": session.session_id, "iteration": iteration,
                         "phase": phase, "error": type(exc2).__name__},
                    )
                raise LoopPhaseError(phase=phase, error_type=type(exc2).__name__) from exc2

        # Completion state + audit event commit atomically: a crash cannot
        # mark the phase completed without its phase_completed event.
        with write_txn(conn):
            store.complete_iteration(conn, row.id, status="completed")
            self._event(
                conn, session.session_id, "phase_completed",
                {"session_id": session.session_id, "iteration": iteration, "phase": phase},
            )
        return result

    def _current_iteration(self, conn, session_id: str) -> tuple[int, set[str]]:
        """(iteration to work on, phases already completed in it)."""
        by_iteration: dict[int, dict[str, str]] = {}
        for row in store.list_iterations(conn, session_id):
            by_iteration.setdefault(row.iteration, {})[row.phase] = row.status
        for iteration in sorted(by_iteration):
            statuses = by_iteration[iteration]
            done = all(
                statuses.get(phase) == "completed" for phase in PHASE_ORDER
            )
            if not done:
                completed = {
                    phase for phase in PHASE_ORDER
                    if statuses.get(phase) == "completed"
                }
                return iteration, completed
        next_iteration = max(by_iteration) + 1 if by_iteration else 1
        return next_iteration, set()

    # ------------------------------------------------------------------
    # Individual phases
    # ------------------------------------------------------------------

    def _planning(self, conn, session, iteration: int) -> None:
        """Persist proposed queries (never executes retrieval)."""
        inputs = sufficiency.build_plan_inputs(conn, session.session_id)
        proposals = self._reasoner.plan_queries(inputs.question, inputs.gaps)
        for proposal in proposals:
            store.add_query(
                conn,
                session_id=session.session_id,
                query_text=proposal.query_text,
                origin=proposal.origin,
                iteration=iteration,
            )
        # Persist the known-evidence baseline for this iteration.
        store.save_iteration_known(
            conn,
            session_id=session.session_id,
            iteration=iteration,
            known_evidence_ids=sorted(
                self._known_evidence_ids(conn, session.session_id)
            ),
        )

    def _retrieving(self, conn, session, iteration: int, accumulator: list) -> None:
        """Execute pending queries across the session's documents.

        Retries each (query, document) pair exactly once. A query is
        ``executed`` if at least one document search succeeded, ``failed``
        otherwise. Evidence creation is delegated to LocalRetriever.
        """
        document_ids = store.list_session_documents(conn, session.session_id)
        pending = [
            q for q in store.get_queries(conn, session.session_id)
            if q.status == "pending"
        ]
        for query in pending:
            results = []
            failures = 0
            for doc_id in document_ids:
                try:
                    results.append(
                        self._retriever.retrieve(
                            doc_id, query.query_text,
                            limit=config.SEARCH_LIMIT_PER_QUERY,
                        )
                    )
                except RetrievalError:
                    try:
                        results.append(
                            self._retriever.retrieve(
                                doc_id, query.query_text,
                                limit=config.SEARCH_LIMIT_PER_QUERY,
                            )
                        )
                    except RetrievalError as exc:
                        failures += 1
                        self._event(
                            conn, session.session_id, "doc_retrieval_failed",
                            {"session_id": session.session_id,
                             "query_id": query.id, "document_id": doc_id,
                             "error": type(exc).__name__},
                        )
            if document_ids and failures == len(document_ids):
                store.set_query_status(conn, query.id, "failed")
                self._event(
                    conn, session.session_id, "query_failed",
                    {"session_id": session.session_id, "query_id": query.id,
                     "query_text": query.query_text},
                )
            else:
                store.set_query_status(conn, query.id, "executed")
                self._event(
                    conn, session.session_id, "query_executed",
                    {"session_id": session.session_id, "query_id": query.id,
                     "query_text": query.query_text,
                     "evidence_count": sum(len(r.evidence) for r in results),
                     "skipped_count": sum(len(r.skipped) for r in results)},
                )
            accumulator.extend(results)
            for result in results:
                for evidence in result.evidence:
                    store.add_iteration_evidence(
                        conn,
                        session_id=session.session_id,
                        iteration=iteration,
                        evidence_id=evidence.evidence_id,
                    )

    def _evidencing_events(self, conn, session, iteration: int, retrieved: list) -> None:
        """Audit-only: record skipped chunks. Counting is side-effect free
        (see ``_count_new_evidence``) so it survives phase skipping."""
        for result in retrieved:
            for skip in result.skipped:
                self._event(
                    conn, session.session_id, "evidence_skipped",
                    {"session_id": session.session_id, "query": result.query,
                     "chunk_id": skip.get("chunk_id"),
                     "reason": str(skip.get("reason"))[:200]},
                )

    @staticmethod
    def _count_new_evidence(conn, session, iteration: int, known_before: set) -> int:
        """Genuinely new evidence = evidence returned in this iteration that
        was not already known to the session before it (C1 dedupe results
        are not new). Reads the PERSISTED iteration evidence set, so the
        count is correct even when retrieving/evidencing were completed
        before a resume."""
        iteration_ids = set(
            store.list_iteration_evidence(conn, session.session_id, iteration)
        )
        return len(iteration_ids - known_before)

    def _claiming(self, conn, session, iteration: int) -> None:
        """Propose claims from the iteration's returned evidence; create +
        link (deduped). Evidence ids come from the PERSISTED iteration
        evidence set, so a resumed iteration claims exactly the evidence
        that was retrieved before the crash."""
        evidence_ids = store.list_iteration_evidence(
            conn, session.session_id, iteration
        )
        all_evidence = [
            c1store.get_evidence(conn, evidence_id)
            for evidence_id in evidence_ids
        ]
        all_evidence = [e for e in all_evidence if e is not None]
        if not all_evidence:
            return
        proposals = self._reasoner.construct_claims(all_evidence)
        for proposal in proposals:
            claim = claims.create_claim_from_proposal(
                conn, session_id=session.session_id, proposal=proposal
            )
            for evidence_id in proposal.evidence_ids:
                claims.link_evidence(
                    conn, claim_id=claim.claim_id,
                    evidence_id=evidence_id, role="supporting",
                )

    def _evaluating(self, conn, session, iteration: int) -> None:
        """Live re-evaluation + contradiction detection for the session."""
        for claim in store.list_claims(conn, session.session_id):
            claims.evaluate_claim(conn, claim.claim_id)
        contradictions.detect_contradictions(conn, session.session_id)

    # ------------------------------------------------------------------
    # Stop conditions
    # ------------------------------------------------------------------

    def _evaluate_stop(
        self, conn, session, iteration: int, new_count: int
    ) -> Optional[str]:
        sufficiency = check_session_sufficiency_fresh(conn, session.session_id)
        open_explicit = [
            c for c in store.list_contradictions(
                conn, session.session_id, status="open"
            )
            if c.kind == "explicit"
        ]
        if sufficiency and not open_explicit:
            return STOP_SYNTHESIS_READY
        if new_count == 0:
            return STOP_NO_NEW_EVIDENCE
        if iteration >= session.max_iterations:
            return STOP_MAX_ITERATIONS
        return None

    def _apply_stop(self, conn, session, stop_reason: str) -> None:
        # Authoritative status transition + audit events commit atomically:
        # a crash cannot leave a terminal status without its events.
        if stop_reason == STOP_SYNTHESIS_READY:
            with write_txn(conn):
                store.update_session_status(conn, session.session_id, "complete")
                self._event(
                    conn, session.session_id, "synthesizing_ready",
                    {"session_id": session.session_id},
                )
                self._event(
                    conn, session.session_id, "session_completed",
                    {"session_id": session.session_id, "status": "complete",
                     "stop_reason": stop_reason},
                )
        else:
            with write_txn(conn):
                store.update_session_status(
                    conn, session.session_id, "complete_with_gaps"
                )
                self._event(
                    conn, session.session_id, "session_completed",
                    {"session_id": session.session_id,
                     "status": "complete_with_gaps", "stop_reason": stop_reason},
                )
        logger.info(
            "research session %s -> %s (stop=%s)",
            session.session_id, store.get_session(conn, session.session_id).status,
            stop_reason,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _connect(self):
        return store.connect(self._db_path)

    def _require_session(self, conn, session_id: str):
        session = store.get_session(conn, session_id)
        if session is None:
            raise SessionNotFoundError(f"session {session_id} not found")
        return session

    def _known_evidence_ids(self, conn, session_id: str) -> set[str]:
        """Evidence already linked to this session's claims."""
        rows = conn.execute(
            "SELECT DISTINCT ce.evidence_id FROM claim_evidence ce "
            "JOIN claims c ON c.id = ce.claim_id "
            "WHERE c.session_id = ?",
            (session_id,),
        ).fetchall()
        return {r["evidence_id"] for r in rows}

    def _event(self, conn, session_id: str, event_type: str, detail: dict) -> None:
        store.add_event(
            conn,
            session_id=session_id,
            event_type=event_type,
            detail=json.dumps(detail, ensure_ascii=False),
        )


def check_session_sufficiency_fresh(conn, session_id: str) -> bool:
    """Deterministic 'synthesis ready' check for stop condition A.

    Uses ``sufficiency.check_session_sufficiency`` on the persisted
    statuses (the evaluating phase has just refreshed them) and requires
    at least one supported claim with NO gaps.
    """
    result = sufficiency.check_session_sufficiency(conn, session_id)
    return bool(result.synthesis_ready)
