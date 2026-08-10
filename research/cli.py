"""C2 research CLI — ``hermes research ...`` / ``python -m research.cli``.

Thin interface layer over the research domain. It NEVER writes SQLite
state directly: session lifecycle goes through ``ResearchLoop``, claims
inspection through ``store`` reads, answers through ``synthesis``.
Domain exceptions become exit code 1 with a short ``error: ...`` message
(no tracebacks by default); ``--json`` makes every output machine-
readable JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import docpipe.db as c0db
import docpipe.config as c0config
from research import store, sufficiency
from research.loop import (
    InvalidSessionTransitionError,
    LoopError,
    NoDocumentsError,
    ResearchLoop,
    SessionNotActiveError,
    SessionNotFoundError,
)
from research.synthesis import SynthesisError, SynthesisNotAllowedError, synthesize


def _emit(args, payload: Any) -> None:
    """Print a payload as JSON (--json) or human text."""
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(payload)


def _error(args, message: str, code: int = 1) -> int:
    """Print an error (JSON when requested) and return the exit code."""
    if getattr(args, "json", False):
        print(json.dumps({"error": message}, ensure_ascii=False))
    else:
        print(f"error: {message}", file=sys.stderr)
    return code


def _db_path_from_args(args) -> Optional[Path]:
    return Path(args.db_path) if getattr(args, "db_path", None) else None


def _session_payload(conn, session) -> dict:
    iterations = store.list_iterations(conn, session.session_id)
    current = max((r.iteration for r in iterations), default=0)
    try:
        suf = sufficiency.check_session_sufficiency(conn, session.session_id)
        sufficiency_payload = {
            "total_claims": suf.total_claims,
            "supported_count": suf.supported_count,
            "insufficient_count": suf.insufficient_count,
            "contradicted_count": suf.contradicted_count,
            "unresolved_count": suf.unresolved_count,
            "has_gaps": suf.has_gaps,
            "synthesis_ready": suf.synthesis_ready,
            "needs_more_retrieval": suf.needs_more_retrieval,
        }
    except ValueError:
        sufficiency_payload = {}
    return {
        **session.to_dict(),
        "document_ids": store.list_session_documents(conn, session.session_id),
        "current_iteration": current,
        "answer_available": session.final_answer is not None,
        "sufficiency": sufficiency_payload,
    }


def _loop_result_payload(result) -> dict:
    return {
        "session_id": result.session_id,
        "session_status": result.session_status,
        "stop_reason": result.stop_reason,
        "iterations_run": result.iterations_run,
        "new_evidence_count": result.new_evidence_count,
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def cmd_new(args) -> int:
    question = (args.question or "").strip()
    if not question:
        return _error(args, "question must not be empty")
    if not args.document_id:
        return _error(args, "at least one --document-id is required")
    db_path = _db_path_from_args(args)
    conn = store.connect(db_path)
    try:
        for doc_id in args.document_id:
            if c0db.get_document(conn, int(doc_id)) is None:
                return _error(args, f"document {doc_id} not found")
        session = store.create_session(conn, question=question)
        for doc_id in args.document_id:
            store.add_session_document(
                conn, session_id=session.session_id, document_id=int(doc_id)
            )
    except ValueError as exc:
        return _error(args, str(exc))
    finally:
        conn.close()
    if getattr(args, "json", False):
        _emit(args, {"session_id": session.session_id})
    else:
        print(session.session_id)
    return 0


def cmd_list(args) -> int:
    conn = store.connect(_db_path_from_args(args))
    try:
        rows = []
        for session in store.list_sessions(conn):
            payload = _session_payload(conn, session)
            rows.append(
                {
                    "session_id": payload["session_id"],
                    "question": payload["question"],
                    "status": payload["status"],
                    "current_iteration": payload["current_iteration"],
                    "answer_available": payload["answer_available"],
                    "synthesis_ready": payload["sufficiency"].get(
                        "synthesis_ready"
                    ),
                }
            )
    finally:
        conn.close()
    if getattr(args, "json", False):
        _emit(args, {"sessions": rows})
    else:
        for row in rows:
            ready = "yes" if row["answer_available"] else "no"
            print(
                f"{row['session_id']}  {row['status']:18s}  iter={row['current_iteration']}  "
                f"answer={ready}  {row['question'][:60]}"
            )
    return 0


def cmd_show(args) -> int:
    conn = store.connect(_db_path_from_args(args))
    try:
        session = store.get_session(conn, args.session_id)
        if session is None:
            return _error(args, f"session {args.session_id} not found")
        payload = _session_payload(conn, session)
    finally:
        conn.close()
    if getattr(args, "json", False):
        _emit(args, payload)
        return 0
    print(f"session_id:        {payload['session_id']}")
    print(f"question:          {payload['question']}")
    print(f"status:            {payload['status']}")
    print(f"current_iteration: {payload['current_iteration']}")
    print(f"documents:         {payload['document_ids']}")
    print(f"answer_available:  {payload['answer_available']}")
    suf = payload["sufficiency"]
    if suf:
        print(
            f"sufficiency:       supported={suf.get('supported_count')} "
            f"insufficient={suf.get('insufficient_count')} "
            f"contradicted={suf.get('contradicted_count')} "
            f"has_gaps={suf.get('has_gaps')}"
        )
    return 0


def cmd_start(args) -> int:
    return _run_loop(args, lambda loop: loop.start(args.session_id))


def cmd_iterate(args) -> int:
    return _run_loop(args, lambda loop: loop.iterate_once(args.session_id))


def cmd_abort(args) -> int:
    try:
        ResearchLoop(db_path=_db_path_from_args(args)).abort(args.session_id)
    except (SessionNotFoundError, InvalidSessionTransitionError) as exc:
        return _error(args, str(exc))
    if getattr(args, "json", False):
        _emit(args, {"session_id": args.session_id, "status": "aborted"})
    else:
        print("aborted")
    return 0


def _run_loop(args, fn) -> int:
    try:
        result = fn(ResearchLoop(db_path=_db_path_from_args(args)))
    except (SessionNotFoundError, SessionNotActiveError,
            InvalidSessionTransitionError, NoDocumentsError, LoopError) as exc:
        return _error(args, str(exc))
    if getattr(args, "json", False):
        _emit(args, _loop_result_payload(result))
    else:
        stop = f" ({result.stop_reason})" if result.stop_reason else ""
        print(
            f"{result.session_id} -> {result.session_status}{stop} | "
            f"{result.iterations_run} iteration(s), "
            f"{result.new_evidence_count} new evidence"
        )
    return 0


def cmd_claims(args) -> int:
    conn = store.connect(_db_path_from_args(args))
    try:
        if store.get_session(conn, args.session_id) is None:
            return _error(args, f"session {args.session_id} not found")
        claims = [c.to_dict() for c in store.list_claims(conn, args.session_id)]
    finally:
        conn.close()
    if getattr(args, "json", False):
        _emit(args, {"session_id": args.session_id, "claims": claims})
    else:
        for claim in claims:
            notes = f"  notes={claim.get('notes')}" if claim.get("notes") else ""
            print(
                f"{claim['claim_id']}  {claim['status']:12s}  "
                f"{claim['statement'][:70]}{notes}"
            )
    return 0


def cmd_answer(args) -> int:
    from research.synthesis import get_answer

    refresh = not getattr(args, "no_refresh", False)
    try:
        conn = store.connect(_db_path_from_args(args))
        try:
            answer = get_answer(conn, args.session_id, refresh=refresh)
        finally:
            conn.close()
    except (SynthesisError, SynthesisNotAllowedError) as exc:
        return _error(args, str(exc))

    payload = {
        "answer": answer.to_dict(),
        "refreshed": refresh,
        "stale_possible": not refresh,
    }
    if getattr(args, "json", False):
        _emit(args, payload)
    else:
        if not refresh:
            print("note: answer may be stale (--no-refresh)")
        print(f"summary: {answer.summary}")
        for stmt in answer.statements:
            citations = ", ".join(c.evidence_id for c in stmt.citations)
            print(f"- {stmt.statement[:80]}  [{citations}]")
        for gap in answer.gaps:
            print(f"  gap: {gap.statement[:60]} ({gap.reason})")
    return 0


def cmd_synthesize(args) -> int:
    try:
        conn = store.connect(_db_path_from_args(args))
        try:
            before = store.get_session(conn, args.session_id).final_answer
            answer = synthesize(conn, args.session_id)
            after = store.get_session(conn, args.session_id).final_answer
        finally:
            conn.close()
    except (SynthesisError, SynthesisNotAllowedError) as exc:
        return _error(args, str(exc))
    payload = {
        "session_id": args.session_id,
        "persisted": before != after,
        "claim_count": len(answer.statements),
        "citation_count": sum(len(s.citations) for s in answer.statements),
        "gap_count": len(answer.gaps),
    }
    if getattr(args, "json", False):
        _emit(args, payload)
    else:
        persisted = "yes" if payload["persisted"] else "no"
        print(
            f"persisted: {persisted} | {payload['claim_count']} statement(s), "
            f"{payload['citation_count']} citation(s), {payload['gap_count']} gap(s)"
        )
    return 0


def cmd_db_path(args) -> int:
    path = str(c0config.default_db_path())
    if getattr(args, "json", False):
        _emit(args, {"db_path": path})
    else:
        print(path)
    return 0


_HANDLERS = {
    "new": cmd_new,
    "list": cmd_list,
    "show": cmd_show,
    "start": cmd_start,
    "iterate": cmd_iterate,
    "abort": cmd_abort,
    "claims": cmd_claims,
    "answer": cmd_answer,
    "synthesize": cmd_synthesize,
    "db-path": cmd_db_path,
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def add_research_subcommands(parser: argparse.ArgumentParser) -> None:
    """Add the research subcommands directly onto ``parser``."""
    sub = parser.add_subparsers(dest="research_command", required=True)

    p_new = sub.add_parser("new", help="Create a research session")
    p_new.add_argument("--question", required=True)
    p_new.add_argument("--document-id", action="append", type=int, required=True,
                       help="C0 document id (repeatable)")
    p_new.set_defaults(handler=cmd_new)

    sub.add_parser("list", help="List research sessions").set_defaults(handler=cmd_list)

    p_show = sub.add_parser("show", help="Show one research session")
    p_show.add_argument("session_id")
    p_show.set_defaults(handler=cmd_show)

    p_start = sub.add_parser("start", help="Start/resume a session")
    p_start.add_argument("session_id")
    p_start.set_defaults(handler=cmd_start)

    p_iterate = sub.add_parser("iterate", help="Run one iteration")
    p_iterate.add_argument("session_id")
    p_iterate.set_defaults(handler=cmd_iterate)

    p_abort = sub.add_parser("abort", help="Abort an active session")
    p_abort.add_argument("session_id")
    p_abort.set_defaults(handler=cmd_abort)

    p_claims = sub.add_parser("claims", help="List a session's claims")
    p_claims.add_argument("session_id")
    p_claims.set_defaults(handler=cmd_claims)

    p_answer = sub.add_parser("answer", help="Retrieve the research answer")
    p_answer.add_argument("session_id")
    p_answer.add_argument("--no-refresh", action="store_true",
                          help="return the stored answer without re-validating "
                          "(result may be stale)")
    p_answer.set_defaults(handler=cmd_answer)

    p_synth = sub.add_parser("synthesize", help="Run synthesis")
    p_synth.add_argument("session_id")
    p_synth.set_defaults(handler=cmd_synthesize)

    sub.add_parser("db-path", help="Print the resolved SQLite path").set_defaults(
        handler=cmd_db_path
    )


def build_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the ``research`` subcommand (docpipe-style pattern)."""
    parser = subparsers.add_parser(
        "research",
        help="C2 research engine: sessions, claims, loop, answer (LLM-free)",
        description="Deterministic research sessions over local documents: "
        "create sessions, run the research loop, inspect claims and "
        "answers. Thin adapter over the research domain.",
    )
    parser.add_argument(
        "--json", action="store_true", help="machine-readable JSON output"
    )
    parser.add_argument(
        "--db", dest="db_path", default=None,
        help="override SQLite path (default: <HERMES_HOME>/docpipe.db)",
    )
    add_research_subcommands(parser)
    return parser


def research_command(args) -> int:
    """Dispatch entry used by ``hermes_cli.main`` (docpipe-style)."""
    handler = getattr(args, "handler", None)
    if handler is None:
        print("unknown research subcommand", file=sys.stderr)
        return 2
    return int(handler(args) or 0)


def main(argv: Optional[list[str]] = None) -> int:
    """Standalone entry: ``python -m research.cli`` (subcommands top-level)."""
    parser = argparse.ArgumentParser(
        prog="research",
        description="Hermes C2 research engine CLI (LLM-free).",
    )
    parser.add_argument(
        "--json", action="store_true", help="machine-readable JSON output"
    )
    parser.add_argument(
        "--db", dest="db_path", default=None,
        help="override SQLite path (default: <HERMES_HOME>/docpipe.db)",
    )
    add_research_subcommands(parser)
    args = parser.parse_args(argv)
    return research_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
