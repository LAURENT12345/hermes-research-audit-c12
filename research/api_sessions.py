"""C2 HTTP API — research sessions (FastAPI router + combined app).

Endpoints (mounted on the same app as the C0/C1 API):

* ``POST /research/sessions``                      create a draft session bound to C0 documents
* ``GET  /research/sessions/{session_id}``         session detail + sufficiency snapshot
* ``POST /research/sessions/{session_id}/start``   bind docs (optional) + run the loop
* ``POST /research/sessions/{session_id}/iterate`` run exactly one more iteration
* ``POST /research/sessions/{session_id}/abort``   abort an active session

Thin orchestration: this module NEVER writes lifecycle state directly —
no ``store.update_session_status``, no claim writes. Every transition
goes through ``ResearchLoop``; reads use the store directly. Error
responses follow the C0/C1 convention: ``{"error": "..."}`` with the
appropriate status code; pydantic request-validation failures are
normalized to ``422 {"error": "invalid request body"}`` (handler
registered by the C1 app).
"""

from __future__ import annotations

import dataclasses
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import docpipe.db as c0db
import evidence.api as c1api

from research import __version__, config, store, sufficiency
from research.loop import (
    InvalidSessionTransitionError,
    LoopError,
    NoDocumentsError,
    ResearchLoop,
    SessionNotActiveError,
    SessionNotFoundError,
)
from research.model import is_valid_session_id

logger = logging.getLogger(__name__)


class CreateSessionRequest(BaseModel):
    """Body for ``POST /research/sessions``."""

    question: str
    document_ids: list[int]
    min_supporting: Optional[int] = None
    max_iterations: Optional[int] = None


class StartSessionRequest(BaseModel):
    """Body for ``POST /research/sessions/{session_id}/start``."""

    document_ids: Optional[list[int]] = None
    iterations: Optional[int] = None


def _db_error_response(exc: Exception) -> JSONResponse:
    logger.error("research database failure: %s", exc)
    return JSONResponse(status_code=500, content={"error": "database failure"})


def _current_iteration(conn, session_id: str) -> int:
    """The highest iteration the loop has started (0 for a fresh draft)."""
    rows = store.list_iterations(conn, session_id)
    return max((row.iteration for row in rows), default=0)


def _sufficiency_dict(result) -> dict:
    return {
        "total_claims": result.total_claims,
        "supported_count": result.supported_count,
        "insufficient_count": result.insufficient_count,
        "contradicted_count": result.contradicted_count,
        "unresolved_count": result.unresolved_count,
        "has_gaps": result.has_gaps,
        "synthesis_ready": result.synthesis_ready,
        "needs_more_retrieval": result.needs_more_retrieval,
        "gaps": result.gaps,
    }


def _session_payload(conn, session_id: str) -> dict:
    """The enriched session dict returned by ``GET`` (caller checked exists)."""
    session = store.get_session(conn, session_id)
    assert session is not None  # callers check existence first
    suff = sufficiency.check_session_sufficiency(conn, session_id)
    return {
        **session.to_dict(),
        "document_ids": store.list_session_documents(conn, session_id),
        "current_iteration": _current_iteration(conn, session_id),
        "answer_available": session.final_answer is not None,
        "sufficiency": _sufficiency_dict(suff),
    }


def _require_documents_exist(
    conn, document_ids: list[int]
) -> Optional[JSONResponse]:
    """404 response if any C0 document is missing, else None."""
    for document_id in document_ids:
        if c0db.get_document(conn, document_id) is None:
            return JSONResponse(
                status_code=404, content={"error": "document not found"}
            )
    return None


def build_research_router(db_path: Path | str | None = None) -> APIRouter:
    """The C2 router; handlers share the app's ``db_path``."""
    router = APIRouter()

    @router.post("/research/sessions")
    def create_session_endpoint(payload: CreateSessionRequest) -> dict:
        question = (payload.question or "").strip()
        if not question:
            return JSONResponse(
                status_code=422, content={"error": "question must not be empty"}
            )
        if not payload.document_ids:
            return JSONResponse(
                status_code=422,
                content={"error": "document_ids must not be empty"},
            )
        try:
            conn = store.connect(db_path)
            try:
                missing = _require_documents_exist(conn, payload.document_ids)
                if missing is not None:
                    return missing
                session = store.create_session(
                    conn,
                    question=question,
                    min_supporting=(
                        payload.min_supporting
                        if payload.min_supporting is not None
                        else config.DEFAULT_MIN_SUPPORTING
                    ),
                    max_iterations=(
                        payload.max_iterations
                        if payload.max_iterations is not None
                        else config.DEFAULT_MAX_ITERATIONS
                    ),
                )
                for document_id in payload.document_ids:
                    store.add_session_document(
                        conn,
                        session_id=session.session_id,
                        document_id=document_id,
                    )
                document_ids = store.list_session_documents(
                    conn, session.session_id
                )
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        # A created session is a DRAFT: never auto-started.
        return {"session": {**session.to_dict(), "document_ids": document_ids}}

    @router.get("/research/sessions/{session_id}")
    def get_session_endpoint(session_id: str) -> dict:
        if not is_valid_session_id(session_id):
            return JSONResponse(
                status_code=400, content={"error": "invalid session ID"}
            )
        try:
            conn = store.connect(db_path)
            try:
                if store.get_session(conn, session_id) is None:
                    return JSONResponse(
                        status_code=404, content={"error": "session not found"}
                    )
                payload = _session_payload(conn, session_id)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        return {"session": payload}

    @router.post("/research/sessions/{session_id}/start")
    def start_session_endpoint(
        session_id: str,
        payload: Optional[StartSessionRequest] = Body(default=None),
    ) -> dict:
        if not is_valid_session_id(session_id):
            return JSONResponse(
                status_code=400, content={"error": "invalid session ID"}
            )
        try:
            if payload is not None and payload.document_ids:
                conn = store.connect(db_path)
                try:
                    missing = _require_documents_exist(
                        conn, payload.document_ids
                    )
                    if missing is not None:
                        return missing
                finally:
                    conn.close()
            result = ResearchLoop(db_path=db_path).start(
                session_id,
                document_ids=payload.document_ids if payload else None,
                iterations=(
                    payload.iterations
                    if payload is not None and payload.iterations is not None
                    else config.ITERATIONS_PER_REQUEST
                ),
            )
        except SessionNotFoundError:
            return JSONResponse(
                status_code=404, content={"error": "session not found"}
            )
        except InvalidSessionTransitionError:
            return JSONResponse(
                status_code=409, content={"error": "invalid session transition"}
            )
        except NoDocumentsError:
            return JSONResponse(
                status_code=422, content={"error": "session has no documents"}
            )
        except LoopError:
            return JSONResponse(
                status_code=500, content={"error": "research loop failure"}
            )
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        return dataclasses.asdict(result)

    @router.post("/research/sessions/{session_id}/iterate")
    def iterate_session_endpoint(session_id: str) -> dict:
        if not is_valid_session_id(session_id):
            return JSONResponse(
                status_code=400, content={"error": "invalid session ID"}
            )
        try:
            result = ResearchLoop(db_path=db_path).iterate_once(session_id)
        except SessionNotFoundError:
            return JSONResponse(
                status_code=404, content={"error": "session not found"}
            )
        except SessionNotActiveError:
            return JSONResponse(
                status_code=409, content={"error": "session not active"}
            )
        except LoopError:
            return JSONResponse(
                status_code=500, content={"error": "research loop failure"}
            )
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        return dataclasses.asdict(result)

    @router.post("/research/sessions/{session_id}/abort")
    def abort_session_endpoint(session_id: str) -> dict:
        if not is_valid_session_id(session_id):
            return JSONResponse(
                status_code=400, content={"error": "invalid session ID"}
            )
        try:
            ResearchLoop(db_path=db_path).abort(session_id)
        except SessionNotFoundError:
            return JSONResponse(
                status_code=404, content={"error": "session not found"}
            )
        except InvalidSessionTransitionError:
            return JSONResponse(
                status_code=409, content={"error": "invalid session transition"}
            )
        except LoopError:
            return JSONResponse(
                status_code=500, content={"error": "research loop failure"}
            )
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        return {"session_id": session_id, "status": "aborted"}

    return router


def create_app(db_path: Path | str | None = None) -> FastAPI:
    """Build the combined C0+C1+C2 FastAPI application.

    ``db_path`` overrides the default ``<HERMES_HOME>/docpipe.db``
    (used by tests and unusual deployments).
    """
    app = c1api.create_app(db_path)
    app.title = "Hermes C0+C1+C2 — Document, Evidence & Research Pipelines"
    app.description = (
        "Local PDF -> chunks -> SQLite + FTS5 -> Evidence -> research "
        "sessions. LLM-free."
    )
    app.version = __version__
    app.include_router(build_research_router(db_path))
    return app
