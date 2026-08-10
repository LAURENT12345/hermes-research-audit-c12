"""C2 HTTP API (FastAPI router + combined app).

The C2 research API is mounted on the same app as C0/C1's. The
sub-routers live in ``research.api_sessions``, ``research.api_claims``
and ``research.api_answer`` (separate commits) and are imported lazily
inside :func:`build_research_router` so this module imports cleanly on
its own.

Error responses follow the C1 convention: ``{"error": "..."}`` with the
appropriate status code. This module owns the *shared* mapping from
research-domain exceptions to those responses (see :data:`_ERROR_MAP`
and :func:`_error_response_for`); request-validation failures are
normalized to ``422 {"error": "invalid request body"}`` by the handler
C1 already registers.

``create_app`` builds the *combined* C0+C1+C2 app: it starts from
``evidence.api.create_app`` (unchanged C0+C1 routes) and includes the
research router.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from evidence import api as c0c1api
from evidence.model import is_valid_evidence_id
from research.claims import ClaimNotFoundError, EvidenceNotFoundError
from research.loop import (
    InvalidSessionTransitionError,
    NoDocumentsError,
    SessionNotActiveError,
    SessionNotFoundError,
)
from research.model import is_valid_claim_id, is_valid_session_id
from research.synthesis import SynthesisError, SynthesisNotAllowedError

logger = logging.getLogger(__name__)


def _error_response(status: int, message: str) -> JSONResponse:
    """C1-convention error body: ``{"error": message}``."""
    return JSONResponse(status_code=status, content={"error": message})


#: Shared C1-convention mapping from research-domain exceptions to
#: ``(HTTP status, error message)``. Order matters: a subclass must
#: precede its base class because the first ``isinstance`` match wins
#: (``SynthesisNotAllowedError`` before ``SynthesisError``).
_ERROR_MAP: tuple[tuple[type[Exception], int, str], ...] = (
    (SessionNotFoundError, 404, "session not found"),
    (ClaimNotFoundError, 404, "claim not found"),
    (EvidenceNotFoundError, 404, "evidence not found"),
    (SessionNotActiveError, 409, "session not active"),
    (InvalidSessionTransitionError, 409, "invalid session transition"),
    (SynthesisNotAllowedError, 409, "synthesis not allowed"),
    (NoDocumentsError, 422, "session has no documents"),
    (SynthesisError, 500, "synthesis failed"),
)


def _error_response_for(exc: Exception) -> JSONResponse | None:
    """Resolve a domain/database exception to an error response.

    Walks :data:`_ERROR_MAP`, then handles the two non-domain classes:
    ``sqlite3.Error`` -> 500 "database failure" (logged) and
    ``ValueError`` -> 400 "invalid input". Returns ``None`` for unknown
    exceptions — the caller then decides (typically a logged 500).
    Callers that need 422 for a ``ValueError`` should call
    ``_error_response(422, "invalid input")`` directly.
    """
    for exc_type, status, message in _ERROR_MAP:
        if isinstance(exc, exc_type):
            return _error_response(status, message)
    if isinstance(exc, sqlite3.Error):
        logger.error("research database failure: %s", exc)
        return _error_response(500, "database failure")
    if isinstance(exc, ValueError):
        return _error_response(400, "invalid input")
    return None


def _require_valid_session_id(session_id: str) -> JSONResponse | None:
    """Return a 400 error response when the session ID is malformed."""
    if not is_valid_session_id(session_id):
        return _error_response(400, "invalid session ID")
    return None


def _require_valid_claim_id(
    claim_id: str,
    evidence_id: str | None = None,
) -> JSONResponse | None:
    """Return a 400 error response when the claim ID is malformed.

    When ``evidence_id`` is given (claim <-> evidence link endpoints),
    it is validated too.
    """
    if not is_valid_claim_id(claim_id):
        return _error_response(400, "invalid claim ID")
    if evidence_id is not None and not is_valid_evidence_id(evidence_id):
        return _error_response(400, "invalid evidence ID")
    return None


def build_research_router(db_path: Path | str | None = None) -> APIRouter:
    """The C2 router; sub-routers share the app's ``db_path``.

    Sub-routers are imported lazily inside the factory (they are created
    by the C2-C7 endpoint workers), so importing this module never
    depends on them. ``api_sessions`` currently exports its sessions
    router as ``build_research_router`` (fallback: ``build_sessions_router``);
    the claims and answer routers are ``build_claims_router`` /
    ``build_answer_router``.
    """
    router = APIRouter()

    from research import api_answer, api_claims, api_sessions

    sessions_builder = getattr(
        api_sessions, "build_sessions_router", None
    ) or getattr(api_sessions, "build_research_router")
    router.include_router(sessions_builder(db_path))
    router.include_router(api_claims.build_claims_router(db_path))
    router.include_router(api_answer.build_answer_router(db_path))
    return router


def create_app(db_path: Path | str | None = None) -> FastAPI:
    """Build the combined C0+C1+C2 FastAPI application.

    ``db_path`` overrides the default ``<HERMES_HOME>/docpipe.db``
    (used by tests and unusual deployments). The C0+C1 app already
    registers the ``RequestValidationError`` -> 422 handler.
    """
    app = c0c1api.create_app(db_path)
    app.title = "Hermes C0+C1+C2 — Document, Evidence & Research Pipelines"
    app.description = (
        "Local PDF -> chunks -> SQLite + FTS5 -> Evidence -> research "
        "sessions, claims and synthesized answers. LLM-free."
    )
    app.include_router(build_research_router(db_path))
    return app
