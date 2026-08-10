"""C2 HTTP API — answer / synthesis endpoints.

* ``GET  /research/sessions/{session_id}/answer?refresh=true|false``
  (default ``refresh=true``) — re-validate against live evidence and
  re-synthesize when stale; ``refresh=false`` returns the stored answer
  untouched and says so explicitly (``stale_possible``).
* ``POST /research/sessions/{session_id}/synthesize`` — build and persist
  the deterministic answer (idempotent).

THIN layer: handlers only call the synthesis domain
(``research.synthesis.get_answer`` / ``synthesize``) plus read-only
existence checks through ``research.store``. No answer is constructed
here and no raw write happens in this module. Error responses follow the
C0/C1 convention: ``{"error": "..."}`` with the appropriate status code;
stack traces never leak.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from research import store
from research.model import is_valid_session_id
from research.synthesis import (
    SynthesisError,
    SynthesisNotAllowedError,
    get_answer,
    load_answer,
    synthesize,
)

logger = logging.getLogger(__name__)


def _error_response(status_code: int, message: str) -> JSONResponse:
    """C0/C1-convention error payload: ``{"error": message}``."""
    return JSONResponse(status_code=status_code, content={"error": message})


def _db_error_response(exc: Exception) -> JSONResponse:
    logger.error("research database failure: %s", exc)
    return _error_response(500, "database failure")


def build_answer_router(db_path: Path | str | None = None) -> APIRouter:
    """The C2 answer/synthesis router; handlers share the app's ``db_path``."""

    router = APIRouter()

    @router.get("/research/sessions/{session_id}/answer")
    def get_session_answer_endpoint(
        session_id: str, refresh: bool = True
    ) -> dict:
        """Stored (or freshly re-synthesized) answer for one session.

        ``refresh=true`` (default) delegates to
        ``synthesis.get_answer(refresh=True)``, which re-validates every
        claim against live C1 evidence and overwrites a stale answer —
        the response is authoritative. ``refresh=false`` reads the stored
        answer without any check and marks the response
        ``stale_possible``.
        """
        if not is_valid_session_id(session_id):
            return _error_response(400, "invalid session ID")
        try:
            conn = store.connect(db_path)
            try:
                if store.get_session(conn, session_id) is None:
                    return _error_response(404, "session not found")
                if not refresh and load_answer(conn, session_id) is None:
                    return _error_response(404, "no stored answer")
                answer = get_answer(conn, session_id, refresh=refresh)
            finally:
                conn.close()
        except SynthesisNotAllowedError:
            return _error_response(409, "synthesis not allowed")
        except SynthesisError as exc:
            logger.error("synthesis failed for session %s: %s", session_id, exc)
            return _error_response(500, "synthesis failed")
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        return {
            "answer": answer.to_dict(),
            "refreshed": bool(refresh),
            "stale_possible": not bool(refresh),
        }

    @router.post("/research/sessions/{session_id}/synthesize")
    def synthesize_session_endpoint(session_id: str) -> dict:
        """Synthesize (and persist) the deterministic answer for a session.

        ``persisted`` reports whether ``final_answer`` actually changed
        (first call: ``True``; an idempotent rebuild with byte-identical
        JSON: ``False``). Session status is never modified here — the
        loop owns lifecycle.
        """
        if not is_valid_session_id(session_id):
            return _error_response(400, "invalid session ID")
        try:
            conn = store.connect(db_path)
            try:
                session = store.get_session(conn, session_id)
                if session is None:
                    return _error_response(404, "session not found")
                before = session.final_answer
                answer = synthesize(conn, session_id)
                after = store.get_session(conn, session_id).final_answer
            finally:
                conn.close()
        except SynthesisNotAllowedError:
            return _error_response(409, "synthesis not allowed")
        except SynthesisError as exc:
            logger.error("synthesis failed for session %s: %s", session_id, exc)
            return _error_response(500, "synthesis failed")
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        return {"answer": answer.to_dict(), "persisted": before != after}

    return router
