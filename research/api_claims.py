"""C2 claims/evidence HTTP API (read-only inspection endpoints).

Endpoints (mounted on the combined C0+C1+C2 app by ``research.api``):

* ``GET /research/sessions/{session_id}/claims``    list a session's claims
* ``GET /research/claims/{claim_id}``               retrieve one claim
* ``GET /research/claims/{claim_id}/evidence``      inspect a claim's
  evidence links with a LIVE C1 validity re-check

Read-only by design: handlers only read through ``research.store`` /
``evidence.store`` / ``evidence.validation`` — no writes, no business
logic. Error responses follow the C0/C1 convention
(``{"error": "..."}`` with the appropriate status code); malformed
session/claim IDs are normalized to ``400``.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from evidence import store as c1store
from evidence import validation as c1validation
from research import store
from research.model import is_valid_claim_id, is_valid_session_id

logger = logging.getLogger(__name__)


def _error_response(status_code: int, message: str) -> JSONResponse:
    """C0/C1-convention error payload: ``{"error": message}``."""
    return JSONResponse(status_code=status_code, content={"error": message})


def _db_error_response(exc: Exception) -> JSONResponse:
    logger.error("research database failure: %s", exc)
    return _error_response(500, "database failure")


def build_claims_router(db_path: Path | str | None = None) -> APIRouter:
    """The C2 claims/evidence router; handlers share the app's ``db_path``."""
    router = APIRouter()

    @router.get("/research/sessions/{session_id}/claims")
    def list_session_claims_endpoint(session_id: str) -> dict:
        if not is_valid_session_id(session_id):
            return _error_response(400, "invalid session ID")
        try:
            conn = store.connect(db_path)
            try:
                if store.get_session(conn, session_id) is None:
                    return _error_response(404, "session not found")
                claims = store.list_claims(conn, session_id)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        return {
            "session_id": session_id,
            "claims": [claim.to_dict() for claim in claims],
        }

    @router.get("/research/claims/{claim_id}")
    def get_claim_endpoint(claim_id: str) -> dict:
        if not is_valid_claim_id(claim_id):
            return _error_response(400, "invalid claim ID")
        try:
            conn = store.connect(db_path)
            try:
                claim = store.get_claim(conn, claim_id)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        if claim is None:
            return _error_response(404, "claim not found")
        return claim.to_dict()

    @router.get("/research/claims/{claim_id}/evidence")
    def get_claim_evidence_endpoint(claim_id: str) -> dict:
        if not is_valid_claim_id(claim_id):
            return _error_response(400, "invalid claim ID")
        try:
            conn = store.connect(db_path)
            try:
                if store.get_claim(conn, claim_id) is None:
                    return _error_response(404, "claim not found")
                links = []
                for link in store.get_links_for_claim(conn, claim_id):
                    validity = c1validation.validate_evidence(
                        conn, link.evidence_id
                    )
                    entry: dict = {
                        "claim_id": link.claim_id,
                        "evidence_id": link.evidence_id,
                        "role": link.role,
                        "created_at": link.created_at,
                        "valid": validity.valid,
                    }
                    evidence = c1store.get_evidence(conn, link.evidence_id)
                    if evidence is not None:
                        entry["evidence"] = evidence.to_dict()
                    links.append(entry)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        return {"claim_id": claim_id, "links": links}

    return router
