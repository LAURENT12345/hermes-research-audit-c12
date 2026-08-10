"""C1 HTTP API (FastAPI router + combined app).

Endpoints (mounted on the same app as C0's API):

* ``POST /evidence``                          create Evidence from a chunk
* ``GET  /evidence/{evidence_id}``            retrieve one Evidence
* ``GET  /evidence/{evidence_id}/validate``   validate against C0 state
* ``GET  /documents/{document_id}/evidence``  list a document's Evidence

``create_app`` builds the *combined* C0+C1 app: it starts from
``docpipe.api.create_app`` (unchanged C0 routes) and includes the
evidence router. Error responses follow the C0 convention:
``{"error": "..."}`` with the appropriate status code; request-validation
failures are normalized to ``422 {"error": "invalid request body"}``.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import docpipe.api as c0api
import docpipe.db as c0db

from evidence import __version__
from evidence import builder, store, validation
from evidence.model import is_valid_evidence_id

logger = logging.getLogger(__name__)


class CreateEvidenceRequest(BaseModel):
    chunk_id: int


def _db_error_response(exc: Exception) -> JSONResponse:
    logger.error("evidence database failure: %s", exc)
    return JSONResponse(status_code=500, content={"error": "database failure"})


def build_evidence_router(db_path: Path | str | None = None) -> APIRouter:
    """The C1 router; handlers share the app's ``db_path``."""
    router = APIRouter()

    @router.post("/evidence")
    def create_evidence_endpoint(payload: CreateEvidenceRequest) -> dict:
        try:
            evidence = builder.create_evidence(payload.chunk_id, db_path=db_path)
        except builder.ChunkNotFoundError:
            return JSONResponse(status_code=404, content={"error": "chunk not found"})
        except builder.DocumentNotFoundError:
            return JSONResponse(
                status_code=404, content={"error": "document not found"}
            )
        except builder.ChunkDocumentMismatchError:
            return JSONResponse(
                status_code=409, content={"error": "chunk/document mismatch"}
            )
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        return evidence.to_dict()

    @router.get("/evidence/{evidence_id}")
    def get_evidence_endpoint(evidence_id: str) -> dict:
        if not is_valid_evidence_id(evidence_id):
            return JSONResponse(
                status_code=400, content={"error": "invalid evidence ID"}
            )
        try:
            conn = store.connect(db_path)
            try:
                evidence = store.get_evidence(conn, evidence_id)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        if evidence is None:
            return JSONResponse(
                status_code=404, content={"error": "evidence not found"}
            )
        return evidence.to_dict()

    @router.get("/evidence/{evidence_id}/validate")
    def validate_evidence_endpoint(evidence_id: str) -> dict:
        if not is_valid_evidence_id(evidence_id):
            return JSONResponse(
                status_code=400, content={"error": "invalid evidence ID"}
            )
        try:
            conn = store.connect(db_path)
            try:
                evidence = store.get_evidence(conn, evidence_id)
                if evidence is None:
                    return JSONResponse(
                        status_code=404, content={"error": "evidence not found"}
                    )
                result = validation.validate_evidence(conn, evidence_id)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        return result.to_dict()

    @router.get("/documents/{document_id}/evidence")
    def list_document_evidence_endpoint(document_id: int) -> dict:
        try:
            conn = store.connect(db_path)
            try:
                if c0db.get_document(conn, document_id) is None:
                    return JSONResponse(
                        status_code=404, content={"error": "document not found"}
                    )
                rows = store.list_evidence(conn, document_id)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        return {
            "document_id": document_id,
            "evidence": [evidence.to_dict() for evidence in rows],
        }

    return router


def create_app(db_path: Path | str | None = None) -> FastAPI:
    """Build the combined C0+C1 FastAPI application.

    ``db_path`` overrides the default ``<HERMES_HOME>/docpipe.db``
    (used by tests and unusual deployments).
    """
    app = c0api.create_app(db_path)
    app.title = "Hermes C0+C1 — Document & Evidence Pipelines"
    app.description = (
        "Local PDF -> chunks -> SQLite + FTS5 -> Evidence. LLM-free."
    )
    app.include_router(build_evidence_router(db_path))

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(request, exc):  # noqa: ANN001
        return JSONResponse(
            status_code=422, content={"error": "invalid request body"}
        )

    return app
