"""C0 HTTP API (FastAPI).

Endpoints (see ``docpipe/README.md`` for the full contract):

* ``GET  /health``                          -> {"status": "ok"}
* ``POST /documents/upload``                multipart ``file`` + optional ``title``
* ``GET  /documents``                       document metadata list
* ``GET  /documents/{id}/search?q=&limit=`` BM25-ranked chunk retrieval

Why a standalone app: Hermes' existing HTTP surfaces are the gateway's
aiohttp OpenAI-compatible server (LLM traffic — C0 must not touch it) and
the dashboard FastAPI app (auth-gated UI server). C0 therefore exposes the
smallest practical interface: the same FastAPI stack (the ``web`` extra)
on its own port (default 127.0.0.1:8731), same conventions as the
gateway API server. ``create_app`` is a plain factory so a future C2 can
mount the router into any ASGI app.

No authentication is implemented (out of scope for C0): bind to loopback
by default; do not expose the port on a LAN without a reverse proxy.
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from docpipe import __version__, config
from docpipe.db import (
    connect,
    get_document,
    list_documents,
)
from docpipe.ingest import (
    DatabaseFailureError,
    EmptyPDFError,
    FileTooLargeError,
    IngestError,
    InvalidFileTypeError,
    InvalidPDFError,
    MissingFileError,
    ingest_pdf,
)
from docpipe.search import InvalidSearchQueryError, search

logger = logging.getLogger(__name__)


def _db_error_response(exc: Exception) -> JSONResponse:
    logger.error("database failure: %s", exc)
    return JSONResponse(status_code=500, content={"error": "database failure"})


def create_app(db_path: Path | str | None = None) -> FastAPI:
    """Build the C0 FastAPI application.

    ``db_path`` overrides the default ``<HERMES_HOME>/docpipe.db``
    (used by tests and unusual deployments).
    """
    app = FastAPI(
        title="Hermes C0 Document Pipeline",
        description="Local PDF -> chunks -> SQLite + FTS5 retrieval. LLM-free.",
        version=__version__,
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/documents/upload")
    def upload_document(
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
    ) -> dict:
        filename = (file.filename or "").strip()
        if not filename.lower().endswith(".pdf"):
            return JSONResponse(
                status_code=400, content={"error": "invalid file type"}
            )

        try:
            with tempfile.TemporaryDirectory(prefix="docpipe-upload-") as tmp:
                tmp_path = Path(tmp) / filename
                size = 0
                with open(tmp_path, "wb") as out:
                    while True:
                        block = file.file.read(1024 * 1024)
                        if not block:
                            break
                        size += len(block)
                        if size > config.MAX_UPLOAD_BYTES:
                            raise FileTooLargeError(
                                f"file exceeds {config.MAX_UPLOAD_BYTES} bytes"
                            )
                        out.write(block)
                result = ingest_pdf(tmp_path, title=title, db_path=db_path)
        except FileTooLargeError:
            return JSONResponse(
                status_code=413, content={"error": "file too large"}
            )
        except InvalidFileTypeError:
            return JSONResponse(
                status_code=400, content={"error": "invalid file type"}
            )
        except InvalidPDFError:
            return JSONResponse(status_code=422, content={"error": "invalid PDF"})
        except EmptyPDFError:
            return JSONResponse(status_code=422, content={"error": "empty PDF"})
        except MissingFileError:
            return JSONResponse(
                status_code=400, content={"error": "missing file"}
            )
        except DatabaseFailureError:
            return JSONResponse(
                status_code=500, content={"error": "database failure"}
            )
        except IngestError as exc:
            logger.error("ingestion failure: %s", exc)
            return JSONResponse(
                status_code=500, content={"error": "ingestion failed"}
            )
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        finally:
            try:
                file.file.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

        if result.duplicate:
            return {"document_id": result.document_id, "status": "duplicate"}
        return {
            "document_id": result.document_id,
            "status": result.status,
            "chunks_created": result.chunks_created,
        }

    @app.get("/documents")
    def documents_list() -> dict:
        try:
            conn = connect(db_path)
            try:
                rows = list_documents(conn)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return _db_error_response(exc)
        return {
            "documents": [
                {
                    "document_id": row["id"],
                    "title": row["title"],
                    "file_path": row["file_path"],
                    "file_hash": row["file_hash"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "chunk_count": row["chunk_count"],
                }
                for row in rows
            ]
        }

    @app.get("/documents/{document_id}/search")
    def document_search(
        document_id: int,
        q: str,
        limit: int = config.SEARCH_DEFAULT_LIMIT,
    ) -> dict:
        try:
            conn = connect(db_path)
            try:
                doc = get_document(conn, document_id)
                if doc is None:
                    return JSONResponse(
                        status_code=404, content={"error": "document not found"}
                    )
                try:
                    chunks = search(conn, document_id, q, limit=limit)
                except InvalidSearchQueryError:
                    return JSONResponse(
                        status_code=400, content={"error": "invalid search query"}
                    )
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return _db_error_response(exc)

        return {
            "document_id": document_id,
            "query": q,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "page_number": c.page_number,
                    "section_path": c.section_path,
                    "content": c.content,
                }
                for c in chunks
            ],
        }

    return app
