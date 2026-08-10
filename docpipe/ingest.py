"""Transactional PDF ingestion: hash -> dedupe -> parse -> chunk -> store.

Ingestion order (failure-safe by construction):

1. ``sha256`` the file (streaming).
2. Look up the hash — a duplicate returns the existing document and
   nothing is parsed, stored, or re-indexed.
3. Parse + chunk **before** opening any write transaction, so a parse or
   chunking failure leaves the database untouched (no orphan rows, no
   ``failed`` status needed).
4. Open a ``BEGIN IMMEDIATE`` transaction, re-check the hash (race
   guard against concurrent uploads of the same file), insert the
   document (status ``ready``) and all chunks (the AFTER INSERT triggers
   index each chunk into FTS5), commit.

Any failure rolls the transaction back; the deliberate statuses are
``uploaded`` (default, unused by this flow), ``ready`` (successfully
ingested) — and nothing else. There is no partial state.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from hermes_cli.sqlite_util import write_txn

from docpipe import config
from docpipe.chunker import chunk_document
from docpipe.db import (
    connect,
    get_document_by_hash,
    insert_chunk,
    insert_document,
)
from docpipe.hashing import sha256_file
from docpipe.parser import (
    EmptyExtractionError as ParserEmptyError,
    InvalidPDFError as ParserInvalidError,
    ParseError as ParserError,
    parse_pdf_normalized,
)

logger = logging.getLogger(__name__)


class IngestError(Exception):
    """Base class for all C0 ingestion failures."""


class MissingFileError(IngestError):
    """The file does not exist."""


class FileTooLargeError(IngestError):
    """The file exceeds ``config.MAX_UPLOAD_BYTES``."""


class InvalidFileTypeError(IngestError):
    """The file is not a PDF."""


class InvalidPDFError(IngestError):
    """The file is not a readable PDF."""


class EmptyPDFError(IngestError):
    """The PDF yielded no extractable text."""


class DatabaseFailureError(IngestError):
    """SQLite write failed; the transaction was rolled back."""


@dataclass(frozen=True)
class IngestResult:
    """Outcome of an ingestion attempt."""

    document_id: int
    status: str  # "ready" | "duplicate"
    chunks_created: int = 0
    duplicate: bool = False


def ingest_pdf(
    path: Path | str,
    *,
    title: str | None = None,
    db_path: Path | str | None = None,
) -> IngestResult:
    """Ingest one PDF into the C0 store. See module docstring for the flow.

    Raises :class:`IngestError` subclasses on any failure; the database is
    never left in a partial state.
    """
    path = Path(path)
    if not path.is_file():
        raise MissingFileError(f"file not found: {path}")

    size = path.stat().st_size
    if size > config.MAX_UPLOAD_BYTES:
        raise FileTooLargeError(
            f"file is {size} bytes; limit is {config.MAX_UPLOAD_BYTES} bytes"
        )
    logger.info("upload started: %s (%d bytes)", path.name, size)

    file_hash = sha256_file(path)
    logger.info("hash calculated: %s", file_hash)

    conn = connect(db_path)
    try:
        # Fast path: duplicate before paying for a parse.
        with write_txn(conn):
            existing = get_document_by_hash(conn, file_hash)
            if existing is not None:
                logger.info("duplicate detected: document_id=%s", existing["id"])
                return IngestResult(
                    document_id=int(existing["id"]),
                    status="duplicate",
                    duplicate=True,
                )

        # Parse + chunk outside any transaction: failures leave no rows.
        logger.info("parse started: %s", path.name)
        try:
            pages = parse_pdf_normalized(path)
        except ParserEmptyError as exc:
            raise EmptyPDFError(str(exc)) from exc
        except ParserInvalidError as exc:
            raise InvalidPDFError(str(exc)) from exc
        except ParserError as exc:
            raise IngestError(str(exc)) from exc
        chunks = chunk_document(pages)
        logger.info("chunks created: %d", len(chunks))

        final_title = (title or path.stem).strip() or path.stem

        # Write transaction: re-check the hash to stay race-safe, then
        # insert document + chunks atomically.
        with write_txn(conn):
            existing = get_document_by_hash(conn, file_hash)
            if existing is not None:
                logger.info("duplicate detected (race): document_id=%s", existing["id"])
                return IngestResult(
                    document_id=int(existing["id"]),
                    status="duplicate",
                    duplicate=True,
                )
            document_id = insert_document(
                conn,
                title=final_title,
                file_path=str(path.resolve()),
                file_hash=file_hash,
                status="ready",
            )
            for chunk in chunks:
                insert_chunk(conn, document_id=document_id, chunk=chunk)

        logger.info(
            "database write completed: document_id=%s chunks=%d",
            document_id,
            len(chunks),
        )
        return IngestResult(
            document_id=document_id,
            status="ready",
            chunks_created=len(chunks),
        )
    except sqlite3.Error as exc:
        logger.error("database failure during ingestion of %s: %s", path.name, exc)
        raise DatabaseFailureError(f"database failure: {exc}") from exc
    finally:
        conn.close()
