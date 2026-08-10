"""C0 CLI: ``hermes docpipe …`` / ``python -m docpipe …``.

Subcommands
-----------
* ``ingest <file.pdf> [--title T]`` — ingest one PDF, print result JSON
* ``list``                          — list documents
* ``search <document_id> <query> [--limit N]`` — FTS5 search, print JSON
* ``serve [--host H] [--port P] [--db PATH]`` — run the HTTP API
* ``db-path``                       — print the resolved SQLite path

Output is JSON so the CLI is scriptable and easy to test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docpipe import config


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def cmd_ingest(args) -> int:
    from docpipe.ingest import IngestError, ingest_pdf

    try:
        result = ingest_pdf(args.file, title=args.title)
    except IngestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print(
        {
            "document_id": result.document_id,
            "status": result.status,
            "chunks_created": result.chunks_created,
        }
    )
    return 0


def cmd_list(args) -> int:
    from docpipe.db import connect, list_documents

    conn = connect(args.db_path)
    try:
        rows = list_documents(conn)
    finally:
        conn.close()
    _print(
        {
            "documents": [
                {
                    "document_id": r["id"],
                    "title": r["title"],
                    "file_path": r["file_path"],
                    "file_hash": r["file_hash"],
                    "status": r["status"],
                    "chunk_count": r["chunk_count"],
                }
                for r in rows
            ]
        }
    )
    return 0


def cmd_search(args) -> int:
    from docpipe.db import connect, get_document
    from docpipe.search import InvalidSearchQueryError, search

    conn = connect(args.db_path)
    try:
        if get_document(conn, args.document_id) is None:
            print(f"error: document {args.document_id} not found", file=sys.stderr)
            return 1
        try:
            results = search(
                conn, args.document_id, args.query, limit=args.limit or config.SEARCH_DEFAULT_LIMIT
            )
        except InvalidSearchQueryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    finally:
        conn.close()
    _print(
        {
            "document_id": args.document_id,
            "query": args.query,
            "chunks": [
                {
                    "chunk_id": r.chunk_id,
                    "page_number": r.page_number,
                    "section_path": r.section_path,
                    "content": r.content,
                }
                for r in results
            ],
        }
    )
    return 0


def cmd_db_path(args) -> int:
    print(config.default_db_path())
    return 0


def build_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the ``docpipe`` subcommand on an argparse subparsers group."""
    parser = subparsers.add_parser(
        "docpipe",
        help="C0 local document pipeline: PDF -> chunks -> SQLite + FTS5 (LLM-free)",
        description="Ingest local PDFs and search their chunks with page-level "
        "provenance. No LLM is involved.",
    )
    sub = parser.add_subparsers(dest="docpipe_command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest one PDF into the C0 store")
    ingest.add_argument("file", help="path to a .pdf file")
    ingest.add_argument("--title", default=None, help="optional document title")
    ingest.add_argument("--db", dest="db_path", default=None)
    ingest.set_defaults(handler=cmd_ingest)

    listing = sub.add_parser("list", aliases=["ls"], help="List ingested documents")
    listing.add_argument("--db", dest="db_path", default=None)
    listing.set_defaults(handler=cmd_list)

    search_p = sub.add_parser("search", help="FTS5 keyword search over one document")
    search_p.add_argument("document_id", type=int)
    search_p.add_argument("query")
    search_p.add_argument("--limit", type=int, default=None)
    search_p.add_argument("--db", dest="db_path", default=None)
    search_p.set_defaults(handler=cmd_search)

    serve = sub.add_parser("serve", help="Run the C0 HTTP API (uvicorn)")
    serve.add_argument("--host", default=config.SERVER_HOST)
    serve.add_argument("--port", type=int, default=config.SERVER_PORT)
    serve.add_argument("--db", dest="db_path", default=None)
    serve.set_defaults(handler=lambda a: _serve(a))

    db_path_p = sub.add_parser("db-path", help="Print the resolved SQLite path")
    db_path_p.set_defaults(handler=cmd_db_path)

    return parser


def _serve(args) -> int:
    from docpipe.serve import main as serve_main

    return serve_main(["--host", args.host, "--port", str(args.port)] + (
        ["--db", args.db_path] if args.db_path else []
    ))


def docpipe_command(args) -> int:
    """Dispatch entry used by ``hermes_cli.main`` (mirrors cmd_kanban)."""
    return args.handler(args)


def main(argv: list[str] | None = None) -> int:
    """Standalone entry: ``python -m docpipe``."""
    parser = argparse.ArgumentParser(
        prog="docpipe",
        description="Hermes C0 local document pipeline (LLM-free).",
    )
    build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(argv)
    if not getattr(args, "docpipe_command", None):
        parser.print_help()
        return 2
    return docpipe_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
