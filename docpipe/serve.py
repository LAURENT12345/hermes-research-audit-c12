"""Standalone C0 API server: ``python -m docpipe.serve`` (uvicorn).

Requires the ``web`` extra (fastapi/uvicorn — already present in any
install that runs the dashboard). Defaults: 127.0.0.1:8731.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from docpipe import config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docpipe serve",
        description="Serve the C0 Document Pipeline HTTP API (FastAPI/uvicorn).",
    )
    parser.add_argument("--host", default=config.SERVER_HOST)
    parser.add_argument("--port", type=int, default=config.SERVER_PORT)
    parser.add_argument("--db", dest="db_path", default=None,
                        help="SQLite DB path (default: <HERMES_HOME>/docpipe.db)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from uvicorn import run
    except ImportError:  # pragma: no cover - depends on install
        logging.error(
            "uvicorn is not installed; install the web extra: uv sync --extra web"
        )
        return 2

    try:
        from hermes_logging import setup_logging
        setup_logging(mode="cli")
    except Exception:  # pragma: no cover - logging must never block serving
        logging.basicConfig(level=logging.INFO)

    from docpipe.api import create_app

    app = create_app(db_path=Path(args.db_path) if args.db_path else None)
    logging.getLogger("docpipe").info(
        "C0 document pipeline API listening on http://%s:%d (docs at /docs)",
        args.host,
        args.port,
    )
    run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
