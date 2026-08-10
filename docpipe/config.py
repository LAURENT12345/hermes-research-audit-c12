"""C0 configuration: defaults, limits, and environment overrides.

Everything here is an engineering default, not an immutable protocol rule.
The only user-facing knobs are ``DOCPIPE_HOST`` / ``DOCPIPE_PORT`` (the
standalone API server bind address). All other settings are code constants
so the behaviour is visible and testable; override them in tests via
``monkeypatch`` on this module.

The database lives at ``<HERMES_HOME>/docpipe.db`` — the same location
convention as ``state.db``, ``kanban.db`` and ``memory_store.db``. There is
no separate env var for the path; point ``HERMES_HOME`` elsewhere to move
the store.
"""

from __future__ import annotations

import os
from pathlib import Path

from hermes_constants import get_hermes_home

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

DB_FILENAME = "docpipe.db"


def default_db_path() -> Path:
    """Absolute path of the C0 SQLite store (profile-aware HERMES_HOME)."""
    return get_hermes_home() / DB_FILENAME


# ---------------------------------------------------------------------------
# Ingestion limits
# ---------------------------------------------------------------------------

#: Hard cap on uploaded file size, in bytes (50 MiB).
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

#: A PDF whose total extracted text is below this many characters is treated
#: as "empty" and rejected. Scanned/image-only PDFs land here.
EMPTY_EXTRACTION_MIN_CHARS = 10

#: Below this total extracted length a warning is logged; the document is
#: still ingested (extremely short extraction is not necessarily an error).
SHORT_EXTRACTION_WARN_CHARS = 200


# ---------------------------------------------------------------------------
# Chunking limits (character-based; no tokenizer dependency)
# ---------------------------------------------------------------------------

#: Preferred chunk size. Chunks in the band [CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS]
#: are accepted; packing aims to get as close to the target as possible.
CHUNK_TARGET_CHARS = 2000

#: Hard maximum chunk length. A paragraph larger than this is split by
#: sentences; if sentence splitting fails, by words; if that fails, by raw
#: character slices.
CHUNK_MAX_CHARS = 3000


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

#: Default number of results returned by search.
SEARCH_DEFAULT_LIMIT = 10

#: Upper bound for the search result limit (clamped, not an error).
SEARCH_MAX_LIMIT = 50


# ---------------------------------------------------------------------------
# Standalone API server (``docpipe serve`` / ``hermes docpipe serve``)
# ---------------------------------------------------------------------------

SERVER_HOST = os.environ.get("DOCPIPE_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("DOCPIPE_PORT", "8731"))
