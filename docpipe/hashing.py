"""File hashing for duplicate detection (SHA-256)."""

from __future__ import annotations

import hashlib
from pathlib import Path

_READ_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path | str) -> str:
    """Streaming SHA-256 hex digest of a file.

    Memory-bounded (1 MiB read buffer) so multi-hundred-MB PDFs hash
    without loading into RAM.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(_READ_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
