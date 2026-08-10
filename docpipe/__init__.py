"""C0 — Hermes Local Document Pipeline.

Converts a local PDF into searchable, provenance-preserving chunks stored
in a local SQLite database (documents / chunks / chunks_fts + FTS5).

This is infrastructure only (Layer C). It is intentionally LLM-free:

* no OpenCode Go integration,
* no LLM calls of any kind,
* no claims / Evidence objects / citation verification,
* no vector search / embeddings / rerankers,
* no knowledge of the Dynamic Research Protocol.

The full Source/Evidence abstraction belongs to C1; this package only
preserves enough provenance for C1 to build on: ``document_id``,
``chunk_id``, ``page_number``, ``section_path``, ``content``.
"""

__version__ = "0.1.0"
