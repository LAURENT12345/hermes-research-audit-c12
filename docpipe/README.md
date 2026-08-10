# C0 — Local Document Pipeline

> Convert a local PDF into searchable, provenance-preserving chunks stored
> in a local SQLite database. **LLM-free by design.**

```
PDF → SHA-256 → dedupe → parse → normalize → chunk → SQLite (documents/chunks) → FTS5 → search
```

C0 is the first infrastructure slice of Hermes Research (Layer C). It does
**not** implement: the research loop, Evidence objects, citations, web
search, embeddings/vector search, MCP, memory, or any LLM call. Those are
C1/C2 concerns; C0 only preserves enough provenance for them to build on:
`document_id`, `chunk_id`, `page_number`, `section_path`, `content`.

---

## Quickstart

```bash
# 1. Install (pypdf for parsing; fastapi/uvicorn for the API)
uv sync --extra docpipe --extra web

# 2. Run the API server (default: 127.0.0.1:8731)
hermes docpipe serve            # or: python -m docpipe.serve

# 3. Ingest a PDF
curl -F "file=@manual.pdf" -F "title=Manual" http://127.0.0.1:8731/documents/upload
# → {"document_id":1,"status":"ready","chunks_created":42}

# 4. Search with provenance
curl "http://127.0.0.1:8731/documents/1/search?q=servo&limit=10"
# → {"document_id":1,"query":"servo","chunks":[{"chunk_id":2,"page_number":5,
#    "section_path":null,"content":"..."}]}
```

CLI equivalents (no server needed):

```bash
hermes docpipe ingest manual.pdf --title Manual     # → JSON result
hermes docpipe list                                  # → document metadata
hermes docpipe search 1 "servo tail" --limit 5       # → ranked chunks
hermes docpipe db-path                               # → SQLite path
```

---

## Data model

| Concept | Table | Fields |
|---|---|---|
| Document (a physical ingested PDF) | `documents` | `id`, `title`, `file_path`, `file_hash` (SHA-256, UNIQUE), `status`, `created_at` |
| Chunk (searchable extracted section) | `chunks` | `id`, `document_id` (FK, cascade), `chunk_index`, `content`, `normalized_content`, `page_number`, `section_path`, `token_count`, `created_at` |
| FTS5 index | `chunks_fts` | external-content FTS5 over `chunks`, kept in sync by AFTER INSERT/UPDATE/DELETE triggers |

The schema lives in `docpipe/db.py` (`SCHEMA_SQL`); `init_db` applies it
idempotently on every connect, which doubles as the migration path.
`rebuild_fts()` repairs a desynced index (`INSERT INTO chunks_fts(chunks_fts)
VALUES('rebuild')`).

Storage: `<HERMES_HOME>/docpipe.db` (profile-aware; WAL mode). Point
`HERMES_HOME` elsewhere to move the store.

## Ingestion flow (transactional)

1. File exists + size ≤ `MAX_UPLOAD_BYTES` (50 MiB)
2. Streaming SHA-256 → hash lookup → **duplicate returns the existing
   document** (nothing re-parsed or re-indexed)
3. Parse + chunk **before** any write transaction — parse/chunk failures
   leave the database untouched (no orphan rows, no partial state)
4. `BEGIN IMMEDIATE` → re-check hash (race guard) → insert document
   (`status='ready'`) + chunks (triggers index FTS5) → commit
5. Any SQLite failure rolls back; nothing is ever left half-written

## Chunking rules

Priority: `page → paragraph → sentence fallback`.

- Blank-line-separated paragraphs become chunks; soft line wraps inside a
  paragraph are joined with a space.
- Target size **2000 chars**, hard maximum **3000 chars** (character-based;
  no tokenizer dependency — engineering defaults in `docpipe/config.py`).
- Oversized paragraphs are split by sentences and packed greedily; if
  sentence splitting fails, at word boundaries; if a single token alone
  exceeds the limit, by raw character slices.
- `chunk_index` is 0-based across the document; `page_number` is the
  1-based physical page; `section_path` is `NULL` (C0 has no heading
  detection); `token_count` is `len(content.split())` (approximation).

## Search

- FTS5 `MATCH` over `chunks_fts`, ranked with BM25 (`ORDER BY
  bm25(chunks_fts)`), filtered by `document_id`.
- **Query safety**: user text is never passed to `MATCH` verbatim.
  `build_fts_query` extracts word tokens, drops a small English stopword
  list, and re-assembles quoted single-token phrases joined with `AND` —
  no FTS syntax (`AND/OR/NOT/NEAR/*/:/"`/parens) can leak through.
- Default limit 10, clamped to `[1, 50]`.
- Matching is case-insensitive and **diacritic-sensitive** (FTS5
  `unicode61` default) and **non-stemming**: `"servo"` matches `servo`,
  not `servos`. Use exact token forms in queries.

## API

| Endpoint | Success | Errors |
|---|---|---|
| `GET /health` | `{"status":"ok"}` | — |
| `POST /documents/upload` (multipart `file`, optional `title`) | `{"document_id":1,"status":"ready","chunks_created":42}` · duplicate: `{"document_id":1,"status":"duplicate"}` | 400 `invalid file type` · 413 `file too large` · 422 `invalid PDF` / `empty PDF` |
| `GET /documents` | `{"documents":[{document_id,title,file_path,file_hash,status,created_at,chunk_count}]}` | 500 `database failure` |
| `GET /documents/{id}/search?q=&limit=` | `{"document_id":1,"query":"...","chunks":[{chunk_id,page_number,section_path,content}]}` | 404 `document not found` · 400 `invalid search query` |

Interactive docs: `http://127.0.0.1:8731/docs` (OpenAPI).

## Configuration

All knobs are code constants in `docpipe/config.py` (see `.env.example`
for the two env-var overrides). No secrets, no `HERMES_HOME`-adjacent
config plumbing: C0 is intentionally self-contained.

## Logging

Structured stdlib logging (`logging.getLogger("docpipe...")`), key events:
`upload started`, `hash calculated`, `duplicate detected`, `parse
started/completed`, `chunks created`, `database write completed`, `search
executed`, `ingestion failure`. Document *contents* are never logged.
`hermes docpipe serve` wires into `hermes_logging.setup_logging()`
(agent.log / errors.log under HERMES_HOME).

## Testing

```bash
uv sync --extra docpipe --extra web --extra dev   # pytest
python -m pytest tests/docpipe -v                 # 70 tests
```

Coverage: DB lifecycle + FTS trigger sync (insert/update/delete/restart/
rebuild), hashing + duplicate detection, parser (valid/invalid/empty/
multi-page/unicode), chunker (paragraphs/long/sentence fallback/pages),
search (relevance/BM25/safety/filtering/limits), and the HTTP API
(health/upload/duplicate/list/search/errors) — including a full
acceptance flow.

## Known limitations (deliberate)

- **No stemming, no diacritic folding** — FTS5 `unicode61` default.
- **Sentence splitting is best-effort** — "e.g." and "3.14" may split;
  the word/char fallback guarantees progress.
- **Extraction quality is pypdf's** — scanned/image-only PDFs yield
  `empty PDF`; complex layouts may have imperfect reading order.
- **English-only stopword list** — Vietnamese function words are not
  filtered (queries still work; just less aggressive).
- **Single-writer ingestion** — concurrent uploads of the *same* file are
  race-safe, but there is no locking between different uploads.
- **No auth** — bind to loopback; do not expose the port on a LAN
  without a reverse proxy. (Auth is an explicit C0 non-goal.)
- **No document versioning** — different bytes = new document; same bytes
  = duplicate, always.

## Roadmap (NOT implemented here)

- **C1 — Evidence Pipeline**: chunks → Evidence objects with full
  provenance. **Implemented** — see the `evidence/` package.
- **C2 — Hermes Integration**: retrieval → evidence → OpenCode Go →
  verified answer.
