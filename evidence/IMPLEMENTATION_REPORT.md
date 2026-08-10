# C1 — Implementation Report

**Task:** C1 — Evidence Pipeline (Hermes Research, Layer C, second
infrastructure slice). C0 chunks → Evidence objects → provenance →
validation → persistence. **LLM-free; no claims; no research agent.**

Date: 2026-08-09 · Hermes Agent 0.17.0 · Python 3.11 · Windows 10

---

## Files changed

```
docpipe/serve.py       +5 lines: serve the combined C0+C1 app
                       (try evidence.api.create_app, fallback docpipe.api.create_app)
docpipe/README.md      roadmap section updated (C1 now implemented)
```

C0 logic itself was NOT modified: `docpipe/db.py`, `api.py`, `chunker.py`,
`search.py`, `ingest.py`, etc. are untouched. C0's test suite passes
unchanged (70/70).

## Files added

```
evidence/__init__.py        version + scope statement
evidence/model.py           Evidence dataclass, E-###### IDs, serialization
evidence/store.py           evidence table schema + CRUD (shared C0 database)
evidence/builder.py         create_evidence service (chunk -> Evidence, dedupe)
evidence/validation.py      validate_evidence + ValidationResult
evidence/api.py             C1 router + combined C0+C1 create_app
evidence/serve.py           uvicorn entry (python -m evidence.serve)
evidence/README.md          documentation
evidence/IMPLEMENTATION_REPORT.md  (this file)

tests/evidence/test_model.py           6 tests
tests/evidence/test_store.py           8 tests
tests/evidence/test_builder.py         8 tests
tests/evidence/test_validation.py      9 tests
tests/evidence/test_evidence_api.py   10 tests
```

(File API test đặt tên `test_evidence_api.py` để tránh xung đột basename
`test_api.py` với `tests/docpipe/` khi chạy `pytest tests/evidence
tests/docpipe` trong cùng một session.)

## Dependencies added

None. C1 uses stdlib (`sqlite3`, `json`, `dataclasses`), the existing C0
package (`docpipe.db`), existing `hermes_cli.sqlite_util.write_txn`, and
the already-installed FastAPI stack (the `web` extra). No pyproject/uv.lock
changes.

## Database changes

One additive table in the existing `<HERMES_HOME>/docpipe.db` (same file,
same conventions as C0):

```sql
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,             -- 'E-000001' (sequential, stable)
    document_id INTEGER NOT NULL,
    chunk_id INTEGER NOT NULL,
    evidence_type TEXT NOT NULL,     -- 'chunk'
    content TEXT NOT NULL,           -- verbatim chunk.content
    page_number INTEGER,
    section_path TEXT,
    source_title TEXT,
    source_type TEXT NOT NULL,       -- 'local_pdf'
    locator TEXT NOT NULL,           -- JSON {"page": N, "chunk_id": N}
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);
CREATE UNIQUE INDEX idx_evidence_chunk ON evidence(chunk_id);
```

Schema is applied idempotently by `evidence.store.init_schema` on every
connect (the same migration convention as C0's `init_db`). Evidence IDs are
derived from `MAX(SUBSTR(id,3))` inside `BEGIN IMMEDIATE` transactions, so
they are deterministic and stable across restarts.

## API endpoints

| Endpoint | Success | Errors |
|---|---|---|
| `POST /evidence` `{"chunk_id":5}` | full Evidence JSON (200); duplicate → same ID | 404 `chunk not found` / `document not found` · 409 `chunk/document mismatch` · 422 `invalid request body` · 500 `database failure` |
| `GET /evidence/{id}` | full Evidence JSON | 400 `invalid evidence ID` · 404 `evidence not found` |
| `GET /evidence/{id}/validate` | `{"evidence_id":…,"valid":true}` | 400/404 as above · invalid: `{…,"valid":false,"reason":…}` |
| `GET /documents/{id}/evidence` | `{"document_id":…,"evidence":[…]}` | 404 `document not found` |

Served by the combined app (`hermes docpipe serve` / `python -m
evidence.serve`); C0 routes are unchanged.

## Tests

- `tests/evidence`: **41 passed, 0 failed** (model 6, store 8, builder 8,
  validation 9, api 10).
- ruff: **All checks passed** (`evidence tests/evidence docpipe/serve.py`).

## C0 regression

- `tests/docpipe`: **70 passed, 0 failed** — unchanged after C1.
- Verified on the live combined server: PDF upload, duplicate detection,
  document listing, FTS5 search, and restart persistence all work
  identically.

## Black-box acceptance

Real PDF `1706.03762v7.pdf` ("Attention Is All You Need", 15 pages), via
the combined server on an isolated `HERMES_HOME`:

```
C0 ingest            → {"document_id":1,"status":"ready","chunks_created":23}
C0 search            → chunk_id=5, page=4 ("scaled dot product")
POST /evidence       → E-000001, document_id=1, chunk_id=5, page=4,
                       source_type=local_pdf,
                       source_title="Attention Is All You Need",
                       content == chunk.content (byte-identical: True)
GET /evidence/E-000001 → full object returned
GET …/validate       → {"evidence_id":"E-000001","valid":true}
Duplicate POST       → same evidence_id E-000001 (no second row)
Tamper (UPDATE content in DB) → validate → {"valid":false,
                       "reason":"content_mismatch"} (row NOT repaired)
Restart server       → evidence persists; validation still reports the
                       tampered state; C0 search still works
Error boundaries     → chunk not found 404, invalid evidence ID 400,
                       malformed body 422, all with {"error": …}
```

## Known limitations

- Only `source_type = "local_pdf"` (no web/scholarly sources — C4).
- One Evidence per chunk, enforced by a UNIQUE index; relaxing it for
  multiple evidence types per chunk is a deliberate future decision.
- Deleting a document that has Evidence raises a foreign-key error (by
  design — provenance protection); no DELETE evidence endpoint exists.
- `created_at` uses SQLite `CURRENT_TIMESTAMP` (UTC).
- Evidence validation is anchored to the current C0 chunk — if C0's chunk
  content is legitimately updated (e.g. re-chunking), existing evidence
  will report `content_mismatch`. This is the documented integrity contract.

## Future work intentionally NOT implemented

Research engine (C2), claims & verification (C3), web/scholarly sources
(C4), full research agent (C5), LLM calls, vector search/embeddings,
MCP, memory, auth, multi-user, UI. Documented in `evidence/README.md`.
