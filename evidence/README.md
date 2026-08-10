# C1 — Evidence Pipeline

> Convert C0 chunks into structured, provenance-preserving Evidence objects
> that can be validated and persisted. **LLM-free by design.**

```
C0 chunk ──► Evidence Builder ──► Evidence ──► Validation ──► Evidence Store (SQLite)
```

C1 is the second infrastructure slice of Hermes Research (Layer C). It does
**not** implement: the research agent, claims, claim verification, answer or
citation generation, web sources, vector search, embeddings, MCP, memory, or
any LLM call. Those are C2+ concerns.

---

## Concepts

| Concept | Meaning |
|---|---|
| **Chunk** | A piece of document data stored by C0 (`docpipe`) — the canonical storage |
| **Evidence** | A structured, validated *reference* to one chunk, used as research evidence |

```
Document 1 ──► Chunk 5 ──► Evidence E-000001
                 (C0)          (C1)
```

Evidence never replaces, deletes, or duplicates the chunk: `docpipe` remains
the canonical document/chunk store. Evidence only *references* it
(`document_id`, `chunk_id`) and snapshots the exact content for validation.

## Evidence object

| Field | Meaning |
|---|---|
| `evidence_id` | Deterministic sequential ID: `E-000001`, `E-000002`, … (stable across restarts) |
| `document_id` / `chunk_id` | Back-reference into C0 |
| `evidence_type` | `"chunk"` (C1 produces only verbatim chunk evidence) |
| `source_type` | `"local_pdf"` (the only supported source in C1) |
| `source_title` | From the C0 document title |
| `content` | **Exactly** `chunk.content` — never summarized, paraphrased, or re-cleaned |
| `page_number`, `section_path` | Copied from the chunk |
| `locator` | Machine-readable pointer: `{"page": 4, "chunk_id": 5}` (JSON in DB) |
| `created_at` | Database timestamp |

## Integrity & tamper detection

Validation re-checks every Evidence against the **current** C0 state:

1. evidence exists
2. document exists
3. chunk exists
4. `chunk.document_id == evidence.document_id`
5. `evidence.page_number == chunk.page_number` (where pages exist)
6. `evidence.content == chunk.content` (exact — catches tampering)
7. `source_type == "local_pdf"`

Failures return structured results, e.g.
`{"evidence_id": "E-000001", "valid": false, "reason": "content_mismatch"}`.
Validation never repairs anything — a corrupted record is reported, never
silently fixed.

## Database

The `evidence` table lives in the **same** SQLite database as C0
(`<HERMES_HOME>/docpipe.db`), using the same conventions (WAL, autocommit +
`BEGIN IMMEDIATE` via `hermes_cli.sqlite_util.write_txn`, idempotent schema
init on connect). Foreign keys reference `documents(id)` and `chunks(id)`
(enforced — `PRAGMA foreign_keys=ON`); a UNIQUE index on `chunk_id`
enforces one-evidence-per-chunk at the DB level.

Deleting a document that has evidence attached fails loudly (FK NO ACTION)
instead of silently destroying provenance — evidence rows are removed
explicitly by future layers.

## API

All endpoints are served by the combined C0+C1 app (default
`http://127.0.0.1:8731`):

| Endpoint | Behavior |
|---|---|
| `POST /evidence` `{"chunk_id": 5}` | Create Evidence (or return the existing one for that chunk) |
| `GET /evidence/{id}` | Full Evidence object; 404 if missing |
| `GET /evidence/{id}/validate` | `{"evidence_id":…,"valid":true}` or `{…,"valid":false,"reason":…}` |
| `GET /documents/{id}/evidence` | All Evidence for a document |

Errors use the C0 convention: `{"error": "…"}` with 400/404/409/422/500.

## Quickstart

```bash
# Serve C0 + C1 from one app (same command as before):
hermes docpipe serve          # or: python -m evidence.serve
# (C0 routes unchanged; /evidence routes added)

# Ingest a PDF (C0), then promote a chunk to Evidence (C1):
curl -F "file=@manual.pdf" http://127.0.0.1:8731/documents/upload
curl "http://127.0.0.1:8731/documents/1/search?q=servo"     # pick a chunk_id
curl -X POST -H "Content-Type: application/json" \
     -d '{"chunk_id": 5}' http://127.0.0.1:8731/evidence
curl http://127.0.0.1:8731/evidence/E-000001/validate       # {"valid": true}
```

## Testing

```bash
python -m pytest tests/evidence -v     # 41 tests
python -m pytest tests/docpipe -q      # C0 regression: 70 tests
```

## Known limitations (deliberate)

- Only `source_type = "local_pdf"`; no web/scholarly sources (C4).
- One Evidence per chunk (UNIQUE index); a future layer needing multiple
  evidence types per chunk must relax the index deliberately.
- Deleting a document with Evidence raises an FK error — by design
  (provenance protection); no DELETE evidence endpoint exists yet.
- No claims, no citations, no verification — those are C3+.

## Roadmap (NOT implemented here)

- **C2 — Research Engine**: planning → retrieval → gathering → evaluation.
- **C3 — Verification**: claims ↔ evidence → citation verification.
- **C4 — External Sources**: web / scholarly sources / paper discovery.
