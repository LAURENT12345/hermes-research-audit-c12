# C0 — Implementation Report

**Task:** C0 — Local Document Pipeline (Hermes Research, Layer C, first
infrastructure slice). PDF → chunks → SQLite + FTS5 → search, with
document/page provenance. **LLM-free; no research loop; no Evidence.**

Date: 2026-08-09 · Hermes Agent 0.17.0 · Python 3.11 · Windows 10

---

## 1. Existing system inspection (summary)

| Component | Found | Reused by C0 |
|---|---|---|
| Python | 3.11.1 (repo venv), `>=3.11,<3.14` per pyproject | — |
| Package manager | uv (exact pins, optional extras, uv.lock) | new `docpipe` extra |
| Storage/database | stdlib sqlite3; WAL + `BEGIN IMMEDIATE` convention (`hermes_cli/sqlite_util.py`); stores at `<HERMES_HOME>/*.db` | `write_txn`, same layout |
| Config system | config.yaml + `HERMES_HOME` env; code constants for infra defaults | `hermes_constants.get_hermes_home()` |
| HTTP/API | aiohttp gateway API server (:8642, LLM traffic); FastAPI dashboard (`hermes_cli/web_server.py`, auth-gated) | neither touched; same FastAPI stack on own port |
| Logging | `hermes_logging.setup_logging()` (agent.log/errors.log) | wired in `docpipe serve` |
| CLI pattern | `hermes_cli/main.py` subparsers + `build_parser()`/`cmd_*` | `hermes docpipe …` |
| Tests | pytest, `tests/`, conftest isolates `HERMES_HOME` per test | `tests/docpipe/` |
| PDF parser | none installed | pypdf 6.15.0 (pure Python, smallest reliable) |

**Integration point chosen:** new top-level package `docpipe/` — the
"Document Pipeline" branch of the architecture — with its own DB
(`<HERMES_HOME>/docpipe.db`) and its own FastAPI app (127.0.0.1:8731,
same stack as the dashboard). No core file was redesigned; the only core
edit is a 16-line CLI subcommand registration.

## 2. Files changed / added

**Added (package):**
```
docpipe/__init__.py        version + scope statement
docpipe/config.py          defaults/limits + env overrides (DOCPIPE_HOST/PORT)
docpipe/hashing.py         streaming SHA-256
docpipe/normalize.py       content vs normalized_content (conservative)
docpipe/parser.py          pypdf wrapper: ParsedPage, error taxonomy
docpipe/chunker.py         page→paragraph→sentence chunker (target 2000 / hard 3000 chars)
docpipe/db.py              schema + FTS5 external-content table + sync triggers
docpipe/search.py          FTS5 query builder (sanitized) + BM25 search
docpipe/ingest.py          transactional ingest: hash → dedupe → parse → chunk → store
docpipe/api.py             FastAPI app (health / upload / list / search)
docpipe/serve.py           uvicorn entry
docpipe/cli.py             hermes docpipe ingest|list|search|serve|db-path
docpipe/README.md          documentation
docpipe/.env.example       configuration documentation
docpipe/IMPLEMENTATION_REPORT.md  (this file)
```

**Added (tests, 70 total):**
```
tests/docpipe/pdf_factory.py        dependency-free minimal PDF builder (incl. Unicode/Type0)
tests/docpipe/test_db.py            12 tests — schema, FTS trigger sync (INSERT/UPDATE/DELETE/
                                    restart/rebuild), BM25 ordering
tests/docpipe/test_hashing.py       3 tests
tests/docpipe/test_parser.py        9 tests — valid/invalid/truncated/empty/multi-page/unicode
tests/docpipe/test_chunker.py       11 tests — paragraphs, long text, sentence fallback, pages
tests/docpipe/test_search.py        11 tests — relevance, safety, filtering, limits, unicode
tests/docpipe/test_ingest.py        13 tests — dedupe, rollback, provenance, errors
tests/docpipe/test_api.py           11 tests — health/upload/duplicate/list/search/errors + full
                                    acceptance flow
```

**Changed:**
```
pyproject.toml      + docpipe extra: pypdf==6.15.0 (exact pin, per project convention)
uv.lock             regenerated (uv lock; pypdf added)
hermes_cli/main.py  + "docpipe" in _BUILTIN_SUBCOMMANDS, + cmd_docpipe wrapper,
                    + parser registration (16 lines, mirrors kanban/project pattern)
```

**Not touched:** OpenCode Go configuration, providers, gateway, agent
core, plugins, web_server, api_server, config.yaml.

## 3. Dependencies added

- `pypdf==6.15.0` (new optional extra `docpipe`) — the only new
  dependency. The API reuses fastapi/uvicorn/python-multipart from the
  existing `web` extra.
- Runtime note: this machine's live Hermes runs from `venv/` while uv
  sync manages `.venv/`; pypdf+pytest were therefore also installed into
  `venv/` (`uv pip install --python venv/Scripts/python.exe pypdf==6.15.0
  pytest==9.0.2 pytest-asyncio==1.3.0`) so `hermes docpipe` works in the
  live install. Both environments are now consistent with the lock.
- Build-environment note: the repo venv shipped setuptools 65.5.0, which
  cannot parse the project's PEP-639 `license` field and broke `uv sync`
  (pre-existing; unrelated to C0). Upgraded to setuptools 82 via pip; uv
  now pins 81.0.0 via the dev extra.

## 4. Commands used to test

```bash
uv lock && uv sync --extra docpipe --extra web --extra dev   # resolve + install
uv pip install --python venv/Scripts/python.exe pypdf==6.15.0 pytest==9.0.2 pytest-asyncio==1.3.0
venv/Scripts/python.exe -m pytest tests/docpipe -q           # → 70 passed
venv/Scripts/python.exe -m pytest tests/cli -q               # → 970 passed, 9 failed (pre-existing, environment)
.venv/Scripts/ruff.exe check docpipe tests/docpipe hermes_cli/main.py   # → All checks passed
# Live acceptance (server + curl): health / upload / duplicate / list / search /
#   404 / 400 / 413 / restart-persistence — all verified end-to-end
```

## 5. Test results

- `tests/docpipe`: **70 passed, 0 failed** (4.4 s)
- ruff: **All checks passed**
- `tests/cli` regression: 970 passed; 9 failures **reproduce without the
  C0 change** (verified via `git stash`) — Windows environment issues in
  WSL/Chrome/tilde/TTY tests, unrelated to C0.
- Live acceptance (real HTTP, real `hermes docpipe` CLI):
  - upload → `{"document_id":1,"status":"ready","chunks_created":3}`
  - duplicate upload → same id, `"status":"duplicate"`, no re-ingestion
  - search `servo tail` → the correct chunk with `page_number: 2`
  - search `DeepSeek` → page-3 chunk; BM25 ordering
  - errors: `invalid file type` (400), `invalid PDF` (422), `empty PDF`
    (422), `document not found` (404), `invalid search query` (400)
  - server restart → documents + FTS index survive (WAL + triggers)
  - `hermes docpipe list/search/db-path` work through the real CLI

## 6. Known limitations (deliberate; documented in README)

1. FTS5 `unicode61`: no stemming, no diacritic folding — queries must use
   exact token forms ("servo" ≠ "servos").
2. Sentence splitting is best-effort ("e.g.", "3.14" may split); word/char
   fallback guarantees termination.
3. Extraction quality = pypdf's (scanned PDFs → `empty PDF`; complex
   layouts may have imperfect reading order). Page numbers are exact
   (1-based physical pages).
4. English-only stopword list in query sanitization.
5. No auth (explicit non-goal): bind to loopback.
6. No document versioning: same bytes → duplicate; different bytes →
   new document.
7. Single-writer ingestion: duplicate-race-safe, but no cross-upload
   locking (single-user local tool by design).
8. `token_count` is an approximation (`len(content.split())`).

## 7. Non-goals honored

No OpenCode Go integration, no LLM calls (verified: zero LLM/provider
imports in `docpipe/`), no research loop, no Evidence/citations, no
vector search/embeddings/reranker, no MCP, no memory, no multi-user, no
auth, no UI. C0 does not know about the Dynamic Research Protocol.

## 8. Handoff notes for C1

C1 (Evidence Pipeline) can consume directly from `docpipe.search.search()`
results (`chunk_id`, `document_id`, `page_number`, `section_path`,
`content`) or from `docpipe.db.list_chunks()`. The FTS index is
trigger-synchronized, so C1 may UPDATE/DELETE chunks safely. `section_path`
is always NULL until a heading-detection pass is added (C1+).
