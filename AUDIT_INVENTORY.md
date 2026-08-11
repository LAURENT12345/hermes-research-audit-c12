# AUDIT INVENTORY

Areas the external reviewer (Kimi K3) should inspect. Everything listed
exists in this repository at the audited commit `4306f15ff`.

## Source modules

| path | role |
|---|---|
| `research/loop.py` | 5-phase ResearchLoop (planning/retrieving/evidencing/claiming/evaluating), idempotency, stop conditions, lease, retry-once |
| `research/reasoner.py` | `ReasonerPort` (Protocol) + `DeterministicReasoner` (loop default) + 5 C3 proposal dataclasses |
| `research/verifier.py` | deterministic proposal validator (`validate_*`), apply functions (`apply_*`), derived coverage (`coverage_for_claim`) |
| `research/claims.py` | claim lifecycle: create (dedupe), decomposition (atomic, provenance), link_evidence, `evaluate_claim` (deterministic recompute) |
| `research/model.py` | frozen dataclasses, status enums, id patterns |
| `research/store.py` | SQLite schema (11 tables), migrations (`PRAGMA user_version` v1→v2→v3), lease table, event log, write_txn usage |
| `research/synthesis.py` | `build_answer`/`synthesize`: statements, citations, ResearchGap, terminal-status exclusion, atomic set_final_answer |
| `research/contradictions.py` | structural contradiction adjudication helper |
| `research/retriever.py` | `LocalRetriever`: FTS5 keyword search → Evidence creation |
| `research/llm_reasoner.py` | `LLMReasoner` (Commit 12): deterministic serialization, `REASONER_SYSTEM_PROMPT_V1`, fail-closed parser, stats, C2 delegation |
| `research/llm_transport.py` | `LLMTransport` Protocol, `TransportConfig`, `ChatCompletionsTransport` (OpenAI-compatible, env-only credentials), `resolve_transport_config` |
| `research/config.py` | engineering defaults incl. LLM defaults (provider `opencode-go`, empty model = fail-closed, timeout) |
| `research/api.py`, `api_sessions.py`, `api_claims.py`, `api_answer.py` | FastAPI routers (sessions/claims/answer; no proposal endpoints) |
| `research/cli.py` | `hermes research` commands (new/list/show/start/iterate/abort/claims/answer/synthesize/db-path/coverage) |
| `evidence/` (store, validation, model, api) | C1 evidence records + tamper-checkable validation (`validate_evidence` compares content snapshot vs live chunk) |
| `docpipe/` (db, search) | C0 documents/chunks + FTS5 (`chunks_fts`), shared STOPWORDS |
| `hermes_cli/` | host CLI layer; `hermes_cli/sqlite_util.py` provides `write_txn` / `add_column_if_missing` used by research |

## Cross-cutting concerns to audit

- **API**: `docs/API_REFERENCE.md`-equivalent surface — see `research/api*.py`
  (12 endpoints).
- **CLI**: `research/cli.py` (11 commands).
- **Database schema**: `research/store.py` (RESEARCH_SCHEMA_SQL), plus
  `evidence/store.py` and `docpipe/db.py`.
- **Migrations**: `research/store.py` — `SCHEMA_VERSION`, migration
  callables, `user_version` anchor, idempotent `add_column_if_missing`.
- **State machines**: session lifecycle (`draft→active→complete/
  complete_with_gaps/failed/aborted`), claim transition matrix in
  `research/claims.py` (`evaluate_claim`), phase lifecycle in
  `research/loop.py`.
- **Transactions**: `write_txn` (hermes_cli/sqlite_util) — atomic
  transitions, migration rollback, atomic decomposition.
- **Concurrency**: `research_session_locks` lease (token + `expires_at`,
  `SESSION_LOCK_TTL_SECONDS`), `SessionBusyError` → HTTP 409.
- **Audit events**: `research_events` append-only log (status changes, phase
  outcomes, stop reasons).
- **Provenance**: documents.file_hash → chunks → evidence (content snapshot +
  locator) → claim_evidence (role-scoped) → claims.parent_claim_id →
  research_iteration_evidence → research_events.
- **Claim/evidence relationships**: `claim_evidence` PK(claim_id,
  evidence_id, role), roles `supporting|contradicting`, idempotent linking.
- **LLM boundary**: `research/llm_reasoner.py` + `llm_transport.py` — pure
  adapter (no store import, no status mutation, no apply calls); enforced by
  `tests/research/test_arch_boundary.py`.
- **Proposal validation**: `research/verifier.py` — fabricated IDs, wrong
  claim↔evidence relationships, invalid enums/labels, out-of-range values
  rejected; parser fail-closed in `llm_reasoner.py`.
- **Synthesis safety**: `research/synthesis.py` — only non-terminal supported
  claims; citations resolve to persisted ids; terminal statuses excluded.

## Tests

- `tests/research/` — 30 test files (acceptance, core, API/CLI, hardening,
  atomicity, tamper, verifier, arch boundary, LLM reasoner, E2E).
- `tests/evidence/`, `tests/docpipe/` — C1/C0 suites.

## Documentation

- `research/README.md` — architecture reference (advisory).
- `research/IMPLEMENTATION_REPORT.md` — commit-by-commit report (advisory).
- `AUDIT_SCOPE.md`, `AUDIT_TIMELINE.md`, `AUDIT_TEST_STATUS.md` — audit
  context (this package).
