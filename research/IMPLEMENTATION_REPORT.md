# C2 — Implementation Report (Commit 10 checkpoint)

Status: **architectural hardening + documentation checkpoint before C3**.
Companion doc: `research/README.md` (user-facing reference).

## Commits 1–9 recap (verified)

| Commit | Contents |
|---|---|
| 1 | research schema + model + store |
| 2 | retriever + reasoner ports |
| 3 | claims + sufficiency |
| 4 | contradictions + resolution |
| 5 | persistent research loop + resume |
| 6 | synthesis + answer |
| 7 | research API + atomicity hardening |
| 8 | research CLI |
| 9 | full system acceptance coverage (53 tests) |

Baseline before Commit 10: 427 tests, ruff clean, hermes.exe acceptance
passing.

## Commit 10 — what changed

### 1. Concurrency (finding A — confirmed)

**Confirmed:** no per-session serialization existed; two concurrent
`start`/`iterate` callers could interleave phases and duplicate audit
events (queries/claims/links were deduplicated, but events and phase
execution were not).

**Solution — per-session execution lease** (`research_session_locks`):

* `store.acquire_session_lock(session_id, token, ttl)` — compare-and-set
  inside `BEGIN IMMEDIATE`; returns False when a live lease exists.
* `store.release_session_lock(session_id, token)` — token-scoped release.
* `loop.start` / `loop._iterate` acquire the lease for the whole logical
  run and release it in `finally`; on contention they raise
  `SessionBusyError` (API 409 "session busy", CLI exit 1).
* Lease TTL (`SESSION_LOCK_TTL_SECONDS = 300`) makes it crash-safe across
  processes: an expired lease can be taken over deterministically.
* `abort` deliberately does NOT take the lease (poison pill): it stops the
  loop at the next iteration boundary; a second abort cannot duplicate the
  audit event (rowcount guard).

Tests: `tests/research/test_research_hardening.py` (acquire/busy,
token-scoped release, expired takeover, loop busy, threaded mutual
exclusion with barriers, lease released after normal run).

### 2. Session lifecycle atomicity (finding B — confirmed)

**Confirmed:** `_apply_stop`, `abort`, `start`(draft→active) and the
failure path each wrote the status change and audit events in separate
autocommit statements — a crash between them left authoritative state
without its event.

**Solution:** each transition now commits status + event(s) in ONE
`write_txn` (loop.py): `_apply_stop` (complete/complete_with_gaps +
events), `abort` (+ rowcount guard), `start` draft→active +
`session_created`, failure path (`failed` + event).

Tests: rollback-on-event-failure for all four paths + a real subprocess
crash mid-transition (os._exit) proving neither state nor event survives.

### 3. Phase atomicity (finding C — already fixed)

Verified: Commit 7 already made phase completion/failure + phase events
atomic; phase *side effects* are intentionally idempotent/retryable (they
are the re-run safety net, not a single transaction). No change.

### 4. Raw store write boundary (finding D — confirmed, docstrings only)

Verified: `store.update_claim_status/update_session_status/set_query_status/
set_session_answer/set_final_answer` are internal helpers; the API/CLI
never call them.

**Solution:** mechanical enforcement via AST —
`tests/research/test_arch_boundary.py` asserts the interface modules
(`api.py`, `api_*`, `cli.py`) never call the raw writers, the reasoner has
no store import, and synthesis does not call the status writers.

### 5. Schema migration (finding E — confirmed missing)

**Solution:** `PRAGMA user_version`-based migration foundation in
`research/store.py` (`SCHEMA_VERSION = 1`, ordered `_MIGRATIONS`,
transactional apply: schema + version bump commit together; future
versions rejected deterministically; statements executed one-by-one inside
the txn because `executescript` would auto-commit).

Tests: fresh DB → version 1; reconnect idempotent; upgrade from a
pre-version database (user_version 0 with existing tables + data
preserved); future version rejected; migration failure rolls back the
version bump.

### 6–10. Findings F–J (verified, no code change)

* **F — structural contradiction advisory**: confirmed correct — synthesis
  and the synthesis guard block only `explicit` open contradictions;
  structural records are advisory. No change.
* **G — research gap semantics**: documented in README (retrieval gap vs
  research gap).
* **H — multi-paper context**: documented (provenance works; contextual
  comparison is a C3/C4 requirement).
* **I — question coverage**: documented (sufficiency ≠ semantic coverage).
* **J — stale answer**: confirmed correct (`refresh=true` default +
  `stale_possible` for `refresh=false`); covered by existing acceptance
  tests.

## Files changed (Commit 10)

```
M research/store.py            lock table + lease + user_version migrations
M research/loop.py             SessionBusyError + lease acquire/release +
                               atomic lifecycle transitions + failure path
M research/config.py           SESSION_LOCK_TTL_SECONDS
M research/api.py              SessionBusyError -> 409 (error map)
M research/api_sessions.py     SessionBusyError import + 409 handlers
A research/README.md           architecture reference
A research/IMPLEMENTATION_REPORT.md   (this file)
A tests/research/test_research_hardening.py   16 tests
A tests/research/test_arch_boundary.py         3 tests
```

C0/C1 implementation logic: **untouched** (verified by diff; docpipe 70 +
evidence 41 tests pass).

## Tests

- New: 19 (16 hardening + 3 boundary).
- Full regression: **446/446** (docpipe 70 + evidence 41 + research 335).
- Ruff: clean.

## Deferred risks (unchanged, documented in README)

- `claim_evidence` source_type/context (C4) — migration 2 will add it.
- `claim_proposed` event crash window (P2).
- Structural contradiction advisory-only (never authoritative).

## Next

Commit 11 recommendation: C3 verification layer (claims ↔ evidence ↔
citations, semantic contradiction via ReasonerPort) — now safe to build on
the serialized, atomic, versioned foundation.
