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

Commit 12 recommendation: C4 external sources (web/Scholar) — the C3
proposal layer is ready; C4 adds retriever adapters + `claim_evidence`
context (migration 3). Or: extend C3 with a real reasoner adapter that
generates decomposition/contradiction/context proposals.

---

# Commit 11 (C3) — Reasoner Verification Layer

## What was added

1. **ReasonerPort extension** (`research/reasoner.py`): five new
   proposal-only methods (`propose_decomposition`,
   `propose_evidence_relations`, `propose_semantic_contradictions`,
   `propose_context`, `propose_coverage`) plus five immutable proposal
   dataclasses (`ClaimDecompositionProposal`, `EvidenceRelationProposal`,
   `SemanticContradictionProposal`, `ClaimContextProposal`,
   `CoverageProposal`). `DeterministicReasoner` implements them with a
   deterministic baseline (decomposition/semantic/context/coverage return
   empty/None; evidence relations mirror the loop's supporting policy).

2. **Deterministic verifier** (`research/verifier.py`): validates every
   proposal against persisted objects before any write:
   - decomposition: source claim exists; child evidence exists, valid,
     and already linked to the source claim (fabricated IDs rejected);
   - evidence relation: claim/evidence exist, evidence valid, role legal;
   - semantic contradiction: both claims exist, evidence on each side
     exists + valid + linked to the corresponding claim, rationale
     non-empty, strength ∈ [0,1], context keys whitelisted;
   - context: claim exists, keys ⊆ CONTEXT_KEYS, values str|None;
   - coverage: label ∈ COVERAGE_LABELS and consistent with the derived
     observation.
   `apply_*` functions persist only validated proposals through the
   domain layer (claims.create_claim / claims.link_evidence /
   store.create_contradiction / store.set_claim_*). Never a direct raw
   write.

3. **Coverage (derived, deterministic)**: `coverage_for_claim` re-evaluates
   the claim and derives exactly one of `directly_supported`,
   `partially_supported`, `unsupported`, `contradictory`. The labels
   `insufficient`/`irrelevant` are accepted proposal intents but never
   derived — "no evidence found" is NEVER a research-gap claim.

4. **Semantic contradiction = observation only**: persisted as
   `claim_contradictions` rows with `kind='semantic'`; neither claim's
   status changes; both findings remain in synthesis; multiple
   observations are distinguishable and idempotent.

5. **Schema migration 2** (via the Commit 10 mechanism): `claims` gains
   `parent_claim_id` (decomposition provenance) and `context` (JSON).
   Migration is an idempotent callable using the repo's
   `add_column_if_missing`. SCHEMA_VERSION=2.

6. **Minimal API/CLI**: read-only `GET /research/claims/{id}/coverage`
   (derived observation) and `GET /research/claims/{id}/contradictions`
   (open observations); CLI `hermes research coverage <claim_id>`. No
   proposal-submission endpoints (proposals are a Python-level interface).

## Tests

29 new tests (`tests/research/test_research_verifier*.py`): proposal
validation, decomposition provenance + unsupported-child exclusion,
semantic contradiction (valid/wrong-side/fabricated/no-auto-invalidation/
distinguishable), context (persistence/backward-compat/malformed),
coverage (all four derivations + no research-gap conflation +
contradicting-proposal rejection), synthesis safety (both sides
traceable, citations resolve to persisted IDs), API/CLI inspection,
migration v2 (fresh + v1→v2 upgrade preserving data + idempotent).

Full regression: **475/475** (docpipe 70 + evidence 41 + research 364).
Ruff: clean.

## Files changed

```
M research/reasoner.py   C3 proposal types + ReasonerPort extension + baseline impl
A research/verifier.py   deterministic validator + apply + coverage
M research/model.py      Claim + parent_claim_id + context
M research/store.py      migration 2 + set_claim_parent/context
M research/api_claims.py coverage + contradictions endpoints (read-only)
M research/cli.py        coverage command
M research/README.md     C3 status
M research/IMPLEMENTATION_REPORT.md  (this section)
A tests/research/test_research_verifier.py    13 tests
A tests/research/test_research_verifier2.py   16 tests
M tests/research/test_research_model.py       Claim.to_dict keys
M tests/research/test_research_hardening.py   v1 upgrade test (realistic v1 DB)
```
