# C2 — Research Engine

Deterministic research sessions over local documents: create a session,
bind C0 documents, run a persistent research loop (planning → retrieval →
evidence → claims → evaluation), and synthesize a provenance-backed answer.
**LLM-free**: no model is called anywhere in this layer.

```
PDF ──► C0 (docpipe) ──► chunks + FTS5 ──► C1 (evidence) ──► C2 (research)
   ingest                search/BM25        snapshot/validate   loop + claims
                                                                 + synthesis
```

## Architecture

```
research/
├── config.py         engineering defaults (chunk of limits, lease TTL)
├── model.py           ResearchSession/ResearchQuery/Claim/ClaimEvidenceLink/
│                      LoopIteration/Contradiction/ResearchEvent + ID helpers
├── store.py           research_* tables + lease + schema-version migrations
├── retriever.py       LocalRetriever: C0 search + C1 evidence (orchestration only)
├── reasoner.py        ReasonerPort (Protocol) + DeterministicReasoner (proposal-only)
├── claims.py          claim creation, transition matrix, linking, evaluator
├── sufficiency.py     claim/session sufficiency assessment (read-only)
├── contradictions.py  explicit/structural detection + resolution + synthesis guard
├── loop.py            ResearchLoop: phases, stop conditions, resume, lease
├── synthesis.py       ResearchAnswer construction + persistence
├── api.py             combined C0+C1+C2 FastAPI app + error mapping
├── api_sessions.py    session endpoints
├── api_claims.py      claims/evidence inspection endpoints
├── api_answer.py      answer/synthesis endpoints
└── cli.py             `hermes research ...` (thin adapter)
```

Data flow: `session → queries → iterations → evidence (C1) → claims →
claim_evidence links → ResearchAnswer (statements + derived citations) →
gaps (insufficient/contradicted claims)`.

## Deterministic core principle

Every authoritative decision is made by deterministic code:

* **Claim status is decided by `claims.evaluate_claim`** — it re-validates
  every linked evidence via C1 (`evidence.validation.validate_evidence`)
  on each call; persisted statuses are never trusted.
* **The reasoner only proposes** (`plan_queries`, `construct_claims`,
  `adjudicate`, `synthesize`). It has no store access and cannot set
  claim/session status (enforced by `tests/research/test_arch_boundary.py`).
* **Synthesis only consumes `supported` claims** whose supporting evidence
  validates right now; citations are derived from `claim_evidence` —
  callers can never inject citation text.
* A future LLM adapter may replace `DeterministicReasoner` behind the
  `ReasonerPort`; it must remain proposal-only.

## Claim lifecycle

```
proposed ──► supported | insufficient | contradicted | invalidated
supported ──► insufficient | contradicted | invalidated
insufficient ──► supported | contradicted | invalidated
contradicted ──► supported | withdrawn | invalidated
withdrawn ──► invalidated            (terminal: never re-evaluated)
```

Every transition commits its status + audit event atomically.

## Evidence validation / provenance

Evidence is a verbatim C1 snapshot of a chunk with full provenance
(`document_id`, `chunk_id`, `page_number`, `section_path`, `locator`).
Validation re-checks content/document/chunk/page/source-type against the
live C0 state — tampering a chunk invalidates the evidence, demotes the
claim, and excludes it from synthesis. Restoration recovers it.

## Synthesis

`ResearchAnswer` = `{session_id, question, summary, statements[],
gaps[]}`; statements carry `{claim_id, statement, citations[]}` where
each citation is `{evidence_id, document_id, page, locator}` derived from
validated supporting links. The summary is a deterministic count — no
prose, no hallucination. Serialization is byte-deterministic
(`sort_keys`), with no timestamps in the payload.

**Stale answers:** `GET .../answer` defaults to `refresh=true` (re-validate
+ re-synthesize; tampered evidence removes the affected statement).
`refresh=false` returns the stored answer and explicitly marks it
`stale_possible=true` — never silently authoritative.

## API (FastAPI, mounted on the C0+C1 app)

| Endpoint | Purpose |
|---|---|
| `POST /research/sessions` | create session (validates documents) |
| `GET /research/sessions/{id}` | session state + sufficiency |
| `POST /research/sessions/{id}/start` · `/iterate` · `/abort` | loop lifecycle |
| `GET /research/sessions/{id}/claims` | list claims |
| `GET /research/claims/{id}` · `/evidence` | claim + link inspection |
| `GET /research/sessions/{id}/answer?refresh=` | answer (safe default) |
| `POST /research/sessions/{id}/synthesize` | run synthesis |

Errors follow `{"error": "..."}`: 404 not found · 409 lifecycle/busy/synthesis · 422 invalid input · 500 internal (no stack traces).

## CLI (`hermes research …` / `python -m research.cli`)

`new --question … --document-id …` (repeatable) · `list` · `show` ·
`start` · `iterate` · `abort` · `claims` · `answer [--no-refresh]` ·
`synthesize` · `db-path` · global `--json`. Exit codes 0/1/2; errors go to
stderr without tracebacks; `--json` keeps stdout pure JSON.

## Persistence model

One SQLite file (`<HERMES_HOME>/docpipe.db`, WAL) shared by C0/C1/C2.
Research tables: `research_sessions`, `research_queries`, `claims`,
`claim_evidence`, `research_iterations`, `research_iteration_evidence`,
`research_iteration_state`, `claim_contradictions`,
`research_session_documents`, `research_session_locks`,
`research_events`. FK cascades: session children delete with the session;
`claim_evidence → evidence` is NO ACTION (deleting linked evidence fails
loudly).

**Schema versioning:** `PRAGMA user_version` anchors the research schema
version (currently 1). Migrations are ordered, applied inside one
transaction each (schema + version bump commit together); a database
newer than the build is rejected deterministically. C0/C1 keep their own
idempotent `CREATE TABLE IF NOT EXISTS`.

## Crash / resume semantics

* Phases are idempotency-bounded by `UNIQUE(session, iteration, phase)`:
  a completed phase is never rerun; a `running` row (crash mid-phase) is
  re-executed — every side effect (queries/claims/links/evidence) is
  deduplicated.
* Evidence returned per iteration and the known-evidence set are
  persisted, so resume works from the database, never from memory.
* Crash between any authoritative transition and its audit event is
  impossible: they commit in one transaction (claims, phases, session
  lifecycle, synthesis).

## Concurrency semantics

A **per-session execution lease** (`research_session_locks`, token +
expiry) serializes `start`/`iterate`/`resume`: one logical run per
session at a time; a concurrent caller gets `SessionBusyError`
(API 409 "session busy"). The lease survives process boundaries and
expires after `SESSION_LOCK_TTL_SECONDS` (300s), so a crashed holder's
session can be taken over. `abort` is a poison pill: it does not take the
lease and stops the loop at the next iteration boundary.

## Transaction / atomicity guarantees

| Operation | Atomic |
|---|---|
| claim transition + `claim_status_changed` | yes |
| evidence link + `evidence_linked` | yes |
| phase completion/failure + phase event | yes |
| session status + lifecycle event (`start`/`abort`/`_apply_stop`/`failed`) | yes |
| `final_answer` + `synthesized` event | yes |
| migration SQL + `user_version` bump | yes |
| phase side effects | intentionally idempotent/retryable (not a single txn) |

## Current limitations (deliberate)

* **FTS5 keyword retrieval only**: no stemming, no diacritic folding, no
  vector search. Queries must use exact token forms.
* **Synthesis summary is a count**, not prose (LLM-free).
* `section_path` is always NULL (no heading detection).
* One evidence per chunk (C1 UNIQUE); one claim per normalized statement
  per session.

## Known deferred risks

* `claim_evidence` lacks `source_type`/context — multi-paper and web
  sources (C4) will need a small schema addition (migration 2).
* `claim_proposed` event could be lost in a tiny crash window between
  claim insert and event (P2; claim itself is deduplicated).
* Structural contradictions are advisory heuristics (Jaccard) — never
  authoritative (see below).

## Roadmap (C3/C4/C5 — NOT implemented)

* **C3** — verification: claims ↔ evidence ↔ citations; semantic
  contradiction adjudication (through the ReasonerPort).
* **C4** — external sources: web/Scholar; requires `claim_evidence`
  context + C1 `source_type` extension.
* **C5** — full research agent.

### Retrieval gap ≠ Research gap

C2's "gap" is a **retrieval/evidence gap**: an `insufficient` or
`contradicted` claim (e.g. "this claim has no supporting evidence in the
bound documents"). C2 does **NOT** perform scientific research-gap
discovery — it never identifies unstudied populations, untested
conditions, methodological limitations, or unanswered questions. Those
are C3+ semantic tasks. Do not confuse the two.

### Multi-paper limitations

Provenance already works across papers (evidence → document), so a claim
can be supported by Paper A and contradicted by Paper B. What does NOT
exist yet: contextual comparison (population/dataset/method/condition),
semantic contradiction, and per-paper gap analysis. Documented for C3/C4.
