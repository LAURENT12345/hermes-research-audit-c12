# AUDIT TIMELINE

Development progression from C0 through Commit 12, based on the actual Git
history of this branch (verified with `git log`) and the implementation
reports in `research/IMPLEMENTATION_REPORT.md`. Nothing here is fabricated;
commit hashes are real.

| Milestone | Commit | Purpose | Major implementation | Architectural decision | Tests added | Known deferred risks |
|---|---|---|---|---|---|---|
| C0 | `6b4dca3c3` | Local document pipeline | `docpipe/`: documents + chunks + SQLite + FTS5 index; `file_hash` UNIQUE dedupe | LLM-free deterministic ingest; FTS5 keyword index as the retrieval substrate | docpipe suite | FTS5 keyword recall limits (no semantic search) |
| C1 | `d3befbeee` | Evidence pipeline | `evidence/`: evidence records with provenance (document/chunk/locator), chunk-unique index, content snapshot | Evidence content is a snapshot checkable against the live chunk (tamper detection later) | evidence suite | — |
| C2 core | `2038ab09f` | C2 schema + model + store | `research/`: sessions, queries, claims, links, loop state, events tables | Domain dataclasses mirror tables; `S-`/`C-` id patterns | store/model tests | — |
| C2 | `04130ab33` | Retriever + reasoner ports | `LocalRetriever` (FTS5 → Evidence), `ReasonerPort` | Retrieval and reasoning behind ports; reasoner may only propose | retriever/reasoner tests | — |
| C2 | `955efb83c` | Claims and sufficiency | `create_claim` (dedupe by normalized statement), `link_evidence`, `evaluate_claim` | Claim status recomputed deterministically from validated evidence | claims/sufficiency tests | — |
| C2 | `c6979d0b1` | Contradictions and resolution | structural contradiction adjudication through ReasonerPort | Contradiction as adjudication outcome | contradictions tests | structural heuristic is overlap-based |
| C2 | `0a61b2308` | Loop state machine | 5-phase loop (planning/retrieving/evidencing/claiming/evaluating), stop conditions, retry-once | Phase idempotency via `UNIQUE(session_id, iteration, phase)` | loop tests | crash window between phase statements (mitigated by idempotency) |
| C2 | `a5900d6a2` | Synthesis and answer | `build_answer` → ResearchAnswer, statements, citations | Citations resolve only to persisted, claim-linked evidence ids | synthesis tests | — |
| C2 fixes | `6580a4d6f`, `0d567a3c7`, `a896c8b98`, `2a8eb7fe4` | Hardening | `set_final_answer` preserves status; terminal claim guard; strict answer deserialization | Terminal claim statuses final; strict deserialization | targeted tests | — |
| C2 API | `0e790adb2` | Research API | `api*.py` endpoints; atomicity hardening; lease → 409 | Lease-based concurrency guard | api tests | — |
| C2 CLI | `9bace8d87` (+`59414e536` fixup) | Research CLI | `hermes research ...` commands | CLI mirrors API surface | cli tests | — |
| C2 | `619e81f92` | Full acceptance coverage | end-to-end pipeline + crash/resume acceptance tests | Black-box acceptance over temp DBs | 5 acceptance files | — |
| C2 | `35152add8` | Architectural hardening + docs | README + IMPLEMENTATION_REPORT checkpoints | Documentation as part of the milestone | — | — |
| **Commit 11 (C3)** | `e57054a2b` | C3 reasoner verification layer | ReasonerPort `propose_*` + 5 proposal types; deterministic verifier (`validate_*`/`apply_*`); derived coverage; semantic contradiction observations; migration 2 (`parent_claim_id`, `context`); coverage/contradictions endpoints | **Proposal-only reasoner; verifier is the only gate; semantic contradiction = observation, never status change** | 29 verifier tests | migration-2 schema delta vs fresh v2; semantic contradiction advisory-only |
| **C3 hardening** | `e6284e7c8` | Atomic decomposition + observation keys | atomic `create_decomposed_claims`; schema v3 (`observation_key` + UNIQUE convergence); migration rollback; tamper re-validation | Contradiction observations idempotent + distinguishable | hardening c3/c3b tests | `claim_proposed` event crash window (P1) |
| **Commit 12** | `4306f15ff` | **LLM reasoner adapter** | `research/llm_reasoner.py` (LLMReasoner: deterministic serialization, `REASONER_SYSTEM_PROMPT_V1`, fail-closed parser), `research/llm_transport.py` (LLMTransport + ChatCompletionsTransport, env-only credentials); config defaults; boundary tests | **LLM output is advisory/proposal data only; parser fail-closed; fabricated IDs pass parser and are rejected by verifier; adapter not wired into loop** | 43 `test_llm_reasoner.py` + 3 `test_llm_e2e.py` + 2 boundary tests | LLMReasoner not wired into ResearchLoop; no real-provider integration test; C4/web/vector/gap engine not started |

## Commit chain (tip of this branch)

```
4306f15ff  Commit 12 — feat(research): add LLM reasoner adapter
e6284e7c8  C3 hardening — fix(research): complete C3 hardening (atomic decomposition, observation keys)
e57054a2b  Commit 11 — feat(research): add C3 reasoner verification layer
35152add8  feat(research): architectural hardening and documentation checkpoint
619e81f92  test(research): add full system acceptance coverage
59414e536  fixup! feat(research): add research CLI
9bace8d87  feat(research): add research CLI
0e790adb2  feat(research): research API with atomicity hardening
2a8eb7fe4  fix(research): harden answer deserialization against malformed fields
a896c8b98  fix(research): strict answer deserialization errors
0d567a3c7  test(research): terminal claim evaluation guard
6580a4d6f  test(research): set_final_answer preserves session status
a5900d6a2  feat(research): synthesis and answer
0a61b2308  feat(research): research loop state machine
c6979d0b1  feat(research): contradictions and resolution
955efb83c  feat(research): claims and sufficiency
04130ab33  feat(research): retriever + reasoner ports
2038ab09f  feat(research): C2 schema + model + store — research sessions, claims, links, loop state, events
d3befbeee  feat(evidence): C1 evidence pipeline — chunks -> validated provenance Evidence (LLM-free)
6b4dca3c3  feat(docpipe): C0 local document pipeline — PDF -> chunks -> SQLite + FTS5 (LLM-free)
```
