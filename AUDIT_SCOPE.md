# AUDIT SCOPE

## AUDIT TARGET

```
Commit 12
4306f15ff029eac80feef4a38dade0e25543a85e
```

## PURPOSE

Independent architectural and correctness audit of the Hermes Research Agent
subsystem (C0 → C1 → C2 → C3 → C3 hardening → Commit 12).

## SCOPE

```
C0  docpipe/   — document/chunk pipeline (PDF → chunks → SQLite + FTS5)
C1  evidence/  — provenance-checked evidence layer (tamper detection)
C2  research/  — session/loop/claims/synthesis core, API, CLI
C3  research/  — reasoner proposals + deterministic verifier + coverage
    hardening  — atomic decomposition, observation keys, migration v3
Commit 12      — LLMReasoner + LLMTransport adapter (proposal-only)
```

## READ-ONLY REVIEW

The external reviewer should not modify the repository. This repository is an
immutable audit snapshot for review purposes only.

## SOURCE OF TRUTH

- Implementation and tests are authoritative.
- Previous review reports (e.g. `research/IMPLEMENTATION_REPORT.md`) are
  advisory only.

## IMPORTANT

- Do not assume previous reviews are correct.
- Verify findings against the actual implementation.

## NO IMPLEMENTATION TASK

This repository exists for audit purposes only. Commit 13 has not begun.

---

## High-level architecture (as implemented)

```
Documents (PDF/text)
   ↓  docpipe/
Chunks (+ FTS5 index)
   ↓  evidence/
Evidence (validated, provenance, tamper-checked)
   ↓  research/ (C2)
Claims + ClaimEvidenceLinks
   ↓  research/verifier.py (C3)
Verifier (proposal validation against persisted state)
   ↓
Coverage / Contradiction observations (derived, authoritative)

ResearchLoop (research/loop.py, 5 phases)
   ↓
ReasonerPort (research/reasoner.py)
   ↓
DeterministicReasoner   OR   LLMReasoner (Commit 12)
   ↓                            ↓
Proposals            (typed proposal objects)
   ↓
Deterministic Verifier (research/verifier.py)
   ↓
Domain Apply (claims.py / store.py — write_txn)
   ↓
Synthesis (research/synthesis.py) → ResearchAnswer
```

Notes on the actual implementation (differences from the conceptual diagram
above are documented, not invented):

- The `ResearchLoop` currently uses `DeterministicReasoner` as its reasoner.
  `LLMReasoner` implements the same `ReasonerPort` but is NOT wired into the
  loop yet — it is a tested, opt-in adapter capability (Commit 12).
- `ResearchGap` objects exist in synthesis output, but there is no engine
  that auto-generates gaps from missing evidence ("no evidence found" is
  deliberately NOT treated as a research gap).
- Retrieval is local FTS5 keyword search only — no web sources, no vector
  DB, no embeddings.
- Semantic contradictions are recorded as observations in
  `claim_contradictions`; they never change claim status.
