# AUDIT TEST STATUS

Verified test results that correspond to the audited commit
`4306f15ff029eac80feef4a38dade0e25543a85e`.

## Primary quality gate

Command (run on the development machine, Windows host, Python 3.11.1,
at HEAD `4306f15ff`, 2026-08-11):

```bash
pytest tests/docpipe tests/evidence tests/research
```

**Actual result: `550 passed`** (docpipe 70 + evidence 41 + research 439).

```text
70 passed  tests/docpipe   (C0)
41 passed  tests/evidence  (C1)
439 passed tests/research  (C2/C3, incl. 43 LLM reasoner + 3 E2E + boundary)
550 passed total, 1 warning in ~143s
```

## Lint gate

```bash
ruff check research/ tests/research/ tests/docpipe/ tests/evidence/
```

**Actual result: `All checks passed!`**

## Pre-existing / environmental failures (outside the audit scope)

The full repository also contains unrelated suites (`tests/acp`,
`tests/agent`, `tests/cli`, `tests/cron`, `tests/acp_adapter`, ...) that
fail on this Windows host for environmental reasons:

- `tests/acp` (2): `OSError: [WinError 6] The handle is invalid` in
  asyncio Proactor/IOCP when spawning subprocesses.
- `tests/agent` (8): tilde-path expansion, sandbox/mirror classification,
  home-directory guards — Windows path semantics.
- `tests/cli`/`tests/cron`/`tests/acp_adapter` (14): tilde/WSL/file-
  permission/secret-capture expectations specific to POSIX environments.

**Verification that these are pre-existing, not Commit-12 regressions:**

1. The same failing tests were run at baseline commit `e57054a2b` (Commit
   11, before Commit 12) in an isolated git worktree — identical failures
   (e.g. `tests/agent` produced the exact same 8 failures with 2313 passed
   at both commits).
2. `tests/acp` fails identically at the baseline.
3. **No test outside `tests/research/` imports `research`** (verified with
   `grep -rln "import research\|from research" tests/`), so Research Agent
   changes cannot affect them.
4. One flaky CLI test (`test_session_not_found_goes_to_stdout_in_full_mode`)
   passed when run in isolation — environmental interaction, not a code
   regression.

## Honest limitations of this record

- The full-suite result for the ENTIRE repository was NOT reproduced to
  completion: a full `pytest` run takes on the order of an hour on this
  host and is dominated by the unrelated suites above. The primary quality
  gate (all research-relevant suites) was run to completion and passed.
- Test counts here were obtained at commit `4306f15ff` on the development
  machine; the audit repository contains the identical tree (same commit
  hash), so results transfer, but if you re-run in a different environment,
  the pre-existing environmental failures may differ.
- No test was modified for this audit snapshot.

## Reproduction

To re-run the primary gate in this repository:

```bash
python -m pytest tests/docpipe tests/evidence tests/research
python -m ruff check research/ tests/research/ tests/docpipe/ tests/evidence/
```
