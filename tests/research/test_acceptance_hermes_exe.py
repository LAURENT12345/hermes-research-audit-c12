"""Acceptance (Scenarios E + G): the real ``venv/Scripts/hermes.exe`` as the research CLI.

Drives the installed console script as an actual subprocess against an
isolated ``HERMES_HOME`` — the same store seeded in-process through
``docpipe.ingest`` — and verifies the C8/C9 CLI contract end-to-end:

* ``hermes research --help`` is fast (exit 0) and lists every subcommand;
  ``research`` is a member of ``_BUILTIN_SUBCOMMANDS`` in
  ``hermes_cli/main.py``, so the CLI fast path parses it without plugin
  discovery.
* new / start / claims / answer / synthesize / show / db-path all work
  through the executable: correct exit codes (0 ok, 1 domain error,
  2 argparse misuse), correct stdout/stderr split, ``--json`` output
  parseable on success AND error paths.
* Scenario G error boundaries: unknown session, invalid input, and even
  unexpected internal failures (corrupted store) exit cleanly with
  ``error: ...`` and NEVER leak a Python traceback on stderr.

Environment-stability rule (per the task): if ``hermes.exe`` is missing or
hangs in this environment, the module fixture skips every test with the
exact reason — assertions are never weakened and nothing is faked as a
pass. Observed on the development machine: ~26 s cold start (first spawn,
bytecode compile + AV scan), ~1-2 s warm; the per-test hard deadline is
120 s as required.

HERMES_HOME isolation note: the child's HERMES_HOME is shaped
``<tmp>/profiles/<name>`` because ``hermes_cli.main._apply_profile_override``
only trusts a pre-set HERMES_HOME when its immediate parent directory is
named ``profiles``; any other layout would be silently redirected to the
sticky active profile (here: ``nckh``) and the test would touch real data.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

CHUNK_TEXTS = [
    "Scaled dot product attention divides by the square root of dk.",
]
QUESTION = "Why does scaled dot product attention divides by the square root of dk?"

#: Per-test hard budget (task requirement: "Mỗi test timeout 120s").
EXE_TIMEOUT_S = 120
#: Upper bound proving ``research --help`` is "fast". Observed ~26 s cold /
#: ~2 s warm on the dev machine; generous enough for slower CI, meaningful
#: against the 120 s per-test cap.
HELP_FAST_BOUND_S = 90

_TERMINAL_STATUSES = ("complete", "complete_with_gaps", "failed", "aborted")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _find_hermes_exe() -> Path | None:
    """Locate the real console script via sys.executable or repo layout."""
    exe_name = "hermes.exe" if os.name == "nt" else "hermes"
    candidates: list[Path] = []
    here = Path(sys.executable).resolve()
    # The venv this test suite is running under: <venv>/Scripts/hermes.exe
    candidates.append(here.parent / exe_name)
    candidates.append(here.parent.parent / "Scripts" / exe_name)
    # The repository checkout layout (either venv name).
    candidates.append(_repo_root() / "venv" / "Scripts" / exe_name)
    candidates.append(_repo_root() / ".venv" / "Scripts" / exe_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _child_env(isolated_home: Path) -> dict:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(isolated_home)
    return env


class _ExeTimeout(Exception):
    """hermes.exe exceeded the per-test deadline (unstable mid-run)."""


def _spawn(
    cmd: list[str],
    isolated_home: Path,
    deadline: float,
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run a command with the isolated HERMES_HOME, bounded by ``deadline``."""
    remaining = deadline - time.monotonic()
    if remaining <= 1:
        raise _ExeTimeout("per-test deadline exhausted before subprocess spawn")
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=remaining,
            env=_child_env(isolated_home),
            cwd=str(cwd or _repo_root()),
            **kwargs,
        )
    except subprocess.TimeoutExpired as exc:
        raise _ExeTimeout(
            f"hermes.exe timed out after {remaining:.0f}s: {' '.join(cmd)}"
        ) from exc


def _run_research(
    exe: Path,
    args: list[str],
    isolated_home: Path,
    deadline: float,
) -> subprocess.CompletedProcess:
    """``hermes research <args>`` as a real subprocess."""
    return _spawn([str(exe), "research", *args], isolated_home, deadline)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hermes_exe(tmp_path_factory) -> Path:
    """The real ``hermes.exe``; skips the whole module when unavailable/hangs.

    Unstable here = the executable cannot be found or ``research --help``
    hangs within the 90 s probe. The probe also warms the bytecode cache, so
    the per-test spawns run at warm speed. A probe that exits non-zero is
    NOT skipped — that is a real regression the tests must surface.
    """
    exe = _find_hermes_exe()
    if exe is None:
        pytest.skip(
            "venv/Scripts/hermes.exe not found (tried sys.executable parent "
            "and repo venv/.venv layouts); cannot run real-executable acceptance"
        )
    probe_home = tmp_path_factory.mktemp("hermes-exe-probe") / "profiles" / "probe"
    probe_home.mkdir(parents=True)
    try:
        _spawn(
            [str(exe), "research", "--help"],
            probe_home,
            deadline=time.monotonic() + 90,
        )
    except _ExeTimeout:
        pytest.skip(
            f"hermes.exe research --help hung for 90s (probe HERMES_HOME="
            f"{probe_home}); executable is unstable in this environment"
        )
    except OSError as exc:  # pragma: no cover - defensive
        pytest.skip(f"hermes.exe failed to start: {exc}")
    return exe


@pytest.fixture
def isolated_home(tmp_path) -> Path:
    """Fresh profile-shaped HERMES_HOME per test (see module docstring)."""
    home = tmp_path / "profiles" / "hermes-exe"
    home.mkdir(parents=True)
    return home


@pytest.fixture
def deadline() -> float:
    return time.monotonic() + EXE_TIMEOUT_S


@pytest.fixture
def seeded_doc(isolated_home) -> int:
    """Ingest a PDF via ``docpipe.ingest`` into the SAME isolated HERMES_HOME.

    The subprocess resolves the store at ``<HERMES_HOME>/docpipe.db``, so
    seeding here is exactly the required setup: ingest in Python first,
    then drive ``hermes.exe research new --document-id <id> ...``.
    """
    import docpipe.ingest as c0ingest

    from tests.docpipe.pdf_factory import build_pdf

    pdf = isolated_home / "paper.pdf"
    pdf.write_bytes(build_pdf(CHUNK_TEXTS))
    ingested = c0ingest.ingest_pdf(pdf, db_path=isolated_home / "docpipe.db")
    return ingested.document_id


# ---------------------------------------------------------------------------
# Scenario E — the real CLI through the executable
# ---------------------------------------------------------------------------


def test_research_help_exits_zero_fast(hermes_exe, isolated_home, deadline):
    """``hermes research --help``: exit 0, fast, and lists every subcommand."""
    started = time.monotonic()
    proc = _run_research(hermes_exe, ["--help"], isolated_home, deadline)
    elapsed = time.monotonic() - started
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert elapsed < HELP_FAST_BOUND_S, (
        f"research --help took {elapsed:.0f}s (bound {HELP_FAST_BOUND_S}s)"
    )
    for name in (
        "new", "list", "show", "start", "iterate", "abort",
        "claims", "answer", "synthesize", "db-path",
    ):
        assert name in proc.stdout, f"subcommand {name!r} missing from --help"
    assert "Traceback" not in proc.stderr


def test_research_registered_as_builtin_subcommand(hermes_exe, isolated_home, deadline):
    """``research`` is a member of ``_BUILTIN_SUBCOMMANDS`` (fast-path skip)."""
    proc = _spawn(
        [
            sys.executable,
            "-c",
            "from hermes_cli.main import _BUILTIN_SUBCOMMANDS; "
            "print('research' in _BUILTIN_SUBCOMMANDS)",
        ],
        isolated_home,
        deadline,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "True"


def test_db_path_resolves_into_isolated_home(hermes_exe, isolated_home, deadline):
    """``db-path`` proves the subprocess really uses the isolated HERMES_HOME."""
    expected = str(isolated_home / "docpipe.db")
    proc = _run_research(hermes_exe, ["db-path"], isolated_home, deadline)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == expected
    assert proc.stderr == ""
    proc = _run_research(hermes_exe, ["--json", "db-path"], isolated_home, deadline)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"db_path": expected}


def test_lifecycle_new_start_claims_synthesize_answer_show(
    hermes_exe, isolated_home, seeded_doc, deadline
):
    """Scenario E: full research lifecycle through the executable.

    new -> start -> (iterate to terminal if needed) -> claims -> synthesize
    -> answer -> show. Every step is a real subprocess against the store
    seeded via ``docpipe.ingest`` in the same HERMES_HOME.
    """
    doc = seeded_doc
    proc = _run_research(
        hermes_exe,
        ["--json", "new", "--question", QUESTION, "--document-id", str(doc)],
        isolated_home,
        deadline,
    )
    assert proc.returncode == 0, proc.stderr
    session_id = json.loads(proc.stdout)["session_id"]
    assert re.fullmatch(r"S-\d{6}", session_id)

    proc = _run_research(hermes_exe, ["--json", "start", session_id], isolated_home, deadline)
    assert proc.returncode == 0, proc.stderr
    status = json.loads(proc.stdout)["session_status"]
    assert status in ("active", "complete", "complete_with_gaps")

    # Iterate until terminal (a single-document session normally completes
    # in the first start, but stay robust to content-dependent loops).
    for _ in range(6):
        proc = _run_research(hermes_exe, ["--json", "show", session_id], isolated_home, deadline)
        assert proc.returncode == 0, proc.stderr
        body = json.loads(proc.stdout)
        if body["status"] in _TERMINAL_STATUSES:
            break
        proc = _run_research(hermes_exe, ["iterate", session_id], isolated_home, deadline)
        assert proc.returncode == 0, proc.stderr

    proc = _run_research(hermes_exe, ["--json", "claims", session_id], isolated_home, deadline)
    assert proc.returncode == 0, proc.stderr
    claims = json.loads(proc.stdout)["claims"]
    assert claims, "expected at least one claim"
    assert claims[0]["claim_id"].startswith("C-")
    assert claims[0]["status"] == "supported"
    assert claims[0]["statement"]

    proc = _run_research(hermes_exe, ["--json", "synthesize", session_id], isolated_home, deadline)
    assert proc.returncode == 0, proc.stderr
    syn = json.loads(proc.stdout)
    assert syn["session_id"] == session_id
    assert syn["claim_count"] >= 1
    assert syn["answer"]["statements"]

    proc = _run_research(hermes_exe, ["--json", "answer", session_id], isolated_home, deadline)
    assert proc.returncode == 0, proc.stderr
    ans = json.loads(proc.stdout)["answer"]
    assert ans["summary"]
    assert ans["statements"], "expected cited statements"
    assert ans["statements"][0]["citations"]

    # Human-mode show carries the session id and question.
    proc = _run_research(hermes_exe, ["show", session_id], isolated_home, deadline)
    assert proc.returncode == 0, proc.stderr
    assert session_id in proc.stdout
    assert "question:" in proc.stdout


# ---------------------------------------------------------------------------
# Scenario G — error boundaries through the executable
# ---------------------------------------------------------------------------


def test_new_rejects_invalid_input_cleanly(
    hermes_exe, isolated_home, seeded_doc, deadline
):
    """Invalid input to ``new``: exit 1/2, clean message, never a traceback."""
    doc = seeded_doc
    # Blank question -> domain error, exit 1, stderr only.
    proc = _run_research(
        hermes_exe,
        ["new", "--question", "   ", "--document-id", str(doc)],
        isolated_home,
        deadline,
    )
    assert proc.returncode == 1
    assert "question must not be empty" in proc.stderr
    assert proc.stdout == ""
    assert "Traceback" not in proc.stderr + proc.stdout
    # Unknown document -> domain error, exit 1, stderr only.
    proc = _run_research(
        hermes_exe,
        ["new", "--question", "Q?", "--document-id", "999"],
        isolated_home,
        deadline,
    )
    assert proc.returncode == 1
    assert "document 999 not found" in proc.stderr
    assert proc.stdout == ""
    assert "Traceback" not in proc.stderr + proc.stdout
    # Missing --document-id -> argparse usage error (exit 2), still clean.
    proc = _run_research(hermes_exe, ["new", "--question", "Q?"], isolated_home, deadline)
    assert proc.returncode == 2
    assert "required" in proc.stderr
    assert "Traceback" not in proc.stderr + proc.stdout
    # JSON mode: the error stays machine-readable on stdout, stderr empty.
    proc = _run_research(
        hermes_exe,
        ["--json", "new", "--question", "Q?", "--document-id", "999"],
        isolated_home,
        deadline,
    )
    assert proc.returncode == 1
    assert proc.stderr == ""
    assert json.loads(proc.stdout) == {"error": "document 999 not found", "exit": 1}


def test_unknown_session_clean_errors(hermes_exe, isolated_home, seeded_doc, deadline):
    """Unknown session across every session-scoped command: exit 1, no traceback."""
    seeded_doc  # ensure a real store exists so the failure is the session, not the DB
    for sub in ("show", "claims", "answer", "synthesize", "start"):
        proc = _run_research(hermes_exe, [sub, "S-999999"], isolated_home, deadline)
        assert proc.returncode == 1, sub
        assert "S-999999 not found" in proc.stderr, sub
        assert proc.stdout == "", sub
        assert "Traceback" not in proc.stderr + proc.stdout, sub
    # JSON mode: machine-readable error on stdout, empty stderr.
    proc = _run_research(hermes_exe, ["--json", "show", "S-999999"], isolated_home, deadline)
    assert proc.returncode == 1
    assert proc.stderr == ""
    assert json.loads(proc.stdout) == {"error": "session S-999999 not found", "exit": 1}
    for sub in ("claims", "answer", "synthesize", "start"):
        proc = _run_research(hermes_exe, ["--json", sub, "S-999999"], isolated_home, deadline)
        assert proc.returncode == 1, sub
        body = json.loads(proc.stdout)
        assert body["error"] and body["exit"] == 1, sub
        assert "Traceback" not in proc.stderr + proc.stdout, sub


def test_unknown_subcommand_usage_error(hermes_exe, isolated_home, deadline):
    """argparse-level misuse: exit 2 with usage on stderr, never a traceback."""
    for args in (["bogus"], ["--json", "bogus"]):
        proc = _run_research(hermes_exe, args, isolated_home, deadline)
        assert proc.returncode == 2
        assert "usage:" in proc.stderr
        assert "Traceback" not in proc.stderr + proc.stdout


def test_internal_error_never_leaks_traceback(hermes_exe, isolated_home, deadline):
    """Unexpected internal failure (corrupted store) -> exit 1, no traceback.

    SQLite raises mid-command; the CLI's top-level boundary converts it to
    ``error: internal error: ...`` — JSON on stdout in ``--json`` mode,
    stderr-only in human mode. Verified on this machine: exit 1, and
    ``Traceback`` appears nowhere.
    """
    (isolated_home / "docpipe.db").write_text(
        "this is not a sqlite database", encoding="utf-8"
    )
    proc = _run_research(hermes_exe, ["--json", "show", "S-000001"], isolated_home, deadline)
    assert proc.returncode == 1
    body = json.loads(proc.stdout)
    assert body["error"].startswith("internal error:")
    assert body["exit"] == 1
    assert "Traceback" not in proc.stderr + proc.stdout

    proc = _run_research(hermes_exe, ["show", "S-000001"], isolated_home, deadline)
    assert proc.returncode == 1
    assert "internal error:" in proc.stderr
    assert proc.stdout == ""
    assert "Traceback" not in proc.stderr
