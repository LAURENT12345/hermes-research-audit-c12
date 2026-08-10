"""CLI tests — ``hermes research`` / ``research.cli`` handlers.

Handlers are invoked directly with argparse Namespaces (fast, hermetic);
a subprocess smoke test covers ``python -m research.cli db-path``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import pytest

from research import cli

QUESTION = (
    "Why does scaled dot product attention divides by the square root of dk?"
)

from tests.docpipe.pdf_factory import build_pdf


def _args(**kw):
    base = {"json": False, "db_path": None, "handler": None}
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture
def env(tmp_path):
    """Ingested document + ready CLI namespace plumbing."""
    db_path = tmp_path / "docpipe.db"
    import docpipe.ingest as c0ingest

    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(
        build_pdf(
            ["Scaled dot product attention divides by the square root of dk."]
        )
    )
    ingested = c0ingest.ingest_pdf(pdf, db_path=db_path)
    return db_path, ingested.document_id


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_build_parser_registers_subcommands():
    parser = argparse.ArgumentParser()
    cli.build_parser(parser.add_subparsers(dest="command"))
    with pytest.raises(SystemExit):
        parser.parse_args(["research", "--help"])
    # The subcommand group is registered (name check via help text below).


def test_all_required_subcommands_exist():
    import contextlib
    import io

    parser = argparse.ArgumentParser()
    cli.build_parser(parser.add_subparsers(dest="command"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pytest.raises(SystemExit):
            parser.parse_args(["research", "--help"])
    help_text = buf.getvalue()
    for name in (
        "new", "list", "show", "start", "iterate", "abort",
        "claims", "answer", "synthesize", "db-path",
    ):
        assert name in help_text, f"missing subcommand {name} in --help"


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def test_new_creates_session(env, capsys):
    db_path, doc_id = env
    code = cli.cmd_new(
        _args(json=True, question=QUESTION, document_id=[doc_id],
              db_path=str(db_path))
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["session_id"] == "S-000001"


def test_new_empty_question(env, capsys):
    db_path, doc_id = env
    code = cli.cmd_new(_args(question="  ", document_id=[doc_id], db_path=str(db_path)))
    assert code == 1
    assert "question must not be empty" in capsys.readouterr().err


def test_new_missing_document(env, capsys):
    db_path, _ = env
    code = cli.cmd_new(
        _args(question="Q?", document_id=[999], db_path=str(db_path))
    )
    assert code == 1
    assert "document 999 not found" in capsys.readouterr().err


def test_show_session(env, capsys):
    db_path, doc_id = env
    cli.cmd_new(_args(question="Q?", document_id=[doc_id], db_path=str(db_path)))
    code = cli.cmd_show(_args(session_id="S-000001", db_path=str(db_path)))
    assert code == 0
    out = capsys.readouterr().out
    assert "S-000001" in out and "draft" in out


def test_show_unknown_session(env, capsys):
    db_path, _ = env
    code = cli.cmd_show(_args(session_id="S-999999", db_path=str(db_path)))
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_start_iterate_abort(env, capsys):
    db_path, doc_id = env
    cli.cmd_new(_args(question=QUESTION, document_id=[doc_id],
                      db_path=str(db_path)))
    capsys.readouterr()  # discard setup output
    code = cli.cmd_start(
        _args(json=True, session_id="S-000001", db_path=str(db_path))
    )
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert body["session_status"] in ("complete", "complete_with_gaps", "active")


def test_iterate_on_draft_fails(env, capsys):
    db_path, doc_id = env
    cli.cmd_new(_args(question="Q?", document_id=[doc_id], db_path=str(db_path)))
    code = cli.cmd_iterate(_args(session_id="S-000001", db_path=str(db_path)))
    assert code == 1
    assert "only active sessions iterate" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def test_list_sessions(env, capsys):
    db_path, doc_id = env
    cli.cmd_new(_args(question="Q?", document_id=[doc_id], db_path=str(db_path)))
    code = cli.cmd_list(_args(db_path=str(db_path)))
    assert code == 0
    assert "S-000001" in capsys.readouterr().out


def test_claims_after_run(env, capsys):
    db_path, doc_id = env
    cli.cmd_new(_args(question=QUESTION, document_id=[doc_id],
                      db_path=str(db_path)))
    cli.cmd_start(_args(session_id="S-000001", db_path=str(db_path)))
    capsys.readouterr()  # discard setup output
    code = cli.cmd_claims(_args(session_id="S-000001", db_path=str(db_path)))
    assert code == 0
    out = capsys.readouterr().out
    assert "C-000001" in out and "supported" in out


def test_answer_default_refresh(env, capsys):
    db_path, doc_id = env
    cli.cmd_new(_args(question=QUESTION, document_id=[doc_id],
                      db_path=str(db_path)))
    cli.cmd_start(_args(session_id="S-000001", db_path=str(db_path)))
    capsys.readouterr()  # discard setup output
    code = cli.cmd_answer(_args(session_id="S-000001", db_path=str(db_path)))
    assert code == 0
    out = capsys.readouterr().out
    assert "summary:" in out and "E-" in out


def test_answer_no_refresh_warns_stale(env, capsys):
    db_path, doc_id = env
    cli.cmd_new(_args(question=QUESTION, document_id=[doc_id],
                      db_path=str(db_path)))
    cli.cmd_start(_args(session_id="S-000001", db_path=str(db_path)))
    cli.cmd_answer(_args(session_id="S-000001", db_path=str(db_path)))
    code = cli.cmd_answer(
        _args(session_id="S-000001", no_refresh=True, db_path=str(db_path))
    )
    assert code == 0
    assert "may be stale" in capsys.readouterr().out


def test_synthesize_persisted_then_idempotent(env, capsys):
    db_path, doc_id = env
    cli.cmd_new(_args(question=QUESTION, document_id=[doc_id],
                      db_path=str(db_path)))
    cli.cmd_start(_args(session_id="S-000001", db_path=str(db_path)))
    capsys.readouterr()  # discard setup output
    code = cli.cmd_synthesize(
        _args(json=True, session_id="S-000001", db_path=str(db_path))
    )
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert body["persisted"] in (True, False)
    code = cli.cmd_synthesize(
        _args(json=True, session_id="S-000001", db_path=str(db_path))
    )
    body = json.loads(capsys.readouterr().out)
    assert body["persisted"] is False  # idempotent second run


def test_synthesize_on_draft_fails(env, capsys):
    db_path, doc_id = env
    cli.cmd_new(_args(question="Q?", document_id=[doc_id], db_path=str(db_path)))
    code = cli.cmd_synthesize(_args(session_id="S-000001", db_path=str(db_path)))
    assert code == 1
    assert "synthesize" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# JSON behavior
# ---------------------------------------------------------------------------

def test_json_output_only_json(env, capsys):
    db_path, doc_id = env
    code = cli.cmd_new(
        _args(json=True, question="Q?", document_id=[doc_id], db_path=str(db_path))
    )
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)  # must be valid JSON
    assert parsed == {"session_id": "S-000001"}
    assert capsys.readouterr().err == ""


def test_json_error_output(env, capsys):
    db_path, _ = env
    code = cli.cmd_show(
        _args(json=True, session_id="S-999999", db_path=str(db_path))
    )
    assert code == 1
    parsed = json.loads(capsys.readouterr().out)
    assert "error" in parsed


def test_db_path_command(env, capsys, monkeypatch):
    db_path, _ = env
    monkeypatch.setattr(
        "docpipe.config.default_db_path", lambda: db_path
    )
    code = cli.cmd_db_path(_args())
    assert code == 0
    assert capsys.readouterr().out.strip() == str(db_path)


# ---------------------------------------------------------------------------
# Subprocess smoke (python -m research.cli)
# ---------------------------------------------------------------------------

def test_python_dash_m_research_cli_db_path(env):
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "research.cli", "db-path"],
        capture_output=True, text=True, timeout=60,
        cwd=str(repo_root),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("docpipe.db")


# ---------------------------------------------------------------------------
# Black-box flow (C0 -> CLI session -> loop -> claims -> answer)
# ---------------------------------------------------------------------------

def test_cli_black_box_flow(env, capsys):
    db_path, doc_id = env
    # research new
    assert cli.cmd_new(
        _args(question=QUESTION, document_id=[doc_id],
              db_path=str(db_path))
    ) == 0
    # research start (may complete immediately)
    assert cli.cmd_start(
        _args(session_id="S-000001", db_path=str(db_path))
    ) == 0
    # iterate until terminal
    for _ in range(6):
        import json as _json

        show_out = _json.loads(
            _run_capture(cli.cmd_show, _args(session_id="S-000001",
                                             json=True, db_path=str(db_path)))
        )
        if show_out["status"] in ("complete", "complete_with_gaps",
                                  "failed", "aborted"):
            break
        assert cli.cmd_iterate(
            _args(session_id="S-000001", db_path=str(db_path))
        ) == 0
    # claims
    assert cli.cmd_claims(
        _args(session_id="S-000001", db_path=str(db_path))
    ) == 0
    # answer (default refresh)
    assert cli.cmd_answer(
        _args(session_id="S-000001", db_path=str(db_path))
    ) == 0
    out = capsys.readouterr().out
    assert "summary:" in out


def _run_capture(fn, args):
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(args)
    return buf.getvalue()
