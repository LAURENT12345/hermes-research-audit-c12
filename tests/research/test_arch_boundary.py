"""Architectural boundary tests (finding D).

Enforces — mechanically, not just by docstring — that the public
interface layers (API, CLI) never call the raw store status writers, and
that the reasoner has no store import (it must remain proposal-only).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Raw status/state writers that bypass domain invariants (transition
#: matrix, lifecycle rules, audit events). Only the domain layers
#: (claims, loop) may touch them.
_RAW_WRITERS = (
    "update_claim_status",
    "update_session_status",
    "set_query_status",
    "set_session_answer",
    "set_final_answer",
)


def _calls_in_module(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    return calls


def _public_interface_modules() -> list[Path]:
    research_dir = REPO_ROOT / "research"
    return [
        research_dir / "api.py",
        research_dir / "api_sessions.py",
        research_dir / "api_claims.py",
        research_dir / "api_answer.py",
        research_dir / "cli.py",
    ]


def test_interface_layers_never_call_raw_writers():
    for module in _public_interface_modules():
        assert module.is_file(), f"missing {module}"
        calls = _calls_in_module(module)
        for writer in _RAW_WRITERS:
            assert writer not in calls, (
                f"{module.name} calls raw writer {writer!r} — "
                "must go through the domain layer instead"
            )


def test_reasoner_has_no_store_import():
    """The reasoner must remain a pure proposal source (finding from the
    LLM-boundary audits): no research.store access at all."""
    reasoner = REPO_ROOT / "research" / "reasoner.py"
    tree = ast.parse(reasoner.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "store" in node.module:
            raise AssertionError(f"reasoner imports {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "store" in alias.name:
                    raise AssertionError(f"reasoner imports {alias.name}")


def test_raw_writers_are_not_called_by_synthesis():
    """Synthesis persists only via set_final_answer+event (atomic); it must
    not call the status writers directly."""
    synthesis = (REPO_ROOT / "research" / "synthesis.py").read_text(encoding="utf-8")
    for writer in ("update_claim_status", "update_session_status", "set_query_status"):
        assert writer not in synthesis
