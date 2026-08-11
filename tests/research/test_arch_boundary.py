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


def _llm_adapter_modules() -> list[Path]:
    research_dir = REPO_ROOT / "research"
    return [
        research_dir / "llm_reasoner.py",
        research_dir / "llm_transport.py",
    ]


def test_llm_adapter_has_no_store_or_domain_import():
    """The LLM adapter must remain a pure proposal source (Commit 12): no
    store module, no sqlite3, no domain apply layers."""
    for module in _llm_adapter_modules():
        assert module.is_file(), f"missing {module}"
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "store" not in node.module.split("."), (
                    f"{module.name} imports {node.module}"
                )
                assert node.module not in (
                    "research.claims", "research.verifier",
                    "research.synthesis", "research.loop",
                ), f"{module.name} imports {node.module}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "store" not in alias.name.split("."), (
                        f"{module.name} imports {alias.name}"
                    )
        assert "import sqlite3" not in module.read_text(encoding="utf-8")


def test_llm_adapter_never_calls_raw_writers_or_apply_functions():
    """The adapter proposes; the deterministic verifier and domain layers
    apply. Mechanical enforcement for the proposal-only principle."""
    for module in _llm_adapter_modules():
        calls = _calls_in_module(module)
        for writer in _RAW_WRITERS:
            assert writer not in calls, (
                f"{module.name} calls raw writer {writer!r}"
            )
        for forbidden in (
            "create_claim", "create_claim_from_proposal", "link_evidence",
            "transition_claim", "apply_decomposition",
            "apply_evidence_relation", "apply_semantic_contradiction",
            "apply_context", "apply_coverage", "create_contradiction",
            "close_contradiction",
        ):
            assert forbidden not in calls, (
                f"{module.name} calls {forbidden!r} — apply belongs to "
                "the verifier/domain layer"
            )
