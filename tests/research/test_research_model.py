"""C2 model tests: ID formats, statement normalization, serialization."""

from __future__ import annotations

import pytest

from research.model import (
    CLAIM_ROLES,
    CLAIM_STATUSES,
    ITERATION_PHASES,
    QUERY_ORIGINS,
    SESSION_STATUSES,
    Claim,
    ResearchEvent,
    ResearchSession,
    format_claim_id,
    format_session_id,
    is_valid_claim_id,
    is_valid_session_id,
    normalize_statement,
)


def test_session_id_format():
    assert format_session_id(1) == "S-000001"
    assert format_session_id(42) == "S-000042"
    assert format_session_id(123456) == "S-123456"
    for good in ("S-000001", "S-999999"):
        assert is_valid_session_id(good)
    for bad in ("S-1", "E-000001", "C-000001", "s-000001", "", "S-0000001"):
        assert not is_valid_session_id(bad)


def test_claim_id_format():
    assert format_claim_id(1) == "C-000001"
    assert format_claim_id(7) == "C-000007"
    for good in ("C-000001", "C-123456"):
        assert is_valid_claim_id(good)
    for bad in ("C-1", "E-000001", "S-000001", "c-000001", "", "C-0000001"):
        assert not is_valid_claim_id(bad)


def test_normalize_statement():
    assert normalize_statement("  Hello   World  ") == "hello world"
    assert normalize_statement("Hello\nWorld\tAgain") == "hello world again"
    assert normalize_statement("Already Normal") == "already normal"
    assert normalize_statement("") == ""
    # Vietnamese diacritics are preserved (no stripping).
    assert normalize_statement("Phương Pháp") == "phương pháp"


def test_normalize_is_lossy_only_in_whitespace_and_case():
    original = "  The   Transformer  "
    assert normalize_statement(original) != original
    assert original.strip() == "The   Transformer  ".strip()


def test_session_to_dict_shape():
    s = ResearchSession(
        session_id="S-000001", question="Q?", status="draft",
        min_supporting=1, max_iterations=5, final_answer=None,
        created_at="t1", updated_at="t2",
    )
    d = s.to_dict()
    assert d["session_id"] == "S-000001"
    assert d["status"] == "draft"
    assert d["final_answer"] is None
    assert set(d) == {
        "session_id", "question", "status", "min_supporting", "max_iterations",
        "final_answer", "created_at", "updated_at",
    }


def test_claim_to_dict_shape():
    c = Claim(
        claim_id="C-000001", session_id="S-000001",
        statement="The dot product has variance d_k.",
        statement_normalized="the dot product has variance d_k.",
        status="proposed", confidence=None, notes=None,
        created_at="t1", updated_at="t2",
    )
    d = c.to_dict()
    assert d["claim_id"] == "C-000001"
    assert d["statement_normalized"] == "the dot product has variance d_k."
    assert set(d) == {
        "claim_id", "session_id", "statement", "statement_normalized",
        "status", "confidence", "notes", "parent_claim_id", "context",
        "created_at", "updated_at",
    }


def test_constant_enum_values():
    assert "draft" in SESSION_STATUSES
    assert "complete_with_gaps" in SESSION_STATUSES
    assert {"proposed", "supported", "insufficient", "contradicted",
            "invalidated", "withdrawn"} <= set(CLAIM_STATUSES)
    assert set(CLAIM_ROLES) == {"supporting", "contradicting"}
    assert set(QUERY_ORIGINS) == {"user", "derived"}
    assert "evaluating" in ITERATION_PHASES


def test_event_to_dict():
    e = ResearchEvent(id=1, session_id="S-000001", event_type="session_created",
                      detail=None, created_at="t")
    assert e.to_dict()["event_type"] == "session_created"
    assert e.to_dict()["detail"] is None
