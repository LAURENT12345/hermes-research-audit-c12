"""Evidence model tests: serialization, required/nullable fields, IDs."""

from __future__ import annotations

import pytest

from evidence.model import (
    Evidence,
    format_evidence_id,
    is_valid_evidence_id,
)


def _sample() -> dict:
    return {
        "evidence_id": "E-000001",
        "document_id": 1,
        "chunk_id": 5,
        "evidence_type": "chunk",
        "source_type": "local_pdf",
        "source_title": "Attention Is All You Need",
        "content": "The quick brown fox.",
        "page_number": 4,
        "section_path": None,
        "locator": {"page": 4, "chunk_id": 5},
        "created_at": "2026-08-09 06:00:00",
    }


def test_to_dict_roundtrip():
    ev = Evidence.from_dict(_sample())
    assert ev.to_dict() == _sample()
    assert Evidence.from_dict(ev.to_dict()) == ev


def test_required_fields():
    data = _sample()
    for field in ("evidence_id", "document_id", "chunk_id", "content",
                  "source_type", "evidence_type"):
        broken = {k: v for k, v in data.items() if k != field}
        with pytest.raises((TypeError, KeyError)):
            Evidence.from_dict(broken)


def test_nullable_fields():
    ev = Evidence.from_dict(
        {
            **_sample(),
            "source_title": None,
            "page_number": None,
            "section_path": None,
            "created_at": None,
            "locator": {"page": None, "chunk_id": 5},
        }
    )
    assert ev.source_title is None
    assert ev.page_number is None
    assert ev.section_path is None
    assert ev.created_at is None
    assert ev.to_dict()["locator"] == {"page": None, "chunk_id": 5}


def test_locator_shape():
    ev = Evidence.from_dict(_sample())
    assert ev.locator == {"page": 4, "chunk_id": 5}


def test_evidence_id_format():
    assert format_evidence_id(1) == "E-000001"
    assert format_evidence_id(42) == "E-000042"
    assert format_evidence_id(123456) == "E-123456"
    for good in ("E-000001", "E-999999"):
        assert is_valid_evidence_id(good)
    for bad in ("E-1", "E-00000", "E-00000a", "e-000001", "abc", "", "E-0000001"):
        assert not is_valid_evidence_id(bad)


def test_evidence_is_frozen():
    ev = Evidence.from_dict(_sample())
    with pytest.raises(Exception):
        ev.content = "changed"  # type: ignore[misc]
