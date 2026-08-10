"""Search tests: relevance, filtering, limits, FTS-syntax safety."""

from __future__ import annotations

import pytest

from docpipe import db
from docpipe.chunker import Chunk
from docpipe.search import InvalidSearchQueryError, build_fts_query, search


@pytest.fixture
def conn(tmp_path):
    handle = db.connect(tmp_path / "docpipe.db")
    yield handle
    handle.close()


def _seed(conn, doc_id: int, texts: list[str]):
    with db.write_txn(conn):
        for i, text in enumerate(texts):
            db.insert_chunk(
                conn,
                document_id=doc_id,
                chunk=Chunk(
                    chunk_index=i,
                    content=text,
                    normalized_content=text,
                    page_number=1,
                    section_path=None,
                    token_count=len(text.split()),
                ),
            )


def test_relevant_query_returns_relevant_chunks(conn):
    doc_id = db.insert_document(
        conn, title="t", file_path="/tmp/a.pdf", file_hash="h1"
    )
    _seed(
        conn,
        doc_id,
        [
            "The servo motor is controlled by PWM pulses.",
            "The ESP32 connects to WiFi and publishes telemetry.",
            "Battery management and charging circuits.",
        ],
    )
    results = search(conn, doc_id, "servo motor")
    assert len(results) == 1
    assert "servo" in results[0].content
    assert results[0].document_id == doc_id


def test_unrelated_query_returns_no_results(conn):
    doc_id = db.insert_document(
        conn, title="t", file_path="/tmp/a.pdf", file_hash="h1"
    )
    _seed(conn, doc_id, ["The robot dog wags its tail."])
    assert search(conn, doc_id, "quantum entanglement") == []


def test_natural_question_query(conn):
    """Stopwords are stripped; content words are ANDed."""
    assert build_fts_query("What is the method used in section 2?") == (
        '"method" AND "used" AND "section" AND "2"'
    )


def test_fts_special_characters_do_not_crash(conn):
    doc_id = db.insert_document(
        conn, title="t", file_path="/tmp/a.pdf", file_hash="h1"
    )
    _seed(conn, doc_id, ["servo AND motor OR testing", "plain servo text"])
    for nasty in [
        'servo "quoted"',
        "AND OR NOT",
        "(parens) [brackets] {braces}",
        "servo* wild*",
        "a:b c^2",
        "servo -motor",
        "NEAR(servo motor)",
        '"',
        "!!! ???",
    ]:
        # Must not crash: either clean results or a clean reject.
        try:
            results = search(conn, doc_id, nasty)
        except InvalidSearchQueryError:
            results = []
        assert isinstance(results, list)


def test_document_filtering(conn):
    doc_a = db.insert_document(
        conn, title="a", file_path="/tmp/a.pdf", file_hash="h1"
    )
    doc_b = db.insert_document(
        conn, title="b", file_path="/tmp/b.pdf", file_hash="h2"
    )
    _seed(conn, doc_a, ["shared keyword alpha"])
    _seed(conn, doc_b, ["shared keyword beta"])

    results_a = search(conn, doc_a, "shared keyword")
    results_b = search(conn, doc_b, "shared keyword")
    assert len(results_a) == 1 and results_a[0].document_id == doc_a
    assert len(results_b) == 1 and results_b[0].document_id == doc_b


def test_result_limit_applies(conn):
    doc_id = db.insert_document(
        conn, title="t", file_path="/tmp/a.pdf", file_hash="h1"
    )
    _seed(conn, doc_id, [f"matching term in chunk {i}" for i in range(20)])
    results = search(conn, doc_id, "matching", limit=3)
    assert len(results) == 3


def test_limit_is_clamped_to_max(conn, monkeypatch):
    from docpipe import config

    monkeypatch.setattr(config, "SEARCH_MAX_LIMIT", 5)
    doc_id = db.insert_document(
        conn, title="t", file_path="/tmp/a.pdf", file_hash="h1"
    )
    _seed(conn, doc_id, [f"matching term in chunk {i}" for i in range(20)])
    assert len(search(conn, doc_id, "matching", limit=999)) == 5


def test_empty_query_rejected(conn):
    with pytest.raises(InvalidSearchQueryError):
        search(conn, 1, "")
    with pytest.raises(InvalidSearchQueryError):
        search(conn, 1, "   ")


def test_stopword_only_query_rejected(conn):
    with pytest.raises(InvalidSearchQueryError):
        search(conn, 1, "what is the and of")


def test_unknown_document_returns_empty(conn):
    assert search(conn, 424242, "anything") == []


def test_unicode_query(conn):
    doc_id = db.insert_document(
        conn, title="t", file_path="/tmp/a.pdf", file_hash="h1"
    )
    _seed(conn, doc_id, ["Phương pháp nghiên cứu định lượng"])
    results = search(conn, doc_id, "phương pháp")
    assert len(results) == 1
