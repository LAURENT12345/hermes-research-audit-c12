"""LocalRetriever tests: C0 search + C1 evidence orchestration."""

from __future__ import annotations

import sqlite3

import pytest

import docpipe.db as c0db
import docpipe.search as c0search
import evidence.store as c1store
from docpipe.chunker import Chunk
from evidence import builder as evidence_builder
from research.retriever import (
    EvidenceCreationError,
    LocalRetriever,
    RetrievalError,
)


@pytest.fixture
def seeded(tmp_path):
    """One document with three searchable chunks."""
    db_path = tmp_path / "docpipe.db"
    conn = c0db.connect(db_path)
    doc_id = c0db.insert_document(
        conn, title="Attention Is All You Need", file_path="/tmp/a.pdf",
        file_hash="ha", status="ready",
    )
    chunk_ids = []
    for i, text in enumerate(
        [
            "Scaled dot product attention divides by the square root of dk.",
            "The softmax function enters regions of extremely small gradients.",
            "Positional encoding injects order information.",
        ]
    ):
        chunk_ids.append(
            c0db.insert_chunk(
                conn, document_id=doc_id,
                chunk=Chunk(chunk_index=i, content=text,
                            normalized_content=text,
                            page_number=i + 1, section_path=None,
                            token_count=len(text.split())),
            )
        )
    conn.close()
    return db_path, doc_id, chunk_ids


def test_retrieve_query_to_evidence_with_provenance(seeded):
    db_path, doc_id, chunk_ids = seeded
    result = LocalRetriever(db_path=db_path).retrieve(doc_id, "scaled dot product")

    assert result.document_id == doc_id
    assert result.query == "scaled dot product"
    assert len(result.evidence) == 1
    assert result.skipped == []

    ev = result.evidence[0]
    conn = c0db.connect(db_path)
    try:
        chunk = c0db.get_chunk(conn, chunk_ids[0])
        # Provenance chain: document -> chunk -> evidence, all consistent.
        assert ev.chunk_id == chunk["id"]
        assert ev.document_id == chunk["document_id"]
        assert ev.page_number == chunk["page_number"]
        assert ev.content == chunk["content"]
        assert ev.locator == {"page": chunk["page_number"], "chunk_id": chunk["id"]}
        assert ev.source_type == "local_pdf"
    finally:
        conn.close()


def test_retrieve_multiple_results(seeded):
    db_path, doc_id, _ = seeded
    result = LocalRetriever(db_path=db_path).retrieve(doc_id, "gradients")
    assert len(result.evidence) == 1

    # All three chunks share no common term; use a broader check: retrieve
    # with the per-chunk terms and confirm hits are distinct chunks.
    r1 = LocalRetriever(db_path=db_path).retrieve(doc_id, "softmax")
    r2 = LocalRetriever(db_path=db_path).retrieve(doc_id, "positional")
    assert {e.chunk_id for e in r1.evidence} != {e.chunk_id for e in r2.evidence}


def test_duplicate_retrieval_reuses_existing_evidence(seeded):
    db_path, doc_id, _ = seeded
    retriever = LocalRetriever(db_path=db_path)

    first = retriever.retrieve(doc_id, "scaled dot product")
    second = retriever.retrieve(doc_id, "scaled dot product")

    assert [e.evidence_id for e in first.evidence] == [
        e.evidence_id for e in second.evidence
    ]
    conn = c1store.connect(db_path)
    try:
        assert c1store.count_evidence(conn) == 1  # no duplicates
    finally:
        conn.close()


def test_retrieve_no_results(seeded):
    db_path, doc_id, _ = seeded
    result = LocalRetriever(db_path=db_path).retrieve(
        doc_id, "xyz_nonexistent_term_12345"
    )
    assert result.evidence == []
    assert result.skipped == []


def test_c0_error_propagation_database(seeded, monkeypatch):
    db_path, doc_id, _ = seeded

    def boom(conn, document_id, query, limit=10):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(c0search, "search", boom)
    with pytest.raises(RetrievalError, match="database failure"):
        LocalRetriever(db_path=db_path).retrieve(doc_id, "anything")


def test_c0_error_propagation_invalid_query(seeded):
    db_path, doc_id, _ = seeded
    with pytest.raises(RetrievalError, match="invalid search query"):
        LocalRetriever(db_path=db_path).retrieve(doc_id, "")


def test_c1_error_handling_skips_chunk(seeded, monkeypatch):
    db_path, doc_id, chunk_ids = seeded

    def skip_everything(chunk_id, **kwargs):
        raise evidence_builder.ChunkNotFoundError(f"chunk {chunk_id} vanished")

    monkeypatch.setattr(evidence_builder, "create_evidence", skip_everything)
    result = LocalRetriever(db_path=db_path).retrieve(doc_id, "scaled dot product")

    assert result.evidence == []
    assert result.skipped == [{"chunk_id": chunk_ids[0], "reason": f"chunk {chunk_ids[0]} vanished"}]


def test_c1_error_partial_skip_keeps_others(seeded, monkeypatch):
    db_path, doc_id, chunk_ids = seeded

    real_create = evidence_builder.create_evidence

    def flaky(chunk_id, **kwargs):
        if chunk_id == chunk_ids[0]:
            raise evidence_builder.DocumentNotFoundError("doc missing")
        return real_create(chunk_id, **kwargs)

    monkeypatch.setattr(evidence_builder, "create_evidence", flaky)
    result = LocalRetriever(db_path=db_path).retrieve(doc_id, "softmax")

    # Only one chunk matches "softmax"; force a multi-hit query instead:
    result2 = LocalRetriever(db_path=db_path).retrieve_many(
        doc_id, ["scaled dot product", "softmax"]
    )
    # First query's chunk is skipped, second query succeeds.
    assert result2[0].skipped and not result2[0].evidence
    assert result2[1].evidence and not result2[1].skipped


def test_c1_database_error_propagates(seeded, monkeypatch):
    db_path, doc_id, _ = seeded

    def boom(chunk_id, **kwargs):
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(evidence_builder, "create_evidence", boom)
    with pytest.raises(EvidenceCreationError, match="database failure"):
        LocalRetriever(db_path=db_path).retrieve(doc_id, "scaled dot product")


def test_retrieve_many_returns_per_query_results(seeded):
    db_path, doc_id, _ = seeded
    results = LocalRetriever(db_path=db_path).retrieve_many(
        doc_id, ["scaled dot product", "positional encoding"]
    )
    assert len(results) == 2
    assert results[0].query == "scaled dot product"
    assert results[1].query == "positional encoding"
    assert all(r.document_id == doc_id for r in results)
    assert len(results[0].evidence) == 1
    assert len(results[1].evidence) == 1
    assert results[0].evidence[0].chunk_id != results[1].evidence[0].chunk_id
