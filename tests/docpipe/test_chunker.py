"""Chunker tests: paragraphs, long paragraphs, sentence fallback, pages."""

from __future__ import annotations

import pytest

from docpipe.chunker import chunk_document
from docpipe.parser import ParsedPage


def _page(number: int, text: str) -> ParsedPage:
    return ParsedPage(page_number=number, text=text)


def test_normal_paragraphs_become_chunks():
    chunks = chunk_document(
        [_page(1, "First paragraph about servo motors.\n\nSecond paragraph about ESP32.")]
    )
    assert len(chunks) == 2
    assert [c.page_number for c in chunks] == [1, 1]
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert "servo motors" in chunks[0].content
    assert "ESP32" in chunks[1].content


def test_soft_line_wraps_joined_within_paragraph():
    chunks = chunk_document([_page(1, "This is a wrapped\nline inside one paragraph.")])
    assert len(chunks) == 1
    assert "wrapped line inside" in chunks[0].content
    assert "\n" not in chunks[0].content


def test_long_paragraph_split_within_hard_max():
    sentences = [
        f"This is sentence number {i} with some padding words to make it long. "
        for i in range(60)
    ]
    text = "".join(sentences)
    chunks = chunk_document([_page(1, text)], target_chars=300, hard_max=500)
    assert len(chunks) > 1
    assert all(len(c.content) <= 500 for c in chunks)
    # Sentence boundaries respected: every chunk (except possibly the last)
    # ends with sentence punctuation.
    for c in chunks[:-1]:
        assert c.content.rstrip().endswith((".", "!", "?"))


def test_sentence_fallback_keeps_sentences_intact():
    text = "Alpha. Beta. Gamma. Delta. Epsilon. Zeta. Eta. Theta. Iota. Kappa."
    chunks = chunk_document([_page(1, text)], target_chars=40, hard_max=60)
    assert all(len(c.content) <= 60 for c in chunks)
    assert all(c.content.endswith(".") for c in chunks)
    assert "Alpha." in chunks[0].content


def test_page_boundaries_preserved():
    chunks = chunk_document(
        [
            _page(1, "Page one paragraph."),
            _page(2, "Page two first.\n\nPage two second."),
            _page(3, "Page three."),
        ]
    )
    assert [c.page_number for c in chunks] == [1, 2, 2, 3]


def test_paragraph_under_max_stays_whole():
    text = "A reasonably sized paragraph. " * 30  # ~1080 chars < default max
    chunks = chunk_document([_page(1, text)])
    assert len(chunks) == 1
    assert chunks[0].content == text.strip()


def test_single_giant_token_hard_sliced():
    token = "X" * 5000  # no spaces, no sentence boundaries
    chunks = chunk_document([_page(1, token)], target_chars=200, hard_max=300)
    assert all(len(c.content) <= 300 for c in chunks)
    assert "".join(c.content for c in chunks) == token


def test_token_count_is_approximate():
    chunks = chunk_document([_page(1, "one two three four five")])
    assert chunks[0].token_count == 5


def test_section_path_is_null_in_c0():
    chunks = chunk_document([_page(1, "hello world")])
    assert chunks[0].section_path is None


def test_empty_pages_produce_no_chunks():
    chunks = chunk_document([_page(1, ""), _page(2, "   ")])
    assert chunks == []
