"""Parser tests: valid, invalid, empty, multi-page, unicode, errors."""

from __future__ import annotations

import pytest

from docpipe import parser
from docpipe.parser import EmptyExtractionError, InvalidPDFError, ParseError

from tests.docpipe.pdf_factory import build_pdf, build_pdf_unicode


def test_parse_single_page(tmp_path):
    pdf = tmp_path / "single.pdf"
    pdf.write_bytes(build_pdf(["Hello world"]))
    pages = parser.parse_pdf(pdf)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Hello world" in pages[0].text


def test_parse_multi_page_preserves_page_numbers(tmp_path):
    pdf = tmp_path / "multi.pdf"
    pdf.write_bytes(
        build_pdf(["First page content", "Second page content", "Third page content"])
    )
    pages = parser.parse_pdf(pdf)
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "Second page" in pages[1].text


def test_parse_missing_file(tmp_path):
    with pytest.raises(ParseError, match="file not found"):
        parser.parse_pdf(tmp_path / "nope.pdf")


def test_parse_invalid_pdf(tmp_path):
    pdf = tmp_path / "garbage.pdf"
    pdf.write_bytes(b"this is definitely not a pdf file")
    with pytest.raises(InvalidPDFError):
        parser.parse_pdf(pdf)


def test_parse_truncated_pdf(tmp_path):
    pdf = tmp_path / "truncated.pdf"
    pdf.write_bytes(build_pdf(["complete page"])[:40])  # cut mid-stream
    with pytest.raises(InvalidPDFError):
        parser.parse_pdf(pdf)


def test_parse_empty_extraction(tmp_path):
    """A valid PDF with no text (blank page) is an empty extraction."""
    pdf = tmp_path / "blank.pdf"
    pdf.write_bytes(build_pdf([""]))
    with pytest.raises(EmptyExtractionError, match="no meaningful text"):
        parser.parse_pdf(pdf)


def test_parse_unicode_pdf(tmp_path):
    pdf = tmp_path / "unicode.pdf"
    pdf.write_bytes(build_pdf_unicode(["Phương pháp nghiên cứu"], {0}))
    pages = parser.parse_pdf(pdf)
    assert "Phương pháp nghiên cứu" in pages[0].text


def test_parse_normalized_drops_blank_pages(tmp_path):
    pdf = tmp_path / "mixed.pdf"
    pdf.write_bytes(build_pdf(["", "Real content here"]))
    pages = parser.parse_pdf_normalized(pdf)
    assert [p.page_number for p in pages] == [2]


def test_parse_normalized_joins_crlf(tmp_path):
    pdf = tmp_path / "crlf.pdf"
    pdf.write_bytes(build_pdf(["Line one\r\nLine two"]))
    pages = parser.parse_pdf_normalized(pdf)
    assert "\r" not in pages[0].text
