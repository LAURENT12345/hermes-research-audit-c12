"""PDF parsing for C0.

Uses ``pypdf`` (pure Python, the smallest reliable parser) with lazy
import so the rest of the package works without the ``docpipe`` extra
installed. Page mapping is exact: ``page_number`` is the 1-based physical
page index reported by the parser — no layout inference is attempted.

The parser is deliberately honest about failure:

* missing file        -> ``ParseError``
* invalid PDF /       -> ``InvalidPDFError``
  encrypted / corrupt
* parser failure      -> ``InvalidPDFError`` (wrapped, original attached)
* empty extraction    -> ``EmptyExtractionError``
  (total text below
  ``config.EMPTY_EXTRACTION_MIN_CHARS``)
* extremely short     -> ingested with a warning logged
  extraction

Extraction quality is the library's, not ours: pypdf extracts the text
layer; image-only/scanned PDFs yield nothing (-> ``EmptyExtractionError``),
and reading order can be imperfect for complex layouts. This is documented
in ``docpipe/README.md`` as a known limitation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from docpipe import config
from docpipe.normalize import normalize_line_breaks

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Base class for all C0 parsing failures."""


class InvalidPDFError(ParseError):
    """The file exists but is not a readable, decrypted PDF."""


class EmptyExtractionError(ParseError):
    """The PDF parsed but yielded no meaningful text (e.g. scanned PDF)."""


@dataclass(frozen=True)
class ParsedPage:
    """One physical page of extracted text."""

    page_number: int  # 1-based physical page index
    text: str  # raw extracted text (line breaks not yet normalized)


def _load_reader(path: Path):
    """Import pypdf lazily and open the file, mapping failures precisely."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ParseError(
            "pypdf is not installed; install the C0 extra with: "
            "uv sync --extra docpipe  (or pip install 'hermes-agent[docpipe]')"
        ) from exc

    try:
        return PdfReader(str(path))
    except Exception as exc:
        raise InvalidPDFError(f"could not open PDF {path.name}: {exc}") from exc


def parse_pdf(path: Path | str) -> list[ParsedPage]:
    """Parse a PDF into 1-based pages of extracted text.

    Raises :class:`ParseError` subclasses on any failure; never returns a
    misleading partial result.
    """
    path = Path(path)
    if not path.is_file():
        raise ParseError(f"file not found: {path}")

    reader = _load_reader(path)
    if reader.is_encrypted:
        raise InvalidPDFError(f"PDF is encrypted and cannot be read: {path.name}")

    pages: list[ParsedPage] = []
    total_chars = 0
    try:
        for page_index in range(len(reader.pages)):
            text = reader.pages[page_index].extract_text() or ""
            pages.append(ParsedPage(page_number=page_index + 1, text=text))
            total_chars += len(text)
    except Exception as exc:
        raise InvalidPDFError(
            f"text extraction failed for {path.name}: {exc}"
        ) from exc

    if total_chars < config.EMPTY_EXTRACTION_MIN_CHARS:
        raise EmptyExtractionError(
            f"no meaningful text extracted from {path.name} "
            f"(total {total_chars} chars) — image-only PDF?"
        )

    if total_chars < config.SHORT_EXTRACTION_WARN_CHARS:
        logger.warning(
            "extremely short extraction: %s yielded only %d chars",
            path.name,
            total_chars,
        )

    logger.info("parse completed: %s -> %d pages, %d chars", path.name, len(pages), total_chars)
    return pages


def parse_pdf_normalized(path: Path | str) -> list[ParsedPage]:
    """Parse a PDF and apply the mechanical line-break cleanup.

    Pages whose normalized text is empty are dropped (blank pages).
    Raises ``EmptyExtractionError`` if nothing remains.
    """
    raw_pages = parse_pdf(path)
    cleaned: list[ParsedPage] = []
    for page in raw_pages:
        text = normalize_line_breaks(page.text)
        if text.strip():
            cleaned.append(ParsedPage(page_number=page.page_number, text=text))
    if not cleaned:
        raise EmptyExtractionError(
            "PDF parsed but every page was blank after normalization"
        )
    return cleaned
