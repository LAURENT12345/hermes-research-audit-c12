"""Text normalization for retrieval.

Two levels exist:

* ``content`` (the chunk's display/provenance text): the extracted text
  with only mechanical line-break cleanup — CRLF/CR → LF, per-line
  whitespace trimmed, blank-line paragraph separation preserved. Words and
  punctuation are never altered.
* ``normalized_content`` (used for FTS5 retrieval): ``content`` with
  repeated whitespace collapsed, unicode spaces folded to ASCII space, and
  stray control characters removed. Case is left untouched — the FTS5
  unicode61 tokenizer folds case at query time.

Normalization is deliberately conservative: no stemming, no diacritic
stripping, no rewriting. Diacritic-sensitive matching is a documented
limitation (a search for "phương pháp" will not match "phuong phap").
"""

from __future__ import annotations

import re
import unicodedata

# Characters we never want inside normalized text (keeps FTS tokens clean).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Any unicode space (including \xa0, \u2003, ...) outside a line break.
_UNICODE_SPACE_RE = re.compile(r"[^\S\n\t]+")
# 3+ consecutive line breaks collapse to a single blank line.
_RUN_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_line_breaks(text: str) -> str:
    """Mechanical line-break cleanup applied to *content*.

    CRLF/CR → LF, control characters that are not LF/TAB removed, trailing
    whitespace stripped per line, and runs of 3+ blank lines collapsed to
    one. This is the only transformation applied to ``content``.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _RUN_NEWLINES_RE.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)


def normalize_content(text: str) -> str:
    """Normalize chunk text for retrieval/search.

    Collapses runs of whitespace (including unicode spaces) to a single
    space, keeps paragraph breaks, drops control characters. Preserves
    words and punctuation.
    """
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _UNICODE_SPACE_RE.sub(" ", text)
    return text.strip()


def is_meaningless(text: str) -> bool:
    """True when normalized text carries no searchable content."""
    return not normalize_content(text)
