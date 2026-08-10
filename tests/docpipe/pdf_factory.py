"""Dependency-free minimal PDF builder for C0 tests.

Constructs valid single-/multi-page PDFs (Helvetica text content streams,
correct xref offsets and /Length values) so the parser tests do not need a
PDF *writer* library — pypdf is only needed to *read*.

Two text modes:

* ``build_pdf`` — ASCII text as literal strings (simple Type1 font).
* ``build_pdf_unicode`` — arbitrary unicode text (e.g. Vietnamese) via a
  Type0 font with Identity-H encoding and a ToUnicode CMap, which is how
  real non-ASCII PDFs encode text; pypdf decodes it back correctly.

Object layout: 1 catalog, 2 pages, then per page ``3+3i`` (page),
``4+3i`` (content), ``5+3i`` (font). Each unicode page additionally gets
a CIDFont and a ToUnicode CMap object allocated *after* the standard
range, so numbering never collides.
"""

from __future__ import annotations

_HEADER = b"%PDF-1.4\n"


def _text_operator(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"({escaped}) Tj".encode("ascii")


def _to_unicode_cmap(text: str) -> bytes:
    bfchar = [f"<{ord(ch):04X}> <{ord(ch):04X}>" for ch in text]
    return (
        "/CIDInit /ProcSet findresource begin\n"
        "12 dict begin\n"
        "begincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def\n"
        "/CMapType 2 def\n"
        "1 begincodespacerange\n"
        "<0000> <FFFF>\n"
        "endcodespacerange\n"
        f"{len(bfchar)} beginbfchar\n"
        + "".join(entry + "\n" for entry in bfchar)
        + "endbfchar\n"
        "endcmap\n"
        "CMapName currentdict /CMap defineresource pop\n"
        "end\n"
        "end\n"
    ).encode("latin-1")


def _build(page_texts: list[str], unicode_pages: set[int]) -> bytes:
    if not page_texts:
        raise ValueError("at least one page required")
    n = len(page_texts)
    unicode_pages = {i for i in unicode_pages if i < n}
    unicode_order = sorted(unicode_pages)
    extra_base = 2 + 3 * n + 1  # first object after the standard layout
    total_objects = extra_base + 2 * len(unicode_order)

    body = bytearray(_HEADER)
    offsets: dict[int, int] = {}

    def emit(obj_num: int, content: bytes) -> None:
        offsets[obj_num] = len(body)
        body.extend(f"{obj_num} 0 obj\n".encode("ascii") + content + b"\nendobj\n")

    emit(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3 + 3 * i} 0 R" for i in range(n))
    emit(2, f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode("ascii"))

    for i, text in enumerate(page_texts):
        page_num, content_num, font_num = 3 + 3 * i, 4 + 3 * i, 5 + 3 * i
        emit(
            page_num,
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_num} 0 R "
                f"/Resources << /Font << /F1 {font_num} 0 R >> >> >>"
            ).encode("ascii"),
        )
        if i in unicode_pages:
            slot = unicode_order.index(i)
            cidfont_num = extra_base + 2 * slot
            cmap_num = cidfont_num + 1
            hexstr = "".join(f"{ord(c):04X}" for c in text)
            stream = f"BT /F1 12 Tf 72 720 Td <{hexstr}> Tj ET".encode("ascii")
            cmap = _to_unicode_cmap(text)
            emit(
                content_num,
                f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
                + stream
                + b"\nendstream",
            )
            emit(
                font_num,
                (
                    b"<< /Type /Font /Subtype /Type0 /BaseFont /TestFont "
                    b"/Encoding /Identity-H /DescendantFonts ["
                    + f"{cidfont_num} 0 R".encode("ascii")
                    + b"] /ToUnicode "
                    + f"{cmap_num} 0 R".encode("ascii")
                    + b" >>"
                ),
            )
            emit(
                cidfont_num,
                (
                    b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /TestFont "
                    b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) "
                    b"/Supplement 0 >> /W [0 [1000]] >>"
                ),
            )
            emit(
                cmap_num,
                f"<< /Length {len(cmap)} >>\nstream\n".encode("ascii")
                + cmap
                + b"\nendstream",
            )
        else:
            stream = b"BT /F1 12 Tf 72 720 Td " + _text_operator(text) + b" ET"
            emit(
                content_num,
                f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
                + stream
                + b"\nendstream",
            )
            emit(font_num, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    xref_offset = len(body)
    body.extend(f"xref\n0 {total_objects}\n0000000000 65535 f \n".encode("ascii"))
    for obj_num in range(1, total_objects):
        body.extend(f"{offsets[obj_num]:010d} 00000 n \n".encode("ascii"))
    body.extend(
        f"trailer\n<< /Size {total_objects} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(body)


def build_pdf(page_texts: list[str]) -> bytes:
    """Build a minimal valid PDF with one ASCII page per entry."""
    return _build(page_texts, unicode_pages=set())


def build_pdf_unicode(page_texts: list[str], unicode_pages: set[int]) -> bytes:
    """Build a PDF where pages in ``unicode_pages`` carry unicode text."""
    return _build(page_texts, unicode_pages=set(unicode_pages))
