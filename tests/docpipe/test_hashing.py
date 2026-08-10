"""Hashing tests: same bytes -> same hash, different bytes -> different hash."""

from __future__ import annotations

from docpipe.hashing import sha256_file


def test_same_file_same_hash(tmp_path):
    p1 = tmp_path / "a.pdf"
    p2 = tmp_path / "b.pdf"
    data = b"%PDF-1.4 fake content \x00\x01\x02" * 100
    p1.write_bytes(data)
    p2.write_bytes(data)
    assert sha256_file(p1) == sha256_file(p2)


def test_different_files_different_hash(tmp_path):
    p1 = tmp_path / "a.pdf"
    p2 = tmp_path / "b.pdf"
    p1.write_bytes(b"content one")
    p2.write_bytes(b"content two")
    assert sha256_file(p1) != sha256_file(p2)


def test_hash_is_hex_sha256(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"x")
    digest = sha256_file(p)
    assert len(digest) == 64
    int(digest, 16)  # hex
