"""Tests for ``lam.model.tokenizer_ve`` string helpers."""

from __future__ import annotations

from lam.model.tokenizer_ve import basic_clean, bytes_to_unicode, get_pairs, whitespace_clean


def test_bytes_to_unicode_is_reversible_dict():
    d = bytes_to_unicode()
    assert len(d) == 256
    assert d[ord("!")] == "!"


def test_get_pairs():
    pairs = get_pairs(("a", "b", "c"))
    assert pairs == {("a", "b"), ("b", "c")}


def test_basic_clean_strips():
    assert basic_clean("  hello  ") == "hello"


def test_whitespace_clean_collapses():
    assert whitespace_clean("a  \n\t  b") == "a b"
