"""Tests for ``lam.agent.helpers.filename_utils``."""

from lam.agent.helpers.filename_utils import sanitize_filename


def test_sanitize_filename_strips_invalid_chars():
    s = sanitize_filename('a<b>:"/\\|?*\x01 prompt')
    assert "<" not in s and ">" not in s
    assert "_" in s


def test_sanitize_filename_truncates_and_keeps_hash():
    long = "x" * 500
    s = sanitize_filename(long, max_length=80)
    assert len(s) <= 80
    assert s.count("_") >= 1
    assert s[-9] == "_"  # ..._<8 hex>


def test_sanitize_empty_becomes_prompt_with_hash():
    s = sanitize_filename("   \n\t  ")
    assert s.startswith("prompt_")
