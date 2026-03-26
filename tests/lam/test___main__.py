"""Smoke test for ``python -m lam`` entry (``lam.__main__``)."""

from __future__ import annotations


def test___main___exports_main():
    import lam.__main__ as m

    assert callable(m.main)
