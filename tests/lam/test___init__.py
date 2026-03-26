"""Tests for ``lam`` package lazy exports."""

from __future__ import annotations

import pytest


def test_lazy_build_lam_image_model_import():
    """Resolving the attribute imports ``lam.model_builder`` (heavy but part of the public API)."""
    import lam

    assert callable(lam.build_lam_image_model)


def test_getattr_unknown_raises():
    import lam

    with pytest.raises(AttributeError):
        _ = lam.not_a_real_export_ever  # type: ignore[attr-defined]
