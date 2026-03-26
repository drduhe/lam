"""Pytest configuration and shared fixtures for the ``lam`` package.

Layout
    Tests mirror the ``lam/`` package tree under ``tests/lam/`` (e.g.
    ``tests/lam/model/test_box_ops.py`` exercises ``lam.model.box_ops``).

    **Pytest imports test modules by basename.** If two files share the same name
    (e.g. ``.../train/test_masks_ops.py`` and ``.../perflib/test_masks_ops.py``),
    collection fails with an import mismatch—use distinct names such as
    ``test_train_masks_ops.py`` vs ``test_perflib_masks_ops.py``.

Running
    .. code-block:: bash

        pytest

    Coverage is enabled by default (``lam`` package, see ``pyproject.toml``).

Optional dependencies
    Some tests ``importorskip`` optional stacks (e.g. OpenCV, rasterio). Install
    ``.[geospatial]`` to collect and run those modules locally or in CI.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def clear_log_level_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure ``LOG_LEVEL`` is unset (``lam.logger.get_logger`` tests)."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
