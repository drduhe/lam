"""Tests for ``lam.geospatial.masks`` (requires OpenCV)."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from lam.geospatial.masks import mask_to_polygon


def test_mask_to_polygon_empty():
    assert mask_to_polygon(np.zeros((5, 5), dtype=np.uint8)) is None


def test_mask_to_polygon_simple_square():
    m = np.zeros((10, 10), dtype=bool)
    m[2:8, 2:8] = True
    poly = mask_to_polygon(m)
    assert poly is not None
    assert poly[0] == poly[-1]
    assert len(poly) >= 4


def test_mask_to_polygon_wrong_dims():
    assert mask_to_polygon(np.zeros((2, 3, 4))) is None
