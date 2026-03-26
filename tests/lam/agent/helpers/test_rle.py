"""Tests for ``lam.agent.helpers.rle`` (GPU path + pycocotools)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_util

from lam.agent.helpers import rle as rle_mod


def test_rle_encode_empty_batch():
    assert rle_mod.rle_encode(torch.zeros(0, 4, 4, dtype=torch.bool)) == []


def test_rle_encode_single_square_round_trip():
    m = torch.zeros(1, 6, 6, dtype=torch.bool, device="cpu")
    m[0, 1:5, 2:5] = True
    enc = rle_mod.rle_encode(m)
    assert len(enc) == 1
    decoded = mask_util.decode(enc[0])
    assert decoded.shape == (6, 6)
    assert bool(decoded[2, 3])


def test_rle_encode_with_areas():
    m = torch.zeros(1, 4, 4, dtype=torch.bool)
    m[0, 0:2, 0:2] = True
    enc = rle_mod.rle_encode(m, return_areas=True)
    assert enc[0]["area"] == 4


def test_robust_rle_encode_cpu_fallback(monkeypatch):
    """Force CPU numpy path after GPU encode raises."""

    def boom(*_a, **_k):
        raise RuntimeError("simulated gpu failure")

    monkeypatch.setattr(rle_mod, "rle_encode", boom)
    m = torch.zeros(1, 3, 3, dtype=torch.bool)
    m[0, 1, 1] = True
    enc = rle_mod.robust_rle_encode(m)
    assert len(enc) == 1
    assert "counts" in enc[0]


def test_ann_to_rle_polygon():
    h, w = 10, 12
    poly = [[0, 0, 10, 0, 10, 10, 0, 10]]
    r = rle_mod.ann_to_rle(poly, {"height": h, "width": w})
    decoded = mask_util.decode(r)
    assert decoded.shape == (h, w)
