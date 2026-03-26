"""Tests for ``lam.prompt_mask.rope``."""

from __future__ import annotations

import torch

from lam.prompt_mask.rope import compute_axial_cis, init_t_xy


def test_init_t_xy_shapes():
    tx, ty = init_t_xy(3, 4, scale=1.0, offset=0, device="cpu")
    assert tx.numel() == 12
    assert ty.numel() == 12


def test_compute_axial_cis_shape():
    dim = 8
    cis = compute_axial_cis(dim, end_x=2, end_y=2, device="cpu")
    assert cis.ndim == 2
    assert cis.shape[0] == 4
