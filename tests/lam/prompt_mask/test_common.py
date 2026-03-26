"""Tests for ``lam.prompt_mask.common`` modules (real forward passes)."""

from __future__ import annotations

import torch

from lam.prompt_mask.common import LayerNorm2d, MLPBlock


def test_mlp_block_shape():
    m = MLPBlock(embedding_dim=16, mlp_dim=32)
    x = torch.randn(2, 10, 16)
    y = m(x)
    assert y.shape == x.shape


def test_layer_norm_2d():
    ln = LayerNorm2d(num_channels=8, eps=1e-5)
    x = torch.randn(2, 8, 4, 4)
    y = ln(x)
    assert y.shape == x.shape
