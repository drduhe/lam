"""Tests for ``lam.model.model_misc`` helpers and small modules."""

from __future__ import annotations

import torch
from torch import nn

from lam.model.model_misc import (
    DotProductScoring,
    LayerScale,
    MultiheadAttentionWrapper,
    get_default_device,
    inverse_sigmoid,
    tensor_to_device,
)


def test_inverse_sigmoid_round_trip():
    x = torch.tensor([0.2, 0.5, 0.9])
    y = torch.sigmoid(inverse_sigmoid(x))
    torch.testing.assert_close(y, x, atol=1e-4, rtol=1e-4)


def test_get_default_device_is_torch_device():
    d = get_default_device()
    assert isinstance(d, torch.device)


def test_tensor_to_device_cpu():
    t = torch.tensor([1.0])
    out = tensor_to_device(t, torch.device("cpu"))
    torch.testing.assert_close(out, t)


def test_dot_product_scoring_forward():
    d_model = 8
    d_proj = 4
    m = DotProductScoring(d_model, d_proj, prompt_mlp=None)
    num_layer, bs, nq, seq = 2, 1, 3, 5
    hs = torch.randn(num_layer, bs, nq, d_model)
    prompt = torch.randn(seq, bs, d_model)
    prompt_mask = torch.zeros(bs, seq, dtype=torch.bool)
    scores = m(hs, prompt, prompt_mask)
    assert scores.shape == (num_layer, bs, nq, 1)


def test_layer_scale():
    ls = LayerScale(4, init_values=0.5)
    x = torch.ones(2, 4)
    y = ls(x)
    assert y.shape == x.shape


def test_multihead_attention_wrapper_disables_need_weights():
    m = MultiheadAttentionWrapper(16, 4, batch_first=True)
    q = torch.randn(2, 5, 16)
    k = torch.randn(2, 5, 16)
    v = torch.randn(2, 5, 16)
    out, w = m(q, k, v, need_weights=True)
    assert w is None
    assert out.shape == q.shape
