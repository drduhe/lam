"""Tests for ``lam.model.act_ckpt_utils``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from lam.model.act_ckpt_utils import activation_ckpt_wrapper, clone_output_wrapper


def test_activation_ckpt_wrapper_disabled_runs_module_normally():
    linear = nn.Linear(4, 2, bias=True)
    wrapped = activation_ckpt_wrapper(linear)
    x = torch.randn(3, 4)
    y = wrapped(x, act_ckpt_enable=False)
    assert y.shape == (3, 2)


def test_activation_ckpt_wrapper_enabled_keyword_only_linear():
    linear = nn.Linear(4, 2, bias=True)
    wrapped = activation_ckpt_wrapper(linear)
    x = torch.randn(3, 4, requires_grad=True)
    y = wrapped(input=x, act_ckpt_enable=True, use_reentrant=False)
    assert y.shape == (3, 2)
    y.sum().backward()


def test_activation_ckpt_wrapper_enabled_rejects_positional_args():
    linear = nn.Linear(4, 2, bias=True)
    wrapped = activation_ckpt_wrapper(linear)
    x = torch.randn(3, 4)
    with pytest.raises(ValueError, match="keyword arguments only"):
        wrapped(x, act_ckpt_enable=True)


def test_clone_output_wrapper_cpu_passthrough():
    @clone_output_wrapper
    def f():
        return torch.tensor([1.0, 2.0])

    out = f()
    torch.testing.assert_close(out, torch.tensor([1.0, 2.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_clone_output_wrapper_cuda_clones():
    inner = torch.tensor([1.0], device="cuda")

    @clone_output_wrapper
    def f():
        return inner

    out = f()
    assert out.is_cuda
    assert out is not inner
    torch.testing.assert_close(out, inner)
