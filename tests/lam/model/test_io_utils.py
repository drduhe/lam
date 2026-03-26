"""Tests for ``lam.model.io_utils`` (lightweight helpers only)."""

from __future__ import annotations

import torch

from lam.model import io_utils


def test_get_float_dtype_cpu_vs_cuda():
    assert io_utils._get_float_dtype(torch.device("cpu")) == torch.float32
    assert io_utils._get_float_dtype(torch.device("meta")) == torch.float16
