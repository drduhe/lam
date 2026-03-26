"""Tests for ``lam.eval.postprocessors``."""

from __future__ import annotations

import torch

from lam.eval.postprocessors import PostProcessNullOp


def test_post_process_null_op_process_results():
    m = PostProcessNullOp()
    stages = [{"a": 1}]
    assert m.process_results(find_stages=stages) is stages


def test_post_process_null_op_forward_returns_none():
    m = PostProcessNullOp()
    assert m.forward(object()) is None
