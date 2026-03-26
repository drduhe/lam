"""Tests for ``lam.model.data_misc``."""

from __future__ import annotations

import torch

from lam.model.data_misc import BatchedPointer, convert_my_tensors, interpolate


def test_interpolate_nonempty():
    x = torch.randn(1, 3, 8, 8)
    y = interpolate(x, size=(4, 4), mode="bilinear", align_corners=False)
    assert y.shape == (1, 3, 4, 4)


def test_interpolate_empty_batch_dimension():
    x = torch.randn(0, 3, 8, 8)
    y = interpolate(x, size=(4, 4))
    assert y.shape == (0, 3, 4, 4)


def test_interpolate_empty_channel_dimension():
    x = torch.randn(2, 0, 8, 8)
    y = interpolate(x, size=(4, 4))
    assert y.shape == (2, 0, 4, 4)


def test_convert_my_tensors_stacks_list_of_tensors():
    t1 = torch.tensor([1, 2], dtype=torch.long)
    t2 = torch.tensor([3, 4], dtype=torch.long)
    bp = BatchedPointer(
        stage_ids=[t1, t2],
        query_ids=[t1, t2],
        object_ids=[t1, t2],
        ptr_mask=[torch.tensor([True, False]), torch.tensor([False, True])],
        ptr_types=[t1, t2],
    )
    convert_my_tensors(bp)
    assert bp.stage_ids.shape == (2, 2)
    assert bp.stage_ids.dtype == torch.long
