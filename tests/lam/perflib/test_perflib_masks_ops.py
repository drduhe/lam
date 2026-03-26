"""Tests for ``lam.perflib.masks_ops`` using synthetic masks (no asset files)."""

from __future__ import annotations

import pytest
import torch

from lam.perflib.masks_ops import mask_iou, masks_to_boxes


def test_masks_to_boxes_single_square():
    m = torch.zeros(1, 10, 12, dtype=torch.float32)
    m[0, 2:6, 3:8] = 1.0
    boxes = masks_to_boxes(m, [1])
    torch.testing.assert_close(
        boxes,
        torch.tensor([[3.0, 2.0, 7.0, 5.0]], dtype=torch.float),
    )


def test_masks_to_boxes_multiple_planes():
    m = torch.zeros(2, 8, 8, dtype=torch.float32)
    m[0, 1:3, 1:4] = 2.0
    m[1, 5:8, 6:8] = 1.0
    boxes = masks_to_boxes(m, [10, 20])
    assert boxes.shape == (2, 4)
    torch.testing.assert_close(boxes[0], torch.tensor([1.0, 1.0, 3.0, 2.0]))
    torch.testing.assert_close(boxes[1], torch.tensor([6.0, 5.0, 7.0, 7.0]))


def test_masks_to_boxes_empty_tensor():
    m = torch.zeros(0, 4, 4)
    out = masks_to_boxes(m, [])
    assert out.shape == (0, 4)


def test_masks_to_boxes_obj_ids_length_mismatch():
    m = torch.zeros(2, 4, 4)
    with pytest.raises(AssertionError):
        masks_to_boxes(m, [1])


def test_mask_iou_disjoint_and_full_overlap():
    a = torch.zeros(2, 4, 4, dtype=torch.bool)
    b = torch.zeros(1, 4, 4, dtype=torch.bool)
    a[0, 0:2, 0:2] = True
    b[0, 2:4, 2:4] = True
    iou = mask_iou(a, b)
    assert iou.shape == (2, 1)
    assert float(iou[0, 0]) == 0.0

    a[1] = b[0]
    iou2 = mask_iou(a[1:2], b)
    assert float(iou2[0, 0]) == 1.0


def test_mask_iou_wrong_dtype_raises():
    with pytest.raises(AssertionError):
        mask_iou(torch.ones(1, 2, 2, dtype=torch.float32), torch.ones(1, 2, 2, dtype=torch.float32))
