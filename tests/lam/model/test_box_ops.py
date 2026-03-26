"""Unit tests for ``lam.model.box_ops`` (real tensor math, no mocks)."""

from __future__ import annotations

import pytest
import torch
from torch.jit import Error as TorchJitError

from lam.model import box_ops


def test_box_conversions_round_trip_xyxy_batched():
    xyxy = torch.tensor([[[10.0, 20.0, 40.0, 60.0], [0.0, 0.0, 5.0, 8.0]]])
    cxcywh = box_ops.box_xyxy_to_cxcywh(xyxy)
    back = box_ops.box_cxcywh_to_xyxy(cxcywh)
    torch.testing.assert_close(back, xyxy)

    xywh = box_ops.box_xyxy_to_xywh(xyxy)
    back2 = box_ops.box_xywh_to_xyxy(xywh)
    torch.testing.assert_close(back2, xyxy)


def test_box_cxcywh_xywh_round_trip():
    cxcywh = torch.tensor([[25.0, 30.0, 10.0, 20.0]])
    xywh = box_ops.box_cxcywh_to_xywh(cxcywh)
    torch.testing.assert_close(xywh, torch.tensor([[20.0, 20.0, 10.0, 20.0]]))
    back = box_ops.box_xywh_to_cxcywh(xywh)
    torch.testing.assert_close(back, cxcywh)


def test_box_area_and_empty_masks_to_boxes():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 20.0]])
    areas = box_ops.box_area(boxes)
    torch.testing.assert_close(areas, torch.tensor([100.0, 150.0]))

    empty = torch.zeros(0, 64, 64, dtype=torch.bool)
    out = box_ops.masks_to_boxes(empty)
    assert out.shape == (0, 4)


def test_masks_to_boxes_empty_mask_plane():
    """All-false mask should yield zero box (invalidated by multiply)."""
    m = torch.zeros(2, 8, 8, dtype=torch.bool)
    m[0, 2:5, 3:6] = True
    boxes = box_ops.masks_to_boxes(m)
    assert boxes[1].sum() == 0


def test_box_iou_known_overlap():
    a = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
    b = torch.tensor([[[5.0, 5.0, 15.0, 15.0]]])
    iou, union = box_ops.box_iou(a, b)
    inter = 5 * 5
    u = 100 + 100 - inter
    assert pytest.approx(float(iou[0, 0, 0])) == inter / u
    assert pytest.approx(float(union[0, 0, 0])) == u


def test_generalized_box_iou_same_box():
    box = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    giou = box_ops.generalized_box_iou(box, box)
    assert giou.shape == (1, 1, 1)
    assert pytest.approx(float(giou[0, 0, 0]), abs=1e-5) == 1.0


def test_fast_diag_box_iou_identical():
    b = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    iou = box_ops.fast_diag_box_iou(b, b)
    assert iou.shape == (2,)
    assert pytest.approx(float(iou[0])) == 1.0


def test_fast_diag_generalized_box_iou_length_mismatch_errors():
    a = torch.zeros(1, 4)
    b = torch.zeros(2, 4)
    with pytest.raises(TorchJitError):
        box_ops.fast_diag_generalized_box_iou(a, b)


def test_fast_diag_box_iou_length_mismatch_errors():
    a = torch.zeros(1, 4)
    b = torch.zeros(2, 4)
    with pytest.raises(TorchJitError):
        box_ops.fast_diag_box_iou(a, b)


def test_fast_diag_box_iou_matches_full_box_iou_diagonal():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    diag = box_ops.fast_diag_box_iou(boxes, boxes)
    full_iou, _ = box_ops.box_iou(boxes.unsqueeze(0), boxes.unsqueeze(0))
    expected = torch.diag(full_iou[0])
    torch.testing.assert_close(diag, expected)


def test_box_xywh_inter_union():
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b2 = torch.tensor([[5.0, 5.0, 10.0, 10.0]])
    inter, union = box_ops.box_xywh_inter_union(b1, b2)
    assert inter.item() == 5 * 5
    assert union.item() == 100 + 100 - 5 * 5
