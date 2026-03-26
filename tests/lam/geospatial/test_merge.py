"""Tests for ``lam.geospatial.merge`` (torchvision NMS, real tensors)."""

from __future__ import annotations

import torch

from lam.geospatial.merge import nms_xyxy


def test_nms_xyxy_empty():
    boxes = torch.zeros(0, 4)
    scores = torch.zeros(0)
    keep = nms_xyxy(boxes, scores, 0.5)
    assert keep.numel() == 0


def test_nms_xyxy_keeps_higher_score_overlap():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
    scores = torch.tensor([0.9, 0.5])
    keep = nms_xyxy(boxes, scores, 0.5)
    assert keep.numel() == 1
    assert int(keep[0]) == 0
