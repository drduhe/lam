"""Tests for ``lam.perflib.nms`` (CPU ``generic_nms`` / ``nms_masks``)."""

from __future__ import annotations

import pytest
import torch

from lam.perflib.nms import generic_nms, generic_nms_cpu, nms_masks


def test_generic_nms_cpu_keeps_highest_score_when_overlapping():
    # Two boxes; IoU between them is 0.5; suppress second when threshold 0.4
    ious = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    kept = generic_nms_cpu(ious, scores, iou_threshold=0.4)
    assert kept.tolist() == [0]

    kept_loose = generic_nms_cpu(ious, scores, iou_threshold=0.5)
    assert set(kept_loose.tolist()) == {0, 1}


def test_generic_nms_cpu_order_follows_scores():
    ious = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    scores = torch.tensor([0.5, 0.9], dtype=torch.float32)
    kept = generic_nms_cpu(ious, scores, iou_threshold=0.1)
    assert kept.tolist() == [1, 0]


def test_generic_nms_routes_to_cpu_when_not_cuda():
    ious = torch.eye(2, dtype=torch.float32)
    scores = torch.tensor([0.5, 0.4], dtype=torch.float32)
    kept = generic_nms(ious, scores, iou_threshold=0.5)
    assert kept.dtype == torch.int64
    assert kept.device == scores.device


def test_generic_nms_asserts_square_matrix():
    ious = torch.zeros(2, 3)
    scores = torch.zeros(2)
    with pytest.raises(AssertionError):
        generic_nms(ious, scores)


def test_nms_masks_all_below_threshold_returns_all_false():
    probs = torch.tensor([0.1, 0.2], dtype=torch.float32)
    masks = torch.ones(2, 4, 4, dtype=torch.float32)
    keep = nms_masks(probs, masks, prob_threshold=0.5, iou_threshold=0.5)
    assert keep.shape == (2,)
    assert not keep.any()


def test_nms_masks_single_detection_kept():
    probs = torch.tensor([0.99], dtype=torch.float32)
    masks = torch.zeros(1, 4, 4, dtype=torch.float32)
    masks[0, 1:3, 1:3] = 1.0
    keep = nms_masks(probs, masks, prob_threshold=0.5, iou_threshold=0.5)
    assert keep.tolist() == [True]
