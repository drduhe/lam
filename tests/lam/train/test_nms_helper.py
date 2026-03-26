"""Tests for ``lam.train.nms_helper`` (numpy + real NMS helpers, numba optional)."""

from __future__ import annotations

import numpy as np
import pytest

from lam.train.nms_helper import (
    apply_frame_nms,
    compute_frame_ious,
    compute_track_iou_matrix,
    convert_bbox_format,
    is_zero_box,
    process_frame_level_nms,
)


@pytest.mark.parametrize(
    "bbox,expected",
    [
        (None, True),
        ([0, 0, 0, 0], True),
        ([-1, -1, -1, -1], True),
        ([1, 2, 3], True),
        ([1, 1, 10, 10], False),
    ],
)
def test_is_zero_box(bbox, expected):
    assert is_zero_box(bbox) is expected


def test_convert_bbox_format():
    assert convert_bbox_format([1.0, 2.0, 3.0, 4.0]) == [1.0, 2.0, 4.0, 6.0]


def test_compute_frame_ious_vector():
    box = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    others = np.array([[5.0, 5.0, 15.0, 15.0], [20.0, 20.0, 30.0, 30.0]], dtype=np.float32)
    ious = compute_frame_ious(box, others)
    inter = 25.0
    union = 100 + 100 - inter
    assert pytest.approx(float(ious[0])) == inter / union
    assert float(ious[1]) == 0.0


def test_apply_frame_nms_suppresses_overlap():
    b = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]], dtype=np.float32)
    s = np.array([0.9, 0.8], dtype=np.float32)
    keep = apply_frame_nms(b, s, 0.5)
    assert keep == [0]


def test_compute_track_iou_matrix_single_track():
    stacked = np.array([[[0.0, 0.0, 10.0, 10.0], [np.nan, np.nan, np.nan, np.nan]]], dtype=np.float32)
    valid = ~np.isnan(stacked).any(axis=2)
    areas = (stacked[:, :, 2] - stacked[:, :, 0]) * (stacked[:, :, 3] - stacked[:, :, 1])
    areas[~valid] = 0
    mat = compute_track_iou_matrix(stacked, valid, areas)
    assert mat.shape == (1, 1)
    assert mat[0, 0] == 0.0


def test_process_frame_level_nms_end_to_end():
    groups = {
        1: [
            {
                "bboxes": [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]],
                "score": 0.9,
            },
            {
                "bboxes": [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]],
                "score": 0.5,
            },
        ]
    }
    out = process_frame_level_nms(groups, nms_threshold=0.5)
    assert out[1][1]["bboxes"][0] is None
