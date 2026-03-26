"""Tests for ``lam.agent.helpers.boxes`` (BoxMode, Boxes)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lam.agent.helpers.boxes import BoxMode, Boxes


def test_box_mode_convert_xyxy_xywh_list_round_trip():
    xyxy = [0.0, 0.0, 10.0, 20.0]
    xywh = BoxMode.convert(xyxy, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    assert xywh == [0.0, 0.0, 10.0, 20.0]
    back = BoxMode.convert(xywh, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    assert back == xyxy


def test_box_mode_noop_same_mode():
    t = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = BoxMode.convert(t, BoxMode.XYXY_ABS, BoxMode.XYXY_ABS)
    torch.testing.assert_close(out, t)


def test_box_mode_xywha_to_xyxy_axis_aligned():
    # center (5,5), w=4, h=2, angle 0 -> axis-aligned xyxy
    box = torch.tensor([[5.0, 5.0, 4.0, 2.0, 0.0]])
    xyxy = BoxMode.convert(box, BoxMode.XYWHA_ABS, BoxMode.XYXY_ABS)
    torch.testing.assert_close(xyxy, torch.tensor([[3.0, 4.0, 7.0, 6.0]]))


def test_box_mode_relative_unsupported():
    with pytest.raises(AssertionError, match="Relative mode"):
        BoxMode.convert([0.1, 0.1, 0.2, 0.2], BoxMode.XYXY_REL, BoxMode.XYWH_ABS)


def test_boxes_area_and_clip():
    b = Boxes(torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 20.0]]))
    torch.testing.assert_close(b.area(), torch.tensor([100.0, 150.0]))
    b.clip((18, 12))
    torch.testing.assert_close(b.tensor[1, 2], torch.tensor(12.0))
    torch.testing.assert_close(b.tensor[1, 3], torch.tensor(18.0))


def test_boxes_numpy_input():
    b = Boxes(np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32))
    assert b.tensor.shape == (1, 4)
