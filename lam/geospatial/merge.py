"""Class-agnostic NMS on axis-aligned boxes in full-raster pixel coordinates."""

from __future__ import annotations

import torch
from torchvision.ops import nms as _tv_nms


def nms_xyxy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Return indices of boxes to keep after torchvision NMS.

    Args:
        boxes: ``(N, 4)`` tensor in xyxy pixel coordinates.
        scores: ``(N,)`` confidence scores.
        iou_threshold: IoU threshold passed to ``torchvision.ops.nms``.

    Returns:
        Long tensor of indices into ``boxes`` / ``scores`` to retain.
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    boxes = boxes.float()
    scores = scores.float()
    return _tv_nms(boxes, scores, iou_threshold)
