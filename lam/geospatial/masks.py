"""Binary mask to polygon (OpenCV), aligned with hosting polygonization behavior."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


def mask_to_polygon(mask: np.ndarray) -> Optional[List[List[float]]]:
    """Convert a binary mask to a single closed exterior polygon in pixel coordinates.

    Args:
        mask: Array shaped ``(H, W)`` or broadcastable to that after squeeze.

    Returns:
        Closed ring ``[[x, y], ...]``, or ``None`` if empty or invalid.
    """
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim != 2:
        return None
    if mask.sum() == 0:
        return None

    if mask.dtype == bool:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = np.clip(mask.astype(np.uint8), 0, 255)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    polygon: List[List[float]] = []
    for point in largest:
        x, y = point[0][0] + 0.5, point[0][1] + 0.5
        polygon.append([float(x), float(y)])
    if polygon and polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    return polygon if len(polygon) >= 3 else None
