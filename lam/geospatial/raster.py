"""Read raster windows as RGB uint8 suitable for the vision model."""

from __future__ import annotations

import numpy as np
from PIL import Image
from rasterio.io import DatasetReader


def window_to_rgb_uint8(dataset: DatasetReader, window) -> Image.Image:
    """Read a raster window and return an RGB ``PIL.Image`` (uint8).

    Args:
        dataset: Open rasterio dataset.
        window: A :class:`rasterio.windows.Window` or window tuple.

    Returns:
        ``RGB`` mode image.

    Raises:
        ValueError: If the window is empty or band count is unsupported (not 1 or 3+).
    """
    arr = dataset.read(window=window)
    if arr.size == 0:
        raise ValueError("empty window")
    count = arr.shape[0]
    if count >= 3:
        rgb = np.stack([arr[0], arr[1], arr[2]], axis=-1)
    elif count == 1:
        g = arr[0]
        rgb = np.stack([g, g, g], axis=-1)
    else:
        raise ValueError(f"unsupported band count: {count}")
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)
