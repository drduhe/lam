"""Map full-raster pixel detections to GeoJSON geometries in a target CRS."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform_geom

from lam.geospatial.masks import mask_to_polygon


def _pixel_ring_to_dst_polygon(
    transform: Affine, crs: CRS, dst_crs: CRS, ring_pixel: List[List[float]]
) -> Dict[str, Any]:
    """Project a closed pixel-space ring to a Polygon geometry in ``dst_crs``.

    Args:
        transform: Raster affine (pixel to CRS of the source raster).
        crs: Source CRS of the raster.
        dst_crs: Target CRS for the output geometry.
        ring_pixel: ``[[col, row], ...]`` closed ring in pixel coordinates.

    Returns:
        GeoJSON-like geometry dict for a Polygon in ``dst_crs``.
    """
    src_coords: List[List[float]] = []
    for col, row in ring_pixel:
        gx, gy = transform * (col, row)
        src_coords.append([float(gx), float(gy)])
    geom: Dict[str, Any] = {"type": "Polygon", "coordinates": [src_coords]}
    return transform_geom(crs, dst_crs, geom, precision=6)


def pixel_detection_to_feature(
    transform: Affine,
    crs: CRS,
    dst_crs: CRS,
    bbox_xyxy: List[float],
    mask_polygon_xy: Optional[List[List[float]]],
    score: float,
    label: str,
) -> Dict[str, Any]:
    """Build one GeoJSON Feature with ``geometry`` in ``dst_crs``.

    Args:
        transform: Raster affine.
        crs: CRS of the raster.
        dst_crs: CRS for the output ``geometry`` field.
        bbox_xyxy: Box ``[x0, y0, x1, y1]`` in full-raster pixel coordinates (column, row).
        mask_polygon_xy: Optional mask outline in pixel coordinates; if ``None``, uses bbox corners.
        score: Detection confidence.
        label: Class / prompt label stored in properties.

    Returns:
        A GeoJSON Feature dict.
    """
    if mask_polygon_xy is not None:
        ring = list(mask_polygon_xy)
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        geometry = _pixel_ring_to_dst_polygon(transform, crs, dst_crs, ring)
    else:
        x0, y0, x1, y1 = bbox_xyxy
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        geometry = _pixel_ring_to_dst_polygon(transform, crs, dst_crs, ring)

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "score": float(score),
            "label": label,
            "imageBBox": list(bbox_xyxy),
        },
    }


def stack_to_geojson_features(
    transform: Affine,
    crs: CRS,
    dst_crs: CRS,
    masks: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    label: str,
) -> List[Dict[str, Any]]:
    """Convert stacked LAM outputs in full-raster pixel space to GeoJSON features.

    Args:
        transform: Raster affine.
        crs: CRS of the raster.
        dst_crs: Target CRS for geometries.
        masks: Tensor ``(N, H, W)`` aligned with the full raster.
        boxes: Tensor ``(N, 4)`` xyxy in full-raster pixels.
        scores: Tensor ``(N,)`` scores.
        label: Stored in each feature's ``properties``.

    Returns:
        List of GeoJSON Feature dicts.
    """
    features: List[Dict[str, Any]] = []
    n = len(scores)
    for i in range(n):
        bbox = boxes[i].cpu().tolist()
        score = float(scores[i].cpu().item())
        m = masks[i].cpu().numpy()
        poly = mask_to_polygon(m)
        features.append(pixel_detection_to_feature(transform, crs, dst_crs, bbox, poly, score, label))
    return features


def features_from_tile_local_masks(
    transform: Affine,
    crs: CRS,
    dst_crs: CRS,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    local_masks: Sequence[np.ndarray],
    offset_xy: Sequence[Tuple[int, int]],
    label: str,
) -> List[Dict[str, Any]]:
    """Like ``stack_to_geojson_features`` but masks are tile-sized; ring coords are shifted by ``offset_xy``.

    Avoids allocating ``(N, H, W)`` full-raster tensors for large GeoTIFFs.
    """
    n = int(scores.shape[0])
    if int(boxes.shape[0]) != n or len(local_masks) != n or len(offset_xy) != n:
        raise ValueError("boxes, scores, local_masks, and offset_xy must have the same length")
    features: List[Dict[str, Any]] = []
    for i in range(n):
        bbox = boxes[i].cpu().tolist()
        score = float(scores[i].cpu().item())
        ox, oy = offset_xy[i]
        poly = mask_to_polygon(local_masks[i])
        if poly is not None:
            poly = [[float(x + ox), float(y + oy)] for x, y in poly]
        features.append(pixel_detection_to_feature(transform, crs, dst_crs, bbox, poly, score, label))
    return features
