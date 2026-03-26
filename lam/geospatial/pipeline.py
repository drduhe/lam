"""End-to-end GeoTIFF tiling, LAM inference, and georeferenced GeoJSON output."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import rasterio
import torch
from rasterio.crs import CRS

from lam.geospatial.geojson_io import features_from_tile_local_masks
from lam.geospatial.merge import nms_xyxy
from lam.geospatial.raster import window_to_rgb_uint8
from lam.geospatial.tiling import iter_tiles
from lam.model.sam3_image_processor import Sam3Processor


@contextmanager
def _open_geotiff(path_or_s3_uri: str) -> Generator[Any, None, None]:
    """Open a local path or ``s3://`` object as a rasterio dataset.

    ``s3://`` uses GDAL ``/vsis3/`` (HTTP range reads); no full-object download.
    """
    if path_or_s3_uri.startswith("s3://"):
        from lam.geospatial.s3_vsis3 import apply_gdal_s3_read_defaults, s3_uri_to_vsis3_path

        apply_gdal_s3_read_defaults()
        vsis3_path = s3_uri_to_vsis3_path(path_or_s3_uri)
        with rasterio.open(vsis3_path) as ds:
            yield ds
    else:
        with rasterio.open(path_or_s3_uri) as ds:
            yield ds


def run_geotiff_inference(
    processor: Sam3Processor,
    geotiff_path: str,
    text_prompt: str,
    *,
    tile_size: int = 1024,
    overlap: int = 128,
    dst_crs: Optional[str] = "EPSG:4326",
    merge_iou_threshold: Optional[float] = 0.45,
) -> Dict[str, Any]:
    """Run text-prompted LAM over a large GeoTIFF and return a FeatureCollection.

    Internally reads overlapping tiles, runs ``processor`` on each, shifts boxes to
    full-raster pixel space, optionally applies class-agnostic NMS, then polygonizes
    tile-local masks (shifted into full-raster coordinates) for GeoJSON in ``dst_crs``.

    Args:
        processor: Configured ``Sam3Processor`` (model already loaded).
        geotiff_path: Path to a GeoTIFF readable by rasterio, or ``s3://bucket/key``
            (GDAL ``/vsis3/`` range reads; AWS credentials via env or IAM role).
        text_prompt: Open-vocabulary prompt passed to the model.
        tile_size: Window width/height in pixels.
        overlap: Overlap between adjacent tiles in pixels.
        dst_crs: Target CRS for output geometries (e.g. ``EPSG:4326``), or ``None``
            to keep the raster's CRS.
        merge_iou_threshold: IoU threshold for cross-tile NMS on xyxy boxes; ``None``
            disables NMS (not recommended when ``overlap > 0``).

    Returns:
        GeoJSON FeatureCollection dict with ``features`` in ``dst_crs`` (or raster CRS).
    """
    boxes_all: List[torch.Tensor] = []
    scores_all: List[torch.Tensor] = []
    local_masks_all: List[np.ndarray] = []
    offset_xy_all: List[tuple[int, int]] = []

    with _open_geotiff(geotiff_path) as ds:
        width = int(ds.width)
        height = int(ds.height)
        transform = ds.transform
        crs = ds.crs or CRS.from_epsg(4326)
        out_crs = CRS.from_string(dst_crs) if dst_crs else crs

        for spec in iter_tiles(width, height, tile_size, overlap):
            pil = window_to_rgb_uint8(ds, spec.window)
            inference_state = processor.set_image(pil)
            out = processor.set_text_prompt(text_prompt, inference_state)
            masks = out["masks"]
            boxes = out["boxes"]
            scores = out["scores"]
            if len(scores) == 0:
                continue

            ox, oy = spec.col_off, spec.row_off
            n = len(scores)
            shift = torch.tensor([ox, oy, ox, oy], device=boxes.device, dtype=boxes.dtype)
            boxes_shift = boxes + shift

            for i in range(n):
                m = masks[i].squeeze(0) if masks[i].dim() == 3 else masks[i]
                local_masks_all.append(m.detach().cpu().numpy())
                offset_xy_all.append((ox, oy))

            boxes_all.append(boxes_shift)
            scores_all.append(scores)

    if not scores_all:
        return {"type": "FeatureCollection", "features": []}

    boxes_cat = torch.cat(boxes_all, dim=0)
    scores_cat = torch.cat(scores_all, dim=0)

    if merge_iou_threshold is not None and boxes_cat.shape[0] > 0:
        keep = nms_xyxy(boxes_cat, scores_cat, float(merge_iou_threshold))
        boxes_cat = boxes_cat[keep]
        scores_cat = scores_cat[keep]
        keep_idx = keep.detach().cpu().tolist()
        local_masks_all = [local_masks_all[j] for j in keep_idx]
        offset_xy_all = [offset_xy_all[j] for j in keep_idx]

    features = features_from_tile_local_masks(
        transform, crs, out_crs, boxes_cat, scores_cat, local_masks_all, offset_xy_all, text_prompt
    )
    return {"type": "FeatureCollection", "features": features}


def write_geojson(path: str, fc: Dict[str, Any]) -> None:
    """Write a FeatureCollection dict to UTF-8 JSON with indentation.

    Args:
        path: Output file path.
        fc: GeoJSON object (typically a FeatureCollection).
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)
