"""GeoJSON serialization and GDAL raster preprocessing for LAM HTTP inference."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from osgeo import gdal

LOG_FMT_UNEXPECTED_4D_MASK = "Unexpected 4D mask shape: {}, expected (N, 1, H, W)"
LOG_FMT_UNEXPECTED_MASK_SHAPE = "Unexpected mask shape: {}, expected (N, H, W) or (N, 1, H, W)"
LOG_FMT_MASK_NORMALIZATION_FAILED = "Mask normalization failed: expected 3D tensor, got shape {}"
LOG_FMT_SHAPE_MISMATCH = "Shape mismatch: non_empty_mask has {} elements, expected {}"
LOG_FMT_GPU_FILTERING_FAILED = "GPU filtering failed: {}. Masks shape: {}, num_detections: {}"
VALUE_ERROR_FMT_CANNOT_PROCESS_MASKS = "Cannot process masks with shape {}"
VALUE_ERROR_FMT_MASK_NORMALIZATION_FAILED = "Mask shape normalization failed: {}"
VALUE_ERROR_FMT_BOOLEAN_MASK_MISMATCH = "Boolean mask shape mismatch: {} vs expected ({},)"


def process_gdal_image_to_rgb(gdal_dataset: gdal.Dataset, logger: logging.Logger) -> np.ndarray:
    """Convert a GDAL dataset to a contiguous uint8 ``H x W x 3`` RGB array.

    Handles band count (1 or 3+), common dtypes, value scaling, and NoData.

    Args:
        gdal_dataset: Open GDAL raster dataset.
        logger: Logger for non-fatal dtype warnings.

    Returns:
        NumPy array shaped ``(height, width, 3)``, dtype ``uint8``, C-contiguous.

    Raises:
        ValueError: If the raster has an unsupported band count (not 1 and < 3).
    """
    num_bands = gdal_dataset.RasterCount
    all_bands = gdal_dataset.ReadAsArray()

    if all_bands.ndim == 2:
        all_bands = all_bands[np.newaxis, :, :]

    if num_bands >= 3:
        image_array = all_bands[:3, :, :]
    elif num_bands == 1:
        image_array = np.repeat(all_bands, 3, axis=0)
    else:
        raise ValueError(f"Unsupported number of bands: {num_bands}")

    for band_idx in range(min(3, num_bands)):
        band_num = band_idx + 1
        band = gdal_dataset.GetRasterBand(band_num)
        nodata_value = band.GetNoDataValue()
        if nodata_value is None:
            continue
        band_data = image_array[band_idx, :, :]
        if np.issubdtype(band_data.dtype, np.floating):
            nodata_mask = np.isclose(band_data, nodata_value, rtol=1e-5, atol=1e-8)
        else:
            nodata_mask = band_data == nodata_value
        image_array[band_idx, nodata_mask] = 0

    dtype = image_array.dtype

    if dtype == np.uint8:
        image_array = np.ascontiguousarray(image_array, dtype=np.uint8)
    elif dtype in (np.uint16, np.int16):
        if dtype == np.int16:
            image_array = image_array.astype(np.float32) - np.iinfo(np.int16).min
        max_val = np.iinfo(dtype).max
        image_array = (image_array.astype(np.float32) / max_val * 255).clip(0, 255).astype(np.uint8)
    elif dtype in (np.float32, np.float64):
        max_val = float(image_array.max())
        min_val = float(image_array.min())
        if max_val <= 1.0 and min_val >= 0.0:
            image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
        elif max_val <= 255.0 and min_val >= 0.0:
            image_array = image_array.clip(0, 255).astype(np.uint8)
        else:
            for b in range(3):
                band_data = image_array[b, :, :]
                band_min = band_data.min()
                band_max = band_data.max()
                if band_max > band_min:
                    image_array[b, :, :] = ((band_data - band_min) / (band_max - band_min) * 255).clip(0, 255)
                else:
                    image_array[b, :, :] = 0
            image_array = image_array.astype(np.uint8)
    else:
        logger.warning("Unknown data type %s, attempting direct cast to uint8", dtype)
        image_array = np.ascontiguousarray(image_array, dtype=np.uint8)

    image_array = np.transpose(image_array, (1, 2, 0))
    if not image_array.flags["C_CONTIGUOUS"]:
        image_array = np.ascontiguousarray(image_array)
    return image_array


def mask_to_polygon(mask: np.ndarray) -> Optional[List[List[float]]]:
    """Extract the largest exterior contour from a binary mask as a closed polygon.

    Args:
        mask: 2D (or squeezable to 2D) array; non-zero pixels treated as foreground.

    Returns:
        Closed ring ``[[x, y], ...]`` in pixel coordinates, or ``None`` if empty or degenerate.
    """
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim != 2 or mask.sum() == 0:
        return None

    mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) <= 3:
        return None

    polygon = (largest_contour.reshape(-1, 2) + 0.5).astype(np.float32).tolist()
    if len(polygon) >= 3 and polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    return polygon if len(polygon) >= 3 else None


def lam_detections_to_geojson(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    detection_type: str,
    *,
    device: str,
    logger: logging.Logger,
    detect_to_feature: Callable[..., Dict],
    model_name: str,
    ontology_version: str,
    max_workers: int,
) -> Dict:
    """Convert LAM mask/box/score tensors into an OSML-style GeoJSON FeatureCollection.

    Args:
        masks: Tensor shaped ``(N, H, W)`` or ``(N, 1, H, W)`` (logits or binary).
        boxes: Tensor shaped ``(N, 4)`` in xyxy image coordinates.
        scores: Tensor shaped ``(N,)`` confidence scores.
        detection_type: Prompt or label string stored per feature.
        device: ``\"cuda\"`` or ``\"cpu\"``; CUDA enables a fast empty-mask prefilter.
        logger: Logger for shape warnings and fallback paths.
        detect_to_feature: Callable that builds one feature dict from bbox, polygon, score, type.
        model_name: Stored in each feature's ``modelMetadata``.
        ontology_version: Stored in each feature's ``modelMetadata``.
        max_workers: Thread pool size for polygonization when ``N > 2``.

    Returns:
        GeoJSON dict: ``{\"type\": \"FeatureCollection\", \"features\": [...]}``.

    Raises:
        ValueError: If mask dimensions cannot be normalized to ``(N, H, W)``.
    """
    features: List[Dict] = []
    num_detections = len(scores)

    if masks.ndim == 4:
        if masks.shape[1] == 1:
            masks = masks.squeeze(1)
        else:
            logger.warning(LOG_FMT_UNEXPECTED_4D_MASK.format(masks.shape))
            if masks.shape[-2] == masks.shape[-1]:
                masks = masks[:, 0, :, :]
    elif masks.ndim == 2:
        masks = masks.unsqueeze(0)
    elif masks.ndim != 3:
        logger.error(LOG_FMT_UNEXPECTED_MASK_SHAPE.format(masks.shape))
        raise ValueError(VALUE_ERROR_FMT_CANNOT_PROCESS_MASKS.format(masks.shape))

    if masks.ndim != 3:
        logger.error(LOG_FMT_MASK_NORMALIZATION_FAILED.format(masks.shape))
        raise ValueError(VALUE_ERROR_FMT_MASK_NORMALIZATION_FAILED.format(masks.shape))

    if device == "cuda" and num_detections > 0:
        try:
            mask_sums = masks.sum(dim=(1, 2))
            non_empty_mask = mask_sums > 0
            if non_empty_mask.shape[0] != num_detections:
                logger.error(LOG_FMT_SHAPE_MISMATCH.format(non_empty_mask.shape[0], num_detections))
                raise ValueError(VALUE_ERROR_FMT_BOOLEAN_MASK_MISMATCH.format(non_empty_mask.shape, num_detections))
            if non_empty_mask.any():
                masks = masks[non_empty_mask]
                boxes = boxes[non_empty_mask]
                scores = scores[non_empty_mask]
                num_detections = len(scores)
            else:
                num_detections = 0
            if num_detections == 0:
                return {"type": "FeatureCollection", "features": []}
        except Exception as e:
            logger.error(LOG_FMT_GPU_FILTERING_FAILED.format(e, masks.shape, num_detections))
            logger.warning("Falling back to CPU processing due to GPU filtering error")

    if device == "cuda":
        torch.cuda.synchronize()

    boxes_cpu, scores_cpu, masks_cpu = boxes.cpu(), scores.cpu(), masks.cpu()
    scores_list = scores_cpu.tolist()
    boxes_list = boxes_cpu.tolist()
    masks_numpy = masks_cpu.numpy()

    if num_detections > 2:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            polygons = list(executor.map(mask_to_polygon, masks_numpy))
    else:
        polygons = [mask_to_polygon(masks_numpy[i]) for i in range(num_detections)]

    for i in range(num_detections):
        feature = detect_to_feature(
            fixed_object_bbox=boxes_list[i],
            fixed_object_mask=polygons[i],
            detection_score=scores_list[i],
            detection_type=detection_type,
        )
        feature["properties"]["modelMetadata"] = {
            "modelName": model_name,
            "ontologyName": detection_type,
            "ontologyVersion": ontology_version,
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def json_dumps_geojson(obj: Dict) -> tuple[str, str]:
    """Serialize a GeoJSON-like dict to a JSON string and MIME type.

    Args:
        obj: Mapping serializable as JSON (typically a FeatureCollection).

    Returns:
        ``(body, mimetype)`` where ``mimetype`` is ``application/json``.
        Uses ``orjson`` when installed, otherwise the standard ``json`` module.
    """
    try:
        import orjson

        return orjson.dumps(obj).decode("utf-8"), "application/json"
    except ImportError:
        import json

        return json.dumps(obj), "application/json"
