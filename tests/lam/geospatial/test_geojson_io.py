"""Tests for ``lam.geospatial.geojson_io`` (requires rasterio + OpenCV via ``lam.geospatial.masks``)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cv2")
pytest.importorskip("rasterio")
from rasterio.crs import CRS
from rasterio.transform import Affine

from lam.geospatial import geojson_io


def test_pixel_detection_to_feature_bbox_fallback():
    crs = CRS.from_epsg(4326)
    feat = geojson_io.pixel_detection_to_feature(
        Affine.identity(),
        crs,
        crs,
        [0.0, 0.0, 10.0, 10.0],
        None,
        0.85,
        "thing",
    )
    assert feat["type"] == "Feature"
    assert feat["properties"]["label"] == "thing"
    assert feat["properties"]["score"] == pytest.approx(0.85)
    assert feat["geometry"]["type"] == "Polygon"


def test_stack_to_geojson_features_empty_masks():
    crs = CRS.from_epsg(4326)
    masks = torch.zeros(0, 8, 8, dtype=torch.bool)
    boxes = torch.zeros(0, 4)
    scores = torch.zeros(0)
    feats = geojson_io.stack_to_geojson_features(
        Affine.identity(),
        crs,
        crs,
        masks,
        boxes,
        scores,
        "lbl",
    )
    assert feats == []
