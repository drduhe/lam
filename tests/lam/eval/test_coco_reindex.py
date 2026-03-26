"""Tests for ``lam.eval.coco_reindex``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lam.eval.coco_reindex import reindex_coco_to_temp


def test_reindex_zero_indexed_coco(tmp_path: Path):
    data = {
        "images": [{"id": 0, "width": 10, "height": 10, "file_name": "a.jpg"}],
        "categories": [{"id": 0, "name": "x"}],
        "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}],
    }
    p = tmp_path / "coco.json"
    p.write_text(json.dumps(data))
    out = reindex_coco_to_temp(str(p))
    assert out is not None
    loaded = json.loads(Path(out).read_text())
    assert loaded["images"][0]["id"] == 1
    assert loaded["annotations"][0]["image_id"] == 1


def test_reindex_missing_file():
    with pytest.raises(FileNotFoundError):
        reindex_coco_to_temp("/nonexistent/coco.json")
