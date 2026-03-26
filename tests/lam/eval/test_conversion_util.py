"""Tests for ``lam.eval.conversion_util`` with minimal on-disk JSON (no I/O mocks)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lam.eval.conversion_util import convert_ytbvis_to_cocovid_gt, convert_ytbvis_to_cocovid_pred


def _minimal_ytvis_ann():
    return {
        "categories": [{"id": 1, "name": "cat"}],
        "videos": [
            {
                "id": 100,
                "file_names": ["v/frame0.jpg", "v/frame1.jpg"],
                "width": 64,
                "height": 48,
                "length": 2,
            }
        ],
        "annotations": [
            {
                "id": 7,
                "category_id": 1,
                "video_id": 100,
                "bboxes": [[10.0, 10.0, 20.0, 30.0], None],
                "areas": [200.0, 0.0],
                "segmentations": [None, None],
                "iscrowd": 0,
            }
        ],
    }


def test_convert_ytbvis_to_cocovid_gt_in_memory(tmp_path: Path):
    src = tmp_path / "in.json"
    src.write_text(json.dumps(_minimal_ytvis_ann()))
    vis = convert_ytbvis_to_cocovid_gt(str(src), save_path=None)
    assert len(vis["videos"]) == 1
    assert len(vis["images"]) == 2
    assert len(vis["tracks"]) == 1
    assert len(vis["annotations"]) == 1
    ann = vis["annotations"][0]
    assert ann["image_id"] == 1
    assert ann["bbox"] == [10.0, 10.0, 20.0, 30.0]


def test_convert_ytbvis_skips_none_bbox_frames(tmp_path: Path):
    src = tmp_path / "in.json"
    src.write_text(json.dumps(_minimal_ytvis_ann()))
    vis = convert_ytbvis_to_cocovid_gt(str(src), save_path=None)
    assert all(a["image_id"] != 2 for a in vis["annotations"])


def test_convert_ytbvis_writes_file(tmp_path: Path):
    src = tmp_path / "in.json"
    out = tmp_path / "out" / "coco.json"
    src.write_text(json.dumps(_minimal_ytvis_ann()))
    convert_ytbvis_to_cocovid_gt(str(src), save_path=str(out))
    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded["categories"][0]["id"] == 1


def _minimal_converted_coco_images():
    return {
        "images": [
            {"id": 1, "video_id": 100, "frame_index": 0, "width": 64, "height": 48},
            {"id": 2, "video_id": 100, "frame_index": 1, "width": 64, "height": 48},
        ]
    }


def test_convert_ytbvis_to_cocovid_pred_writes_annotations(tmp_path: Path, capsys):
    pred_path = tmp_path / "pred.json"
    coco_path = tmp_path / "coco.json"
    out_path = tmp_path / "out_pred.json"
    pred_path.write_text(
        json.dumps(
            [
                {
                    "video_id": 100,
                    "category_id": 1,
                    "bboxes": [[1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0]],
                    "score": 0.95,
                }
            ]
        )
    )
    coco_path.write_text(json.dumps(_minimal_converted_coco_images()))
    convert_ytbvis_to_cocovid_pred(str(pred_path), str(coco_path), str(out_path))
    rows = json.loads(out_path.read_text())
    assert len(rows) == 1
    assert rows[0]["image_id"] == 1
    assert rows[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert rows[0]["video_id"] == 100
    captured = capsys.readouterr()
    assert "Converted" in captured.out


def test_convert_ytbvis_to_cocovid_pred_missing_image_raises(tmp_path: Path):
    pred_path = tmp_path / "pred.json"
    coco_path = tmp_path / "coco.json"
    out_path = tmp_path / "out_pred.json"
    pred_path.write_text(
        json.dumps(
            [
                {
                    "video_id": 999,
                    "category_id": 1,
                    "bboxes": [[1.0, 2.0, 3.0, 4.0]],
                    "score": 0.5,
                }
            ]
        )
    )
    coco_path.write_text(json.dumps(_minimal_converted_coco_images()))
    with pytest.raises(RuntimeError, match="does not match any images"):
        convert_ytbvis_to_cocovid_pred(str(pred_path), str(coco_path), str(out_path))
