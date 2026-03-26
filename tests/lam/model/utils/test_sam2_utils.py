"""Tests for ``lam.model.utils.sam2_utils`` (frame loading and routing)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from lam.model.utils import sam2_utils


def test_load_video_frames_rejects_non_path_types():
    with pytest.raises(NotImplementedError, match="Only MP4 video and JPEG folder"):
        sam2_utils.load_video_frames(123, image_size=32, offload_video_to_cpu=True)


def test_load_video_frames_from_jpg_images_requires_directory():
    with pytest.raises(NotImplementedError, match="Only JPEG frames"):
        sam2_utils.load_video_frames_from_jpg_images(
            "/definitely/not/a/real/dir/00000",
            image_size=32,
            offload_video_to_cpu=True,
        )


def test_load_video_frames_from_jpg_images_empty_folder_raises(tmp_path):
    jdir = tmp_path / "frames"
    jdir.mkdir()
    with pytest.raises(RuntimeError, match="no images found"):
        sam2_utils.load_video_frames_from_jpg_images(
            str(jdir),
            image_size=16,
            offload_video_to_cpu=True,
        )


def test_load_video_frames_from_jpg_images_sync(tmp_path):
    jdir = tmp_path / "frames"
    jdir.mkdir()
    for i in range(2):
        img = Image.new("RGB", (40, 30), color=(i * 80, 10, 20))
        img.save(jdir / f"{i}.jpg")

    images, vh, vw = sam2_utils.load_video_frames_from_jpg_images(
        str(jdir),
        image_size=16,
        offload_video_to_cpu=True,
        compute_device=torch.device("cpu"),
    )
    assert images.shape == (2, 3, 16, 16)
    assert vh == 30 and vw == 40
    assert images.dtype == torch.float32


def test_load_video_frames_from_jpg_images_async_waits_for_thread(tmp_path):
    jdir = tmp_path / "frames"
    jdir.mkdir()
    for i in range(3):
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(jdir / f"{i}.jpg")

    loader, _, _ = sam2_utils.load_video_frames_from_jpg_images(
        str(jdir),
        image_size=8,
        offload_video_to_cpu=True,
        async_loading_frames=True,
        compute_device=torch.device("cpu"),
    )
    for i in range(len(loader)):
        _ = loader[i]
    loader.thread.join(timeout=10.0)
    assert not loader.thread.is_alive()


def test_load_img_as_tensor_uint8_resize(tmp_path):
    path = tmp_path / "a.jpg"
    Image.new("RGB", (20, 10), color=(128, 64, 32)).save(path)
    tensor, h, w = sam2_utils._load_img_as_tensor(str(path), image_size=8)
    assert tensor.shape == (3, 8, 8)
    assert h == 10 and w == 20
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_load_img_as_tensor_unknown_dtype_raises(tmp_path):
    """``np.uint8`` is expected after decode; exercise the error branch via a mock."""
    path = tmp_path / "a.jpg"
    Image.new("RGB", (4, 4)).save(path)
    with patch.object(sam2_utils.np, "array", return_value=np.ones((4, 4, 3), dtype=np.float32)):
        with pytest.raises(RuntimeError, match="Unknown image dtype"):
            sam2_utils._load_img_as_tensor(str(path), image_size=4)
