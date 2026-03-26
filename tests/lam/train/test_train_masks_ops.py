"""Tests for ``lam.train.masks_ops``."""

from __future__ import annotations

import pytest
import torch

from lam.train import masks_ops


def test_instance_masks_to_semantic_masks():
    # two images: first has 2 instances, second has 0
    inst = torch.zeros(2, 4, 4, dtype=torch.bool)
    inst[0, 1:3, 1:3] = True
    inst[1, 2:4, 2:4] = True
    num = torch.tensor([2, 0])
    sem = masks_ops.instance_masks_to_semantic_masks(inst, num)
    assert sem.shape == (2, 4, 4)
    assert sem[0].any()
    assert not sem[1].any()


def test_mask_intersection():
    a = torch.zeros(2, 3, 3, dtype=torch.bool)
    b = torch.zeros(2, 3, 3, dtype=torch.bool)
    a[0, 0, 0] = a[1, 1, 1] = True
    b[0, 0, 0] = b[1, 2, 2] = True
    inter = masks_ops.mask_intersection(a, b, block_size=8)
    assert inter[0, 0] == 1
    assert inter[1, 1] == 0


def test_mask_iom():
    a = torch.zeros(1, 2, 2, dtype=torch.bool)
    b = torch.zeros(1, 2, 2, dtype=torch.bool)
    a[0, :, :] = True
    b[0, 0, 0] = True
    out = masks_ops.mask_iom(a, b)
    assert out.shape == (1, 1)
    assert 0 < float(out[0, 0]) <= 1


def test_compute_boundary():
    seg = torch.zeros(1, 4, 4, dtype=torch.bool)
    seg[0, 1:3, 1:3] = True
    b = masks_ops.compute_boundary(seg)
    assert b.dtype == torch.bool
    assert b.sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA dilation path")
def test_dilation_cuda():
    m = torch.zeros(1, 5, 5, dtype=torch.bool, device="cuda")
    m[0, 2, 2] = True
    out = masks_ops.dilation(m, kernel_size=3)
    assert out.shape == m.shape


def test_dilation_cpu_with_cv2():
    pytest.importorskip("cv2")
    m = torch.zeros(1, 5, 5, dtype=torch.bool)
    m[0, 2, 2] = True
    out = masks_ops.dilation(m, kernel_size=3)
    assert out[0, 2, 2]
