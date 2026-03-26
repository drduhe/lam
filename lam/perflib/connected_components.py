"""Batched 2D connected-components labeling (cc_torch, Triton, or CPU skimage)."""

from __future__ import annotations

import logging

import torch

try:
    from cc_torch import get_connected_components

    HAS_CC_TORCH = True
except ImportError:
    logging.debug(
        "cc_torch not found. Consider installing for better performance. Command line:"
        " pip install git+https://github.com/ronghanghu/cc_torch.git"
    )
    HAS_CC_TORCH = False


def connected_components_cpu_single(values: torch.Tensor):
    """Label one ``H x W`` map on CPU; returns label ids and per-pixel component sizes."""
    assert values.dim() == 2
    from skimage.measure import label

    labels, num = label(values.cpu().numpy(), return_num=True)
    labels = torch.from_numpy(labels)
    counts = torch.zeros_like(labels)
    for i in range(1, num + 1):
        cur_mask = labels == i
        cur_count = cur_mask.sum()
        counts[cur_mask] = cur_count
    return labels, counts


def connected_components_cpu(input_tensor: torch.Tensor):
    """Batched CPU labeling via ``skimage.measure.label`` per slice."""
    out_shape = input_tensor.shape
    if input_tensor.dim() == 4 and input_tensor.shape[1] == 1:
        input_tensor = input_tensor.squeeze(1)
    else:
        assert input_tensor.dim() == 3, "Input tensor must be (B, H, W) or (B, 1, H, W)."

    batch_size = input_tensor.shape[0]
    # Handle empty batch case
    if batch_size == 0:
        return (
            torch.zeros(out_shape, dtype=torch.int64, device=input_tensor.device),
            torch.zeros(out_shape, dtype=torch.int64, device=input_tensor.device),
        )

    labels_list = []
    counts_list = []
    for b in range(batch_size):
        labels, counts = connected_components_cpu_single(input_tensor[b])
        labels_list.append(labels)
        counts_list.append(counts)
    labels_tensor = torch.stack(labels_list, dim=0).to(input_tensor.device)
    counts_tensor = torch.stack(counts_list, dim=0).to(input_tensor.device)
    return labels_tensor.view(out_shape), counts_tensor.view(out_shape)


def connected_components(input_tensor: torch.Tensor):
    """Label foreground pixels per batch item using the best backend.

    Args:
        input_tensor: ``(B, H, W)`` or ``(B, 1, H, W)``; non-zero (or True) is foreground.

    Returns:
        ``(labels, counts)`` with the same shape as ``input_tensor``: dense labels (0 =
        background) and per-pixel component area.
    """
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(1)

    assert input_tensor.dim() == 4 and input_tensor.shape[1] == 1, "Input tensor must be (B, H, W) or (B, 1, H, W)."

    if input_tensor.is_cuda:
        if HAS_CC_TORCH:
            return get_connected_components(input_tensor.to(torch.uint8))
        else:
            # triton fallback
            from lam.perflib.triton.connected_components import (
                connected_components_triton,
            )

            return connected_components_triton(input_tensor)

    # CPU fallback
    return connected_components_cpu(input_tensor)
