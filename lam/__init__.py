"""LAM (Locate Anything Model): open-vocabulary segmentation and detection.

Public factories are exposed as lazy attributes to keep import time low.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.2.1"

__all__ = ["build_lam_image_model", "build_sam3_image_model"]


def __getattr__(name: str) -> Any:
    """Load ``build_lam_image_model`` / ``build_sam3_image_model`` from ``lam.model_builder`` on demand."""
    if name in __all__:
        from lam.model_builder import build_lam_image_model, build_sam3_image_model

        return {
            "build_lam_image_model": build_lam_image_model,
            "build_sam3_image_model": build_sam3_image_model,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
