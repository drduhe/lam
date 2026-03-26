"""GeoTIFF tiling, georeferencing, and tiled LAM inference."""

from __future__ import annotations

from typing import Any

__all__ = ["run_geotiff_inference"]


def __getattr__(name: str) -> Any:
    """Lazy-export ``run_geotiff_inference`` from ``lam.geospatial.pipeline``."""
    if name == "run_geotiff_inference":
        from lam.geospatial.pipeline import run_geotiff_inference

        return run_geotiff_inference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
