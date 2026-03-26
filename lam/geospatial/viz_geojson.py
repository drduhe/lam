"""Overlay LAM GeoJSON detections on the source GeoTIFF (matplotlib PNG or display)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.warp import transform_geom
from shapely.geometry import shape


def _read_rgb_uint8(src: DatasetReader) -> np.ndarray:
    """Normalize the first 1 or 3 bands to uint8 RGB ``(H, W, 3)`` for display."""
    if src.count >= 3:
        rgb = src.read([1, 2, 3])
    else:
        rgb = np.repeat(src.read(1)[None, :, :], 3, axis=0)
    rgb = np.transpose(rgb, (1, 2, 0))
    lo, hi = float(rgb.min()), float(rgb.max())
    if hi <= lo:
        return np.zeros_like(rgb, dtype=np.uint8)
    return ((rgb - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _geom_to_patches(geom: Any) -> List[MplPolygon]:
    """Turn a Shapely geometry into matplotlib :class:`~matplotlib.patches.Polygon` patches."""
    patches: List[MplPolygon] = []
    if geom.geom_type == "Polygon":
        patches.append(MplPolygon(np.array(geom.exterior.coords)[:, :2], closed=True))
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            patches.append(MplPolygon(np.array(poly.exterior.coords)[:, :2], closed=True))
    elif geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            patches.extend(_geom_to_patches(g))
    return patches


def visualize_geojson_on_geotiff(
    geotiff_path: str,
    geojson_path: str,
    *,
    output_path: str | None,
    geojson_crs: str,
    min_score: float,
    dpi: int = 150,
) -> None:
    """Plot GeoJSON polygons over a GeoTIFF (reprojecting geometries to the raster CRS).

    Args:
        geotiff_path: Path to the reference GeoTIFF (must define a CRS).
        geojson_path: Path to a LAM FeatureCollection JSON file.
        output_path: If set, save a PNG to this path; otherwise call ``plt.show()``.
        geojson_crs: CRS of the GeoJSON geometries (e.g. ``EPSG:4326``).
        min_score: Skip features with ``properties.score`` below this value.
        dpi: Figure resolution when saving.

    Raises:
        ValueError: If the GeoTIFF has no CRS.
    """
    import matplotlib.pyplot as plt

    with open(geojson_path, encoding="utf-8") as f:
        fc: Dict[str, Any] = json.load(f)

    features = fc.get("features") or []
    if not features:
        print("No features in GeoJSON.", file=sys.stderr)
        return

    src_crs = CRS.from_string(geojson_crs)

    with rasterio.open(geotiff_path) as src:
        rgb = _read_rgb_uint8(src)
        bounds = src.bounds
        dst_crs = src.crs
        if dst_crs is None:
            raise ValueError("GeoTIFF has no CRS; cannot align with GeoJSON.")

    patches: List[MplPolygon] = []
    for feat in features:
        props = feat.get("properties") or {}
        score = float(props.get("score", 1.0))
        if score < min_score:
            continue
        raw = feat.get("geometry")
        if not raw:
            continue
        try:
            tgeom = transform_geom(src_crs, dst_crs, raw, precision=9)
            g = shape(tgeom)
            if not g.is_valid:
                g = g.buffer(0)
            patches.extend(_geom_to_patches(g))
        except Exception as e:
            print(f"Skipping feature (transform/plot): {e}", file=sys.stderr)

    fig_w = rgb.shape[1] / dpi
    fig_h = rgb.shape[0] / dpi
    fig, ax = plt.subplots(figsize=(max(fig_w, 4), max(fig_h, 4)))
    extent: Tuple[float, float, float, float] = (
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top,
    )
    ax.imshow(rgb, extent=extent, origin="upper")
    if patches:
        ax.add_collection(
            PatchCollection(
                patches,
                facecolor="lime",
                edgecolor="darkgreen",
                linewidth=0.8,
                alpha=0.35,
            )
        )
    ax.set_title(f"LAM overlay ({len(patches)} polygons, score ≥ {min_score})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    """CLI entry for ``lam-viz``."""
    parser = argparse.ArgumentParser(description="Draw LAM GeoJSON polygons on the source GeoTIFF (matplotlib).")
    parser.add_argument("geotiff", help="Source GeoTIFF (same file used for lam inference)")
    parser.add_argument("geojson", help="LAM output GeoJSON")
    parser.add_argument(
        "-o",
        "--output",
        help="Save PNG path (omit to open an interactive window)",
    )
    parser.add_argument(
        "--geojson-crs",
        default="EPSG:4326",
        help="CRS of GeoJSON geometries (default matches lam --dst-crs default)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum detection score to draw",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as _plt  # noqa: F401
    except ImportError:
        print(
            "matplotlib is required. Install geospatial extras: pip install '.[geospatial]'",
            file=sys.stderr,
        )
        sys.exit(1)

    visualize_geojson_on_geotiff(
        args.geotiff,
        args.geojson,
        output_path=args.output,
        geojson_crs=args.geojson_crs,
        min_score=args.min_score,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
