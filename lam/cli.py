"""CLI entrypoint for LAM GeoTIFF inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def _default_checkpoint_path() -> str | None:
    """Resolve checkpoint: ``CHECKPOINT_PATH`` env, else ``<repo>/assets/weights/sam3.pt`` if present."""
    env = os.environ.get("CHECKPOINT_PATH")
    if env:
        return env
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "assets" / "weights" / "sam3.pt"
    return str(candidate) if candidate.is_file() else None


def main() -> None:
    """Parse CLI arguments and run tiled GeoTIFF inference; writes GeoJSON to disk."""
    parser = argparse.ArgumentParser(description="LAM: Locate Anything Model — GeoTIFF tiling + grounded detection")
    parser.add_argument(
        "geotiff",
        help="Path to input GeoTIFF, or s3://bucket/key (GDAL /vsis3/ range reads; AWS creds via env or IAM role)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="objects",
        help="Text prompt for open-vocabulary detection",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="lam_out.geojson",
        help="Output GeoJSON path",
    )
    parser.add_argument("--tile-size", type=int, default=1008)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Weights .pt; overrides defaults. If omitted: $CHECKPOINT_PATH, else assets/weights/sam3.pt "
        "under the repo root when that file exists, otherwise Hugging Face.",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Do not download from Hugging Face (requires --checkpoint)",
    )
    parser.add_argument(
        "--dst-crs",
        default="EPSG:4326",
        help="CRS for output geometries (set empty to keep raster CRS)",
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.45,
        help="Cross-tile NMS IoU threshold; use negative to disable",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint if args.checkpoint is not None else _default_checkpoint_path()

    try:
        import rasterio  # noqa: F401
    except ImportError:
        print(
            "Install geospatial extras: pip install '.[geospatial]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from lam.geospatial.pipeline import run_geotiff_inference, write_geojson
    from lam.model.sam3_image_processor import Sam3Processor
    from lam.model_builder import build_lam_image_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_from_hf = not args.no_hf
    if args.no_hf and not checkpoint:
        print(
            "--no-hf requires a checkpoint ($CHECKPOINT_PATH, assets/weights/sam3.pt, or --checkpoint)", file=sys.stderr
        )
        sys.exit(1)

    model = build_lam_image_model(
        device=device,
        checkpoint_path=checkpoint,
        load_from_HF=load_from_hf,
    )
    model = model.to(device)
    processor = Sam3Processor(model, device=device)

    dst_crs = args.dst_crs.strip() or None
    merge_iou = None if args.merge_iou < 0 else args.merge_iou
    fc = run_geotiff_inference(
        processor,
        args.geotiff,
        args.prompt,
        tile_size=args.tile_size,
        overlap=args.overlap,
        dst_crs=dst_crs,
        merge_iou_threshold=merge_iou,
    )
    write_geojson(args.output, fc)
    print(f"Wrote {args.output} ({len(fc['features'])} features)")


if __name__ == "__main__":
    main()
