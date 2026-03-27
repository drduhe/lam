#!/usr/bin/env python3
"""Wall-clock benchmark for ``python -m lam`` on a GeoTIFF.

Matches typical SageMaker notebook usage (subprocess, fresh interpreter per run).
Use ``--raster-info-only`` to print dimensions and tile counts without loading the model.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _count_tiles(width: int, height: int, tile_size: int, overlap: int) -> int:
    from lam.geospatial.tiling import iter_tiles

    return sum(1 for _ in iter_tiles(width, height, tile_size, overlap))


def _raster_summary(path: str, tile_size: int, overlap: int) -> dict[str, Any]:
    import rasterio

    with rasterio.open(path) as ds:
        w, h = int(ds.width), int(ds.height)
        crs_wkt = ds.crs.to_wkt() if ds.crs else None
        return {
            "path": os.path.abspath(path),
            "width": w,
            "height": h,
            "count": int(ds.count),
            "dtype": str(ds.dtypes[0]) if ds.dtypes else None,
            "crs_wkt": crs_wkt,
            "tile_count_default_cli": _count_tiles(w, h, tile_size, overlap),
        }


def _torch_cuda_summary() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda_available": False}

    out: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        out["cuda_device"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        out["cuda_total_memory_gb"] = round(props.total_memory / (1024**3), 2)
    return out


def _build_lam_argv(
    geotiff: str,
    prompt: str,
    output: str,
    tile_size: int,
    overlap: int,
    extra: list[str],
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "lam",
        "--prompt",
        prompt,
        "-o",
        output,
        "--tile-size",
        str(tile_size),
        "--overlap",
        str(overlap),
        geotiff,
    ]
    return argv + list(extra)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark wall time for python -m lam on a GeoTIFF (subprocess, like SageMaker notebooks).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("geotiff", help="Path to input GeoTIFF")
    parser.add_argument("--prompt", default="aircraft", help="Same as lam --prompt")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output GeoJSON path (default: temp file, removed unless --keep-output)",
    )
    parser.add_argument("--tile-size", type=int, default=1008)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument(
        "--repeat", type=int, default=1, help="Number of timed runs (same process invocation each time)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Untimed runs before measuring (loads weights and CUDA kernels)",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Do not delete the output GeoJSON when -o was omitted",
    )
    parser.add_argument(
        "--raster-info-only",
        action="store_true",
        help="Print raster + tile count and exit (no model load)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a single JSON object with results to stdout",
    )
    parser.epilog = (
        "Any arguments this script does not recognize are forwarded to lam "
        "(e.g. --checkpoint /path/sam3.pt --merge-iou 0.3). "
        "You can still use a bare -- before them if you prefer."
    )
    args, lam_extra = parser.parse_known_args()

    geotiff = os.path.abspath(args.geotiff)
    if not os.path.isfile(geotiff):
        print(f"Input not found: {geotiff}", file=sys.stderr)
        return 2

    raster = _raster_summary(geotiff, args.tile_size, args.overlap)
    if args.raster_info_only:
        payload = {"raster": raster, "host": {"platform": platform.platform(), **_torch_cuda_summary()}}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Raster: {raster['width']} x {raster['height']} px, tiles ~{raster['tile_count_default_cli']}")
        return 0

    out_path = args.output
    tmp_path: str | None = None
    if out_path is None:
        fd, tmp_path = tempfile.mkstemp(suffix=".geojson", prefix="lam_benchmark_")
        os.close(fd)
        out_path = tmp_path

    argv_template = _build_lam_argv(
        geotiff,
        args.prompt,
        out_path,
        args.tile_size,
        args.overlap,
        lam_extra,
    )

    host = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": os.getcwd(),
        **_torch_cuda_summary(),
    }

    for _ in range(args.warmup):
        subprocess.run(argv_template, check=True, cwd=str(_repo_root()))

    wall_seconds: list[float] = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        subprocess.run(argv_template, check=True, cwd=str(_repo_root()))
        wall_seconds.append(time.perf_counter() - t0)

    result: dict[str, Any] = {
        "raster": raster,
        "lam_argv": argv_template,
        "host": host,
        "wall_seconds": wall_seconds,
        "wall_seconds_mean": sum(wall_seconds) / len(wall_seconds),
        "output_path": os.path.abspath(out_path),
    }

    if tmp_path and not args.keep_output:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("LAM subprocess:", " ".join(argv_template))
        print(f"Wall time (s): {wall_seconds}  mean={result['wall_seconds_mean']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
