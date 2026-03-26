# LAM — Locate Anything Model

**LAM** is an open-vocabulary detection and segmentation stack for **still imagery**. You describe what to find in natural language; the model returns **masks, boxes, and scores**, exposed as **GeoJSON-friendly features** for GIS workflows and downstream services.

At its core, LAM uses the same **text-prompted grounding idea as SAM 3**: a single RGB view goes through the image backbone and decoder, and a text prompt selects which objects to segment. That per-image behavior is what you get when you call the processor on one PIL image or when the HTTP service decodes one request body.

**LAM extends that core with a native, in-repo geospatial path for large GeoTIFFs.** You do not have to pre-tile rasters in a separate GIS application or pre-processing logic. The **`lam` CLI** (and the underlying `run_geotiff_inference` API) opens the mosaic with **rasterio**, walks the raster in configurable **windows** (tile size and overlap), and runs the same SAM 3–class inference on each window. Detections are **lifted from tile-local coordinates into full-mosaic pixel space**: bounding boxes are offset by each window’s origin, and masks are **embedded into a full-height, full-width raster canvas** so every instance is expressed in one common pixel grid. Because neighboring tiles overlap, the same object can appear more than once; LAM then applies **class-agnostic non-maximum suppression (NMS)** on those full-raster boxes (by default using IoU on axis-aligned boxes) so the merged result is a single set of instances for the whole extent.

**Georeferencing is handled inside that same pipeline, not as a post-processing hack.** Each GeoTIFF carries an **affine geotransform** and a **coordinate reference system (CRS)**. After inference and optional cross-tile merging, mask outlines and boxes are transformed from pixel space into **real-world coordinates** using that transform and CRS. You can **reproject outputs** to another CRS (for example **EPSG:4326** for WGS 84 longitude/latitude) before writing **GeoJSON**, so features are ready for web maps, GIS databases, or orchestration stacks—while the HTTP **`/invocations`** path still returns **image-coordinate** GeoJSON for single-image payloads where no geotransform is in play.

The codebase is a **standalone packaging** of the SAM 3–class image architecture and compatible weights (for example [`facebook/sam3`](https://huggingface.co/facebook/sam3) on Hugging Face). It does **not** depend on Meta’s upstream `sam3` Python package—everything ships under this repository’s `lam` package.

---

## What you can do with LAM

| Mode | Best for | Output |
|------|-----------|--------|
| **HTTP server** (`app.py`) | One image per request (tiles, chips, small rasters in memory) | **GeoJSON** in **pixel / image** coordinates |
| **CLI** (`lam`) | Large **GeoTIFF** mosaics | **Georeferenced GeoJSON** (reprojectable; cross-tile **NMS** to drop duplicates) |
| **Python** | Custom pipelines | `build_lam_image_model` + `Sam3Processor` for `set_image` / `set_text_prompt` |

---

## Requirements

- **Python** 3.10+
- **PyTorch** and **torchvision** (CUDA build on GPU hosts; see [pytorch.org](https://pytorch.org/))
- **Geospatial CLI**: optional extras pull in **rasterio**, **shapely**, **pyproj**, etc.
- **Visualization** (upstream-style overlays): optional **`[viz]`** extra for **`lam.visualization_utils`** (`plot_mask`, `COLORS`, video/COCO helpers)—see Python API below.
- **HTTP server**: GDAL-backed decode via **`osgeo.gdal`** (provided by the conda environment below or your system GDAL bindings)

---

## Installation

### Option A — Conda (recommended, especially on macOS)

Using **conda-forge** for PyTorch, NumPy, TorchVision, and GDAL avoids common **OpenMP / libomp** conflicts when mixing wheels.

```bash
# From the repository root (directory containing pyproject.toml)
conda env create -f conda/lam-sagemaker.yml
conda activate lam_sagemaker
pip install -e . --no-deps
```

If this environment previously had **PyPI** `torch`, `torchvision`, or `numpy`, remove them so only conda-forge builds load:

```bash
pip uninstall -y torch torchvision numpy
conda env update -f conda/lam-sagemaker.yml
```

**Linux + NVIDIA GPU (e.g. SageMaker):** conda-forge’s `pytorch` is often CPU-only. Use `scripts/setup_sagemaker_notebook.sh` with **`LAM_PYTORCH_CUDA=12.4`** (maps to pip index **`cu124`**) so the script removes conda `pytorch`/`torchvision` and installs CUDA wheels from `download.pytorch.org`. Override the index with **`LAM_PYTORCH_WHL=cu124`** if needed. On macOS, do not set these (keep conda torch for OpenMP sanity).

### Option B — pip only

```bash
pip install -e .
pip install -e ".[geospatial]"   # large GeoTIFF CLI
pip install -e ".[serve]"        # Flask + Waitress + server deps
pip install -e ".[viz]"          # matplotlib/OpenCV/pandas/sklearn/skimage for lam.visualization_utils
```

Install a matching **PyTorch** build for your platform. On macOS, if you see duplicate **libomp** errors, prefer the conda route above or, as a last resort, `KMP_DUPLICATE_LIB_OK=TRUE`.

---

## Quick start

### GeoTIFF → GeoJSON (CLI)

Install **`[geospatial]`** so the **`lam`** and **`lam-viz`** entry points are available (`pip install -e ".[geospatial]"`).

**Run inference** (writes a georeferenced **FeatureCollection**):

```bash
lam --prompt "aircraft" -o out.geojson assets/images/small.tif
```

**Visualize** the same GeoJSON on top of the source GeoTIFF (matplotlib). Save a PNG:

```bash
lam-viz assets/images/small.tif out.geojson -o out_overlay.png
```

Or open an interactive plot (needs a display; omit **`-o`**):

```bash
lam-viz assets/images/small.tif out.geojson
```

Use **`--min-score`** (for example **`0.35`**) to hide low-confidence polygons. **`--geojson-crs`** must match the CRS you used when writing the GeoJSON: the default is **`EPSG:4326`**, same as **`lam --dst-crs`**; if you ran **`lam --dst-crs ""`** to keep the raster CRS, pass that CRS to **`lam-viz`** (for example **`--geojson-crs EPSG:32633`**).

- Tiling defaults: **1008** px tiles, **128** px overlap (override with **`--tile-size`** / **`--overlap`**).
- Cross-tile deduplication uses **class-agnostic NMS** on boxes in raster space (default IoU **0.45**). Disable with **`--merge-iou -1`**.
- Output CRS: **`--dst-crs EPSG:4326`** by default; set **`--dst-crs`** empty to keep the raster’s CRS.
- Weights: **`assets/weights/sam3.pt`** at the repo root is used when that file exists (populate with **`hf sync hf://buckets/drduhe/lam-weights ./assets/weights/`**); else **`$CHECKPOINT_PATH`** if set; else Hugging Face. Pass **`--checkpoint`** to override; **`--no-hf`** requires a local path (env, that file, or **`--checkpoint`**).

### HTTP server (tiled raster per request)

```bash
python app.py
```

By default, weights load from **Hugging Face** when no local checkpoint is configured (`LOAD_FROM_HF` defaults to **`true`** in `app.py`). Set **`LOAD_FROM_HF=false`** or **`CHECKPOINT_PATH`** when you want offline / image-bundled weights only.

- Listens on **`0.0.0.0:8080`** by default (override with **`SAGEMAKER_BIND_TO_PORT`**).
- **`GET /ping`** — health check (HTTP 200, plain-text body `healthy`).
- **`POST /invocations`** — same **tiling pipeline** as **`python -m lam`**: overlapping windows (defaults **1008** px, overlap **128**), cross-tile NMS, georeferenced GeoJSON in **`LAM_TILE_DST_CRS`** (default **`EPSG:4326`**). Send **raw raster bytes** in the body (for example GeoTIFF); **`Content-Type`** should match the format (`image/tiff`, `image/jpeg`, …). Alternatively pass **`s3_uri`** in **`X-Amzn-SageMaker-Custom-Attributes`** or a JSON body (see below).

Per-request parameters can be set in **`X-Amzn-SageMaker-Custom-Attributes`** (comma-separated `key=value`, values URL-encoded as needed) **and/or** in a JSON body when **`Content-Type: application/json`**. If both are present, **JSON overrides** the header for the same key.

| Parameter | Purpose | Accepted keys (case-insensitive; `-` and `_` equivalent) |
|-----------|---------|------------------------------------------------------------|
| Text prompt | Open-vocabulary query | `text_prompt`, `prompt`, `lam_text_prompt` |
| S3 input | Read raster from S3 instead of body | `s3_uri`, `lam_s3_uri` |
| Tile size | Window size (px) | `tile_size`, `lam_tile_size` |
| Overlap | Tile overlap (px) | `overlap`, `lam_tile_overlap` |
| Output CRS | GeoJSON geometry CRS | `dst_crs`, `lam_tile_dst_crs` (empty value keeps raster CRS) |
| Cross-tile NMS | Box merge IoU | `merge_iou`, `lam_tile_merge_iou` (**`-1`** disables) |

Environment variables **`LAM_TILE_*`** still set defaults when a key is omitted.

```bash
curl -s -X POST http://127.0.0.1:8080/invocations \
  -H "Content-Type: image/tiff" \
  -H "X-Amzn-SageMaker-Custom-Attributes: text_prompt=building" \
  --data-binary @/assets/images/small.tif
```

Response body is a **GeoJSON FeatureCollection** with geometries in the requested output CRS (raster CRS if **`dst_crs`** is empty).

---

## Server configuration (environment)

| Variable | Purpose |
|----------|---------|
| `LOAD_FROM_HF` | Default **`true`** — allow Hugging Face when no local checkpoint; set **`false`** to require local weights |
| `CHECKPOINT_PATH` | Path to local weights (convention: **`assets/weights/sam3.pt`**); when set, Hugging Face load is skipped |
| `DEFAULT_TEXT_PROMPT` | Prompt if the request does not set `text_prompt` (default `objects`) |
| `CONFIDENCE_THRESHOLD` | Detection confidence cutoff (default `0.3`) |
| `FUSE_DETECTIONS_IOU_THRESHOLD` | Optional IoU threshold for fusing overlapping detections **within each tile** |
| `LAM_TILE_SIZE` | HTTP `/invocations` tile width/height in pixels (default `1008`, same as CLI) |
| `LAM_TILE_OVERLAP` | HTTP tile overlap in pixels (default `128`) |
| `LAM_TILE_DST_CRS` | HTTP output CRS, e.g. `EPSG:4326` (empty string keeps raster CRS) |
| `LAM_TILE_MERGE_IOU` | Cross-tile box NMS IoU threshold (default `0.45`; negative disables) |
| `ENABLE_TORCH_COMPILE` | `torch.compile` on CUDA (default on when CUDA is available) |
| `TORCH_COMPILE_MODE` | Passed to `torch.compile` (default `reduce-overhead`) |
| `MIXED_PRECISION` | `bf16`, `fp16`, or `fp32` (CUDA; BF16 falls back if unsupported) |
| `PREWARM_GPU` | Run a dummy forward on startup when CUDA is available |
| `SAGEMAKER_BIND_TO_PORT` | Listen port (default `8080`) |

---

## Docker and SageMaker

Production-style images and local run examples (including GPU and large GeoTIFF entrypoint overrides) are documented in **[`docker/README.md`](docker/README.md)**.

Weights for the image must be present under **`assets/weights/`** (for example **`sam3.pt`**). Sync them from the project Hugging Face bucket, then build from the **repository root** (the directory that contains `pyproject.toml` and `app.py`):

```bash
hf sync hf://buckets/drduhe/lam-weights ./assets/weights/
docker build -f docker/Dockerfile.lam-sagemaker -t lam-sagemaker:latest .
```

---

## Repository layout

| Path | Contents |
|------|----------|
| `assets/weights/` | Local **`sam3.pt`** (gitignored); populate with **`hf sync hf://buckets/drduhe/lam-weights ./assets/weights/`** for Docker builds and offline use; or set **`CHECKPOINT_PATH`** / **`--checkpoint`** |
| `lam/` | Model, training, evaluation, inference helpers, **`build_lam_image_model`**, tokenizer asset under `lam/assets/` |
| `lam/geospatial/` | GeoTIFF tiling pipeline and georeferenced GeoJSON output |
| `lam/visualization_utils.py` | Upstream-aligned plotting helpers (`plot_mask`, `COLORS`, video/COCO tools); needs **`[viz]`** (conda env includes these via pip) |
| `app.py` | Flask app: GDAL decode → LAM → GeoJSON |
| `lam/hosting/` | Server helpers, SageMaker-oriented parsing, GeoJSON serialization |
| `docker/` | Multi-stage SageMaker-oriented Dockerfile |
| `conda/lam-sagemaker.yml` | Reference conda environment for GDAL + aligned scientific stack |

---

## Python API (sketch)

```python
import torch
from lam import build_lam_image_model
from lam.model.sam3_image_processor import Sam3Processor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_lam_image_model(device=device, load_from_HF=True).to(device).eval()
processor = Sam3Processor(model, device=device, confidence_threshold=0.3)

# PIL Image or similar → state → prompt
state = processor.set_image(pil_image)
out = processor.set_text_prompt("your prompt here", state)
# out contains masks, boxes, scores
```

Install **`[viz]`** for **`lam.visualization_utils`** (same role as upstream `sam3.visualization_utils`: `COLORS`, `plot_mask`, `plot_bbox`, `plot_results`, video/COCO helpers).

```python
from lam.visualization_utils import COLORS, plot_mask

# After processor.set_text_prompt(...): overlay each mask on the PIL image in matplotlib
```

---

## Development

For **branch workflow, reviews, and commit conventions**, see [CONTRIBUTING.md](CONTRIBUTING.md). **Release notes** live in [CHANGELOG.md](CHANGELOG.md).

Install dev tools and enable [pre-commit](https://pre-commit.com/) hooks (Ruff lint + format, YAML/TOML checks, whitespace):

```bash
pip install '.[dev]'
pre-commit install
```

Run on all tracked files once: `pre-commit run --all-files`. Hooks use the Ruff settings in [`pyproject.toml`](pyproject.toml).

---

## Community and support

**Maintenance** — LAM is supported **on a best-effort basis**. There is no service-level agreement and response times will vary.

**Where to ask** — Use **GitHub Issues** on **this repository** for **bug reports** and **feature requests** (issue templates help triage). Use **GitHub Discussions** for **questions**, how-tos, and broader design or integration topics when Discussions are enabled here; if they are not enabled yet, opening an Issue for a question is fine.

**Help wanted** — The maintainer is actively looking for **community members who want to share ownership**: reviews, documentation, issue triage, and steering the roadmap. If you rely on LAM and want to co-maintain, say so in a Discussion, Issue, or PR thread.

**Why this repo** — LAM is an independent packaging aimed at **geospatial tiling**, **HTTP serving**, and **day-to-day integration** on top of the SAM 3–class image model. One reason it exists is **limited ongoing development** in Meta’s upstream **[`facebookresearch/sam3`](https://github.com/facebookresearch/sam3)** repository for those downstream concerns; LAM is meant to move them forward **with the community**. Point `[project.urls]` in [`pyproject.toml`](pyproject.toml) at this repo when you publish so PyPI and GitHub show the right home.

---

## License

Model and code are subject to the **SAM License** in [`LICENSE`](LICENSE) (Meta). Review the agreement before use or redistribution.

---

## Acknowledgments

LAM builds on research and open releases from the **Segment Anything** / SAM family. Weights and architecture align with publicly released SAM 3 image checkpoints (for example on Hugging Face); this repository is an independent packaging for geospatial and serving workflows.
