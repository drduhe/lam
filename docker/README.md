# LAM SageMaker Docker

Build context must be the **LAM repository root** (the directory that contains `pyproject.toml`, `app.py`, `lam/`, and `conda/`).

## Files

- `Dockerfile.lam-sagemaker` — multi-stage image with Miniconda, GDAL (conda-forge), `pip install -e ".[serve,geospatial]"`, and weights copied from **`assets/weights/sam3.pt`** (synced from the Hugging Face bucket below) to **`/opt/checkpoint/sam3.pt`** in the image.
- `../conda/lam-sagemaker.yml` — Python 3.13 + **conda-forge** NumPy, PyTorch, TorchVision, GDAL, rasterio/shapely/pyproj/opencv, plus pip-installed extras (`timm`, Flask, …). LAM itself is **`pip install -e lam --no-deps`** so pip does not replace torch/numpy wheels.

## Weights (before build)

Sync checkpoint files into **`./assets/weights/`** using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`hf`). The image build expects **`sam3.pt`** at **`assets/weights/sam3.pt`**.

```bash
cd /path/to/lam
hf sync hf://buckets/drduhe/lam-weights ./assets/weights/
```

## Build

```bash
cd /path/to/lam
docker build -f docker/Dockerfile.lam-sagemaker -t lam-sagemaker:latest .
```

## Run locally (HTTP server — tiled raster, same as ``python -m lam``)

Default entrypoint starts `app.py` on port **8080**. **`POST /invocations`** uses the **tiling pipeline** (overlapping windows, cross-tile NMS, georeferenced GeoJSON). Send **raw raster bytes** (GeoTIFF, JPEG, …) or pass **`s3_uri`** via custom attributes / JSON. Tune with **`LAM_TILE_SIZE`**, **`LAM_TILE_OVERLAP`**, **`LAM_TILE_DST_CRS`**, **`LAM_TILE_MERGE_IOU`** (see root **README**).

```bash
docker run --rm -p 8080:8080 lam-sagemaker:latest
```

Health (expect HTTP 200 and "healthy"):

```bash
curl -i --noproxy '*' http://127.0.0.1:8080/ping
```

**If you see ``Not Found`` but container logs show Waitress started**, the request is usually **not** reaching this Flask app. Common causes:

1. **HTTP proxy** — ``curl`` may send even ``127.0.0.1`` through ``http_proxy``. Use ``--noproxy '*'`` (as above) or ``NO_PROXY=127.0.0.1,localhost``.
2. **Wrong host** — With SSH / remote development, ``curl`` must run on the **same machine** where Docker published the port (or use SSH ``-L 8080:localhost:8080``).
3. **Dev container** — ``127.0.0.1`` inside a dev container is **not** the host; from the host use ``localhost:8080``; from another container use ``host.docker.internal:8080`` (Mac/Win Docker Desktop).
4. **Confirm traffic hits LAM** — ``curl -s --noproxy '*' http://127.0.0.1:8080/`` should return ``LAM inference server``. From a **second** terminal while the container runs:
   ``docker exec "$(docker ps -q -f ancestor=lam-sagemaker:latest | head -1)" curl -s -i http://127.0.0.1:8080/ping``
   If that returns 200 but the host ``curl`` does not, the published port is not reaching your shell’s loopback.

Set ``LAM_LOG_HTTP=true`` on ``docker run`` (``-e LAM_LOG_HTTP=true``) to log each request path to the container logs.

Waitress tuning matches the OSML Models ``server_utils`` pattern: set ``WAITRESS_THREADS``, ``WAITRESS_CHANNEL_TIMEOUT``, or ``WAITRESS_CONNECTION_LIMIT`` if needed.

Tiled inference (example: GeoTIFF bytes; large rasters behave like the CLI):

```bash
curl -s -X POST http://127.0.0.1:8080/invocations \
  -H "Content-Type: image/tiff" \
  -H "X-Amzn-SageMaker-Custom-Attributes: text_prompt=building" \
  --data-binary @/path/to/small_tile.tif
```

Add **`--gpus all`** if the image is built with GPU-capable PyTorch and you want CUDA inside the container.

## Run locally (GeoTIFF — tiled `lam` CLI)

The same image installs the **`lam`** console script (`pip install -e ".[serve,geospatial]"`). For **large GeoTIFFs** (internal tiling + NMS), override the entrypoint and mount your data:

```bash
docker run --rm \
  -v /path/to/your/data:/data \
  --entrypoint /bin/bash \
  lam-sagemaker:latest \
  -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate lam_sagemaker && cd /home/lam && lam --prompt "ships" --output /data/out.geojson /data/your_image.tif'
```

Use **`--gpus all`** when PyTorch in the container is CUDA-enabled and you want GPU inference.

On Apple Silicon, omit **`--gpus`** (CPU inference). For a quick CPU-only smoke test, use a **small** GeoTIFF or reduce **`--tile-size`** if memory is tight.
