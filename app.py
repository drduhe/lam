"""Flask app: single-image LAM inference with GDAL decode and GeoJSON response.

Intended for SageMaker-style ``/ping`` and ``/invocations`` HTTP serving.
"""

from __future__ import annotations

import os
from secrets import token_hex
from typing import Any, Dict, Optional

import torch
from flask import Response, request
from osgeo import gdal
from PIL import Image

from lam.geospatial.s3_vsis3 import apply_gdal_s3_read_defaults, parse_s3_uri
from lam.hosting.sagemaker_geojson import json_dumps_geojson
from lam.hosting.server_utils import (
    build_flask_app,
    build_logger,
    parse_custom_attributes,
    setup_server,
    simulate_model_latency,
)
from lam.model.sam3_image_processor import Sam3Processor
from lam.model_builder import build_lam_image_model

gdal.UseExceptions()

app = build_flask_app(build_logger())


@app.before_request
def _log_request_path() -> None:
    if os.environ.get("LAM_LOG_HTTP", "").lower() in ("1", "true", "yes"):
        app.logger.info("%s %s", request.method, request.path)


# Fingerprint for local debugging (SageMaker uses /ping and /invocations only).
@app.route("/", methods=["GET"])
def root() -> Response:
    return Response(response="LAM inference server\n", status=200, mimetype="text/plain")


# SageMaker-style health check: HTTP 200 is what SageMaker checks; body is plain text for humans (`curl -s`).
@app.route("/ping", methods=["GET", "HEAD"], strict_slashes=False)
def healthcheck() -> Response:
    """SageMaker-style health check (200 OK)."""
    app.logger.debug("Responding to health check")
    return Response(response="healthy\n", status=200, mimetype="text/plain")


CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3"))
FUSE_DETECTIONS_IOU_THRESHOLD = (
    float(os.environ.get("FUSE_DETECTIONS_IOU_THRESHOLD")) if os.environ.get("FUSE_DETECTIONS_IOU_THRESHOLD") else None
)
TEXT_PROMPT = os.environ.get("DEFAULT_TEXT_PROMPT", "objects")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", None)
LOAD_FROM_HF = os.environ.get("LOAD_FROM_HF", "true").lower() == "true"

ENABLE_TORCH_COMPILE = (
    os.environ.get("ENABLE_TORCH_COMPILE", "true" if torch.cuda.is_available() else "false").lower() == "true"
)
MIXED_PRECISION = os.environ.get("MIXED_PRECISION", "bf16" if torch.cuda.is_available() else "fp32").lower()
PREWARM_GPU = os.environ.get("PREWARM_GPU", "true" if torch.cuda.is_available() else "false").lower() == "true"
TORCH_COMPILE_MODE = os.environ.get("TORCH_COMPILE_MODE", "reduce-overhead")

# Tiled GeoTIFF inference — same defaults as ``python -m lam`` (``lam/cli.py``).
TILE_SIZE = int(os.environ.get("LAM_TILE_SIZE", "1008"))
TILE_OVERLAP = int(os.environ.get("LAM_TILE_OVERLAP", "128"))
TILE_DST_CRS = os.environ.get("LAM_TILE_DST_CRS", "EPSG:4326").strip()
try:
    TILE_MERGE_IOU = float(os.environ.get("LAM_TILE_MERGE_IOU", "0.45"))
except ValueError:
    TILE_MERGE_IOU = 0.45

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _coerce_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_request_keys(mapping: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Lowercase keys and map hyphens to underscores (``dst-crs`` → ``dst_crs``)."""
    if not mapping:
        return {}
    out: Dict[str, Any] = {}
    for k, v in mapping.items():
        nk = str(k).strip().lower().replace("-", "_")
        out[nk] = v
    return out


def _layer_get(layer: Dict[str, Any], aliases: tuple[str, ...]) -> tuple[bool, Any]:
    """Return (True, value) if any alias is present (even when value is empty string)."""
    for a in aliases:
        key = a.lower().replace("-", "_")
        if key in layer:
            return True, layer[key]
    return False, None


def _request_attr_layers(
    custom_attrs: Dict[str, str],
    payload: Optional[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Ordered layers: custom attributes first, then JSON body (later overrides earlier)."""
    layers: list[Dict[str, Any]] = []
    layers.append(_normalize_request_keys(custom_attrs))
    layers.append(_normalize_request_keys(payload))
    return layers


def _resolve_invocation_params(
    custom_attrs: Dict[str, str],
    payload: Optional[Dict[str, Any]],
) -> tuple[str, Optional[str], int, int, str, float]:
    """Resolve text_prompt, s3_uri, and tiling options from attributes and JSON.

    Attribute / JSON key aliases (case-insensitive, ``-`` or ``_``):

    * **text_prompt** — ``text_prompt``, ``prompt``, ``lam_text_prompt``
    * **s3_uri** — ``s3_uri``, ``lam_s3_uri``
    * **tile_size** — ``tile_size``, ``lam_tile_size``
    * **overlap** — ``overlap``, ``lam_tile_overlap``
    * **dst_crs** — ``dst_crs``, ``lam_tile_dst_crs`` (empty string keeps raster CRS)
    * **merge_iou** — ``merge_iou``, ``lam_tile_merge_iou`` (negative disables cross-tile NMS)
    """
    layers = _request_attr_layers(custom_attrs, payload)

    text_prompt = TEXT_PROMPT
    s3_uri: Optional[str] = None
    tile_size = TILE_SIZE
    overlap = TILE_OVERLAP
    dst_crs = TILE_DST_CRS
    merge_iou = TILE_MERGE_IOU

    for layer in layers:
        if not layer:
            continue
        present, v = _layer_get(layer, ("text_prompt", "prompt", "lam_text_prompt"))
        if present and v is not None:
            text_prompt = str(v)
        present, v = _layer_get(layer, ("s3_uri", "lam_s3_uri"))
        if present and v is not None:
            s3_uri = str(v).strip() or None
        present, v = _layer_get(layer, ("tile_size", "lam_tile_size"))
        if present and v is not None:
            tile_size = _coerce_int(v, tile_size)
        present, v = _layer_get(layer, ("overlap", "lam_tile_overlap"))
        if present and v is not None:
            overlap = _coerce_int(v, overlap)
        present, v = _layer_get(layer, ("dst_crs", "lam_tile_dst_crs"))
        if present and v is not None:
            dst_crs = str(v).strip()
        present, v = _layer_get(layer, ("merge_iou", "lam_tile_merge_iou"))
        if present and v is not None:
            merge_iou = _coerce_float(v, merge_iou)

    return text_prompt, s3_uri, tile_size, overlap, dst_crs, merge_iou


def _resolve_bpe_path() -> Optional[str]:
    import lam

    candidates = [
        os.path.join(os.path.dirname(lam.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz"),
        "/opt/conda/envs/lam_sagemaker/lib/python3.13/site-packages/lam/assets/bpe_simple_vocab_16e6.txt.gz",
    ]
    for p in candidates:
        if os.path.isfile(p):
            app.logger.info("Using BPE vocabulary at %s", p)
            return p
    return None


if torch.cuda.is_available():
    app.logger.info(
        "CUDA %s, PyTorch %s, devices=%s",
        torch.version.cuda,
        torch.__version__,
        torch.cuda.device_count(),
    )
else:
    app.logger.warning("No CUDA — CPU inference")

app.logger.info("Loading LAM on device=%s", DEVICE)
BPE_PATH = _resolve_bpe_path()

if CHECKPOINT_PATH:
    app.logger.info("Local checkpoint: %s", CHECKPOINT_PATH)
    model = build_lam_image_model(
        device=DEVICE,
        checkpoint_path=CHECKPOINT_PATH,
        load_from_HF=False,
        bpe_path=BPE_PATH,
    )
else:
    app.logger.info("Loading from Hugging Face (LOAD_FROM_HF=%s)", LOAD_FROM_HF)
    model = build_lam_image_model(device=DEVICE, load_from_HF=LOAD_FROM_HF, bpe_path=BPE_PATH)

model = model.to(DEVICE)
model.eval()

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if ENABLE_TORCH_COMPILE:
    try:
        if hasattr(torch, "compile") and DEVICE == "cuda":
            app.logger.info("torch.compile(mode=%s)", TORCH_COMPILE_MODE)
            model = torch.compile(model, mode=TORCH_COMPILE_MODE)
        elif DEVICE == "cpu":
            app.logger.info("Skipping torch.compile on CPU")
    except Exception as e:
        app.logger.warning("torch.compile failed: %s", e)

EFFECTIVE_MIXED_PRECISION = MIXED_PRECISION
if DEVICE == "cuda" and EFFECTIVE_MIXED_PRECISION == "bf16" and not torch.cuda.is_bf16_supported():
    app.logger.warning("BF16 unsupported; using FP16")
    EFFECTIVE_MIXED_PRECISION = "fp16"

processor = Sam3Processor(
    model,
    device=DEVICE,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    fuse_detections_iou_threshold=FUSE_DETECTIONS_IOU_THRESHOLD,
)
app.logger.info("LAM model loaded")

if PREWARM_GPU and DEVICE == "cuda":
    try:
        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        with torch.inference_mode():
            if EFFECTIVE_MIXED_PRECISION in ("fp16", "bf16"):
                dtype = torch.bfloat16 if EFFECTIVE_MIXED_PRECISION == "bf16" else torch.float16
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                    st = processor.set_image(dummy)
                    processor.set_text_prompt("objects", st)
            else:
                st = processor.set_image(dummy)
                processor.set_text_prompt("objects", st)
        torch.cuda.empty_cache()
        app.logger.info("GPU prewarm done")
    except Exception as e:
        app.logger.warning("GPU prewarm failed: %s", e)


@app.route("/invocations", methods=["POST"])
def predict() -> Response:
    """Run LAM on raw raster bytes or ``s3://`` using the same tiling path as ``python -m lam``."""
    try:
        from lam.geospatial.pipeline import run_geotiff_inference
    except ImportError:
        return Response(
            response=(
                "HTTP /invocations requires geospatial dependencies (rasterio). "
                "Install: pip install 'lam[geospatial]' or '.[serve,geospatial]'.\n"
            ),
            status=500,
            mimetype="text/plain",
        )
    app.logger.debug("Invoking LAM model endpoint")
    simulate_model_latency()
    custom_attrs = parse_custom_attributes()

    payload: Optional[Dict[str, Any]] = None
    content_type = (request.content_type or "").lower()
    if "application/json" in content_type:
        raw = request.get_json(silent=True)
        if isinstance(raw, dict):
            payload = raw

    text_prompt, s3_uri, tile_size, overlap, dst_crs, merge_iou = _resolve_invocation_params(
        custom_attrs,
        payload,
    )
    merge_iou_threshold = None if merge_iou < 0 else merge_iou
    dst_crs_opt = dst_crs.strip() or None

    temp_ds_name: Optional[str] = None
    gdal_dataset = None
    path_for_tiles: str
    try:
        if s3_uri:
            try:
                parse_s3_uri(s3_uri)
            except ValueError as e:
                app.logger.warning("Invalid s3_uri: %s", e)
                return Response(response=str(e), status=400)
            apply_gdal_s3_read_defaults()
            path_for_tiles = s3_uri
        else:
            body = request.get_data()
            if not body:
                return Response(response="Empty request body (or use JSON / custom attributes with s3_uri)", status=400)
            temp_ds_name = "/vsimem/" + token_hex(16)
            gdal.FileFromMemBuffer(temp_ds_name, body)
            try:
                gdal_dataset = gdal.Open(temp_ds_name)
            except RuntimeError:
                app.logger.warning("GDAL could not parse request body")
                return Response(response="Unable to parse image from request!", status=400)
            if gdal_dataset is None:
                return Response(response="Unable to parse image from request!", status=400)
            del gdal_dataset
            gdal_dataset = None
            path_for_tiles = temp_ds_name

        with torch.inference_mode():
            if DEVICE == "cuda" and EFFECTIVE_MIXED_PRECISION in ("fp16", "bf16"):
                dtype = torch.bfloat16 if EFFECTIVE_MIXED_PRECISION == "bf16" else torch.float16
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
                    fc = run_geotiff_inference(
                        processor,
                        path_for_tiles,
                        text_prompt,
                        tile_size=tile_size,
                        overlap=overlap,
                        dst_crs=dst_crs_opt,
                        merge_iou_threshold=merge_iou_threshold,
                    )
            else:
                fc = run_geotiff_inference(
                    processor,
                    path_for_tiles,
                    text_prompt,
                    tile_size=tile_size,
                    overlap=overlap,
                    dst_crs=dst_crs_opt,
                    merge_iou_threshold=merge_iou_threshold,
                )

        nfeat = len(fc.get("features", []))
        app.logger.info(
            "LAM tiled inference: %d features (tile_size=%s overlap=%s dst_crs=%s merge_iou=%s)",
            nfeat,
            tile_size,
            overlap,
            dst_crs_opt,
            merge_iou_threshold,
        )
        body_json, mime = json_dumps_geojson(fc)
        return Response(response=body_json, status=200, mimetype=mime)

    except Exception as err:
        msg = str(err)
        app.logger.warning("LAM inference failed", exc_info=True)
        lower = msg.lower()
        if "404" in msg or "not found" in lower or "nosuchkey" in lower:
            return Response(response=f"Unable to open raster: {msg}", status=404)
        if "403" in msg or "access denied" in lower:
            return Response(response=f"Unable to open raster: {msg}", status=403)
        return Response(response=f"Unable to process request: {msg}", status=500)
    finally:
        if gdal_dataset is not None:
            del gdal_dataset
        if temp_ds_name is not None:
            gdal.Unlink(temp_ds_name)


if __name__ == "__main__":
    setup_server(app)
