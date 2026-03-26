"""Flask and SageMaker request helpers for LAM HTTP serving."""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from secrets import token_hex
from typing import Dict, List, Optional, Union
from urllib.parse import unquote

import json_logging
from flask import Flask, request
from osgeo import gdal

gdal.UseExceptions()


def build_logger(level: int = logging.INFO) -> logging.Logger:
    """Create a logger that emits JSON-formatted records to stdout.

    Args:
        level: Minimum log level for the logger.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(__name__)

    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        logging.root.addHandler(handler)

    logger.setLevel(level)
    return logger


def setup_server(app: Flask) -> None:
    """Start Waitress to serve the Flask app (intended for container entrypoints).

    Binds to ``0.0.0.0`` and port ``SAGEMAKER_BIND_TO_PORT`` (default ``8080``).
    Tunables match ``aws.osml.models.server_utils`` (osml-models): ``WAITRESS_THREADS``,
    ``WAITRESS_CHANNEL_TIMEOUT``, ``WAITRESS_CONNECTION_LIMIT``.

    Args:
        app: Flask application instance.
    """
    port = int(os.getenv("SAGEMAKER_BIND_TO_PORT", 8080))
    default_threads = 16
    threads = int(os.getenv("WAITRESS_THREADS", default_threads))
    channel_timeout = int(os.getenv("WAITRESS_CHANNEL_TIMEOUT", 120))
    connection_limit = int(os.getenv("WAITRESS_CONNECTION_LIMIT", 100))

    app.logger.info("Initializing LAM Flask server on port %s", port)
    app.logger.info(
        "Waitress: threads=%s, channel_timeout=%ss, connection_limit=%s",
        threads,
        channel_timeout,
        connection_limit,
    )
    from waitress import serve

    serve(
        app,
        host="0.0.0.0",
        port=port,
        threads=threads,
        channel_timeout=channel_timeout,
        connection_limit=connection_limit,
        clear_untrusted_proxy_headers=True,
    )


def build_flask_app(logger: logging.Logger) -> Flask:
    """Build a Flask app wired to the given logger and JSON logging for requests.

    Args:
        logger: Logger whose handlers and level are attached to ``app.logger``.

    Returns:
        Configured :class:`flask.Flask` application.
    """
    app = Flask(__name__)
    app.logger.handlers.clear()
    for handler in logger.handlers:
        app.logger.addHandler(handler)
    app.logger.setLevel(logger.level)

    if json_logging._current_framework is None:
        json_logging.init_flask(enable_json=True)

    return app


def detect_to_feature(
    fixed_object_bbox: List[float],
    fixed_object_mask: Optional[List[List[float]]] = None,
    detection_score: Optional[float] = 1.0,
    detection_type: Optional[str] = "sample_object",
) -> Dict[str, Union[str, list]]:
    """Build one GeoJSON-like feature dict for a detection in image coordinates.

    World coordinates are not available in this serving path; a placeholder
    point ``(0, 0)`` is used when no mask polygon is supplied.

    Args:
        fixed_object_bbox: Bounding box in image space (format expected by downstream consumers).
        fixed_object_mask: Optional closed polygon ring ``[[x, y], ...]`` for the instance mask.
        detection_score: Confidence score stored in feature properties.
        detection_type: Label / class string stored in feature properties.

    Returns:
        A dict with ``type``, ``id``, ``geometry``, and ``properties`` suitable for serialization.
    """
    feature = {
        "type": "Feature",
        "geometry": None,
        "id": token_hex(16),
        "properties": {
            "imageGeometry": {"type": "Point", "coordinates": [0.0, 0.0]},
            "imageBBox": fixed_object_bbox,
            "featureClasses": [{"iri": detection_type, "score": detection_score}],
            "modelMetadata": {
                "modelName": "centerpoint",
                "ontologyName": "centerpoint",
                "ontologyVersion": "1.0.0",
            },
            "image_id": token_hex(16),
        },
    }

    if fixed_object_mask is not None:
        feature["properties"]["imageGeometry"] = {
            "type": "Polygon",
            "coordinates": [fixed_object_mask],
        }

    return feature


def parse_custom_attributes() -> Dict[str, str]:
    """Parse ``X-Amzn-SageMaker-Custom-Attributes`` from the current Flask request.

    Expects comma-separated ``key=value`` pairs. Values are URL-decoded
    (e.g. ``text_prompt=sport%20cars`` → ``sport cars``).

    Returns:
        Mapping of attribute names to decoded values, or an empty dict if the header
        is missing or cannot be parsed.

    Note:
        Must be called inside a Flask request context.

        For LAM ``app.py`` ``/invocations``, the same keys can appear here as in a JSON
        body (later source wins when both are set). Tiling and input routing accept
        aliases such as ``lam_tile_size``, ``lam_tile_overlap``, ``lam_tile_dst_crs``,
        ``lam_tile_merge_iou``, ``prompt``, ``lam_s3_uri``, and hyphenated spellings (normalized
        to underscores). See ``_resolve_invocation_params`` in ``app.py``.
    """
    custom_attributes = request.headers.get("X-Amzn-SageMaker-Custom-Attributes", "")

    if not custom_attributes:
        return {}

    attributes: Dict[str, str] = {}
    try:
        for pair in custom_attributes.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                decoded_value = unquote(value.strip())
                attributes[key.strip()] = decoded_value
    except Exception:
        return {}

    return attributes


def simulate_model_latency() -> None:
    """Optionally sleep to emulate inference latency (testing / demos).

    Reads ``mock_latency_mean`` and optional ``mock_latency_std`` (milliseconds) from
    parsed custom attributes (normal distribution). If ``mock_latency_std`` is omitted,
    it defaults to 10% of the mean. Does nothing if ``mock_latency_mean`` is absent.
    """
    attributes = parse_custom_attributes()
    if "mock_latency_mean" not in attributes:
        return

    try:
        mean_ms = float(attributes["mock_latency_mean"])
        if "mock_latency_std" in attributes:
            std_ms = float(attributes["mock_latency_std"])
        else:
            std_ms = mean_ms * 0.1
        latency_ms = max(0.0, random.gauss(mean_ms, std_ms))
        time.sleep(latency_ms / 1000.0)
    except (ValueError, TypeError):
        return
