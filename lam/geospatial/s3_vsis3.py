"""Map ``s3://`` URIs to GDAL ``/vsis3/`` paths for range-based reads.

GDAL (and rasterio) open remote GeoTIFFs over HTTPS using byte-range requests,
so the full object is not downloaded up front. Credentials follow GDAL’s rules:
``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` / ``AWS_SESSION_TOKEN``, optional
``AWS_DEFAULT_REGION``, instance/role metadata on SageMaker or EC2, etc.

No boto3 is required. For public buckets you may need ``AWS_NO_SIGN_REQUEST=YES`` (GDAL
env). For requester-pays buckets set ``AWS_REQUEST_PAYER=requester``.
"""

from __future__ import annotations

import os
from typing import Tuple
from urllib.parse import quote


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``.

    Args:
        uri: S3 URI with a non-empty bucket and key (keys may contain ``/``).

    Returns:
        ``(bucket_name, object_key)``.

    Raises:
        ValueError: If ``uri`` is missing, not an ``s3://`` URI, or has no key.
    """
    u = (uri or "").strip()
    if not u.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {uri!r}")
    rest = u[5:]
    if "/" not in rest:
        raise ValueError(f"S3 URI must include a key after the bucket: {uri!r}")
    bucket, _, key = rest.partition("/")
    bucket, key = bucket.strip(), key.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3:// URI (empty bucket or key): {uri!r}")
    return bucket, key


def apply_gdal_s3_read_defaults() -> None:
    """Set GDAL env defaults once per process for efficient /vsis3/ raster opens.

    ``GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`` avoids unnecessary LIST operations
    on the bucket when opening a single object (typical for GeoTIFF/COG).
    """
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


def s3_uri_to_vsis3_path(uri: str) -> str:
    """Convert ``s3://bucket/key`` to GDAL virtual path ``/vsis3/bucket/key``.

    The object key is URL-encoded per segment-preserving rules (slashes kept) so
    spaces and other reserved characters work.

    Args:
        uri: ``s3://bucket/key``.

    Returns:
        Path suitable for :func:`rasterio.open` or :func:`osgeo.gdal.Open`.

    Raises:
        ValueError: If ``uri`` is malformed.
    """
    bucket, key = parse_s3_uri(uri)
    enc_bucket = quote(bucket, safe="")
    enc_key = quote(key, safe="/")
    return f"/vsis3/{enc_bucket}/{enc_key}"
