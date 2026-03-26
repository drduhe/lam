"""Sanitize LLM / SAM text prompts for use in filesystem paths."""

from __future__ import annotations

import hashlib
import re


def sanitize_filename(text_prompt: str, max_length: int = 200) -> str:
    """
    Turn a free-text prompt into a short, filename-safe stem with a hash suffix.

    Replaces invalid path characters, truncates to ``max_length`` (including the
    hash suffix), and appends an 8-character MD5 prefix so colliding truncations
    stay distinct.

    Args:
        text_prompt: Raw prompt text.
        max_length: Maximum total length of the returned stem (default 200).

    Returns:
        Sanitized string safe to embed in a filename (no extension).
    """
    prompt_hash = hashlib.md5(text_prompt.encode("utf-8")).hexdigest()[:8]
    hash_suffix = f"_{prompt_hash}"

    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', "_", text_prompt).strip("_.")
    sanitized = sanitized or "prompt"

    available_length = max_length - len(hash_suffix)
    if len(sanitized) > available_length:
        sanitized = sanitized[: max(0, available_length)]

    return f"{sanitized}{hash_suffix}"
