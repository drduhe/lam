"""Optional fast paths for NMS, connected components, masks, and ``torch.compile`` helpers."""

from __future__ import annotations

import os

is_enabled = False
if os.getenv("USE_PERFLIB", "1") == "1":
    is_enabled = True
