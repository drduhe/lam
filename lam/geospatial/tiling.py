"""Fixed-step tiling over raster width/height in pixel space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from rasterio.windows import Window


@dataclass(frozen=True)
class TileSpec:
    """One tile: top-left offset and size within the full raster."""

    col_off: int
    row_off: int
    width: int
    height: int

    @property
    def window(self) -> Window:
        """Rasterio window for this tile."""
        return Window(self.col_off, self.row_off, self.width, self.height)


def iter_tiles(width: int, height: int, tile_size: int, overlap: int) -> Iterator[TileSpec]:
    """Yield tile specs covering ``[0, width) x [0, height)`` with a fixed step.

    Args:
        width: Raster width in pixels.
        height: Raster height in pixels.
        tile_size: Target window size (clipped at edges).
        overlap: Overlap in pixels; step is ``max(1, tile_size - overlap)``.

    Yields:
        :class:`TileSpec` instances in row-major order.

    Raises:
        ValueError: If ``tile_size`` or ``overlap`` are invalid.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be in [0, tile_size)")
    step = max(1, tile_size - overlap)
    row = 0
    while row < height:
        col = 0
        while col < width:
            w = min(tile_size, width - col)
            h = min(tile_size, height - row)
            if w > 0 and h > 0:
                yield TileSpec(col, row, w, h)
            col += step
        row += step
