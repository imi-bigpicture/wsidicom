from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

def compose(
    canvas: NDArray[Any],
    tiles: Sequence[NDArray[Any]],
    positions: NDArray[np.float64],
    filter: str = "bilinear",
    pad: int = 255,
    crops: NDArray[np.int64] | None = None,
) -> None:
    """Low-level binding; see :func:`wsidicom_kernels.compose`."""

def downscale(
    image: NDArray[Any], factor: int, filter: str = "box", pad: int | None = None
) -> NDArray[Any]:
    """Low-level binding; see :func:`wsidicom_kernels.downscale`."""

def resize(
    image: NDArray[Any],
    out_h: int,
    out_w: int,
    filter: str = "box",
    pad: int | None = None,
) -> NDArray[Any]:
    """Low-level binding; see :func:`wsidicom_kernels.resize`."""

def downscale_grid_box(
    tiles: Sequence[NDArray[Any]],
    grid_width: int,
    factor: int,
    size: tuple[int, int] | None = None,
    fill: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Low-level binding; see :func:`wsidicom_kernels.downscale_grid`."""

def framed(
    image: NDArray[Any], size: tuple[int, int], fill: NDArray[Any]
) -> NDArray[Any]:
    """Low-level binding; see :func:`wsidicom_kernels.downscale_grid`."""

def stitch_grid(tiles: Sequence[NDArray[Any]], grid_width: int) -> NDArray[Any]:
    """Low-level binding; see :func:`wsidicom_kernels.downscale_grid`."""

def rgb_to_rgbx(rgb: NDArray[Any], pad: int = 0) -> NDArray[np.uint8]:
    """Low-level binding; see :func:`wsidicom_kernels.rgb_to_rgbx`."""

def set_simd(enabled: bool) -> None:
    """Test/benchmark hook: force scalar kernels (False) or restore SIMD (True)."""
