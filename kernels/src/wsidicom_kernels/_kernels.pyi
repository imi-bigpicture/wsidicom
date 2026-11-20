from collections.abc import Sequence
from typing import Any

from numpy.typing import NDArray

def downscale(image: NDArray[Any], factor: int, resample: str) -> NDArray[Any]:
    """Low-level binding; see :func:`wsidicom_kernels.downscale`."""

def resize(
    image: NDArray[Any], height: int, width: int, resample: str
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

def set_simd(enabled: bool) -> None:
    """Test/benchmark hook: force scalar kernels (False) or restore SIMD (True)."""
