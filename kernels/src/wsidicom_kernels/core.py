#    Copyright 2026 SECTRA AB
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Implementation of the wsidicom_kernels public API (see the package docstring)."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

import wsidicom_kernels._kernels as lib
from wsidicom_kernels.models import ResizeFilter


def compose(
    canvas: NDArray[Any],
    tiles: Sequence[NDArray[Any]],
    positions: NDArray[np.int64],
    crops: NDArray[np.int64] | None = None,
) -> NDArray[Any]:
    """Paste ``tiles`` onto ``canvas`` in place; overlaps resolve last-write-wins.

    Each tile is copied into ``canvas`` with its top-left at its integer
    ``(x, y)`` position, row by row. Tiles may hang off the canvas on any side;
    only the pixels inside it are written.

    With ``crops``, each placed piece is a rectangle of one of ``tiles``, so
    a tile that packs several pieces is passed once, decoded, and never
    copied apart.

    Images are planar ``(H, W)`` or interleaved HWC ``(H, W, C)`` with 1 or 3
    channels. Subsampled chroma (differently sized planes) cannot be
    interleaved, so compose those per plane.

    Parameters
    ----------
    canvas: numpy.ndarray
        The destination, dtype uint8 / uint16, C-contiguous. Planar
        ``(H, W)`` or interleaved HWC ``(H, W, C)``, C = 1 or 3. Mutated in
        place.
    tiles: Sequence[numpy.ndarray]
        The source tiles, same dtype and channel count as ``canvas``
        (individual sizes may differ). Each row must be contiguous: a
        C-contiguous array, or a slice of rows and columns of one
        (``tile[y0:y1, x0:x1]``), read without a copy.
    positions: numpy.ndarray
        C-contiguous ``(N, 2)`` int64 top-left ``(x, y)`` = ``(column, row)``
        positions in canvas pixels, one per tile (per crop, with ``crops``).
    crops: numpy.ndarray | None = None
        ``(N, 5)`` int64 ``(tile, x, y, width, height)`` rows: the pieces to
        place, each a nonempty rectangle of ``tiles[tile]`` with its
        top-left at ``(x, y)`` in that tile. ``None`` places each tile whole.
        Pieces overlap in their order, as tiles do.

    Returns
    -------
    numpy.ndarray
        ``canvas``, mutated in place and returned for convenience.

    Notes
    -----
    Performance: each call carries a few microseconds of fixed cost (borrow
    setup). For tile loops, pass all tiles in ONE call; batching amortizes the
    fixed cost to nothing.
    Reuse large canvases across calls where possible: first writes to a
    freshly allocated array page-fault, which can cost more than the
    compose itself (see the README's performance notes).
    """
    lib.compose(canvas, list(tiles), positions, crops)
    return canvas


def downscale(
    image: NDArray[Any],
    factor: int,
    filter: ResizeFilter,
) -> NDArray[Any]:
    """Reduce ``image`` by integer ``factor`` per spatial axis, antialiased.

    Downscaling must low-pass filter before decimating or high frequencies
    alias; every filter here does so except ``NEAREST`` (which subsamples and
    can alias). The kernel support is scaled to the reduction ratio — unlike
    ``cv2.resize``, where only ``INTER_AREA`` antialiases and
    ``INTER_LINEAR``/``CUBIC``/``LANCZOS4`` are fixed-kernel interpolators
    that alias when minifying. A power-of-2 ``factor`` with ``BOX`` uses
    native SIMD box kernels; other filters and factors resample natively for
    uint8 and uint16 (within 1 LSB of the exact convolution at any factor;
    identical on every CPU; uint8 ``BILINEAR`` and ``CATMULLROM`` at
    ``factor`` 2 and ``BILINEAR`` at ``factor`` 4 exact where the factor
    divides the size). ``NEAREST`` is
    native too. ``BOX`` is fastest and mipmap-standard; ``LANCZOS3`` is
    sharpest, and the windowed filters (``CATMULLROM``/``LANCZOS3``) also
    suppress the low-frequency beat that pure area averaging leaves on
    strongly periodic content (gratings, halftones).

    The scale is exactly ``factor``, and output dimensions are
    ``ceil(dim / factor)``, as Pillow's ``reduce`` and libvips' ``shrink``:
    where the factor doesn't divide the size, the last row and column cover
    what remains of the image (with ``BOX``, the mean of the pixels there),
    rather than the image being stretched to whole blocks or cropped.

    Images are planar ``(H, W)`` or interleaved HWC ``(H, W, C)`` with 1 or 3
    channels, each filtered independently. For subsampled chroma, downscale
    each plane separately.

    Parameters
    ----------
    image: numpy.ndarray
        The source, C-contiguous, dtype uint8 / uint16. Planar
        ``(H, W)`` or interleaved HWC ``(H, W, C)``, C = 1 or 3.
    factor: int
        Integer reduction factor applied to both spatial axes.
    filter: ResizeFilter
        Antialiasing filter.

    Returns
    -------
    numpy.ndarray
        A new, reduced array with the same dtype and channel count as ``image``.
    """
    return lib.downscale(image, factor, filter)


def downscale_grid(
    tiles: Sequence[NDArray[Any]],
    grid_width: int,
    factor: int,
    filter: ResizeFilter,
    size: tuple[int, int] | None = None,
    fill: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Reduce a grid of tiles to one image: ``downscale`` of the stitched grid.

    The step that builds a pyramid level from the one below: e.g. the four
    children of a tile, ``[top_left, top_right, bottom_left, bottom_right]``
    with ``grid_width=2``, reduced by 2 into their parent. The result equals
    ``downscale`` of the tiles stitched row-major, without building the
    stitched image: with ``BOX`` and a power-of-2 ``factor``, when every tile
    but those of the last grid row and column divides by ``factor`` (the box
    windows then never cross a seam), each tile is reduced on its own straight
    into its place in the output. Other filters, whose windows reach across
    the seams, stitch first.

    Parameters
    ----------
    tiles: Sequence[numpy.ndarray]
        Row-major C-contiguous tiles, ``grid_width`` per grid row, all of one
        dtype and channel count; the tiles of a grid row share their height
        and those of a grid column their width (so edge tiles may be smaller).
    grid_width: int
        Tiles per grid row; ``len(tiles)`` must be a multiple of it.
    factor: int
        Integer reduction factor applied to both spatial axes.
    filter: ResizeFilter
        Antialiasing filter.
    size: tuple[int, int] | None = None
        If set, the ``(height, width)`` of the result, at least the reduced
        grid's: the reduced grid at its top left, ``fill`` around it. A parent
        tile from a partial block at the image's edge then keeps the full tile
        size.
    fill: numpy.ndarray | None = None
        The pixel around the reduced grid, required with ``size``: a ``(C,)``
        array in the tiles' dtype, one sample per channel.

    Returns
    -------
    numpy.ndarray
        The reduced grid, ``ceil(sum of tile heights / factor)`` by
        ``ceil(sum of tile widths / factor)`` (or ``size``), with the tiles'
        dtype and channels.
    """
    tile_list = list(tiles)
    if not tile_list or grid_width < 1 or len(tile_list) % grid_width:
        raise ValueError("tiles must fill whole grid rows of grid_width tiles")
    if (size is None) != (fill is None):
        raise ValueError("size and fill go together")
    if factor < 1:
        raise ValueError("factor must be >= 1")
    last_col = grid_width - 1
    inner_rows = tile_list[: len(tile_list) - grid_width]
    inner_cols = [
        tile for index, tile in enumerate(tile_list) if index % grid_width != last_col
    ]
    seams_divide = all(tile.shape[0] % factor == 0 for tile in inner_rows) and all(
        tile.shape[1] % factor == 0 for tile in inner_cols
    )
    power_of_two = factor > 0 and factor & (factor - 1) == 0
    if filter == ResizeFilter.BOX and power_of_two and seams_divide:
        return lib.downscale_grid_box(tile_list, grid_width, factor, size, fill)
    stitched = lib.stitch_grid(tile_list, grid_width)
    reduced = downscale(stitched, factor, filter)
    if size is None or fill is None:
        return reduced
    return lib.framed(reduced, size, fill)


def resize(
    image: NDArray[Any],
    out_h: int,
    out_w: int,
    filter: ResizeFilter,
) -> NDArray[Any]:
    """Resample ``image`` to an exact ``(out_h, out_w)`` — any ratio, antialiased.

    Unlike :func:`downscale`, the ratio is not restricted to an integer factor:
    any target size works, including fractional reductions (e.g. resampling to a
    requested resolution) and upscaling. The filter kernel is scaled to
    the reduction ratio, so every filter here antialiases at every ratio — there
    is no aliasing fast path to fall into.

    An exact power-of-2 reduction with ``BOX`` (same factor on both axes) takes
    the same native SIMD kernels as :func:`downscale`; other resizes resample
    natively for uint8 and uint16 (within 1 LSB of the exact convolution at
    any ratio; identical on every CPU;
    uint8 ``BILINEAR`` and ``CATMULLROM`` to exactly half the size and
    ``BILINEAR`` to exactly a quarter or twice the size exact). ``NEAREST``
    is native too.

    ``BOX`` is the cheapest antialiased choice and the right default for
    downscaling; for *upscaling* it degenerates to blocky nearest-like output, so
    pass ``BILINEAR`` / ``CATMULLROM`` / ``LANCZOS3`` when enlarging.

    Images are planar ``(H, W)`` or interleaved HWC ``(H, W, C)`` with 1 or 3
    channels, each filtered independently. For subsampled chroma, resize
    each plane separately.

    Parameters
    ----------
    image: numpy.ndarray
        The source, C-contiguous, dtype uint8 / uint16. Planar
        ``(H, W)`` or interleaved HWC ``(H, W, C)``, C = 1 or 3.
    out_h: int
        Target height in pixels.
    out_w: int
        Target width in pixels.
    filter: ResizeFilter
        Resampling filter.

    Returns
    -------
    numpy.ndarray
        A new ``(out_h, out_w)`` (or ``(out_h, out_w, C)``) array, same dtype
        and channel count as ``image``.
    """
    return lib.resize(image, out_h, out_w, filter)
