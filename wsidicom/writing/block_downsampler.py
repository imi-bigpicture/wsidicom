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

"""Downsampling of a block of tiles into one tile of the level above."""

from collections.abc import Sequence

import numpy as np

from wsidicom.downsampler import Downsampler
from wsidicom.geometry import Size
from wsidicom.stitcher import NumpyStitcher


class BlockDownsampler:
    """Downsamples a block of up to 2x2 tiles into one tile of the level above."""

    def __init__(
        self,
        downsampler: Downsampler,
        stitcher: NumpyStitcher,
        tile_size: Size,
        blank_tile: np.ndarray,
    ) -> None:
        """Create a block downsampler.

        Parameters
        ----------
        downsampler: Downsampler
            Downsampler that halves the tiles.
        stitcher: NumpyStitcher
            Stitcher used to combine the tiles of a block.
        tile_size: Size
            Size of the output tile.
        blank_tile: np.ndarray
            Background tile of the output tile size, used to pad a downsampled
            edge block back to a full tile. Treated as read-only.
        """
        self._downsampler = downsampler
        self._stitcher = stitcher
        self._tile_size = tile_size
        self._blank_tile = blank_tile

    def downsample(self, tiles: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
        """Downsample a block of tiles into one output tile.

        Chooses between reduce-each-then-stitch (fast path when the downsampler
        supports it) and stitch-then-halve (standard path). Both halve the block
        by exactly 2x; the result is then padded to the full output tile size,
        which matters at the right and bottom edges of a level where the input
        block is smaller than 2x2.

        Parameters
        ----------
        tiles: Sequence[Sequence[np.ndarray]]
            The tiles of the block, as a row-major 2D matrix of up to 2x2 tiles.

        Returns
        -------
        np.ndarray
            The output tile.
        """
        if self._downsampler.commutes_with_stitch:
            reduced = [
                [self._downsampler.reduce_by_half(tile) for tile in row]
                for row in tiles
            ]
            block = self._stitcher.stitch_grid(reduced)
        else:
            composite = self._stitcher.stitch_grid(tiles)
            # Halve the composite, rather than resizing it to the output tile
            # size: a partial edge block is smaller than 2x2, and resizing it to
            # a full tile would stretch it instead of downsampling it.
            half = Size(composite.shape[1] // 2, composite.shape[0] // 2)
            block = self._downsampler.downsample(composite, half)
        return self._pad_to_tile(block)

    def _pad_to_tile(self, block: np.ndarray) -> np.ndarray:
        """Pad a downsampled block to the full output tile size.

        At the right and bottom edges of a level the input block can be smaller
        than 2x2 (the level below has an odd number of tiles), so the halved
        block covers only part of the output tile. The remainder lies outside
        the image and is filled with the background color, matching how source
        edge tiles are padded.

        Parameters
        ----------
        block: np.ndarray
            The halved block, at most the output tile size.

        Returns
        -------
        np.ndarray
            The block padded to the full output tile size.
        """
        height, width = block.shape[0], block.shape[1]
        if (width, height) == (self._tile_size.width, self._tile_size.height):
            return block
        # blank_tile is a cached, shared array - copy before writing into it.
        canvas = self._blank_tile.copy()
        canvas[:height, :width] = block
        return canvas
