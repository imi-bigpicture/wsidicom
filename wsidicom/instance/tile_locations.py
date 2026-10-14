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

"""Where the encoded tiles of an image are stored."""

from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray

from wsidicom.geometry import Size


class FrameReader(Protocol):
    """Reads frames stored at positions of a file, ready to serve. Is thread safe.

    A reader may do more than read the bytes stored, e.g. put the jpeg tables of a
    tiff page in place of the header of an abbreviated jpeg frame."""

    def read(self, position: int, length: int, /) -> bytes:
        """Return the frame of `length` bytes stored at `position`.

        Parameters
        ----------
        position: int
            Position the frame is stored at.
        length: int
            Number of bytes the frame is stored as.

        Returns
        -------
        bytes
            The frame, ready to serve.
        """
        ...


@dataclass(frozen=True, slots=True)
class TileLocations:
    """Where the encoded tiles of a plane (focal plane and optical path) of an image
    are stored, for reading a tile with a single read instead of through the image
    data.

    The encoded tile at (x, y) is stored as the `lengths[i]` bytes at
    `positions[i]` of file `file_indices[i]` of the image data (or its only file if
    there are no file indices), where `i = y * tiled_size.width + x`, and is read
    with the frame reader of that file (see `ImageData.get_frame_readers()`). A
    tile whose length is `READ_THROUGH_IMAGE_DATA` is not stored as a frame, and is
    to be read through the image data.

    Positions, lengths and file indices are read-only memoryviews of int64 arrays
    rather than the arrays themselves, as an item of a memoryview is a plain int got
    several times faster than one of an array. They are not lists, which would cost
    about as little to index but take four to five times the memory, and copy
    arrays that can instead be shared with the image data that made them.
    """

    READ_THROUGH_IMAGE_DATA: ClassVar[int] = -1
    """Length of a tile that is not stored as a frame."""

    tiled_size: Size
    """Number of tiles in each direction."""
    whole_tiled_size: Size
    """Number of tiles in each direction that lie wholly within the image, and so
    are not cropped to the image boundary."""
    positions: memoryview
    """Position of the bytes of each tile, in row-major order."""
    lengths: memoryview
    """Number of bytes of each tile, or `READ_THROUGH_IMAGE_DATA`."""
    file_indices: memoryview | None = None
    """Index of the file of each tile, or None if the image data has one file."""

    def __post_init__(self) -> None:
        tile_count = self.tiled_size.area
        if len(self.positions) != tile_count or len(self.lengths) != tile_count:
            raise ValueError(
                f"Got {len(self.positions)} positions and {len(self.lengths)} "
                f"lengths, expected one of each for the {tile_count} tiles."
            )
        if self.file_indices is not None and len(self.file_indices) != tile_count:
            raise ValueError(
                f"Got {len(self.file_indices)} file indices, expected one for each "
                f"of the {tile_count} tiles."
            )

    @classmethod
    def from_arrays(
        cls,
        image_size: Size,
        tile_size: Size,
        positions: NDArray[np.int64],
        lengths: NDArray[np.int64],
        file_indices: NDArray[np.int64] | None = None,
    ) -> "TileLocations":
        """Create tile locations from arrays of the positions and lengths.

        The arrays are not copied if they already are contiguous int64 arrays, but
        held through read-only views, so they are not to be changed after.

        Parameters
        ----------
        image_size: Size
            Size of the image in pixels.
        tile_size: Size
            Size of the tiles in pixels.
        positions: NDArray[np.int64]
            Position of the bytes of each tile, in row-major order.
        lengths: NDArray[np.int64]
            Number of bytes of each tile, or `READ_THROUGH_IMAGE_DATA`.
        file_indices: NDArray[np.int64] | None = None
            Index of the file of each tile, if the image data has more than one.

        Returns
        -------
        TileLocations
            The tile locations.
        """
        return cls(
            image_size.ceil_div(tile_size),
            Size(
                image_size.width // tile_size.width,
                image_size.height // tile_size.height,
            ),
            cls._read_only_view(positions),
            cls._read_only_view(lengths),
            cls._read_only_view(file_indices) if file_indices is not None else None,
        )

    @staticmethod
    def _read_only_view(values: NDArray[np.int64]) -> memoryview:
        """Return a read-only memoryview of the values as contiguous int64, sharing
        their memory when they already are, without changing the array given."""
        view = np.ascontiguousarray(values, dtype=np.int64).view()
        view.flags.writeable = False
        return memoryview(view)
