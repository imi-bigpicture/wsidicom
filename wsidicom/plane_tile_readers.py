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

"""Readers of encoded tiles at their byte ranges, one for each plane of a level."""

import threading
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass

from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.instance.tile_byte_ranges import FrameReader, TileByteRanges
from wsidicom.series import Pyramids

PlaneKey = tuple[int, int, float | None, str | None]
"""Pyramid index, level, and focal plane and optical path as asked for."""


@dataclass(frozen=True, slots=True)
class PlaneTileReader:
    """Reads the encoded tiles of a plane with a single read at their byte ranges."""

    byte_ranges: TileByteRanges
    """Where the encoded tiles of the plane are stored."""
    readers: Sequence[FrameReader]
    """Readers of the files of the image data, in the order of the file indices of the
    byte ranges."""

    def read(self, x: int, y: int, crop_to_image_boundary: bool) -> bytes | None:
        """Return the encoded tile at (x, y), read with a single read at its byte range.
        Returns None if the tile is to be read through the level instead.

        A tile is read through the level if it is outside the image (so that the level
        raises), if it is not stored as a single run of bytes, or if it is to be cropped
        to the image boundary (which requires re-encoding it).

        Parameters
        ----------
        x: int
            Column of the tile.
        y: int
            Row of the tile.
        crop_to_image_boundary: bool
            If a tile that extends outside the image is to be cropped.

        Returns
        -------
        bytes | None
            The encoded tile, or None.
        """
        byte_ranges = self.byte_ranges
        servable = (
            byte_ranges.whole_tiled_size
            if crop_to_image_boundary
            else byte_ranges.tiled_size
        )
        # Checked, because a position outside the tiled size would otherwise index the
        # byte range of another tile.
        if not (0 <= x < servable.width and 0 <= y < servable.height):
            return None
        index = y * byte_ranges.tiled_size.width + x
        length = byte_ranges.lengths[index]
        if length < 0:
            return None
        if byte_ranges.file_indices is None:
            reader = self.readers[0]
        else:
            reader = self.readers[byte_ranges.file_indices[index]]
        return reader.read(byte_ranges.positions[index], length)


class PlaneTileReaders:
    """Readers of the encoded tiles of the planes of the levels of pyramids.

    The reader of a plane is made when a tile of the plane is first read, and kept for
    later reads. A read of a plane whose reader is being made waits for it. Reads of
    other planes do not wait.
    """

    def __init__(self, pyramids: Pyramids):
        """
        Parameters
        ----------
        pyramids: Pyramids
            Pyramids whose tiles to read.
        """
        self._pyramids = pyramids
        self._readers: dict[PlaneKey, PlaneTileReader | None] = {}
        self._being_made: dict[PlaneKey, Future[PlaneTileReader | None]] = {}
        """Readers of planes that are being made, for reads of the same plane to wait
        for. The lock guards both dicts. It is not held while a reader is made, so
        reads of other planes do not wait."""
        self._lock = threading.Lock()

    def read(
        self,
        pyramid: int,
        level: int,
        tile: tuple[int, int],
        z: float | None,
        path: str | None,
        crop_to_image_boundary: bool,
    ) -> bytes | None:
        """Return the encoded tile, read with a single read at its byte range. Returns
        None if the tile is to be read through the level instead.

        Parameters
        ----------
        pyramid: int
            Index of pyramid.
        level: int
            Pyramid level.
        tile: tuple[int, int]
            Column and row of the tile.
        z: float | None
            Z coordinate as asked for, the default of the instance if None.
        path: str | None
            Optical path as asked for, the default of the instance if None.
        crop_to_image_boundary: bool
            If a tile that extends outside the image is to be cropped.

        Returns
        -------
        bytes | None
            The encoded tile, or None.
        """
        try:
            reader = self._readers[(pyramid, level, z, path)]
        except KeyError:
            reader = self._get_reader((pyramid, level, z, path))
        if reader is None:
            return None
        return reader.read(tile[0], tile[1], crop_to_image_boundary)

    def clear(self) -> None:
        """Forget the readers made."""
        with self._lock:
            self._readers.clear()

    def _get_reader(self, key: PlaneKey) -> PlaneTileReader | None:
        """Return the reader of a plane, made by this read or by another read of the
        plane at the same time.

        Parameters
        ----------
        key: PlaneKey
            Pyramid index, level, and focal plane and optical path as asked for.

        Returns
        -------
        PlaneTileReader | None
            Reader of the plane, or None if its tiles are not to be read so.
        """
        with self._lock:
            # Checked again with the lock held, so that reads of the plane at the same
            # time make the reader only once.
            if key in self._readers:
                return self._readers[key]
            being_made = self._being_made.get(key)
            makes_here = being_made is None
            if being_made is None:
                being_made = Future()
                self._being_made[key] = being_made
        if not makes_here:
            # Waits only for the read making the reader of this plane, not for those of
            # other planes.
            return being_made.result()
        try:
            reader = self._create_reader(*key)
        except BaseException as exception:
            # Left unmade, so that a later read tries again.
            with self._lock:
                del self._being_made[key]
            being_made.set_exception(exception)
            raise
        with self._lock:
            self._readers[key] = reader
            del self._being_made[key]
        being_made.set_result(reader)
        return reader

    def _create_reader(
        self, pyramid: int, level: int, z: float | None, path: str | None
    ) -> PlaneTileReader | None:
        """Make the reader of a plane.

        Returns None if a tile of the plane cannot be read with a single read, or if the
        level or plane is not in the pyramids. Reading a tile through the level then
        scales it or raises.

        Parameters
        ----------
        pyramid: int
            Index of pyramid.
        level: int
            Pyramid level.
        z: float | None
            Z coordinate as asked for, the default of the instance if None.
        path: str | None
            Optical path as asked for, the default of the instance if None.

        Returns
        -------
        PlaneTileReader | None
            Reader of the plane, or None.
        """
        try:
            instance = self._pyramids.get(pyramid).get(level).get_instance(z, path)
        except WsiDicomNotFoundError:
            return None
        image_data = instance.image_data
        # Readers first, so that no byte ranges are made for image data whose tiles
        # cannot be read with them.
        readers = image_data.get_frame_readers()
        if readers is None:
            return None
        byte_ranges = image_data.get_tile_byte_ranges(
            z if z is not None else instance.default_z,
            path if path is not None else instance.default_path,
        )
        if byte_ranges is None:
            return None
        return PlaneTileReader(byte_ranges, readers)
