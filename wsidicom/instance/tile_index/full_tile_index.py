#    Copyright 2021, 2022, 2023 SECTRA AB
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

from collections.abc import Sequence
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.geometry import Point
from wsidicom.instance.dataset import WsiDataset
from wsidicom.instance.tile_index.tile_index import TileIndex


class FullTileIndex(TileIndex):
    """Index for mapping tile position to frame number for datasets containing
    full tiles. Pixel data tiles are ordered by column, row, z and path, thus
    the frame index for a tile can directly be calculated."""

    def __init__(self, datasets: Sequence[WsiDataset]):
        """Create full tile index for frames in datasets. Requires equal tile
        size for all tile planes.

        Parameters
        ----------
        datasets: Sequence[Dataset]
            List of datasets containing full tiled image data.
        """
        super().__init__(datasets)

    @property
    def focal_planes(self) -> list[float]:
        return self._focal_planes

    @cached_property
    def _focal_planes(self) -> list[float]:
        return self._read_focal_planes_from_datasets()

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(self, indent: int = 0, depth: int | None = None) -> str:
        string = (
            f"Full tile index tile size: {self.tile_size}"
            f", plane size: {self.tiled_size}"
        )
        if depth is not None:
            depth -= 1
            if depth < 0:
                return string
        string += f" of z: {self.focal_planes} and path: {self.optical_paths}"

        return string

    def get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Return frame index for a Point tile, z coordinate, and optical path
        from full tile index. Assumes that tile, z, and path are valid.

        Parameters
        ----------
        tile: Point
            Tile xy to get.
        z: float
            Z coordinate to get.
        path: str
            ID of optical path to get.

        Returns
        -------
        int
            Frame index.
        """
        plane_offset = tile.x + self.tiled_size.width * tile.y
        z_offset = self._get_focal_plane_index(z) * self.tiled_size.area
        path_offset = (
            self._get_optical_path_index(path)
            * len(self.focal_planes)
            * self.tiled_size.area
        )
        return plane_offset + z_offset + path_offset

    def get_plane_frame_indices(self, z: float, path: str) -> NDArray[np.int64]:
        """Return the frame index of every tile of a plane, which in a full tile
        index are consecutive from the frame of the first tile."""
        first = self.get_frame_index(Point(0, 0), z, path)
        return np.arange(first, first + self.tiled_size.area, dtype=np.int64)

    def _read_focal_planes_from_datasets(
        self,
    ) -> list[float]:
        """Return list of focal planes in datasets. Values in Pixel Measures
        Sequence are in mm.

        Returns
        -------
        list[float]
            Focal planes, specified in um.

        """
        MM_TO_MICRON = 1000.0
        focal_planes: set[float] = set()
        for dataset in self._datasets:
            slice_spacing = dataset.spacing_between_slices
            number_of_focal_planes = dataset.number_of_focal_planes
            if slice_spacing is None:
                if number_of_focal_planes == 1:
                    slice_spacing = 0.0
                else:
                    raise ValueError(
                        "Slice spacing must be known if multiple focal planes."
                    )
            elif slice_spacing == 0 and number_of_focal_planes != 1:
                raise ValueError(
                    "Slice spacing must be non-zero if multiple focal planes."
                )

            for plane in range(number_of_focal_planes):
                z = round(
                    dataset.z_offset + plane * slice_spacing * MM_TO_MICRON,
                    self.Z_DECIMALS,
                )
                focal_planes.add(z)
        return sorted(focal_planes)

    @cached_property
    def _optical_path_indices(self) -> dict[str, int]:
        """Index of each optical path identifier in the optical path sequence.

        A dict rather than a cached method, as it is looked up for every tile read
        and a dict lookup costs a fraction of a cached method call."""
        indices: dict[str, int] = {}
        for index, path in enumerate(self._optical_paths):
            indices.setdefault(path, index)
        return indices

    @cached_property
    def _focal_plane_indices(self) -> dict[float, int]:
        """Index of each focal plane (in um), see `_optical_path_indices`."""
        indices: dict[float, int] = {}
        for index, plane in enumerate(self.focal_planes):
            indices.setdefault(plane, index)
        return indices

    def _get_optical_path_index(self, path: str) -> int:
        """Return index of the optical path in instance.
        This assumes that all files in a concatenated set contains all the
        optical path identifiers of the set.

        Parameters
        ----------
        path: str
            Optical path identifier to search for.

        Returns
        -------
        int
            The index of the optical path identifier in the optical path
            sequence.
        """
        try:
            return self._optical_path_indices[path]
        except KeyError:
            raise WsiDicomNotFoundError(f"Optical path {path}", str(self)) from None

    def _get_focal_plane_index(self, z: float) -> int:
        """Return index of the focal plane of z.

        Parameters
        ----------
        z: float
            The z coordinate (in um) to search for.

        Returns
        -------
        int
            Focal plane index for z coordinate.
        """
        try:
            return self._focal_plane_indices[z]
        except KeyError:
            raise WsiDicomNotFoundError(f"Z {z} in instance", str(self)) from None
