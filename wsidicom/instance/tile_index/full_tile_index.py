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

from functools import cached_property, lru_cache
from typing import List, Optional, Sequence, Set

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

    @cached_property
    def focal_planes(self) -> List[float]:
        return self._read_focal_planes_from_datasets()

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
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
        ----------
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

    def _read_focal_planes_from_datasets(
        self,
    ) -> List[float]:
        """Return list of focal planes in datasets. Values in Pixel Measures
        Sequence are in mm.

        Returns
        ----------
        List[float]
            Focal planes, specified in um.

        """
        MM_TO_MICRON = 1000.0
        DECIMALS = 3
        focal_planes: Set[float] = set()
        for dataset in self._datasets:
            slice_spacing = dataset.spacing_between_slices
            number_of_focal_planes = getattr(dataset, "TotalPixelMatrixFocalPlanes", 1)
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

            try:
                z_offset = (
                    dataset.SharedFunctionalGroupsSequence[0]
                    .PlanePositionSlideSequence[0]
                    .ZOffsetInSlideCoordinateSystem
                )
            except AttributeError:
                z_offset = 0

            for plane in range(number_of_focal_planes):
                z = z_offset + round(plane * slice_spacing * MM_TO_MICRON, DECIMALS)
                focal_planes.add(z)
        return sorted(list(focal_planes))

    @lru_cache
    def _get_optical_path_index(self, path: str) -> int:
        """Return index of the optical path in instance.
        This assumes that all files in a concatenated set contains all the
        optical path identifiers of the set.

        Parameters
        ----------
        path: str
            Optical path identifier to search for.

        Returns
        ----------
        int
            The index of the optical path identifier in the optical path
            sequence.
        """
        try:
            return next(
                (
                    index
                    for index, plane_path in enumerate(self._optical_paths)
                    if plane_path == path
                )
            )
        except StopIteration:
            raise WsiDicomNotFoundError(f"Optical path {path}", str(self))

    @lru_cache
    def _get_focal_plane_index(self, z: float) -> int:
        """Return index of the focal plane of z.

        Parameters
        ----------
        z: float
            The z coordinate (in um) to search for.

        Returns
        ----------
        int
            Focal plane index for z coordinate.
        """
        try:
            return next(
                index for index, plane in enumerate(self.focal_planes) if plane == z
            )
        except StopIteration:
            raise WsiDicomNotFoundError(f"Z {z} in instance", str(self))
