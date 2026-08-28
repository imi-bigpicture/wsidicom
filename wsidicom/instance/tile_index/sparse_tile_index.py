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

from collections import defaultdict
from collections.abc import Sequence
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from wsidicom.errors import WsiDicomError, WsiDicomNotFoundError
from wsidicom.geometry import Point, Size
from wsidicom.instance.dataset import WsiDataset
from wsidicom.instance.per_frame_group_positions import PerFrameGroupPositions
from wsidicom.instance.tile_index.tile_index import TileIndex


class SparseTilePlane:
    """Hold frame indices for the tiles in a sparse tiled file. Empty (sparse)
    frames are represented by -1."""

    def __init__(self, tiled_size: Size):
        """Create a SparseTilePlane of specified size.

        Parameters
        ----------
        tiled_size: Size
            Size of the tiling
        """
        self._shape = tiled_size
        self.plane = np.full(tiled_size.to_tuple(), -1, dtype=np.dtype(int))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._shape})"

    def __str__(self) -> str:
        return self.pretty_str()

    def __getitem__(self, position: Point) -> int:
        """Get frame index from tile index at position.

        Parameters
        ----------
        position: Point
            Position in plane to get the frame index from

        Returns
        -------
        int
            Frame index
        """
        frame_index = int(self.plane[position.x, position.y])
        return frame_index

    def add_frames(
        self,
        columns: NDArray[np.int64],
        rows: NDArray[np.int64],
        frame_indices: NDArray[np.int64],
    ):
        """Add the frame index of many tiles at once.

        Parameters
        ----------
        columns: NDArray[np.int64]
            Column in the plane of each frame.
        rows: NDArray[np.int64]
            Row in the plane of each frame.
        frame_indices: NDArray[np.int64]
            Frame index to add at each of those positions.
        """
        self.plane[columns, rows] = frame_indices

    def pretty_str(self, indent: int = 0, depth: int | None = None) -> str:
        return "Sparse tile plane"


class SparseTileIndex(TileIndex):
    """Index for mapping tile position to frame number for datasets containing
    sparse tiles. Frame indices are retrieved from tile position, z, and path
    by finding the corresponding matching SparseTilePlane (z and path) and
    returning the frame index at tile position. If the tile is missing (due to
    the sparseness), -1 is returned."""

    def __init__(self, datasets: Sequence[WsiDataset]):
        """Create sparse tile index for frames in datasets. Requires equal tile
        size for all tile planes. Pixel data tiles are identified by the Per
        Frame Functional Groups Sequence that contains tile column, row, z,
        path, and frame index. These are stored in a SparseTilePlane
        (one plane for every combination of z and path).

        Parameters
        ----------
        datasets: Sequence[Dataset]
            List of datasets containing sparse tiled image data.
        """
        super().__init__(datasets)

    @cached_property
    def planes(self) -> dict[tuple[float, str], SparseTilePlane]:
        return self._read_planes_from_datasets()

    @property
    def focal_planes(self) -> list[float]:
        return self._focal_planes

    @cached_property
    def _focal_planes(self) -> list[float]:
        return self._get_focal_planes()

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(self, indent: int = 0, depth: int | None = None) -> str:
        return (
            f"Sparse tile index tile size: {self.tile_size}, "
            f"plane size: {self.tiled_size}"
        )

    def get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Return frame index for a Point tile, z coordinate, and optical
        path.

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
        try:
            plane = self.planes[(z, path)]
        except KeyError:
            raise WsiDicomNotFoundError(
                f"Plane with z {z}, path {path}", str(self)
            ) from None
        frame_index = plane[tile]
        return frame_index

    def _get_focal_planes(self) -> list[float]:
        """Return list of focal planes defined in planes.

        Returns
        -------
        list[float]
            Focal planes, specified in um.
        """
        focal_planes: set[float] = set()
        for z, _ in self.planes:
            focal_planes.add(z)
        return sorted(list(focal_planes))

    def _read_planes_from_datasets(self) -> dict[tuple[float, str], SparseTilePlane]:
        """Return SparseTilePlane from planes in datasets.

        Returns
        -------
        dict[tuple[float, str], SparseTilePlane]
            Dict of planes with focal plane and optical identifier as key.
        """
        planes: dict[tuple[float, str], SparseTilePlane] = defaultdict(
            lambda: SparseTilePlane(self.tiled_size)
        )
        tile_size = self.tile_size
        for dataset in self._datasets:
            positions = dataset.frame_positions
            frame_count = len(positions.columns)
            frame_indices = np.arange(frame_count) + dataset.frame_offset
            tile_columns = (positions.columns - 1) // tile_size.width
            tile_rows = (positions.rows - 1) // tile_size.height
            self._check_within_tiling(tile_columns, tile_rows, positions)
            z_offsets = self._plane_z_offsets(
                positions.z_offsets, frame_count, dataset.read_z_offset()
            )
            optical_identifiers = self._plane_identifiers(
                positions.optical_path_identifiers,
                frame_count,
                dataset.read_optical_path_identifier(),
            )
            unique_optical_identifiers = np.unique(optical_identifiers)
            for z in np.unique(z_offsets):
                for optical_identifier in unique_optical_identifiers:
                    # Comparing an array to a value compares every element of it, and
                    # & is likewise elementwise, so this is one bool per frame. Numpy
                    # types both as Any, hence the annotation.
                    frames_in_plane: NDArray[np.bool_] = (z_offsets == z) & (
                        optical_identifiers == optical_identifier
                    )
                    if not frames_in_plane.any():
                        continue
                    # Every array is one value per frame, so the same frames are taken
                    # out of each.
                    planes[(float(z), str(optical_identifier))].add_frames(
                        tile_columns[frames_in_plane],
                        tile_rows[frames_in_plane],
                        frame_indices[frames_in_plane],
                    )
        return planes

    def _check_within_tiling(
        self,
        tile_columns: NDArray[np.int64],
        tile_rows: NDArray[np.int64],
        positions: PerFrameGroupPositions,
    ) -> None:
        """Raise if a frame states a position that is not a tile of this image.

        Placing a frame above the tiling raises an IndexError that says nothing about
        the file it came from. Placing one below it raises nothing at all, as a
        negative index counts from the far end, and the frame is quietly put somewhere
        it does not belong. Both are the file stating a position the image has no room
        for, so both are refused here rather than left to numpy.

        Parameters
        ----------
        tile_columns: NDArray[np.int64]
            Column in the tiling of each frame.
        tile_rows: NDArray[np.int64]
            Row in the tiling of each frame.
        positions: PerFrameGroupPositions
            Positions the frames state, to say in the message what one of them said.

        Raises
        ------
        WsiDicomError
            If any frame is outside the tiling.
        """
        outside = (
            (tile_columns < 0)
            | (tile_columns >= self.tiled_size.width)
            | (tile_rows < 0)
            | (tile_rows >= self.tiled_size.height)
        )
        if not outside.any():
            return
        frame = int(outside.argmax())
        raise WsiDicomError(
            f"{int(outside.sum())} of {len(outside)} frames state a position the image "
            f"has no tile for. Frame {frame} states "
            f"({positions.columns[frame]}, {positions.rows[frame]}), which is tile "
            f"({tile_columns[frame]}, {tile_rows[frame]}) of a tiling that is "
            f"{self.tiled_size.width} by {self.tiled_size.height} tiles."
        )

    def _plane_z_offsets(
        self, z_offsets: NDArray[np.float64] | None, frame_count: int, default: float
    ) -> NDArray[np.float64]:
        """Return the z offset that puts each frame in a plane.

        Parameters
        ----------
        z_offsets: NDArray[np.float64] | None
            Z offset of each frame, or None if the frames do not state one.
        frame_count: int
            Number of frames.
        default: float
            Z offset of the frames of a dataset that does not state one per frame.

        Returns
        -------
        NDArray[np.float64]
            Z offset of each frame, rounded to the precision planes are keyed by.
        """
        if z_offsets is None:
            return np.full(frame_count, default)
        return np.round(z_offsets, self.Z_DECIMALS)

    @staticmethod
    def _plane_identifiers(
        identifiers: NDArray[np.str_] | None, frame_count: int, default: str
    ) -> NDArray[np.str_]:
        """Return the optical path identifier that puts each frame in a plane.

        Parameters
        ----------
        identifiers: NDArray[np.str_] | None
            Optical path identifier of each frame, or None if the frames do not state
            one.
        frame_count: int
            Number of frames.
        default: str
            Identifier of the frames of a dataset that does not state one per frame.

        Returns
        -------
        NDArray[np.str_]
            Optical path identifier of each frame.
        """
        if identifiers is None:
            return np.full(frame_count, default)
        return identifiers
