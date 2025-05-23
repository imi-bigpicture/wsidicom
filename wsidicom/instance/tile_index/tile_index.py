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

from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence, Set, Tuple

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence

from wsidicom.geometry import Point, Size
from wsidicom.instance.dataset import WsiDataset
from wsidicom.tags import (
    ColumnPositionInTotalImagePixelMatrixTag,
    OpticalPathIdentifierTag,
    PlanePositionSlideSequenceTag,
    RowPositionInTotalImagePixelMatrixTag,
    ZOffsetInSlideCoordinateSystemTag,
)


class TileIndex(metaclass=ABCMeta):
    """Index for mapping tile position to frame number. Is subclassed into
    FullTileIndex and SparseTileIndex."""

    def __init__(self, datasets: Sequence[WsiDataset]):
        """Create tile index for frames in datasets. Requires equal tile
        size for all tile planes.

        Parameters
        ----------
        datasets: Sequence[Dataset]
            List of datasets containing tiled image data.

        """
        self._datasets = datasets
        self._image_size = datasets[0].image_size
        self._tile_size = datasets[0].tile_size
        self._optical_paths = self._read_optical_paths_from_datasets(datasets)
        self._tiled_size = self.image_size.ceil_div(self.tile_size)

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} with image size {self.image_size}, "
            f"tile size {self.tile_size}, tiled size {self.tiled_size}, "
            f"optical paths {self.optical_paths}, "
            f"focal planes {self.focal_planes}"
        )

    @property
    @abstractmethod
    def focal_planes(self) -> List[float]:
        """Return list of focal planes in index."""
        raise NotImplementedError()

    @property
    def image_size(self) -> Size:
        """Return image size in pixels."""
        return self._image_size

    @property
    def tile_size(self) -> Size:
        """Return tile size in pixels."""
        return self._tile_size

    @property
    def tiled_size(self) -> Size:
        """Return size of tiling (columns x rows)."""
        return self._tiled_size

    @property
    def optical_paths(self) -> List[str]:
        """Return list of optical paths in index."""
        return self._optical_paths

    @abstractmethod
    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Abstract method for getting the frame index for a tile"""
        raise NotImplementedError()

    @staticmethod
    def _read_frame_count_from_datasets(datasets: Sequence[Dataset]) -> int:
        """Return total frame count from files.

        Parameters
        ----------
        datasets: Sequence[Dataset]
           List of datasets.

        Returns
        -------
        int
            Total frame count.

        """
        count = 0
        for dataset in datasets:
            count += dataset.frame_count
        return count

    @classmethod
    def _read_optical_paths_from_datasets(
        cls, datasets: Sequence[WsiDataset]
    ) -> List[str]:
        """Return list of optical path identifiers from files.

        Parameters
        ----------
        datasets: Sequence[Dataset]
           List of datasets.

        Returns
        -------
        List[str]
            Optical identifiers.

        """
        paths: Set[str] = set()
        for dataset in datasets:
            paths.update(cls._get_path_identifers(dataset.optical_path_sequence))
        if len(paths) == 0:
            return ["0"]
        return list(paths)

    @staticmethod
    def _get_path_identifers(
        optical_path_sequence: Optional[DicomSequence],
    ) -> List[str]:
        """Parse optical path sequence and return list of optical path
        identifiers

        Parameters
        ----------
        optical_path_sequence: DicomSequence
            Optical path sequence.

        Returns
        -------
        List[str]
            List of optical path identifiers.
        """
        if optical_path_sequence is None:
            return ["0"]
        return list(
            {
                str(optical_ds[OpticalPathIdentifierTag].value)
                for optical_ds in optical_path_sequence
            }
        )

    def _read_frame_coordinates(self, frame: Dataset) -> Tuple[Point, float]:
        """Return frame coordinate (Point(x, y) and float z) of the frame.
        In the Plane Position Slide Sequence x and y are defined in mm and z in
        um.

        Parameters
        ----------
        frame: Dataset
            Pydicom frame sequence.

        Returns
        -------
        Point, float
            The frame xy coordinate and z coordinate
        """
        DECIMALS = 3
        position: Dataset = frame[PlanePositionSlideSequenceTag][0]
        y = int(position[RowPositionInTotalImagePixelMatrixTag].value) - 1
        x = int(position[ColumnPositionInTotalImagePixelMatrixTag].value) - 1
        z_offset = position.get(ZOffsetInSlideCoordinateSystemTag, None)
        if z_offset is None:
            z = 0
        else:
            z = round(float(z_offset.value), DECIMALS)
        tile = Point(x=x, y=y) // self.tile_size
        return tile, z
