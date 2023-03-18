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

import io
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Sequence, Set, Tuple, Union

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import UID
from wsidicom.dataset import TileType

from wsidicom.errors import WsiDicomNotFoundError, WsiDicomOutOfBoundsError
from wsidicom.file import WsiDicomFile
from wsidicom.geometry import Point, Region, Size, SizeMm
from wsidicom.image_data import ImageData, ImageOrigin


class WsiDicomImageData(ImageData):
    """Represents image data read from dicom file(s). Image data can
    be sparsly or fully tiled and/or concatenated."""

    def __init__(self, files: Union[WsiDicomFile, Sequence[WsiDicomFile]]) -> None:
        """Create WsiDicomImageData from frame data in files.

        Parameters
        ----------
        files: Union[WsiDicomFile, Sequence[WsiDicomFile]]
            Single or list of WsiDicomFiles containing frame data.
        """
        if not isinstance(files, Sequence):
            files = [files]

        # Key is frame offset
        self._files = OrderedDict(
            (file.frame_offset, file)
            for file in sorted(files, key=lambda file: file.frame_offset)
        )

        base_file = files[0]
        datasets = [file.dataset for file in self._files.values()]
        if base_file.dataset.tile_type == TileType.FULL:
            self.tiles = FullTileIndex(datasets)
        else:
            self.tiles = SparseTileIndex(datasets)

        self._pixel_spacing = datasets[0].pixel_spacing
        self._transfer_syntax = base_file.transfer_syntax
        self._default_z: Optional[float] = None
        self._photometric_interpretation = datasets[0].photometric_interpretation
        self._samples_per_pixel = datasets[0].samples_per_pixel
        self._image_origin = ImageOrigin.from_dataset(datasets[0])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._files.values()})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of files {self._files.values()}"

    @property
    def files(self) -> List[Path]:
        return [file.filepath for file in self._files.values()]

    @property
    def transfer_syntax(self) -> UID:
        """The uid of the transfer syntax of the image."""
        return self._transfer_syntax

    @property
    def image_size(self) -> Size:
        """The pixel size of the image."""
        return self.tiles.image_size

    @property
    def tile_size(self) -> Size:
        """The pixel tile size of the image."""
        return self.tiles.tile_size

    @property
    def focal_planes(self) -> List[float]:
        """Focal planes avaiable in the image defined in um."""
        return self.tiles.focal_planes

    @property
    def optical_paths(self) -> List[str]:
        """Optical paths avaiable in the image."""
        return self.tiles.optical_paths

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        """Size of the pixels in mm/pixel."""
        return self._pixel_spacing

    @property
    def photometric_interpretation(self) -> str:
        """Return photometric interpretation."""
        return self._photometric_interpretation

    @property
    def samples_per_pixel(self) -> int:
        """Return samples per pixel (1 or 3)."""
        return self._samples_per_pixel

    @property
    def image_origin(self) -> ImageOrigin:
        return self._image_origin

    def _get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        frame_index = self._get_frame_index(tile, z, path)
        if frame_index == -1:
            return self.blank_encoded_tile
        return self._get_tile_frame(frame_index)

    def _get_decoded_tile(self, tile_point: Point, z: float, path: str) -> PILImage:
        frame_index = self._get_frame_index(tile_point, z, path)
        if frame_index == -1:
            return self.blank_tile
        frame = self._get_tile_frame(frame_index)
        return Image.open(io.BytesIO(frame))

    def _get_file(self, frame_index: int) -> WsiDicomFile:
        """Return file contaning frame index. Raises WsiDicomNotFoundError if
        frame is not found.

        Parameters
        ----------
        frame_index: int
             Frame index to get

        Returns
        ----------
        WsiDicomFile
            File containing the frame
        """
        for frame_offset, file in self._files.items():
            if (
                frame_index < frame_offset + file.frame_count
                and frame_index >= frame_offset
            ):
                return file

        raise WsiDicomNotFoundError(f"Frame index {frame_index}", "instance")

    def _get_tile_frame(self, frame_index: int) -> bytes:
        """Return tile frame for frame index.

        Parameters
        ----------
        frame_index: int
             Frame index to get

        Returns
        ----------
        bytes
            The frame in bytes
        """
        file = self._get_file(frame_index)
        tile_frame = file.read_frame(frame_index)
        return tile_frame

    def _get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Return frame index for tile. Raises WsiDicomOutOfBoundsError if
        tile, z, or path is not valid.

        Parameters
        ----------
        tile: Point
             Tile coordinate
        z: float
            Z coordinate
        path: str
            Optical identifier

        Returns
        ----------
        int
            Tile frame index
        """
        tile_region = Region(position=tile, size=Size(0, 0))
        if not self.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBoundsError(
                f"Tile region {tile_region}", f"plane {self.tiles.tiled_size}"
            )
        frame_index = self.tiles.get_frame_index(tile, z, path)
        return frame_index

    def is_sparse(self, tile: Point, z: float, path: str) -> bool:
        return self.tiles.get_frame_index(tile, z, path) == -1

    def close(self) -> None:
        for file in self._files.values():
            file.close()


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
        """Get frame index from tile index at plane_position.

        Parameters
        ----------
        plane_position: Point
            Position in plane to get the frame index from

        Returns
        ----------
        int
            Frame index
        """
        frame_index = int(self.plane[position.x, position.y])
        return frame_index

    def __setitem__(self, position: Point, frame_index: int):
        """Add frame index to tile index.

        Parameters
        ----------
        plane_position: Point
            Position in plane to add the frame index
        frame_index: int
            Frame index to add to the index
        """
        self.plane[position.x, position.y] = frame_index

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
        return "Sparse tile plane"


class TileIndex(metaclass=ABCMeta):
    """Index for mapping tile position to frame number. Is subclassed into
    FullTileIndex and SparseTileIndex."""

    def __init__(self, datasets: Sequence[Dataset]):
        """Create tile index for frames in datasets. Requires equal tile
        size for all tile planes.

        Parameters
        ----------
        datasets: Sequence[Dataset]
            List of datasets containing tiled image data.

        """
        base_dataset = datasets[0]
        self._image_size = base_dataset.image_size
        self._tile_size = base_dataset.tile_size
        self._frame_count = self._read_frame_count_from_datasets(datasets)
        self._optical_paths = self._read_optical_paths_from_datasets(datasets)
        self._tiled_size = self.image_size.ceil_div(self.tile_size)

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} with image size {self.image_size}, "
            f"tile size {self.tile_size}, tiled size {self.tiled_size}, "
            f"optical paths {self.optical_paths}, "
            f"focal planes {self.focal_planes}, "
            f"and frame count {self.frame_count}"
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
    def frame_count(self) -> int:
        """Return total number of frames in index."""
        return self._frame_count

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
        ----------
        int
            Total frame count.

        """
        count = 0
        for dataset in datasets:
            count += dataset.frame_count
        return count

    @classmethod
    def _read_optical_paths_from_datasets(
        cls, datasets: Sequence[Dataset]
    ) -> List[str]:
        """Return list of optical path identifiers from files.

        Parameters
        ----------
        datasets: Sequence[Dataset]
           List of datasets.

        Returns
        ----------
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
        ----------
        List[str]
            List of optical path identifiers.
        """
        if optical_path_sequence is None:
            return ["0"]
        return list(
            {
                str(optical_ds.OpticalPathIdentifier)
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
        ----------
        Point, float
            The frame xy coordinate and z coordinate
        """
        DECIMALS = 3
        position = frame.PlanePositionSlideSequence[0]
        y = int(position.RowPositionInTotalImagePixelMatrix) - 1
        x = int(position.ColumnPositionInTotalImagePixelMatrix) - 1
        z_offset = getattr(position, "ZOffsetInSlideCoordinateSystem", 0.0)
        z = round(float(z_offset), DECIMALS)
        tile = Point(x=x, y=y) // self.tile_size
        return tile, z


class FullTileIndex(TileIndex):
    """Index for mapping tile position to frame number for datasets containing
    full tiles. Pixel data tiles are ordered by colum, row, z and path, thus
    the frame index for a tile can directly be calculated."""

    def __init__(self, datasets: Sequence[Dataset]):
        """Create full tile index for frames in datasets. Requires equal tile
        size for all tile planes.

        Parameters
        ----------
        datasets: Sequence[Dataset]
            List of datasets containing full tiled image data.
        """
        super().__init__(datasets)
        self._focal_planes = self._read_focal_planes_from_datasets(datasets)

    @property
    def focal_planes(self) -> List[float]:
        return self._focal_planes

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
            * len(self._focal_planes)
            * self.tiled_size.area
        )
        return plane_offset + z_offset + path_offset

    def _read_focal_planes_from_datasets(
        self, datasets: Sequence[Dataset]
    ) -> List[float]:
        """Return list of focal planes in datasets. Values in Pixel Measures
        Sequene are in mm.

        Parameters
        ----------
        datasets: Sequence[Dataset]
           List of datasets to read focal planes from.

        Returns
        ----------
        List[float]
            Focal planes, specified in um.

        """
        MM_TO_MICRON = 1000.0
        DECIMALS = 3
        focal_planes: Set[float] = set()
        for dataset in datasets:
            slice_spacing = dataset.spacing_between_slices
            number_of_focal_planes = dataset.number_of_focal_planes
            if slice_spacing is None:
                if number_of_focal_planes == 1:
                    slice_spacing = 0.0
                else:
                    raise ValueError(
                        "Slice spacing must be known if multple focal planes."
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


class SparseTileIndex(TileIndex):
    """Index for mapping tile position to frame number for datasets containing
    sparse tiles. Frame indices are retrieved from tile position, z, and path
    by finding the corresponding matching SparseTilePlane (z and path) and
    returning the frame index at tile position. If the tile is missing (due to
    the sparseness), -1 is returned."""

    def __init__(self, datasets: Sequence[Dataset]):
        """Create sparse tile index for frames in datasets. Requires equal tile
        size for all tile planes. Pixel data tiles are identified by the Per
        Frame Functional Groups Sequence that contains tile colum, row, z,
        path, and frame index. These are stored in a SparseTilePlane
        (one plane for every combination of z and path).

        Parameters
        ----------
        datasets: Sequence[Dataset]
            List of datasets containing sparse tiled image data.
        """
        super().__init__(datasets)
        self._planes = self._read_planes_from_datasets(datasets)
        self._focal_planes = self._get_focal_planes()

    @property
    def focal_planes(self) -> List[float]:
        return self._focal_planes

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
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
        ----------
        int
            Frame index.
        """
        try:
            plane = self._planes[(z, path)]
        except KeyError:
            raise WsiDicomNotFoundError(f"Plane with z {z}, path {path}", str(self))
        frame_index = plane[tile]
        return frame_index

    def _get_focal_planes(self) -> List[float]:
        """Return list of focal planes defiend in planes.

        Returns
        ----------
        List[float]
            Focal planes, specified in um.
        """
        focal_planes: Set[float] = set()
        for z, _ in self._planes.keys():
            focal_planes.add(z)
        return sorted(list(focal_planes))

    def _read_planes_from_datasets(
        self, datasets: Sequence[Dataset]
    ) -> Dict[Tuple[float, str], SparseTilePlane]:
        """Return SparseTilePlane from planes in datasets.

        Parameters
        ----------
        datasets: Sequence[Dataset]
           List of datasets to read planes from.

        Returns
        ----------
        Dict[Tuple[float, str], SparseTilePlane]
            Dict of planes with focal plane and optical identifier as key.
        """
        planes: Dict[Tuple[float, str], SparseTilePlane] = {}

        for dataset in datasets:
            frame_sequence = dataset.frame_sequence
            for i, frame in enumerate(frame_sequence):
                (tile, z) = self._read_frame_coordinates(frame)
                identifier = dataset.read_optical_path_identifier(frame)

                try:
                    plane = planes[(z, identifier)]
                except KeyError:
                    plane = SparseTilePlane(self.tiled_size)
                    planes[(z, identifier)] = plane
                plane[tile] = i + dataset.frame_offset

        return planes
