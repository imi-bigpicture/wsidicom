from abc import ABCMeta, abstractmethod
from functools import cached_property
from typing import Dict, List, OrderedDict, Set, Tuple

import numpy as np
import pydicom
from pydicom.sequence import Sequence as DicomSequence

from wsidicom.errors import (WsiDicomNotFoundError, WsiDicomOutOfBondsError,
                             WsiDicomSparse)
from wsidicom.file import WsiDicomFile, WsiDataset
from wsidicom.geometry import Point, Region, Size
from wsidicom.optical import OpticalManager


class ImageData(metaclass=ABCMeta):
    """Generic class for image data. Abstract functions for getting tile and
    abstract functions for presenting image properties."""

    @property
    @abstractmethod
    def image_size(self) -> Size:
        raise NotImplementedError

    @property
    @abstractmethod
    def tile_size(self) -> Size:
        raise NotImplementedError

    @property
    @abstractmethod
    def tiled_size(self) -> Size:
        raise NotImplementedError

    @property
    @abstractmethod
    def focal_planes(self) -> List[float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def optical_paths(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_tile(
        self,
        tile: Point,
        z: float,
        path: str
    ) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @cached_property
    def default_z(self) -> float:
        """Return default focal plane in um."""
        default = 0
        if(len(self.focal_planes) > 1):
            smallest = min(self.focal_planes)
            largest = max(self.focal_planes)
            middle = (largest - smallest)/2
            default = min(range(len(self.focal_planes)),
                          key=lambda i: abs(self.focal_planes[i]-middle))

        return self.focal_planes[default]

    @property
    def default_path(self) -> str:
        """Return default optical path identifier."""
        return self.optical_paths[0]

    @cached_property
    def plane_region(self) -> Region:
        return Region(position=Point(0, 0), size=self.tiled_size - 1)

    def valid_tiles(self, region: Region, z: float, path: str) -> bool:
        """Check if tile region is inside tile geometry and z coordinate and
        optical path exists.

        Parameters
        ----------
        region: Region
            Tile region.
        z: float
            Z coordiante.
        path: str
            Optical path.
        """
        return (
            region.is_inside(self.plane_region) and
            (z in self.focal_planes) and
            (path in self.optical_paths)
        )


class SparseTilePlane:
    def __init__(self, tiled_size: Size):
        """Hold frame indices for the tiles in a sparse tiled file.
        Empty (sparse) frames are represented by -1.

        Parameters
        ----------
        tiled_size: Size
            Size of the tiling
        """
        self.plane = np.full(tiled_size.to_tuple(), -1, dtype=int)

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
        if frame_index == -1:
            raise WsiDicomSparse(position)
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

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        return ("Sparse tile plane")


class TileIndex(metaclass=ABCMeta):
    def __init__(
        self,
        datasets: List[WsiDataset]
    ):
        """Index for tiling of pixel data in datasets. Requires equal tile
        size for all tile planes.

        Parameters
        ----------
        datasets: List[WsiDataset]
            List of datasets containing tiled image data.

        """
        base_dataset = datasets[0]
        self._image_size = base_dataset.image_size
        self._tile_size = base_dataset.tile_size
        self._frame_count = self._read_frame_count_from_datasets(datasets)
        self._optical_paths = self._read_optical_paths_from_datasets(datasets)

    @property
    @abstractmethod
    def focal_planes(self) -> List[float]:
        """Return list of focal planes in index."""
        raise NotImplementedError

    @property
    def image_size(self) -> Size:
        """Return image size in pixels."""
        return self._image_size

    @property
    def tile_size(self) -> Size:
        """Return tile size in pixels."""
        return self._tile_size

    @cached_property
    def tiled_size(self) -> Size:
        """Return size of tiling (columns x rows)."""
        return self.image_size / self.tile_size

    @property
    def frame_count(self) -> int:
        """Return total number of frames in index."""
        return self._frame_count

    @property
    def optical_paths(self) -> List[str]:
        """Return list of optical paths in index."""
        return self._optical_paths

    @abstractmethod
    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Abstract method for getting the frame index for a tile"""
        raise NotImplementedError

    @staticmethod
    def _read_frame_count_from_datasets(
        datasets: List[WsiDataset]
    ) -> int:
        """Return total frame count from files.

        Parameters
        ----------
        datasets: List[WsiDataset]
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

    @staticmethod
    def _read_optical_paths_from_datasets(
        datasets: List[WsiDataset]
    ) -> List[str]:
        """Return list of optical path identifiers from files.

        Parameters
        ----------
        datasets: List[WsiDataset]
           List of datasets.

        Returns
        ----------
        List[str]
            Optical identifiers.

        """
        paths: Set[str] = set()
        for dataset in datasets:
            paths.update(OpticalManager.get_path_identifers(
                dataset.optical_path_sequence
            ))
        return list(paths)


class FullTileIndex(TileIndex):
    def __init__(
        self,
        datasets: List[WsiDataset]
    ):
        """Index for tiling of full tiled pixel data in datasets. Requires
        equal tile size for all tile planes. Pixel data tiles are ordered by
        colum, row, z and path, thus the frame index for a tile can directly be
        calculated.
        """
        super().__init__(datasets)
        self._focal_planes = self._read_focal_planes_from_datasets(datasets)

    @property
    def focal_planes(self) -> List[float]:
        return self._focal_planes

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f"Full tile index tile size: {self.tile_size}"
            f", plane size: {self.tiled_size}"
        )
        if depth is not None:
            depth -= 1
            if(depth < 0):
                return string
        string += (
            f" of z: {self.focal_planes} and path: {self.optical_paths}"
        )

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
            * len(self._focal_planes) * self.tiled_size.area
        )
        return plane_offset + z_offset + path_offset

    def _read_focal_planes_from_datasets(
        self,
        datasets: List[WsiDataset]
    ) -> List[float]:
        """Return list of focal planes in datasets. Values in Pixel Measures
        Sequene are in mm.

        Parameters
        ----------
        datasets: List[WsiDataset]
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
            if slice_spacing == 0 and number_of_focal_planes != 1:
                raise ValueError
            for plane in range(number_of_focal_planes):
                z = round(plane * slice_spacing * MM_TO_MICRON, DECIMALS)
                focal_planes.add(z)
        return list(focal_planes)

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
                (index for index, plane_path in enumerate(self._optical_paths)
                 if plane_path == path)
            )
        except StopIteration:
            raise WsiDicomNotFoundError(f"Optical path {path}", self)

    def _get_focal_plane_index(self, z: float) -> int:
        """Return index of the focal plane of z.

        Parameters
        ----------
        z: float
            The z coordinate (in um) to search for.

        Returns
        ----------
        int
            Focal plane index for z coordiante.
        """
        try:
            return next(index for index, plane in enumerate(self.focal_planes)
                        if plane == z)
        except StopIteration:
            raise WsiDicomNotFoundError(f"Z {z} in instance", self)


class SparseTileIndex(TileIndex):
    def __init__(
        self,
        datasets: List[WsiDataset]
    ):
        """Index for sparse tiled pixel data in datasets. Requires equal tile
        size for all tile planes Pixel data tiles are identified by the Per
        Frame Functional Groups Sequence that contains tile colum, row, z,
        path, and frame index. These are stored in a SparseTilePlane
        (one plane for every combination of z and path). Frame indices are
        retrieved from tile position, z, and path by finding the corresponding
        matching SparseTilePlane (z and path) and returning the frame index at
        tile position. If the tile is missing (due to the sparseness), -1 is
        returned.

        Parameters
        ----------
        datasets: List[WsiDataset]
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

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f"Sparse tile index tile size: {self.tile_size}"
            f", plane size: {self.tiled_size}"
        )
        return string

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
            raise WsiDicomNotFoundError(
                f"Plane with z {z}, path {path}", self
            )
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
        for z, path in self._planes.keys():
            focal_planes.add(z)
        return list(focal_planes)

    def _read_planes_from_datasets(
        self,
        datasets: List[WsiDataset]
    ) -> Dict[Tuple[float, str], SparseTilePlane]:
        """Return SparseTilePlane from planes in datasets.

        Parameters
        ----------
        datasets: List[WsiDataset]
           List of datasets to read planes from.

        Returns
        ----------
        Dict[Tuple[float, str], SparseTilePlane]
            Dict of planes with focal plane and optical identifier as key.
        """
        planes: Dict[Tuple[float, str], SparseTilePlane] = {}

        for dataset in datasets:
            file_offset = dataset.file_offset
            frame_sequence = dataset.frame_sequence

            for i, frame in enumerate(frame_sequence):
                (tile, z) = self._read_frame_coordinates(frame)
                identifier = dataset.read_optical_path_identifier(frame)
                try:
                    plane = planes[(z, identifier)]
                except KeyError:
                    plane = SparseTilePlane(self.tiled_size)
                    planes[(z, identifier)] = plane
                plane[tile] = i + file_offset

        return planes

    def _read_frame_coordinates(
            self,
            frame: DicomSequence
    ) -> Tuple[Point, float]:
        """Return frame coordinate (Point(x, y) and float z) of the frame.
        In the Plane Position Slide Sequence x and y are defined in mm and z in
        um.

        Parameters
        ----------
        frame: DicomSequence
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
        z = round(float(position.ZOffsetInSlideCoordinateSystem), DECIMALS)
        tile = Point(x=x, y=y) // self.tile_size
        return tile, z


class DicomImageData(ImageData):
    """Generic class reading image data from dicom file(s). Image data can
    be sparsly or fully tiled-"""
    def __init__(self, files: List[WsiDicomFile]) -> None:
        # Key is frame offset
        self._files = OrderedDict(
            (file.frame_offset, file) for file
            in sorted(files, key=lambda file: file.frame_offset)
        )

        base_file = files[0]
        datasets = [file.dataset for file in self._files.values()]
        if base_file.dataset.tile_type == 'TILED_FULL':
            self.tiles = FullTileIndex(datasets)
        else:
            self.tiles = SparseTileIndex(datasets)

    @property
    def image_size(self) -> Size:
        return self.tiles.image_size

    @property
    def tile_size(self) -> Size:
        return self.tiles.tile_size

    @property
    def tiled_size(self) -> Size:
        return self.tiles.tiled_size

    @property
    def focal_planes(self) -> List[float]:
        return self.tiles.focal_planes

    @property
    def optical_paths(self) -> List[str]:
        return self.tiles.optical_paths

    def get_tile(self, tile: Point, z: float, path: str) -> bytes:
        frame_index = self._get_frame_index(tile, z, path)
        return self._get_tile_frame(frame_index)

    def get_filepointer(
        self,
        tile: Point,
        z: float,
        path: str
    ) -> Tuple[pydicom.filebase.DicomFileLike, int, int]:
        """Return file pointer, frame position, and frame lenght for tile with
        z and path. If frame is inside tile geometry but no tile exists in
        frame data (sparse) WsiDicomSparse is raised.

        Parameters
        ----------
        tile: Point
            Tile coordinate to get.
        z: float
            z coordinate to get tile for.
        path: str
            Optical path to get tile for.

        Returns
        ----------
        Tuple[pydicom.filebase.DicomFileLike, int, int]:
            File pointer, frame offset and frame lenght in number of bytes.
        """
        frame_index = self._get_frame_index(tile, z, path)
        file = self._get_file(frame_index)
        return file.get_filepointer(frame_index)

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
            if (frame_index < frame_offset + file.frame_count and
                    frame_index >= frame_offset):
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
        tile_frame = file._read_frame(frame_index)
        return tile_frame

    def _get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Return frame index for tile. Raises WsiDicomOutOfBondsError if
        tile, z, or path is not valid. Raises WsiDicomSparse if index is sparse
        and tile is not in frame data.

        Parameters
        ----------
        tile: Point
             Tile coordiante
        z: float
            Z coordiante
        path: str
            Optical identifier

        Returns
        ----------
        int
            Tile frame index
        """
        tile_region = Region(position=tile, size=Size(0, 0))
        if not self.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBondsError(
                f"Tile region {tile_region}",
                f"plane {self.tiles.tiled_size}"
            )
        frame_index = self.tiles.get_frame_index(tile, z, path)
        return frame_index

    def is_sparse(self, tile: Point, z: float, path: str) -> bool:
        try:
            self.tiles.get_frame_index(tile, z, path)
            return False
        except WsiDicomSparse:
            return True

    def close(self) -> None:
        for file in self._files.values():
            file.close()


class Tiler(metaclass=ABCMeta):
    @property
    @abstractmethod
    def level_count(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def label_count(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def overview_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_level(self, level: int) -> ImageData:
        raise NotImplementedError

    @abstractmethod
    def get_label(self, index: int = 0) -> ImageData:
        raise NotImplementedError

    @abstractmethod
    def get_overview(self, index: int = 0) -> ImageData:
        raise NotImplementedError

    @abstractmethod
    def get_tile(
        self,
        level: int,
        tile_position: Tuple[int, int]
    ) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def close() -> None:
        raise NotImplementedError
