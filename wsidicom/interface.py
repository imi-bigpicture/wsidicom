import io
import math
import os
import warnings
from abc import ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Set, Tuple, Union

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import UID as Uid

from .errors import (WsiDicomFileError, WsiDicomMatchError,
                     WsiDicomNotFoundError, WsiDicomOutOfBondsError,
                     WsiDicomSparse, WsiDicomUidDuplicateError)
from .file import WsiDicomFile
from .geometry import Point, PointMm, Region, RegionMm, Size, SizeMm
from .optical import OpticalManager, OpticalPath
from .stringprinting import dict_pretty_str, list_pretty_str, str_indent
from .uid import BaseUids


class SparseTilePlane:
    def __init__(self, plane_size: Size):
        """Hold frame indices for the tiles in a sparse tiled file.
        Empty (sparse) frames are represented by -1.

        Parameters
        ----------
        plane_size: Size
            Size of the tiling
        """
        self.plane = np.full(plane_size.to_tuple(), -1, dtype=int)

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
        files: Dict[int, WsiDicomFile],
    ):
        """Index for the tiling of pixel data.
        Requires same tile size for all tile planes

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
            Dict of file frame offset and wsi file.

        """
        self._image_size = files[0].image_size
        self._tile_size = files[0].tile_size
        self._plane_size: Size = self.image_size / self.tile_size
        self._frame_count = self._get_frame_count_from_files(files)
        self._focal_planes = self._get_focal_planes_from_files(files)
        self._optical_paths = self._get_optical_paths_from_files(files)
        self._default_z: float = self._select_default_z(self.focal_planes)
        self._default_path = self.optical_paths[0]

    @property
    def tile_size(self) -> Size:
        """Return tile size in pixels"""
        return self._tile_size

    @property
    def plane_size(self) -> Size:
        """Return size of tiling (columns x rows)"""
        return self._plane_size

    @property
    def default_z(self) -> float:
        """Return default focal plane in um"""
        return self._default_z

    @property
    def default_path(self) -> str:
        """Return default optical path identifier"""
        return self._default_path

    @property
    def frame_count(self) -> int:
        """Return total number of frames"""
        return self._frame_count

    @property
    def focal_planes(self) -> List[float]:
        """Return total number of focal planes"""
        return self._focal_planes

    @property
    def optical_paths(self) -> List[str]:
        """Return total number of optical paths"""
        return self._optical_paths

    @property
    def image_size(self) -> Size:
        """Return image size in pixels"""
        return self._image_size

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

    @abstractmethod
    def _get_focal_planes_from_files(
        self,
        files: Dict[int, WsiDicomFile]
    ) -> List[float]:
        """Abstract method for getting focal planes from file"""
        raise NotImplementedError

    def valid_tiles(self, region: Region, z: float, path: str) -> bool:
        """Check if tile region is inside tile geometry and z coordinate and
        optical path exists.

        Parameters
        ----------
        region: Region
            Tile region
        z: float
            z coordiante
        path: str
            optical path
        """
        plane_region = Region(
            position=Point(0, 0),
            size=self.plane_size - 1
        )
        return (
            region.is_inside(plane_region) and
            (z in self.focal_planes) and
            (path in self.optical_paths)
        )

    @staticmethod
    def _select_default_z(focal_planes: List[float]) -> float:
        """Select default z coordinate to use if specific plane not set.
        If more than one focal plane available the middle one is selected.

        Parameters
        ----------
        focal_planes: List[float]
           List of focal planes to select from

        Returns
        ----------
        float
            Default z coordinate

        """
        default = 0
        if(len(focal_planes) > 1):
            smallest = min(focal_planes)
            largest = max(focal_planes)
            middle = (largest - smallest)/2
            default = min(range(len(focal_planes)),
                          key=lambda i: abs(focal_planes[i]-middle))

        return focal_planes[default]

    @staticmethod
    def _get_frame_count_from_files(
        files: Dict[int, WsiDicomFile]
    ) -> int:
        """Return total frame count from files.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
           Dict of wsi dicom files

        Returns
        ----------
        int
            Total frame count

        """
        count = 0
        for file in files.values():
            count += file.frame_count
        return count

    @staticmethod
    def _get_optical_paths_from_files(
        files: Dict[int, WsiDicomFile]
    ) -> List[str]:
        """Return list of optical path identifiers from files.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
           Dict of wsi dicom files

        Returns
        ----------
        List[str]
            Optical identifiers

        """
        paths: Set[str] = set()
        for file in files.values():
            paths.update(OpticalManager.get_path_identifers(
                file.optical_path_sequence)
            )
        return list(paths)


class FullTileIndex(TileIndex):
    """Index for full tiled pixel data.
    Requires same tile size for all tile planes.
    Pixel data tiles are ordered by colum, row, z and path, thus
    the frame index for a tile can directly be calculated.
    """
    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f"Full tile index tile size: {self.tile_size}"
            f", plane size: {self.plane_size}"
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
        """Return frame index for a Point tile, z coordinate,
        and optical path from full tile index. Assumes that tile, z, and path
        are valid.

        Parameters
        ----------
        tile: Point
            tile xy to get
        z: float
            z coordinate to get
        path: str
            ID of optical path to get

        Returns
        ----------
        int
            Frame index
        """
        z_index = self._focal_plane_index(z)
        plane_offset = tile.x + self.plane_size.width*tile.y
        tiles_in_plane = self.plane_size.width * self.plane_size.height
        z_offset = z_index * tiles_in_plane
        path_index = self._optical_path_index(path)
        path_offset = (
            path_index * len(self._focal_planes) * tiles_in_plane
        )
        return plane_offset + z_offset + path_offset

    def _get_focal_planes_from_files(
        self,
        files: Dict[int, WsiDicomFile]
    ) -> List[float]:
        """Return list of focal planes from files.
        Values in Pixel Measures Sequene are in mm.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
           Dict of wsi dicom files

        Returns
        ----------
        List[float]
            Focal planes

        """
        MM_TO_MICRON = 1000.0
        DECIMALS = 3
        focal_planes: Set[float] = set()
        for file in files.values():
            sequence = file.frame_sequence[0]
            spacing = getattr(
                sequence.PixelMeasuresSequence[0],
                'SpacingBetweenSlices',
                0
            )
            if spacing == 0 and file.focal_planes != 1:
                raise WsiDicomFileError(
                    file.filepath,
                    "Multipe focal planes and zero plane spacing"
                )
            for plane in range(file.focal_planes):
                z = round(plane * spacing * MM_TO_MICRON, DECIMALS)
                focal_planes.add(z)
        return list(focal_planes)

    def _optical_path_index(self, path: str) -> int:
        """Return index of the optical path in instance.
        This assumes that all files in a concatenated set contains all the
        optical path identifiers of the set.

        Parameters
        ----------
        path: str
            Optical path identifier

        Returns
        ----------
        int
            The index of the optical path identifier in the optical path
            sequence
        """
        try:
            return next(
                (index for index, plane_path in enumerate(self._optical_paths)
                 if plane_path == path)
            )
        except StopIteration:
            raise WsiDicomNotFoundError(
                f"Optical path {path}",
                "full tile index"
            )

    def _focal_plane_index(self, z: float) -> int:
        """Return index of the focal plane of z.

        Parameters
        ----------
        z: float
            The z coordinate to search for

        Returns
        ----------
        int
            Focal plane index
        """
        try:
            return next(index for index, plane in enumerate(self.focal_planes)
                        if plane == z)
        except StopIteration:
            raise WsiDicomNotFoundError(
                f"Z {z} in instance", "full tile index"
            )


class SparseTileIndex(TileIndex):
    def __init__(
        self,
        files: Dict[int, WsiDicomFile]
    ):
        """Index for sparse tiled pixel data.
        Requires same tile size for all tile planes.
        Pixel data tiles are identified by the Per Frame Functional Groups
        Sequence that contains tile colum, row, z, path, and frame index. These
        are stored in a SparseTilePlane (one plane for every combination of z
        and path). Frame indices are retrieved from tile position, z, and path
        by finding the corresponding matching SparseTilePlane (z and path) and
        returning the frame index at tile position. If the tile is missing (due
        to the sparseness), -1 is returned.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
            Dict of file frame offset and wsi file.

        """
        super().__init__(files)
        self._planes = self._get_planes_from_files(files)

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f"Sparse tile index tile size: {self.tile_size}"
            f", plane size: {self.plane_size}"
        )
        return string

    def get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Return frame index for a Point tile, z coordinate, and optical
        path.

        Parameters
        ----------
        tile: Point
            tile xy to get
        z: float
            z coordinate to get
        path: str
            ID of optical path to get

        Returns
        ----------
        int
            Frame index
        """
        plane = self._get_plane(z, path)
        frame_index = plane[tile]
        return frame_index

    def _get_focal_planes_from_files(
        self,
        files: Dict[int, WsiDicomFile]
    ) -> List[float]:
        """Return list of focal planes from files.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
           Dict of wsi dicom files

        Returns
        ----------
        List[float]
            Focal planes
        """
        focal_planes: Set[float] = set()
        for file in files.values():
            for frame in file.frame_sequence:
                try:
                    (tile, z) = self._get_frame_coordinates(frame)
                except AttributeError:
                    raise WsiDicomFileError(
                        file.filepath,
                        "Invalid plane position slide sequence"
                    )
                focal_planes.add(z)
        return list(focal_planes)

    def _get_planes_from_files(
        self,
        files: Dict[int, WsiDicomFile]
    ) -> Dict[Tuple[float, str], SparseTilePlane]:
        """Return SparseTilePlane from planes read from files.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
           Dict of wsi dicom files

        Returns
        ----------
        Dict[Tuple[float, str], SparseTilePlane]
            Dict of planes with focal plane and optical identifier as key.
        """
        planes: Dict[Tuple[float, str], SparseTilePlane] = {}

        for file_offset, file in files.items():
            for i, frame in enumerate(file.frame_sequence):
                (tile, z) = self._get_frame_coordinates(frame)
                optical_sequence = getattr(
                    frame,
                    'OpticalPathIdentificationSequence',
                    file.optical_path_sequence
                )
                identifier = getattr(
                    optical_sequence[0],
                    'OpticalPathIdentifier',
                    '0'
                )
                try:
                    plane = planes[(z, identifier)]
                except KeyError:
                    plane = SparseTilePlane(self.plane_size)
                    planes[(z, identifier)] = plane
                plane[tile] = i + file_offset
        return planes

    def _get_plane(self, z: float, path: str) -> SparseTilePlane:
        """Return plane with z coordinate and optical path.

        Parameters
        ----------
        z: float
            Z coordinate to search for
        path: str
            Optical path identifer to search for

        Returns
        ----------
        SparseTilePlane
            The plane for z coordinate and path
        """
        try:
            return self._planes[(z, path)]
        except KeyError:
            raise WsiDicomNotFoundError(
                f"Plane with z {z}, path {path}", "sparse tile index"
            )

    def _get_frame_coordinates(
            self,
            frame: DicomSequence
    ) -> Tuple[Point, float]:
        """Return frame coordinate (Point(x, y) and float z) of the frame.
        In the Plane Position Slide Sequence x and y are defined in mm and z in
        um.

        Parameters
        ----------
        frame: DicomSequence
            Pydicom frame sequence

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


class Tiler(metaclass=ABCMeta):
    @abstractmethod
    def get_encoded_tile(self, tile: Point) -> bytes:
        raise NotImplementedError


class WsiInstance(metaclass=ABCMeta):
    def __init__(
        self,
        size: Size,
        pixel_spacing: SizeMm,
        tile_size: Size,
        wsi_type: str,
        samples_per_pixel: int,
        transfer_syntax: Uid,
        photometric_interpretation: str,
        slice_thickness: float,
        slice_spacing: float,
        mm_size: SizeMm,
        mm_depth: float,
        focus_method: str,
        ext_depth_of_field: bool,
        ext_depth_of_field_planes: int = None,
        ext_depth_of_field_plane_distance: float = None
    ):
        self._size = size
        self._pixel_spacing = pixel_spacing
        self._tile_size = tile_size
        self._wsi_type = wsi_type
        self._samples_per_pixel = samples_per_pixel
        self._transfer_syntax = transfer_syntax
        self._photometric_interpretation = photometric_interpretation
        self._slice_thickness = slice_thickness
        self._slice_spacing = slice_spacing
        self._blank_color = self._get_blank_color(
            self._photometric_interpretation
        )
        if(self._samples_per_pixel == 1):
            self._image_mode = "L"
        else:
            self._image_mode = "RGB"
        self._mm_size = mm_size
        self._mm_depth = mm_depth
        self._focus_method = focus_method
        self._ext_depth_of_field = ext_depth_of_field
        self._ext_depth_of_field_planes = ext_depth_of_field_planes
        self._ext_depth_of_field_plane_distance = (
            ext_depth_of_field_plane_distance
        )

    @property
    def wsi_type(self) -> str:
        """Return wsi type"""
        return self._wsi_type

    @property
    def blank_color(self) -> Tuple[int, int, int]:
        """Return RGB background color"""
        return self._blank_color

    @property
    def blank_tile(self) -> Image:
        """Return background tile"""
        return self._blank_tile

    @property
    def blank_encoded_tile(self) -> bytes:
        """Return encoded background tile"""
        return self._blank_encoded_tile

    @property
    def size(self) -> Size:
        """Return image size in pixels"""
        return self._size

    @property
    def tile_size(self) -> Size:
        """Return tile size in pixels"""
        return self._tile_size

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um/pixel"""
        return self.pixel_spacing*1000.0

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm/pixel"""
        return self._pixel_spacing

    @property
    def mm_size(self) -> SizeMm:
        """Return slide size in mm"""
        return self._mm_size

    @property
    def mm_depth(self) -> float:
        """Return imaged depth in mm."""
        return self._mm_depth

    @property
    def slice_thickness(self) -> float:
        """Return slice thickness"""
        return self._slice_thickness

    @property
    def slice_spacing(self) -> float:
        """Return slice spacing"""
        return self._slice_spacing

    @property
    def focus_method(self) -> str:
        return self._focus_method

    @property
    def ext_depth_of_field(self) -> bool:
        return self._ext_depth_of_field

    @property
    def ext_depth_of_field_planes(self) -> Optional[int]:
        return self._ext_depth_of_field_planes

    @property
    def ext_depth_of_field_plane_distance(self) -> Optional[float]:
        return self._ext_depth_of_field_plane_distance

    @property
    def photometric_interpretation(self) -> str:
        """Return photometric interpretation"""
        return self._photometric_interpretation

    @property
    def samples_per_pixel(self) -> int:
        """Return samples per pixel (1 or 3)"""
        return self._samples_per_pixel

    @property
    def identifier(self) -> Uid:
        """Return identifier (instance uid for single file instance or
        concatenation uid for multiple file instance)"""
        return self._identifier

    @property
    @abstractmethod
    def default_z(self) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def default_path(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def focal_planes(self) -> List[float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def optical_paths(self) -> List[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def plane_size(self) -> Size:
        raise NotImplementedError

    @property
    def uids(self) -> Optional[BaseUids]:
        raise NotImplementedError

    def stitch_tiles(self, region: Region, path: str, z: float) -> Image:
        """Stitches tiles together to form requested image.

        Parameters
        ----------
        region: Region
             Pixel region to stitch to image
        path: str
            Optical path
        z: float
            Z coordinate

        Returns
        ----------
        Image
            Stitched image
        """
        stitching_tiles = self.get_tile_range(region, z, path)
        image = Image.new(mode=self._image_mode, size=region.size.to_tuple())
        write_index = Point(x=0, y=0)
        tile = stitching_tiles.position
        for x, y in stitching_tiles.iterate_all(include_end=True):
            tile = Point(x=x, y=y)
            tile_image = self.get_tile(tile, z, path)
            tile_crop = self.crop_tile(tile, region)
            tile_image = tile_image.crop(box=tile_crop.box)
            image.paste(tile_image, write_index.to_tuple())
            write_index = self._write_indexer(
                write_index,
                tile_crop.size,
                region.size
            )
        return image

    def get_tile_range(
        self,
        pixel_region: Region,
        z: float,
        path: str
    ) -> Region:
        """Return range of tiles to cover pixel region.

        Parameters
        ----------
        pixel_region: Region
            Pixel region of tiles to get
        z: float
            Z coordinate of tiles to get
        path: str
            Optical path identifier of tiles to get

        Returns
        ----------
        Region
            Region of tiles for stitching image
        """
        start = pixel_region.start // self.tiles.tile_size
        end = pixel_region.end / self.tiles.tile_size - 1
        tile_region = Region.from_points(start, end)
        if not self.tiles.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBondsError(
                f"Tile region {tile_region}", f"plane {self.tiles.plane_size}"
            )
        return tile_region

    def crop_encoded_tile_to_level(
        self,
        tile: Point,
        tile_frame: bytes
    ) -> bytes:
        cropped_tile_region = self.crop_to_level_size(tile)
        if cropped_tile_region.size != self.tile_size:
            image = Image.open(io.BytesIO(tile_frame))
            image.crop(box=cropped_tile_region.box_from_origin)
            tile_frame = self.encode(image)
        return tile_frame

    def crop_tile_to_level(
        self,
        tile: Point,
        tile_image: Image
    ) -> Image:
        cropped_tile_region = self.crop_to_level_size(tile)
        if cropped_tile_region.size != self.tile_size:
            tile_image = tile_image.crop(
                box=cropped_tile_region.box_from_origin
            )
        return tile_image

    def encode(self, image: Image) -> bytes:
        """Encode image using transfer syntax.

        Parameters
        ----------
        image: Image
            Image to encode

        Returns
        ----------
        bytes
            Encoded image as bytes

        """
        (image_format, image_options) = self._image_settings(
            self._transfer_syntax
        )
        with io.BytesIO() as buffer:
            image.save(buffer, format=image_format, **image_options)
            return buffer.getvalue()

    def crop_tile(self, tile: Point, stitching: Region) -> Region:
        """Crop tile at edge of stitching region so that the tile after croping
        is inside the stitching region.

        Parameters
        ----------
        tile: Point
            Position of tile to crop
        stitching : Region
            Region of stitched image

        Returns
        ----------
        Region
            Region of tile inside stitching region
        """
        tile_region = Region(
            position=tile * self.tile_size,
            size=self.tile_size
        )
        cropped_tile_region = stitching.crop(tile_region)
        cropped_tile_region.position = (
            cropped_tile_region.position % self.tile_size
        )
        return cropped_tile_region

    def crop_to_level_size(self, item: Union[Point, Region]) -> Region:
        """Crop tile or region so that the tile (Point) or region (Region)
        after cropping is inside the image size of the level.

        Parameters
        ----------
        item: Union[Point, Region]
            Position of tile or region to crop

        Returns
        ----------
        Region
            Region of tile or region inside level image
        """
        level_region = Region(
            position=Point(x=0, y=0),
            size=self.size
        )
        if isinstance(item, Point):
            return self.crop_tile(item, level_region)
        return level_region.crop(item)

    @staticmethod
    def _write_indexer(
        index: Point,
        previous_size: Size,
        image_size: Size
    ) -> Point:
        """Increment index in x by previous width until index x exceds image
        size. Then resets index x to 0 and increments index y by previous
        height. Requires that tiles are scanned row by row.

        Parameters
        ----------
        index: Point
            The last write index position
        previouis_size: Size
            The size of the last written last tile
        image_size: Size
            The size of the image to be written

        Returns
        ----------
        Point
            The position (upper right) in image to insert the next tile into
        """
        index.x += previous_size.width
        if(index.x >= image_size.width):
            index.x = 0
            index.y += previous_size.height
        return index

    def _create_blank_tile(self) -> Image:
        """Create blank tile for instance.

        Returns
        ----------
        Image
            Blank tile image
        """
        if(self._samples_per_pixel == 1):
            self._image_mode = "L"
        else:
            self._image_mode = "RGB"
        return Image.new(
            mode=self._image_mode,
            size=self.tile_size.to_tuple(),
            color=self.blank_color[:self._samples_per_pixel]
        )

    @staticmethod
    def _get_blank_color(
        photometric_interpretation: str
    ) -> Tuple[int, int, int]:
        """Return color to use blank tiles.

        Parameters
        ----------
        photometric_interpretation: str
            The photomoetric interpretation of the dataset

        Returns
        ----------
        int, int, int
            RGB color

        """
        BLACK = 0
        WHITE = 255
        if(photometric_interpretation == "MONOCHROME2"):
            return (BLACK, BLACK, BLACK)  # Monocrhome2 is black
        return (WHITE, WHITE, WHITE)

    @abstractmethod
    def get_tile(self, tile: Point, z: float, path: str) -> Image:
        raise NotImplementedError

    @abstractmethod
    def get_encoded_tile(
        self,
        tile: Point,
        z: float,
        path: str,
        crop: bool
    ) -> bytes:
        raise NotImplementedError

    @staticmethod
    def check_duplicate_instance(
        instances: List['WsiInstance'],
        self: object
    ) -> None:
        """Check for duplicates in list of instances. Instances are duplicate
        if instance identifier (file instance uid or concatenation uid) match.
        Stops at first found duplicate and raises WsiDicomUidDuplicateError.

        Parameters
        ----------
        instances: List['WsiInstance']
            List of instances to check.
        caller: Object
            Object that the instances belongs to.
        """
        instance_identifiers: List[str] = []
        for instance in instances:
            instance_identifier = instance.identifier
            if instance_identifier not in instance_identifiers:
                instance_identifiers.append(instance_identifier)
            else:
                raise WsiDicomUidDuplicateError(str(instance), str(self))


class WsiGenericInstance(WsiInstance):
    def __init__(
        self,
        tiler: Tiler,
        size: Size,
        pixel_spacing: SizeMm,
        tile_size: Size,
        optical_paths: List[str],
        focal_planes: List[float],
        wsi_type: str,
        samples_per_pixel: int,
        transfer_syntax: Uid,
        photometric_interpretation: str,
        slice_thickness: float,
        slice_spacing: float,
        focus_method: str,
        ext_depth_of_field: bool,
        ext_depth_of_field_planes: int = None,
        ext_depth_of_field_plane_distance: float = None
    ):
        super().__init__(
            size,
            pixel_spacing,
            tile_size,
            wsi_type,
            samples_per_pixel,
            transfer_syntax,
            photometric_interpretation,
            slice_thickness,
            slice_spacing
        )
        self.tiler = tiler
        self._focal_planes = focal_planes
        self._optical_paths = optical_paths
        self._identifier = pydicom.uid.generate_uid()
        self._uids = None

    @property
    def default_z(self) -> float:
        return self._focal_planes[0]

    @property
    def default_path(self) -> str:
        return self._optical_paths[0]

    @property
    def focal_planes(self) -> List[float]:
        return self._focal_planes

    @property
    def optical_paths(self) -> List[str]:
        return self._optical_paths

    @property
    def plane_size(self) -> Size:
        return self.size//self.tile_size

    @property
    def uids(self) -> Optional[BaseUids]:
        return None

    def get_tile(
        self,
        tile: Point,
        z: float,
        path: str
    ) -> Image:
        tile_frame = self.tiler.get_encoded_tile(tile)
        image = Image.open(io.BytesIO(tile_frame))
        return self.crop_tile_to_level(tile, image)

    def get_encoded_tile(
        self,
        tile: Point,
        z: float,
        path: str,
        crop: bool = True
    ) -> bytes:
        tile_frame = self.tiler.get_encoded_tile(tile)
        if not crop:
            return tile_frame
        return self.crop_encoded_tile_to_level(tile, tile_frame)

    def get_tile_range(
        self,
        pixel_region: Region,
        z: float,
        path: str
    ) -> Region:
        start = pixel_region.start // self.tile_size
        end = pixel_region.end / self.tile_size - 1
        tile_region = Region.from_points(start, end)
        return tile_region


class WsiDicomInstance(WsiInstance):
    def __init__(self, files: List[WsiDicomFile]):
        """Represents a single SOP instance or a concatenated SOP instance.
        The instance can contain multiple focal planes and optical paths.

        Files needs to match in UIDs and have the same image and tile size.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Files to include in the instance.
        """
        # Key is frame offset
        self._files = OrderedDict(
            (file.frame_offset, file) for file
            in sorted(files, key=lambda file: file.frame_offset)
        )
        (
            self._identifier,
            self._uids,
            wsi_type
        ) = self._validate_instance()

        self.tiles = self._create_tileindex(self._files)
        base_file = files[0]
        super().__init__(
            base_file.image_size,
            base_file.pixel_spacing,
            base_file.tile_size,
            wsi_type,
            base_file.samples_per_pixel,
            base_file.transfer_syntax,
            base_file.photometric_interpretation,
            base_file.slice_thickness,
            base_file.slice_spacing,
            base_file.mm_size,
            base_file.mm_depth,
            base_file.focus_method,
            base_file.ext_depth_of_field,
            base_file.ext_depth_of_field_planes,
            base_file.ext_depth_of_field_plane_distance
        )
        self._blank_tile = self._create_blank_tile()
        self._blank_encoded_tile = self.encode(self._blank_tile)

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f"Instance default z: {self.tiles.default_z,}"
            f" default path: { self.tiles.default_path}"
        )
        if depth is not None:
            depth -= 1
            if(depth < 0):
                return string
        string += (
            '\n' + str_indent(indent) + 'Files'
            + dict_pretty_str(self._files, indent+1, depth, 1, 1) + '\n'
            + str_indent(indent) + self.tiles.pretty_str(indent+1, depth)
        )
        return string

    @property
    def default_z(self) -> float:
        return self.tiles.default_z

    @property
    def default_path(self) -> str:
        return self.tiles.default_path

    @property
    def focal_planes(self) -> List[float]:
        return self.tiles.focal_planes

    @property
    def optical_paths(self) -> List[str]:
        return self.tiles.optical_paths

    @property
    def plane_size(self) -> Size:
        return self.tiles.plane_size

    @property
    def files(self) -> List[WsiDicomFile]:
        """Return list of files"""
        return list(self._files.values())

    @property
    def uids(self) -> Optional[BaseUids]:
        """Return base uids"""
        return self._uids

    @classmethod
    def open(
        cls,
        files: List[WsiDicomFile],
        series_uids: BaseUids,
        series_tile_size: Size = None
    ) -> List['WsiDicomInstance']:
        """Return list of instances created from files. Only files matching
        series uid and tile_size, if defined, are opened.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Files to create instances from.
        series_uids: BaseUids
            Uid to match against.
        series_tile_size: Size
            Tile size to match against (for level instances)

        Returns
        ----------
        List[WsiDicomInstance]
            List of created instances.
        """
        filtered_files = cls._filter_files(
            files,
            series_uids,
            series_tile_size
        )
        instances: List[WsiDicomInstance] = []
        files_grouped_by_instance = cls._group_files(filtered_files)

        for instance_files in files_grouped_by_instance.values():
            new_instance = WsiDicomInstance(instance_files)
            instances.append(new_instance)

        return instances

    def matches(self, other_instance: 'WsiDicomInstance') -> bool:
        """Return true if other instance is of the same stack as self.

        Parameters
        ----------
        other_instance: WsiDicomInstance
            Instance to check

        Returns
        ----------
        bool
            True if instanes are of same stack.

        """
        return (
            self.uids == other_instance.uids and
            self.size == other_instance.size and
            self.tile_size == other_instance.tile_size and
            self.wsi_type == other_instance.wsi_type
        )

    def create_ds(self) -> Dataset:
        """Create a base pydicom dataset based on first file in instance.

        Returns
        ----------
        pydicom.dataset
            Pydicom dataset with common attributes for the levels.
        """
        INCLUDE = [0x0002, 0x0008, 0x0010, 0x0018, 0x0020, 0x0040]
        DELETE = ['ImageType', 'SOPInstanceUID', 'ContentTime',
                  'InstanceNumber', 'DimensionOrganizationSequence']

        base_file_path = self._files[0].filepath
        base_ds = pydicom.dcmread(base_file_path, stop_before_pixels=True)
        ds = pydicom.Dataset()
        for group in INCLUDE:
            group_ds = base_ds.group_dataset(group)
            for element in group_ds.iterall():
                ds.add(element)
        for delete in DELETE:
            ds.pop(delete)
        return ds

    def get_tile(self, tile: Point, z: float, path: str) -> Image:
        """Get tile image at tile coordinate x, y.
        If frame is inside tile geometry but no tile exists in
        frame data (sparse) returns blank image

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate
        z: float
            Z coordinate
        path: str
            Optical path

        Returns
        ----------
        Image
            Tile image
        """
        try:
            frame_index = self._get_frame_index(tile, z, path)
            tile_frame = self._get_tile_frame(frame_index)
            image = Image.open(io.BytesIO(tile_frame))
        except WsiDicomSparse:
            image = self.blank_tile
        return self.crop_tile_to_level(tile, image)

    def get_encoded_tile(
        self,
        tile: Point,
        z: float,
        path: str,
        crop: bool = True
    ) -> bytes:
        """Get tile bytes at tile coordinate x, y
        If frame is inside tile geometry but no tile exists in
        frame data (sparse) returns encoded blank image.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate.
        z: float
            Z coordinate.
        path: str
            Optical path.
        crop: bool
            If the tile should be croped to image region.

        Returns
        ----------
        bytes
            Tile image as bytes
        """

        try:
            frame_index = self._get_frame_index(tile, z, path)
            tile_frame = self._get_tile_frame(frame_index)
        except WsiDicomSparse:
            tile_frame = self.blank_encoded_tile

        if not crop:
            return tile_frame
        return self.crop_encoded_tile_to_level(tile, tile_frame)

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
            File pointer, frame offset and frame lenght in number of bytes
        """
        frame_index = self._get_frame_index(tile, z, path)
        file = self._get_file(frame_index)
        return file.get_filepointer(frame_index)

    def close(self) -> None:
        """Close all files in the instance."""
        for file in self._files.values():
            file.close()

    def _validate_instance(
        self,
    ) -> Tuple[str, BaseUids, str]:
        """Check that no files in instance are duplicate, that all files in
        instance matches (uid, type and size).
        Raises WsiDicomMatchError otherwise.
        Returns the matching file uid.


        Returns
        ----------
        Tuple[str, BaseUids, str]
            Instance identifier uid, base uids and wsi type
        """
        WsiDicomFile.check_duplicate_file(self.files, self)
        base_file = self.files[0]
        for file in self.files[1:]:
            if not base_file.matches_instance(file):
                raise WsiDicomMatchError(
                    str(file.filepath), str(self)
                )
        return (
            base_file.uids.identifier,
            base_file.uids.base,
            base_file.wsi_type
        )

    @staticmethod
    def _filter_files(
        files: List[WsiDicomFile],
        series_uids: BaseUids,
        series_tile_size: Size = None
    ) -> List[WsiDicomFile]:
        """Filter list of wsi dicom files to only include matching uids and
        tile size if defined.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Wsi files to filter
        series_uids: Uids
            Uids to check against
        series_tile_size: Size
            Tile size to check against

        Returns
        ----------
        List[WsiDicomFile]
            List of matching wsi dicom files
        """
        valid_files: List[WsiDicomFile] = []

        for file in files:
            if file.matches_series(series_uids, series_tile_size):
                valid_files.append(file)
            else:
                file.close()

        return valid_files

    @classmethod
    def _group_files(
        cls,
        files: List[WsiDicomFile]
    ) -> Dict[str, List[WsiDicomFile]]:
        """Return files grouped by instance identifier (instances).

        Parameters
        ----------
        files: List[WsiDicomFile]
            Files to group into instances

        Returns
        ----------
        Dict[str, List[WsiDicomFile]]
            Files grouped by instance, with instance identifier as key.
        """
        grouped_files: Dict[str, List[WsiDicomFile]] = {}
        for file in files:
            try:
                grouped_files[file.uids.identifier].append(file)
            except KeyError:
                grouped_files[file.uids.identifier] = [file]
        return grouped_files

    @staticmethod
    def _create_tileindex(
        files: Dict[int, WsiDicomFile]
    ) -> TileIndex:
        """Return a tile index created from files. Add optical paths to optical
        manager.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
            Files to add.

        Returns
        ----------
        TileIndex
            Created tile index
        """
        base_file = files[0]
        if(base_file.tile_type == 'TILED_FULL'):
            return FullTileIndex(files)
        return SparseTileIndex(files)

    @staticmethod
    def _image_settings(
        transfer_syntax: pydicom.uid
    ) -> Tuple[str, Dict[str, int]]:
        """Return image format and options for creating encoded tiles as in the
        used transfer syntax.

        Parameters
        ----------
        transfer_syntax: pydicom.uid
            Transfer syntax to match image format and options to

        Returns
        ----------
        tuple[str, dict[str, int]]
            image format and image options

        """
        if(transfer_syntax == pydicom.uid.JPEGBaseline8Bit):
            image_format = 'jpeg'
            image_options = {'quality': 95}
        elif(transfer_syntax == pydicom.uid.JPEG2000):
            image_format = 'jpeg2000'
            image_options = {'irreversible': True}
        elif(transfer_syntax == pydicom.uid.JPEG2000Lossless):
            image_format = 'jpeg2000'
            image_options = {'irreversible': False}
        else:
            raise NotImplementedError(
                "Only supports jpeg and jpeg2000"
            )
        return (image_format, image_options)

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
        if not self.tiles.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBondsError(
                f"Tile region {tile_region}",
                f"plane {self.tiles.plane_size}"
            )
        frame_index = self.tiles.get_frame_index(tile, z, path)
        return frame_index

    def is_sparse(self, tile: Point, z: float, path: str) -> bool:
        try:
            self.tiles.get_frame_index(tile, z, path)
            return False
        except WsiDicomSparse:
            return True


class WsiDicomStack(metaclass=ABCMeta):
    def __init__(
        self,
        instances: List[WsiInstance]
    ):
        """Represents a stack of instances having the same size,
        but possibly different z coordinate and/or optical path.
        Instances should match in the common uids, wsi type, and tile size.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to build the stack
        """
        self._instances = {  # key is identifier (str)
            instance.identifier: instance for instance in instances
        }
        self._validate_stack()

        base_instance = instances[0]
        self._wsi_type = base_instance.wsi_type
        self._uids = base_instance.uids

        self._size = base_instance.size
        self._pixel_spacing = base_instance.pixel_spacing
        self._default_instance_uid: str = base_instance.identifier

    def __getitem__(self, index) -> WsiInstance:
        return self.instances[index]

    @property
    def uids(self) -> BaseUids:
        """Return uids"""
        return self._uids

    @property
    def wsi_type(self) -> str:
        """Return wsi type"""
        return self._wsi_type

    @property
    def size(self) -> Size:
        """Return image size in pixels"""
        return self._size

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um/pixel"""
        return self.pixel_spacing*1000.0

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm/pixel"""
        return self._pixel_spacing

    @property
    def instances(self) -> Dict[str, WsiInstance]:
        """Return contained instances"""
        return self._instances

    @property
    def default_instance(self) -> WsiInstance:
        """Return default instance"""
        return self.instances[self._default_instance_uid]

    @property
    def files(self) -> List[WsiDicomFile]:
        """Return contained files"""
        instance_files = [
            instance.files for instance in self.instances.values()
        ]
        return [file for sublist in instance_files for file in sublist]

    @property
    def optical_paths(self) -> List[str]:
        return list({
            path
            for instance in self.instances.values()
            for path in instance.optical_paths
        })

    @property
    def focal_planes(self) -> List[float]:
        return list({
            focal_plane
            for innstance in self.instances.values()
            for focal_plane in innstance.focal_planes
        })

    @classmethod
    def open(
        cls,
        instances: List[WsiInstance],
    ) -> List['WsiDicomStack']:
        """Return list of stacks created from wsi files.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Files to create stacks from.

        Returns
        ----------
        List[WsiDicomStack]
            List of created stacks.

        """
        stacks: List[WsiDicomStack] = []

        instances_grouped_by_stack = cls._group_instances(instances)

        for instance_group in instances_grouped_by_stack.values():
            new_stack = WsiDicomStack(instance_group)
            stacks.append(new_stack)

        return stacks

    def matches(self, other_stack: 'WsiDicomStack') -> bool:
        """Check if stack is valid (Uids and tile size match).
        The common Uids should match for all series.
        """
        return (
            other_stack.uids == self.uids and
            other_stack.wsi_type == self.wsi_type
        )

    def valid_pixels(self, region: Region) -> bool:
        """Check if pixel region is withing the size of the stack image size.

        Parameters
        ----------
        region: Region
            Pixel region to check

        Returns
        ----------
        bool
            True if pixel position and size is within image
        """
        # Return true if inside pixel plane.
        image_region = Region(Point(0, 0), self.size)
        return region.is_inside(image_region)

    def get_instance(
        self,
        z: float = None,
        path: str = None
    ) -> Tuple[WsiInstance, float, str]:
        """Search for instance fullfilling the parameters.
        The behavior when z and/or path is none could be made more
        clear.

        Parameters
        ----------
        z: float
            Z coordinate of the searched instance
        path: str
            Optical path of the searched instance

        Returns
        ----------
        WsiInstance, float, str
            The instance containing selected path and z coordinate,
            selected or default focal plane and optical path
        """
        if z is None and path is None:
            instance = self.default_instance
            z = instance.default_z
            path = instance.default_path
            return self.default_instance, z, path

        # Sort instances by number of focal planes (prefer simplest instance)
        sorted_instances = sorted(
            list(self._instances.values()),
            key=lambda i: len(i.focal_planes)
        )
        try:
            if z is None:
                # Select the instance with selected optical path
                instance = next(i for i in sorted_instances if
                                path in i.optical_paths)
            elif path is None:
                # Select the instance with selected z
                instance = next(i for i in sorted_instances
                                if z in i.focal_planes)
            else:
                # Select by both path and z
                instance = next(i for i in sorted_instances
                                if (z in i.focal_planes and
                                    path in i.optical_paths))
        except StopIteration:
            raise WsiDicomNotFoundError(
                f"Instance for path: {path}, z: {z}",
                "stack"
            )
        if z is None:
            z = instance.default_z
        if path is None:
            path = instance.default_path
        return instance, z, path

    def get_default_full(self) -> Image:
        """Read full image using default z coordinate and path.

        Returns
        ----------
        Image
            Full image of the stack
        """
        instance = self.default_instance
        z = instance.tiles.default_z
        path = instance.tiles.default_path
        region = Region(position=Point(x=0, y=0), size=self.size)
        image = self.get_region(region, z, path)
        return image

    def get_region(
        self,
        region: Region,
        z: float = None,
        path: str = None,
    ) -> Image:
        """Read region defined by pixels.

        Parameters
        ----------
        location: int, int
            Upper left corner of region in pixels
        size: int
            Size of region in pixels
        z: float
            Z coordinate, optional
        path: str
            optical path, optional

        Returns
        ----------
        Image
            Region as image
        """

        (instance, z, path) = self.get_instance(z, path)
        image = instance.stitch_tiles(region, path, z)
        return image

    def get_region_mm(
        self,
        region: RegionMm,
        z: float = None,
        path: str = None
    ) -> Image:
        """Read region defined by mm.

        Parameters
        ----------
        location: float, float
            Upper left corner of region in mm
        size: float
            Size of region in mm
        z: float
            Z coordinate, optional
        path:
            optical path, optional

        Returns
        ----------
        Image
            Region as image
        """
        pixel_region = self.mm_to_pixel(region)
        image = self.get_region(pixel_region, z, path)
        return image

    def get_tile(
        self,
        tile: Point,
        z: float = None,
        path: str = None
    ) -> Image:
        """Return tile at tile coordinate x, y as image.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate
        z: float
            Z coordinate
        path: str
            Optical path

        Returns
        ----------
        Image
            The tile as image
        """

        (instance, z, path) = self.get_instance(z, path)
        return instance.get_tile(tile, z, path)

    def get_encoded_tile(
        self,
        tile: Point,
        z: float = None,
        path: str = None,
        crop: bool = True
    ) -> bytes:
        """Return tile at tile coordinate x, y as bytes.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate
        z: float
            Z coordinate
        path: str
            Optical path
        crop: bool
            If the tile should be croped to image region.

        Returns
        ----------
        bytes
            The tile as bytes
        """
        (instance, z, path) = self.get_instance(z, path)
        return instance.get_encoded_tile(tile, z, path, crop)

    def mm_to_pixel(self, region: RegionMm) -> Region:
        """Convert region in mm to pixel region.

        Parameters
        ----------
        region: RegionMm
            Region in mm

        Returns
        ----------
        Region
            Region in pixels
        """
        pixel_region = Region(
            position=region.position // self.pixel_spacing,
            size=region.size // self.pixel_spacing
        )
        if not self.valid_pixels(pixel_region):
            raise WsiDicomOutOfBondsError(
                f"Region {region}", f"level size {self.size}"
            )
        return pixel_region

    def close(self) -> None:
        """Close all instances on the stack."""
        for instance in self._instances.values():
            instance.close()

    def _create_image_type_attribute(self) -> List[str]:
        value_1 = 'DERIVED'
        value_4 = 'RESAMPLED'
        if isinstance(self, WsiDicomLevel):
            if self.level == 0:
                value_1 = 'ORGINAL'
                value_4 = 'None'
        value_2 = 'PRIMARY'
        value_3 = self.wsi_type
        return [value_1, value_2, value_3, value_4]

    def _create_shared_functional_groups_sequence(self) -> DicomSequence:
        pixel_measure_item = Dataset()
        instance = self.default_instance
        pixel_measure_item.SliceThickness = instance.slice_thickness
        if instance.slice_spacing != 0:
            pixel_measure_item.SpacingBetweenSlices = instance.slice_spacing
        pixel_measure_item.PixelSpacing = [
            self.pixel_spacing.width,
            self.pixel_spacing.height
        ]
        pixel_measure_sequence = DicomSequence([pixel_measure_item])
        frame_type_item = Dataset()
        frame_type_item.FrameType = self._create_image_type_attribute()
        frame_type_sequence = DicomSequence([frame_type_item])
        sequence_item = Dataset()
        sequence_item.PixelMeasuresSequence = pixel_measure_sequence
        sequence_item.WholeSlideMicroscopyImageFrameTypeSequence = (
            frame_type_sequence
        )
        return DicomSequence([sequence_item])

    @staticmethod
    def write_preamble(fp: pydicom.filebase.DicomFileLike):
        """Writes file preamble to file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomFileLike
            Filepointer to file to write.
        """
        preamble = b'\x00' * 128
        fp.write(preamble)
        fp.write(b'DICM')

    def write_file_meta(self, fp: pydicom.filebase.DicomFileLike, uid: Uid):
        """Writes file meta dataset to file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomFileLike
            Filepointer to file to write.
        uid: Uid
            SOP instance uid to include in file.
        """
        WSI_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.77.1.6'
        meta_ds = pydicom.dataset.FileMetaDataset()
        meta_ds.TransferSyntaxUID = self.default_instance._transfer_syntax
        meta_ds.MediaStorageSOPInstanceUID = uid
        meta_ds.MediaStorageSOPClassUID = WSI_SOP_CLASS_UID
        pydicom.dataset.validate_file_meta(meta_ds)
        pydicom.filewriter.write_file_meta_info(fp, meta_ds)

    def write_base(
        self,
        fp: pydicom.filebase.DicomFileLike,
        optical: OpticalManager,
        uid: Uid,
        focal_planes: List[str],
        optical_paths: List[float]
    ):
        """Writes base dataset to file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomFileLike
            Filepointer to file to write.
        optical: OpticalManager
            Manager container optical paths.
        uid: Uid
            SOP instance uid to include in file.
        focal_planes: List[float]
            List of focal planes to include in file.
        optical_paths: List[str]
            List of optical paths to include in file.
        """
        if(isinstance(self.default_instance, WsiDicomInstance)):
            ds = self.default_instance.create_ds()
        else:
            ds = Dataset()
        ds.ImageType = self._create_image_type_attribute()
        ds.SOPInstanceUID = uid
        ds.DimensionOrganizationType = 'TILED_FULL'
        plane_size = self.default_instance.plane_size
        number_of_frames = (
            plane_size.width
            * plane_size.height
            * len(optical_paths)
            * len(focal_planes)
        )
        ds.NumberOfFrames = number_of_frames
        now = datetime.now()
        ds.ContentDate = datetime.date(now).strftime('%Y%m%d')
        ds.ContentTime = datetime.time(now).strftime('%H%M%S.%f')
        ds.TotalPixelMatrixFocalPlanes = len(focal_planes)
        ds.SharedFunctionalGroupsSequence = (
            self._create_shared_functional_groups_sequence()
        )

        ds = optical.insert_into_ds(ds)

        instance = self.default_instance

        ds.Rows = instance.tile_size.width
        ds.Columns = instance.tile_size.height
        ds.SamplesPerPixel = instance.samples_per_pixel
        ds.PhotometricInterpretation = instance.photometric_interpretation
        ds.ImagedVolumeWidth = instance.mm_size.width
        ds.ImagedVolumeHeight = instance.mm_size.height
        ds.ImagedVolumeDepth = instance.mm_depth
        ds.FocusMethod = instance.focus_method
        if instance.ext_depth_of_field:
            ds.ExtendedDepthOfField = 'YES'
        else:
            ds.ExtendedDepthOfField = 'NO'
        if instance.ext_depth_of_field_planes is not None:
            ds.NumberOfFocalPlanes = instance.ext_depth_of_field_planes
        if instance.ext_depth_of_field_plane_distance is not None:
            ds.DistanceBetweenFocalPlanes = (
                instance.ext_depth_of_field_plane_distance
            )
        ds.TotalPixelMatrixColumns = self.size.width
        ds.TotalPixelMatrixRows = self.size.height

        pydicom.filewriter.write_dataset(fp, ds)

    def write_pixel_data(
        self,
        fp: pydicom.filebase.DicomFileLike,
        focal_planes: List[float],
        optical_paths: List[str]
    ):
        """Writes pixel data to file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomFileLike
            Filepointer to file to write.
        focal_planes: List[float]
            List of focal planes to include in file.
        optical_paths: List[str]
            List of optical paths to include in file.
        """
        plane_size = self.default_instance.plane_size
        tile_geometry = Region(Point(0, 0), plane_size)
        # Generator for the tiles
        tiles = (
            self.get_encoded_tile(Point(x_tile, y_tile), z, path, False)
            for path in optical_paths
            for z in focal_planes
            for x_tile, y_tile in tile_geometry.iterate_all()
        )

        pixel_data_element = pydicom.dataset.DataElement(
            0x7FE00010,
            'OB',
            0,
            is_undefined_length=True
            )

        # Write pixel data tag
        fp.write_tag(pixel_data_element.tag)

        if not fp.is_implicit_VR:
            # Write pixel data VR (OB), two empty bytes (PS3.5 7.1.2)
            fp.write(bytes(pixel_data_element.VR, "iso8859"))
            fp.write_US(0)
        # Write unspecific length
        fp.write_UL(0xFFFFFFFF)

        # Write item tag and (empty) length for BOT
        fp.write_tag(pydicom.tag.ItemTag)
        fp.write_UL(0)

        # itemize and and write the tiles
        for tile in tiles:
            for frame in pydicom.encaps.itemize_frame(tile, 1):
                fp.write(frame)

        # end sequence
        fp.write_tag(pydicom.tag.SequenceDelimiterTag)
        fp.write_UL(0)

    @staticmethod
    def create_filepointer(path: Path) -> pydicom.filebase.DicomFileLike:
        """Return a dicom filepointer.

        Parameters
        ----------
        path: Path
            Path to filepointer.
        Returns
        ----------
        pydicom.filebase.DicomFileLike
            Created filepointer.
        """
        fp = pydicom.filebase.DicomFile(path, mode='wb')
        fp.is_little_endian = True
        fp.is_implicit_VR = False
        return fp

    def save(
        self,
        path: Path,
        optical: OpticalManager,
        focal_planes: List[float] = None,
        optical_paths: List[str] = None
    ) -> None:
        """Writes stack to to file. File is written as TILED_FULL.
        Writing of optical path sequence is not yet implemented.

        Parameters
        ----------
        path: Path
            Path to directory to write to.
        optical: OpticalManager
            Manager containing optical paths.
        optical_paths: List[str]
            List of optical paths to include in file.
        focal_planes: List[float]
            List of focal planes to include in file.
        """
        uid = pydicom.uid.generate_uid()
        file_path = os.path.join(path, uid+'.dcm')

        fp = self.create_filepointer(file_path)
        self.write_preamble(fp)
        self.write_file_meta(fp, uid)

        if optical_paths is not None:
            if not all(path in self.optical_paths for path in optical_paths):
                raise ValueError("Requested optical paths not found")
        else:
            optical_paths = self.optical_paths

        if focal_planes is not None:
            if not all(plane in self.focal_planes for plane in focal_planes):
                raise ValueError("Requested focal planes not found")
        else:
            focal_planes = self.focal_planes

        # Here we should check that the instances in the stack are possible
        # to put in the same file. Start by getting the unique instances
        # needed for the optical paths and focal planes
        # instances = (optical_paths, focal_planes)
        # Then group these, with the required attributes that needs to match
        # (in the dicom file) as key. Attributes that needs to match:
        # photometric_interpretation
        # transfer_syntax
        # samples_per_pixel
        # focal method?

        self.write_base(fp, optical, uid, focal_planes, optical_paths)
        self.write_pixel_data(fp, focal_planes, optical_paths)

        # close the file
        fp.close()

    def _validate_stack(self):
        """Check that no file or instance in stack is duplicate, instances in
        stack matches and that the optical manager matches by base uid.
        Raises WsiDicomMatchError otherwise.
        """
        instances = list(self.instances.values())
        if(isinstance(instances[0], WsiDicomInstance)):
            WsiDicomFile.check_duplicate_file(self.files, self)
            base_instance = instances[0]
            for instance in instances[1:]:
                if not base_instance.matches(instance):
                    raise WsiDicomMatchError(str(instance), str(self))
        WsiInstance.check_duplicate_instance(instances, self)

    @classmethod
    def _group_instances(
        cls,
        instances: List[WsiInstance]
    ) -> Dict[Size, List[WsiInstance]]:
        """Return instances grouped by image size (stacks).

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to group into stacks.

        Returns
        ----------
        Dict[Size, List[WsiInstance]]:
            Instances grouped by size, with size as key.

        """
        grouped_instances: Dict[Size, List[WsiInstance]] = {}
        for instance in instances:
            try:
                grouped_instances[instance.size].append(instance)
            except KeyError:
                grouped_instances[instance.size] = [instance]
        return grouped_instances


class WsiDicomLevel(WsiDicomStack):
    def __init__(
        self,
        instances: List[WsiInstance],
        base_pixel_spacing: SizeMm
    ):
        """Represents a level in the pyramid and contains one or more
        instances having the same level, pixel spacing, and size but possibly
        different focal planes and/or optical paths and present in
        different files.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to build the stack.
        base_pixel_spacing: SizeMm
            Pixel spacing of base level.
        """
        super().__init__(instances)
        self._level = self._assign_level(base_pixel_spacing)

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f'Level: {self.level}, size: {self.size} px, mpp: {self.mpp} um/px'
        )
        if depth is not None:
            depth -= 1
            if(depth < 0):
                return string
        string += dict_pretty_str(self.instances, indent, depth)
        return string

    @property
    def pyramid(self) -> str:
        """Return string representatin of the level"""
        return (
            f'Level [{self.level}]'
            f' tiles: {self.default_instance.tiles.plane_size},'
            f' size: {self.size}, mpp: {self.mpp} um/px'
        )

    @property
    def level(self) -> int:
        """Return pyramid level"""
        return self._level

    @classmethod
    def open_levels(
        cls,
        instances: List[WsiInstance],
    ) -> List['WsiDicomLevel']:
        """Return list of level stacks created wsi files.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Files to create stacks from

        Returns
        ----------
        List[WsiDicomLevel]
            List of created level stacks.

        """
        levels: List[WsiDicomLevel] = []
        instances_grouped_by_stack = cls._group_instances(instances)
        largest_size = max(instances_grouped_by_stack.keys())
        base_group = instances_grouped_by_stack[largest_size]
        base_pixel_spacing = base_group[0].pixel_spacing
        for level_group in instances_grouped_by_stack.values():
            new_level = WsiDicomLevel(level_group, base_pixel_spacing)
            levels.append(new_level)

        return levels

    def matches(self, other_stack: 'WsiDicomStack') -> bool:
        """Check if stack is valid (Uids and tile size match).
        The common Uids should match for all series. For level series the tile
        size should also match. It is assumed that the instances in the stacks
        are matching each other.
        """
        other_instance = other_stack.default_instance
        this_instance = self.default_instance
        return (
            other_stack.uids == self.uids and
            other_instance.tile_size == this_instance.tile_size
        )

    def get_highest_level(self) -> int:
        """Return highest deep zoom scale that can be produced
        from the image in the level.

        Returns
        ----------
        int
            Relative level where the pixel size becomes 1x1
        """
        return math.ceil(math.log2(max(self.size.width, self.size.height)))

    def get_scaled_tile(
        self,
        tile: Point,
        level: int,
        z: float = None,
        path: str = None
    ) -> Image:
        """Return tile in another level by scaling a region.
        If the tile is an edge tile, the resulting tile is croped
        to remove part outside of the image (as defiend by level size).

        Parameters
        ----------
        tile: Point
            Non scaled tile coordinate
        level: int
            Level to scale from
        z: float
            Z coordinate
        path: str
            Optical path

        Returns
        ----------
        Image
            A tile image
        """
        scale = self.calculate_scale(level)
        (instance, z, path) = self.get_instance(z, path)
        scaled_region = Region.from_tile(tile, instance.tile_size) * scale
        cropped_region = instance.crop_to_level_size(scaled_region)
        if not self.valid_pixels(cropped_region):
            raise WsiDicomOutOfBondsError(
                f"Region {cropped_region}", f"level size {self.size}"
            )
        image = self.get_region(cropped_region, z, path)
        tile_size = cropped_region.size/scale
        image = image.resize(
            tile_size.to_tuple(),
            resample=Image.BILINEAR
        )
        return image

    def get_scaled_encoded_tile(
        self,
        tile: Point,
        scale: int,
        z: float = None,
        path: str = None
    ) -> bytes:
        """Return encoded tile in another level by scaling a region.

        Parameters
        ----------
        tile: Point
            Non scaled tile coordinate
        level: int
           Level to scale from
        z: float
            Z coordinate
        path: str
            Optical path

        Returns
        ----------
        bytes
            A transfer syntax encoded tile
        """
        image = self.get_scaled_tile(tile, scale, z, path)
        (instance, z, path) = self.get_instance(z, path)
        return instance.encode(image)

    def calculate_scale(self, level_to: int) -> int:
        """Return scaling factor to given level.

        Parameters
        ----------
        level_to -- index of level to scale to

        Returns
        ----------
        int
            Scaling factor between this level and given level
        """
        return int(2 ** (level_to - self.level))

    def _assign_level(self, base_pixel_spacing: SizeMm) -> int:
        """Return (2^level scale factor) based on pixel spacing.
        Will round to closest integer. Raises NotImplementedError if level is
        to far from integer.

        Parameters
        ----------
        base_pixel_spacing: SizeMm
            The pixel spacing of the base lavel

        Returns
        ----------
        int
            The pyramid order of the level
        """
        float_level = math.log2(
            self.pixel_spacing.width/base_pixel_spacing.width
        )
        level = int(round(float_level))
        TOLERANCE = 1e-2
        if not math.isclose(float_level, level, rel_tol=TOLERANCE):
            raise NotImplementedError("Levels needs to be integer")
        return level


class WsiDicomSeries(metaclass=ABCMeta):
    wsi_type: str

    def __init__(self, stacks: List[WsiDicomStack]):
        """Holds a series of stacks of same image flavor

        Parameters
        ----------
        stacks: List[WsiDicomStack]
            List of stacks to include in the series
        """
        self._stacks: List[WsiDicomStack] = stacks

        if len(self.stacks) != 0 and self.stacks[0].uids is not None:
            self._uids = self._validate_series(self.stacks)
        else:
            self._uids = None

    def __getitem__(self, index: int) -> WsiDicomStack:
        """Get stack by index.

        Parameters
        ----------
        index: int
            Index in series to get

        Returns
        ----------
        WsiDicomStack
            The stack at index in the series
        """
        return self.stacks[index]

    @property
    def stacks(self) -> List[WsiDicomStack]:
        """Return contained stacks"""
        return self._stacks

    @property
    def uids(self) -> Optional[BaseUids]:
        """Return uids"""
        return self._uids

    @property
    def mpps(self) -> List[SizeMm]:
        """Return contained mpp (um/px)"""
        return [stack.mpp for stack in self.stacks]

    @property
    def files(self) -> List[WsiDicomFile]:
        """Return contained files"""
        series_files = [series.files for series in self.stacks]
        return [file for sublist in series_files for file in sublist]

    @property
    def instances(self) -> List[WsiInstance]:
        """Return contained instances"""
        series_instances = [
            series.instances.values() for series in self.stacks
        ]
        return [
            instance for sublist in series_instances for instance in sublist
        ]

    def _validate_series(
            self,
            stacks: Union[List[WsiDicomStack], List[WsiDicomLevel]]
    ) -> Optional[BaseUids]:
        """Check that no files or instances in series is duplicate, all stacks
        in series matches and that the optical manager matches by base uid.
        Raises WsiDicomMatchError otherwise.
        Returns the matching base uid. If list of stacks is empty, return None.

        Parameters
        ----------
        stacks: Union[List[WsiDicomStack], List[WsiDicomLevel]]
            List of stacks to check

        Returns
        ----------
        Optional[BaseUids]:
            Matching uids
        """
        WsiDicomFile.check_duplicate_file(self.files, self)
        WsiDicomInstance.check_duplicate_instance(self.instances, self)

        try:
            base_stack = stacks[0]
            if base_stack.wsi_type != self.wsi_type:
                raise WsiDicomMatchError(
                    str(base_stack), str(self)
                )
            for stack in stacks[1:]:
                if not stack.matches(base_stack):
                    raise WsiDicomMatchError(
                        str(stack), str(self)
                    )
            return base_stack.uids
        except IndexError:
            return None

    def close(self) -> None:
        """Close all stacks in the series."""
        for stack in self.stacks:
            stack.close()

    def save(
        self,
        path: Path,
        optical: OpticalManager,
        focal_planes: List[float] = None,
        optical_paths: List[str] = None,
    ) -> None:
        """Saves series to files in path. Optionaly only included selected
        focal planes, and/or optical paths.

        Parameters
        ----------
        path: Path
            Folder path to save files to.
        optical: OpticalManager
            Optical manager containing optical paths for levels.
        focal_planes: List[float]
            Focal planes to save
        optical_paths: List[str]
            Optical paths to save.
        """

        for stack in self.stacks:
            stack.save(path, optical, focal_planes, optical_paths)


class WsiDicomLabels(WsiDicomSeries):
    wsi_type = 'LABEL'

    @classmethod
    def open(
        cls,
        instances: List[WsiInstance]
    ) -> 'WsiDicomLabels':
        """Return label series created from wsi files.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to create levels from.

        Returns
        ----------
        WsiDicomLabels
            Created label series
        """
        labels = WsiDicomStack.open(instances)
        return WsiDicomLabels(labels)


class WsiDicomOverviews(WsiDicomSeries):
    wsi_type = 'OVERVIEW'

    @classmethod
    def open(
        cls,
        instances: List[WsiInstance]
    ) -> 'WsiDicomOverviews':
        """Return overview series created from wsi files.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to create levels from.

        Returns
        ----------
        WsiDicomOverviews
            Created overview series
        """
        overviews = WsiDicomStack.open(instances)
        return WsiDicomOverviews(overviews)


class WsiDicomLevels(WsiDicomSeries):
    wsi_type = 'VOLUME'

    def __init__(self, levels: List[WsiDicomLevel]):
        """Holds a stack of levels.

        Parameters
        ----------
        stacks: List[WsiDicomLevel]
            List of level stacks to include in series
        """
        self._levels = OrderedDict(
            (level.level, level)
            for level in sorted(levels, key=lambda level: level.level)
        )
        if len(self.stacks) != 0 and self.stacks[0].uids is not None:
            self._uids = self._validate_series(self.stacks)
        else:
            self._uids = None

    @property
    def pyramid(self) -> str:
        """Return string representation of pyramid"""
        return (
            'Pyramid levels in file:\n'
            + '\n'.join(
                [str_indent(2) + level.pyramid
                 for level in self._levels.values()]
            )
        )

    @property
    def stacks(self) -> List[WsiDicomStack]:
        """Return contained stacks"""
        return list(self._levels.values())

    @property
    def levels(self) -> List[int]:
        """Return contained levels"""
        return list(self._levels.keys())

    @property
    def highest_level(self) -> int:
        """Return highest valid pyramid level (which results in a 1x1 image)"""
        return self.base_level.get_highest_level()

    @property
    def base_level(self) -> WsiDicomLevel:
        """Return the base level of the pyramid"""
        return self._levels[0]

    @classmethod
    def open(
        cls,
        instances: List[WsiInstance]
    ) -> 'WsiDicomLevels':
        """Return level series created from wsi instances.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to create levels from.

        Returns
        ----------
        WsiDicomLevels
            Created level series
        """
        levels = WsiDicomLevel.open_levels(instances)
        return WsiDicomLevels(levels)

    def save(
        self,
        path: Path,
        optical: OpticalManager,
        focal_planes: List[float] = None,
        optical_paths: List[str] = None,
        levels: List[int] = None
    ) -> None:
        """Saves levels to files in path. Optionaly only included selected
        levels, focal planes, and/or optical paths.

        Parameters
        ----------
        path: Path
            Folder path to save files to.
        optical: OpticalManager
            Optical manager containing optical paths for levels.
        focal_planes: List[float]
            Focal planes to save
        optical_paths: List[str]
            Optical paths to save.
        levels: List[int]
            Levels to save.
        """
        if levels is None:
            levels = self.levels

        levels_to_save = [
            level_stack for (level, level_stack) in self._levels.items()
            if level in levels
        ]

        for level in levels_to_save:
            level.save(path, optical, focal_planes, optical_paths)

    def valid_level(self, level: int) -> bool:
        """Check that given level is less or equal to the highest level
        (1x1 pixel level).

        Parameters
        ----------
        level: int
            The level to check

        Returns
        ----------
        bool
            True if level is valid
        """
        return level <= self.highest_level

    def get_level(self, level: int) -> WsiDicomLevel:
        """Return wsi level.

        Parameters
        ----------
        level: int
            The level of the wsi level to return

        Returns
        ----------
        WsiDicomLevel
            The searched level
        """
        try:
            return self._levels[level]
        except KeyError:
            raise WsiDicomNotFoundError(
                f"Level of {level}", "level series"
            )

    def get_closest_by_level(self, level: int) -> WsiDicomLevel:
        """Search for level that is closest to and smaller than the given
        level.

        Parameters
        ----------
        level: int
            The level to search for

        Returns
        ----------
        WsiDicomLevel
            The level closest to searched level
        """
        if not self.valid_level(level):
            raise WsiDicomOutOfBondsError(
                f"Level {level}", f"maximum level {self.highest_level}"
            )
        closest_level = 0
        closest = None
        for wsi_level in self._levels.values():
            if((level >= wsi_level.level) and
               (closest_level <= wsi_level.level)):
                closest_level = wsi_level.level
                closest = wsi_level
        if closest is None:
            raise WsiDicomNotFoundError(
                f"Level for {level}", "level series"
            )
        return closest

    def get_closest_by_size(self, size: Size) -> WsiDicomLevel:
        """Search for level that by size is closest to and larger than the
        given size.

        Parameters
        ----------
        size: Size
            The size to search for

        Returns
        ----------
        WsiDicomLevel
            The level with size closest to searched size
        """
        closest_size = self.stacks[0].size
        closest = None
        for wsi_level in self._levels.values():
            if((size.width <= wsi_level.size.width) and
               (wsi_level.size.width <= closest_size.width)):
                closest_size = wsi_level.size
                closest = wsi_level
        if closest is None:
            raise WsiDicomNotFoundError(
                f"Level for size {size}", "level series"
            )
        return closest

    def get_closest_by_pixel_spacing(
        self,
        pixel_spacing: SizeMm
    ) -> WsiDicomLevel:
        """Search for level that by pixel spacing is closest to and smaller
        than the given pixel spacing. Only the spacing in x-axis is used.

        Parameters
        ----------
        pixel_spacing: SizeMm
            Pixel spacing to search for

        Returns
        ----------
        WsiDicomLevel
            The level with pixel spacing closest to searched spacing
        """
        closest_pixel_spacing: float = 0
        closest = None
        for wsi_level in self._levels.values():
            if((pixel_spacing.width >= wsi_level.pixel_spacing.width) and
               (closest_pixel_spacing <= wsi_level.pixel_spacing.width)):
                closest_pixel_spacing = wsi_level.pixel_spacing.width
                closest = wsi_level
        if closest is None:
            raise WsiDicomNotFoundError(
                f"Level for pixel spacing {pixel_spacing}", "level series")
        return closest


class TileManager(metaclass=ABCMeta):
    def __init__(
        self,
        filepaths: List[Path],
        optical_paths: List[str],
        focal_planes: List[float]
    ):
        self.filepaths = filepaths
        self.optical_paths = optical_paths
        self.focal_planes = focal_planes

    @abstractmethod
    def level_instances(self) -> List[WsiGenericInstance]:
        raise NotImplementedError


class WsiDicom:
    def __init__(
        self,
        series: List[WsiDicomSeries],
        optical: OpticalManager
    ):
        """Holds wsi dicom levels, labels and overviews.

        Parameters
        ----------
        series: List[WsiDicomSeries]
            List of series (levels, labels, overviews).
        optical: OpticalManager
            Manager for optical paths.
        """

        self._levels: WsiDicomLevels = self._assign(series, 'levels')
        self._labels: WsiDicomLabels = self._assign(series, 'labels')
        self._overviews: WsiDicomOverviews = self._assign(series, 'overviews')

        if self.levels.uids is not None:
            self.uids = self._validate_collection(
                [self.levels, self.labels, self.overviews],
            )
        else:
            self.uids = None
        self.optical = optical
        base = self.levels.base_level.default_instance
        # self.ds = base.create_ds()
        self.__enter__()

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def __str__(self) -> str:
        return self.pretty_str()

    @property
    def levels(self) -> WsiDicomLevels:
        """Return contained levels"""
        if self._levels is not None:
            return self._levels
        raise WsiDicomNotFoundError("levels", str(self))

    @property
    def labels(self) -> WsiDicomLabels:
        """Return contained labels"""
        if self._labels is not None:
            return self._labels
        raise WsiDicomNotFoundError("labels", str(self))

    @property
    def overviews(self) -> WsiDicomOverviews:
        """Return contained overviews"""
        if self._overviews is not None:
            return self._overviews
        raise WsiDicomNotFoundError("overviews", str(self))

    @property
    def files(self) -> List[WsiDicomFile]:
        """Return contained files"""
        return self.levels.files + self.labels.files + self.overviews.files

    @property
    def instances(self) -> List[WsiInstance]:
        """Return contained instances"""
        return (
            self.levels.instances
            + self.labels.instances
            + self.overviews.instances
        )

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = self.__class__.__name__
        if depth is not None:
            depth -= 1
            if(depth < 0):
                return string
        return (
            string + ' of levels:\n'
            + list_pretty_str(self.levels.stacks, indent, depth, 0, 2)
        )

    @classmethod
    def open(cls, path: Union[str, List[str]]) -> 'WsiDicom':
        """Open valid wsi dicom files in path and return a WsiDicom object.
        Non-valid files are ignored.

        Parameters
        ----------
        paths: List[Path]
            Path to files to open

        Returns
        ----------
        WsiDicom
            Object created from wsi dicom files in path
        """
        filepaths = cls._get_filepaths(path)
        level_files: List[WsiDicomFile] = []
        label_files: List[WsiDicomFile] = []
        overview_files: List[WsiDicomFile] = []

        for filepath in cls._filter_paths(filepaths):
            dicom_file = WsiDicomFile(filepath)
            if(dicom_file.wsi_type == 'VOLUME'):
                level_files.append(dicom_file)
            elif(dicom_file.wsi_type == 'LABEL'):
                label_files.append(dicom_file)
            elif(dicom_file.wsi_type == 'OVERVIEW'):
                overview_files.append(dicom_file)
            else:
                dicom_file.close()

        base_file = cls._get_base_file(level_files)

        optical = OpticalManager.open(level_files+label_files+overview_files)
        level_instances = WsiDicomInstance.open(
            level_files,
            base_file.uids.base,
            base_file.tile_size
        )
        label_instances = WsiDicomInstance.open(
            label_files,
            base_file.uids.base
        )
        overview_instances = WsiDicomInstance.open(
            overview_files,
            base_file.uids.base
        )

        levels = WsiDicomLevels.open(level_instances)
        labels = WsiDicomLabels.open(label_instances)
        overviews = WsiDicomOverviews.open(overview_instances)

        return WsiDicom([levels, labels, overviews], optical=optical)

    @classmethod
    def open_generic(
        cls,
        level_instances: WsiGenericInstance,
        label_instanaces: WsiGenericInstance,
        overview_instances: WsiGenericInstance
    ) -> 'WsiDicom':
        optical = OpticalManager()
        levels = WsiDicomLevels.open(level_instances)
        labels = WsiDicomLabels(label_instanaces)
        overviews = WsiDicomOverviews(overview_instances)
        return cls([levels, labels, overviews], optical)

    @staticmethod
    def _get_filepaths(path: Union[str, List[str]]) -> List[Path]:
        """Return file paths to files in path.
        If path is folder, return list of folder files in path.
        If path is single file, return list of that path.
        If path is list, return list of paths that are files.
        Raises WsiDicomNotFoundError if no files found

        Parameters
        ----------
        path: Union[str, List[str]]
            Path to folder, file or list of files

        Returns
        ----------
        List[Path]
            List of found file paths
        """
        if isinstance(path, str):
            single_path = Path(path)
            if single_path.is_dir():
                return list(single_path.iterdir())
            elif single_path.is_file():
                return [single_path]
        elif isinstance(path, list):
            multiple_paths = [
                Path(file_path) for file_path in path
                if Path(file_path).is_file()
            ]
            if multiple_paths != []:
                return multiple_paths

        raise WsiDicomNotFoundError("No files found", str(path))

    @staticmethod
    def _get_base_file(
        files: List[WsiDicomFile]
    ) -> WsiDicomFile:
        """Return file with largest image (width) from list of files.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Wsi files to check.

        Returns
        ----------
        WsiDicomFile
            Base layer file.
        """
        base_size = Size(0, 0)
        base_file: WsiDicomFile
        for file in files:
            if file.image_size.width > base_size.width:
                base_file = file
                base_size = file.image_size
        return base_file

    @staticmethod
    def _filter_paths(filepaths: List[Path]) -> List[Path]:
        """Filter list of paths to only include valid dicom files.

        Parameters
        ----------
        filepaths: List[Path]
            Paths to filter

        Returns
        ----------
        List[Path]
            List of paths with dicom files
        """
        return [
            path for path in filepaths
            if path.is_file() and pydicom.misc.is_dicom(path)
        ]

    def _validate_collection(
        self,
        series: List[WsiDicomSeries]
    ) -> BaseUids:
        """Check that no files or instance in collection is duplicate, that all
        series and optical manager all have the same base uids.
        Raises WsiDicomMatchError otherwise.
        Returns the matching base uid.

        Parameters
        ----------
        series: List[WsiDicomSeries]
            List of series to check

        Returns
        ----------
        BaseUids
            Matching uids
        """
        WsiDicomFile.check_duplicate_file(self.files, self)
        WsiDicomInstance.check_duplicate_instance(self.instances, self)

        try:
            base_uids = next(
                item.uids for item in series if item.uids is not None
            )
        except StopIteration:
            raise WsiDicomNotFoundError("Valid series", "in collection")
        for item in series:
            if item.uids is not None and item.uids != base_uids:
                raise WsiDicomMatchError(str(item), str(self))
        return base_uids

    @staticmethod
    def _assign(series: List[WsiDicomSeries], assign_as: str):
        """Returns first series in list that matches assign as parameter.
        Returns None if none found.

        Parameters
        ----------
        series: List[WsiDicomSeries]
            List of series to check
        assign_as: str
            Type of series to get

        Returns
        ----------
        Optional[
            WsiDicomLevels,
            WsiDicomLabels,
            WsiDicomOverviews
        ]
            Series of assign as type
        """
        series_types = {
            'levels': WsiDicomLevels,
            'labels': WsiDicomLabels,
            'overviews': WsiDicomOverviews
        }
        return next(
            (item for item in series
             if type(item) == series_types[assign_as]), None
        )

    def save(
        self,
        path: Path,
        focal_planes: List[float] = None,
        optical_paths: List[str] = None,
        levels: List[int] = None,
    ) -> None:
        """Saves wsi to files in path. Optionaly only included selected
        levels, focal planes, and/or optical paths.

        Parameters
        ----------
        path: Path
            Folder path to save files to.
        focal_planes: List[float]
            Focal planes to save
        optical_paths: List[str]
            Optical paths to save.
        levels: List[int]
            Levels to save.
        """
        self.levels.save(
            path,
            self.optical,
            focal_planes,
            optical_paths,
            levels
        )
        series = [self.labels, self.overviews]
        for item in series:
            item.save(path, self.optical, focal_planes, optical_paths)

    def read_label(self, index: int = 0) -> Image:
        """Read label image of the whole slide. If several label
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int
            Index of the label image to read

        Returns
        ----------
        Image
            label as image
        """
        try:
            label = self.labels[index]
            return label.get_default_full()
        except IndexError:
            raise WsiDicomNotFoundError("label", "series")

    def read_overview(self, index: int = 0) -> Image:
        """Read overview image of the whole slide. If several overview
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int
            Index of the overview image to read

        Returns
        ----------
        Image
            Overview as image
        """
        try:
            overview = self.overviews[index]
            return overview.get_default_full()
        except IndexError:
            raise WsiDicomNotFoundError("overview", "series")

    def read_thumbnail(
        self,
        size: Tuple[int, int],
        z: float = None,
        path: str = None
    ) -> Image:
        """Read thumbnail image of the whole slide with dimensions
        no larger than given size.

        Parameters
        ----------
        size: int, int
            Upper size limit for thumbnail
        z: float
            Z coordinate, optional
        path:
            optical path, optional

        Returns
        ----------
        Image
            Thumbnail as image
        """
        thumbnail_size = Size.from_tuple(size)
        level = self.levels.get_closest_by_size(thumbnail_size)
        region = Region(position=Point(0, 0), size=level.size)
        image = level.get_region(region, z, path)
        image.thumbnail((size), resample=Image.BILINEAR)
        return image

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        z: float = None,
        path: str = None
    ) -> Image:
        """Read region defined by pixels.

        Parameters
        ----------
        location: int, int
            Upper left corner of region in pixels
        level: int
            Level in pyramid
        size: int
            Size of region in pixels
        z: float
            Z coordinate, optional
        path:
            optical path, optional

        Returns
        ----------
        Image
            Region as image
        """
        wsi_level = self.levels.get_closest_by_level(level)
        scale_factor = wsi_level.calculate_scale(level)
        scaled_region = Region(
            position=Point.from_tuple(location),
            size=Size.from_tuple(size)
        ) * scale_factor

        if not wsi_level.valid_pixels(scaled_region):
            raise WsiDicomOutOfBondsError(
                f"Region {scaled_region}", f"level size {wsi_level.size}"
            )
        image = wsi_level.get_region(scaled_region, z, path)
        if(scale_factor != 1):
            image = image.resize((size), resample=Image.BILINEAR)
        return image

    def read_region_mm(
        self,
        location: Tuple[float, float],
        level: int,
        size: Tuple[float, float],
        z: float = None,
        path: str = None
    ) -> Image:
        """Read image from region defined in mm.

        Parameters
        ----------
        location: float, float
            Upper left corner of region in mm
        level: int
            Level in pyramid
        size: float
            Size of region in mm
        z: float
            Z coordinate, optional
        path:
            optical path, optional

        Returns
        ----------
        Image
            Region as image
        """
        wsi_level = self.levels.get_closest_by_level(level)
        scale_factor = wsi_level.calculate_scale(level)
        region = RegionMm(
            position=PointMm.from_tuple(location),
            size=SizeMm.from_tuple(size)
        )
        image = wsi_level.get_region_mm(region, z, path)
        image_size = (
            Size(width=image.size[0], height=image.size[1]) // scale_factor
        )
        return image.resize(image_size.to_tuple(), resample=Image.BILINEAR)

    def read_region_mpp(
        self,
        location: Tuple[float, float],
        mpp: float,
        size: Tuple[float, float],
        z: float = None,
        path: str = None
    ) -> Image:
        """Read image from region defined in mm with set pixel spacing.

        Parameters
        ----------
        location: float, float
            Upper left corner of region in mm
        mpp: float
            Requested pixel spacing (um/mm)
        size: float
            Size of region in mm
        z: float
            Z coordinate, optional
        path:
            optical path, optional

        Returns
        ----------
        Image
            Region as image
        """
        pixel_spacing = mpp/1000.0
        wsi_level = self.levels.get_closest_by_pixel_spacing(
            SizeMm(pixel_spacing, pixel_spacing)
        )
        region = RegionMm(
            position=PointMm.from_tuple(location),
            size=SizeMm.from_tuple(size)
        )
        image = wsi_level.get_region_mm(region, z, path)
        image_size = SizeMm(width=size[0], height=size[1]) // pixel_spacing
        return image.resize(image_size.to_int_tuple(), resample=Image.BILINEAR)

    def read_tile(
        self,
        level: int,
        tile: Tuple[int, int],
        z: float = None,
        path: str = None
    ) -> Image:
        """Read tile in pyramid level as image.

        Parameters
        ----------
        level: int
            Pyramid level
        tile: int, int
            tile xy coordinate
        z: float
            Z coordinate, optional
        path:
            optical path, optional

        Returns
        ----------
        Image
            Tile as image
        """
        tile_point = Point.from_tuple(tile)
        try:
            wsi_level = self.levels.get_level(level)
            return wsi_level.get_tile(tile_point, z, path)
        except WsiDicomNotFoundError:
            # Scale from closest level
            wsi_level = self.levels.get_closest_by_level(level)
            return wsi_level.get_scaled_tile(
                tile_point,
                level,
                z,
                path)

    def read_encoded_tile(
        self,
        level: int,
        tile: Tuple[int, int],
        z: float = None,
        path: str = None
    ) -> bytes:
        """Read tile in pyramid level as encoded bytes. For non-existing levels
        the tile is scaled down from a lower level, using the similar encoding.

        Parameters
        ----------
        level: int
            Pyramid level
        tile: int, int
            tile xy coordinate
        z: float
            Z coordinate, optional
        path:
            optical path, optional

        Returns
        ----------
        bytes
            Tile in file encoding.
        """
        tile_point = Point.from_tuple(tile)
        try:
            wsi_level = self.levels.get_level(level)
            return wsi_level.get_encoded_tile(tile_point, z, path)
        except WsiDicomNotFoundError:
            # Scale from closest level
            wsi_level = self.levels.get_closest_by_level(level)
            return wsi_level.get_scaled_encoded_tile(
                tile_point,
                level,
                z,
                path
            )

    def get_instance(
        self,
        level: int,
        z: float = None,
        path: str = None
    ) -> Tuple[WsiInstance, float, str]:
        """Return instance fullfilling level, z and/or path.

        Parameters
        ----------
        level: int
            Pyramid level
        z: float
            Z coordinate, optional
        path:
            optical path, optional

        Returns
        ----------
        Tuple[WsiInstance, float, str]:
            Instance, selected z and path
        """
        wsi_level = self.levels.get_level(level)
        return wsi_level.get_instance(z, path)

    def close(self) -> None:
        """Close all files."""
        for series in [self.levels, self.overviews, self.labels]:
            series.close()
