import io
import math
import os
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import (Callable, Dict, List, Optional, OrderedDict, Tuple,
                    Union)

import pydicom
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.uid import UID as Uid

from .errors import (WsiDicomError, WsiDicomMatchError, WsiDicomNotFoundError,
                     WsiDicomOutOfBondsError, WsiDicomSparse,
                     WsiDicomUidDuplicateError)
from .file import WsiDicomFile, WsiDataset
from .geometry import Point, PointMm, Region, RegionMm, Size, SizeMm
from .image_data import DicomImageData, ImageData, Tiler
from .optical import OpticalManager
from .stringprinting import dict_pretty_str, list_pretty_str, str_indent
from .uid import BaseUids, WSI_SOP_CLASS_UID


class WsiInstance:
    def __init__(
        self,
        datasets: Union[WsiDataset, List[WsiDataset]],
        image_data: ImageData,
        transfer_syntax_uid: Uid
    ):
        if not isinstance(datasets, list):
            datasets = [datasets]
        self._datasets = datasets

        self._transfer_syntax = transfer_syntax_uid
        self._image_data = image_data

        self._identifier, self._uids = self._validate_instance(self.datasets)

        if self.ext_depth_of_field:
            if self.ext_depth_of_field_planes is None:
                raise WsiDicomError("Instance Missing NumberOfFocalPlanes")
            if self.ext_depth_of_field_plane_distance is None:
                raise WsiDicomError(
                    "Instance Missing DistanceBetweenFocalPlanes"
                )

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f"default z: {self.default_z} "
            f"default path: { self.default_path}"
        )
        if depth is not None:
            depth -= 1
            if(depth < 0):
                return string
        string += (
            ' ImageData ' + self._image_data.pretty_str(indent+1, depth)
        )
        return string

    @cached_property
    def wsi_type(self) -> str:
        """Return wsi type."""
        return self.dataset.get_supported_wsi_dicom_type(
            self._transfer_syntax
        )

    @property
    def blank_color(self) -> Tuple[int, int, int]:
        """Return RGB background color."""
        return self._get_blank_color(self.photometric_interpretation)

    @cached_property
    def image_mode(self) -> str:
        if(self.samples_per_pixel == 1):
            return "L"
        else:
            return "RGB"

    @property
    def datasets(self) -> List[WsiDataset]:
        return self._datasets

    @property
    def dataset(self) -> WsiDataset:
        return self.datasets[0]

    @property
    def size(self) -> Size:
        """Return image size in pixels."""
        return self._image_data.image_size

    @property
    def tile_size(self) -> Size:
        """Return tile size in pixels."""
        return self._image_data.tile_size

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um/pixel."""
        return self.pixel_spacing*1000.0

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm/pixel."""
        return self._image_data.pixel_spacing

    @property
    def mm_size(self) -> SizeMm:
        """Return slide size in mm."""
        return self.dataset.mm_size

    @property
    def mm_depth(self) -> float:
        """Return imaged depth in mm."""
        return self.dataset.mm_depth

    @property
    def slice_thickness(self) -> float:
        """Return slice thickness."""
        return self.dataset.slice_thickness

    @property
    def slice_spacing(self) -> float:
        """Return slice spacing."""
        return self.dataset.spacing_between_slices

    @property
    def focus_method(self) -> str:
        return self.dataset.focus_method

    @property
    def ext_depth_of_field(self) -> bool:
        return self.dataset.ext_depth_of_field

    @property
    def ext_depth_of_field_planes(self) -> Optional[int]:
        return self.dataset.ext_depth_of_field_planes

    @property
    def ext_depth_of_field_plane_distance(self) -> Optional[float]:
        return self.dataset.ext_depth_of_field_plane_distance

    @property
    def photometric_interpretation(self) -> str:
        """Return photometric interpretation."""
        return self.dataset.photophotometric_interpretation

    @property
    def samples_per_pixel(self) -> int:
        """Return samples per pixel (1 or 3)."""
        return self.dataset.samples_per_pixel

    @property
    def identifier(self) -> Uid:
        """Return identifier (instance uid for single file instance or
        concatenation uid for multiple file instance)."""
        return self._identifier

    @property
    def instance_number(self) -> int:
        return self.dataset.instance_number

    @property
    def default_z(self) -> float:
        return self._image_data.default_z

    @property
    def default_path(self) -> str:
        return self._image_data.default_path

    @property
    def focal_planes(self) -> List[float]:
        return self._image_data.focal_planes

    @property
    def optical_paths(self) -> List[str]:
        return self._image_data.optical_paths

    @property
    def tiled_size(self) -> Size:
        return self._image_data.tiled_size

    @property
    def datasets(self) -> List[WsiDataset]:
        return self._datasets

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
    ) -> List['WsiInstance']:
        """Create instances from Dicom files. Only files with matching series
        uid and tile size, if defined, are used. Other files are closed.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Files to create instances from.
        series_uids: BaseUids
            Uid to match against.
        series_tile_size: Size
            Tile size to match against (for level instances).

        Returns
        ----------
        List[WsiInstancece]
            List of created instances.
        """
        filtered_files = WsiDicomFile._filter_files(
            files,
            series_uids,
            series_tile_size
        )
        files_grouped_by_instance = WsiDicomFile._group_files(filtered_files)
        return [
            WsiInstance(
                [file.dataset for file in instance_files],
                DicomImageData(instance_files),
                instance_files[0].transfer_syntax
            )
            for instance_files in files_grouped_by_instance.values()
        ]

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
        image = Image.new(mode=self.image_mode, size=region.size.to_tuple())
        write_index = Point(x=0, y=0)
        tile = stitching_tiles.position
        for tile in stitching_tiles.iterate_all(include_end=True):
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
        start = pixel_region.start // self._image_data.tile_size
        end = pixel_region.end / self._image_data.tile_size - 1
        tile_region = Region.from_points(start, end)
        if not self._image_data.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBondsError(
                f"Tile region {tile_region}",
                f"tiled size {self._image_data.tiled_size}"
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

    def get_tile(self, tile: Point, z: float, path: str) -> Image:
        """Get tile image at tile coordinate x, y.
        If frame is inside tile geometry but no tile exists in
        frame data (sparse) returns blank image.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        ----------
        Image
            Tile image.
        """
        try:
            tile_frame = self._image_data.get_tile(tile, z, path)
            image = Image.open(io.BytesIO(tile_frame))
        except WsiDicomSparse:
            image = self.blank_tile
        return self.crop_tile_to_level(tile, image)

    def get_encoded_tile(
        self,
        tile: Point,
        z: float,
        path: str,
        crop: bool
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
            Tile image as bytes.
        """
        try:
            tile_frame = self._image_data.get_tile(tile, z, path)
        except WsiDicomSparse:
            tile_frame = self.blank_encoded_tile
        if not crop:
            return tile_frame
        return self.crop_encoded_tile_to_level(tile, tile_frame)

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

    def _validate_instance(
        self,
        datasets: List[WsiDataset]
    ) -> Tuple[str, BaseUids, str]:
        """Check that no files in instance are duplicate, that all files in
        instance matches (uid, type and size).
        Raises WsiDicomMatchError otherwise.
        Returns the matching file uid.


        Returns
        ----------
        Tuple[str, BaseUids, str]
            Instance identifier uid and base uids
        """
        WsiDataset.check_duplicate_dataset(datasets, self)

        base_dataset = datasets[0]
        for dataset in datasets[1:]:
            if not base_dataset.matches_instance(dataset):
                raise WsiDicomError()
        return (
            base_dataset.uids.identifier,
            base_dataset.uids.base,
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

    def save(self, path: Path) -> None:
        """Write instance to file. File is written as TILED_FULL.

        Parameters
        ----------
        path: Path
            Path to directory to write to.
        """

        file_path = os.path.join(path, self.dataset.instance_uid+'.dcm')

        fp = self._create_filepointer(file_path)
        self._write_preamble(fp)
        self._write_file_meta(fp, self.dataset.instance_uid)
        self._write_base(fp, self.dataset)
        self._write_pixel_data(fp)

        # close the file
        fp.close()
        print(f"Wrote file {file_path}")

    @staticmethod
    def _write_preamble(fp: pydicom.filebase.DicomFileLike):
        """Writes file preamble to file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomFileLike
            Filepointer to file to write.
        """
        preamble = b'\x00' * 128
        fp.write(preamble)
        fp.write(b'DICM')

    def _write_file_meta(self, fp: pydicom.filebase.DicomFileLike, uid: Uid):
        """Writes file meta dataset to file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomFileLike
            Filepointer to file to write.
        uid: Uid
            SOP instance uid to include in file.
        """
        meta_ds = pydicom.dataset.FileMetaDataset()
        meta_ds.TransferSyntaxUID = self._transfer_syntax
        meta_ds.MediaStorageSOPInstanceUID = uid
        meta_ds.MediaStorageSOPClassUID = WSI_SOP_CLASS_UID
        pydicom.dataset.validate_file_meta(meta_ds)
        pydicom.filewriter.write_file_meta_info(fp, meta_ds)

    def _write_base(
        self,
        fp: pydicom.filebase.DicomFileLike,
        dataset: WsiDataset
    ) -> None:
        """Writes base dataset to file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomFileLike
            Filepointer to file to write.
        dataset: WsiDataset

        """
        dataset.DimensionOrganizationType = 'TILED_FULL'
        now = datetime.now()
        dataset.ContentDate = datetime.date(now).strftime('%Y%m%d')
        dataset.ContentTime = datetime.time(now).strftime('%H%M%S.%f')
        dataset.NumberOfFrames = (
            self.tiled_size.width * self.tiled_size.height
        )
        pydicom.filewriter.write_dataset(fp, dataset)

    def _write_pixel_data(
        self,
        fp: pydicom.filebase.DicomFileLike
    ):
        """Writes pixel data to file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomFileLike
            Filepointer to file to write.
        focal_planes: List[float]
        """
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

        tile_geometry = Region(Point(0, 0), self.tiled_size)
        # Generator for the tiles
        tile_jobs = (
            self._image_data.get_encoded_tiles(tile_geometry.iterate_all())
        )
        # itemize and and write the tiles
        for tile_job in tile_jobs:
            for tile in tile_job:
                for frame in pydicom.encaps.itemize_frame(tile, 1):
                    fp.write(frame)

        # This method tests faster, but also writes all data at once
        # (takes a lot of memory)
        # fp.write(
        #     self.tiler.get_encapsulated_tiles(tile_geometry.iterate_all())
        # )

        # end sequence
        fp.write_tag(pydicom.tag.SequenceDelimiterTag)
        fp.write_UL(0)

    @staticmethod
    def _create_filepointer(path: Path) -> pydicom.filebase.DicomFile:
        """Return a dicom filepointer.

        Parameters
        ----------
        path: Path
            Path to filepointer.
        Returns
        ----------
        pydicom.filebase.DicomFile
            Created filepointer.
        """
        fp = pydicom.filebase.DicomFile(path, mode='wb')
        fp.is_little_endian = True
        fp.is_implicit_VR = False
        return fp

    @cached_property
    def blank_tile(self) -> Image:
        """Return background tile."""
        return self._create_blank_tile()

    @cached_property
    def blank_encoded_tile(self) -> bytes:
        """Return encoded background tile."""
        return self.encode(self.blank_tile)

    def matches(self, other_instance: 'WsiInstance') -> bool:
        """Return true if other instance is of the same group as self.

        Parameters
        ----------
        other_instance: WsiInstance
            Instance to check.

        Returns
        ----------
        bool
            True if instanes are of same group.

        """
        return (
            self.uids == other_instance.uids and
            self.size == other_instance.size and
            self.tile_size == other_instance.tile_size and
            self.wsi_type == other_instance.wsi_type
        )

    def is_sparse(self, tile: Point, z: float, path: str) -> bool:
        try:
            self._image_data.get_tile(tile, z, path)
            return False
        except WsiDicomSparse:
            return True

    def _create_blank_tile(self) -> Image:
        """Create blank tile for instance.

        Returns
        ----------
        Image
            Blank tile image
        """
        return Image.new(
            mode=self.image_mode,
            size=self.tile_size.to_tuple(),
            color=self.blank_color[:self.samples_per_pixel]
        )

    def close(self) -> None:
        self._image_data.close()


class WsiDicomGroup:
    def __init__(
        self,
        instances: List[WsiInstance]
    ):
        """Represents a group of instances having the same size,
        but possibly different z coordinate and/or optical path.
        Instances should match in the common uids, wsi type, and tile size.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to build the group.
        """
        self._instances = {  # key is identifier (str)
            instance.identifier: instance for instance in instances
        }
        self._validate_group()

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
    def datasets(self) -> List[Dataset]:
        """Return contained datasets."""
        instance_datasets = [
            instance.datasets for instance in self.instances.values()
        ]
        return [
            dataset for sublist in instance_datasets for dataset in sublist
        ]

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
    ) -> List['WsiDicomGroup']:
        """Return list of groups created from wsi instances.

        Parameters
        ----------
        files: List[WsiInstance]
            Instances to create groups from.

        Returns
        ----------
        List[WsiDicomGroup]
            List of created groups.

        """
        groups: List[WsiDicomGroup] = []

        grouped_instances = cls._group_instances(instances)

        for group in grouped_instances.values():
            groups.append(WsiDicomGroup(group))

        return groups

    def matches(self, other_group: 'WsiDicomGroup') -> bool:
        """Check if group is valid (Uids and tile size match).
        The common Uids should match for all series.
        """
        return (
            other_group.uids == self.uids and
            other_group.wsi_type == self.wsi_type
        )

    def valid_pixels(self, region: Region) -> bool:
        """Check if pixel region is withing the size of the group image size.

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
                f"Instance for path: {path}, z: {z}", "group"
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
            Full image of the group.
        """
        instance = self.default_instance
        z = instance.default_z
        path = instance.default_path
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
        """Close all instances on the group."""
        for instance in self._instances.values():
            instance.close()

    def _validate_group(self):
        """Check that no file or instance in group is duplicate, instances in
        group matches and that the optical manager matches by base uid.
        Raises WsiDicomMatchError otherwise.
        """
        WsiDataset.check_duplicate_dataset(self.datasets, self)
        instances = list(self.instances.values())
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
        """Return instances grouped by image size.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to group by image size.

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


class WsiDicomLevel(WsiDicomGroup):
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
            Instances to build the level.
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
        string += (
            ' Instances: ' + dict_pretty_str(self.instances, indent, depth)
        )
        return string

    @property
    def pyramid(self) -> str:
        """Return string representatin of the level"""
        return (
            f'Level [{self.level}]'
            f' tiles: {self.default_instance.tiled_size},'
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
        """Return list of levels created wsi files.

        Parameters
        ----------
        files: List[WsiInstance]
            Instances to create levels from.

        Returns
        ----------
        List[WsiDicomLevel]
            List of created levels.

        """
        levels: List[WsiDicomLevel] = []
        instances_grouped_by_level = cls._group_instances(instances)
        largest_size = max(instances_grouped_by_level.keys())
        base_group = instances_grouped_by_level[largest_size]
        base_pixel_spacing = base_group[0].pixel_spacing
        for level in instances_grouped_by_level.values():
            levels.append(WsiDicomLevel(level, base_pixel_spacing))

        return levels

    def matches(self, other_group: 'WsiDicomGroup') -> bool:
        """Check if group is valid (Uids and tile size match).
        The common Uids should match for all series. For level series the tile
        size should also match. It is assumed that the instances in the groups
        are matching each other.
        """
        other_instance = other_group.default_instance
        this_instance = self.default_instance
        return (
            other_group.uids == self.uids and
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


class WsiDicomSeries:
    wsi_type: str

    def __init__(self, groups: List[WsiDicomGroup]):
        """Holds a series of image groups of same image flavor

        Parameters
        ----------
        groups: List[WsiDicomGroup]
            List of groups to include in the series
        """
        self._groups: List[WsiDicomGroup] = groups

        if len(self.groups) != 0 and self.groups[0].uids is not None:
            self._uids = self._validate_series(self.groups)
        else:
            self._uids = None

    def __getitem__(self, index: int) -> WsiDicomGroup:
        """Get group by index.

        Parameters
        ----------
        index: int
            Index in series to get

        Returns
        ----------
        WsiDicomGroup
            The group at index in the series
        """
        return self.groups[index]

    @property
    def groups(self) -> List[WsiDicomGroup]:
        """Return contained groups."""
        return self._groups

    @property
    def uids(self) -> Optional[BaseUids]:
        """Return uids."""
        return self._uids

    @property
    def mpps(self) -> List[SizeMm]:
        """Return contained mpp (um/px)."""
        return [group.mpp for group in self.groups]

    @property
    def files(self) -> List[WsiDicomFile]:
        """Return contained files."""
        series_files = [series.files for series in self.groups]
        return [file for sublist in series_files for file in sublist]

    @property
    def datasets(self) -> List[Dataset]:
        """Return contained datasets."""

        series_datasets = [
            series.datasets for series in self.groups
        ]
        return [
            dataset for sublist in series_datasets for dataset in sublist
        ]

    @property
    def instances(self) -> List[WsiInstance]:
        """Return contained instances"""
        series_instances = [
            series.instances.values() for series in self.groups
        ]
        return [
            instance for sublist in series_instances for instance in sublist
        ]

    def _validate_series(
            self,
            groups: Union[List[WsiDicomGroup], List[WsiDicomLevel]]
    ) -> Optional[BaseUids]:
        """Check that no files or instances in series is duplicate and that
        all groups in series matches.
        Raises WsiDicomMatchError otherwise.
        Returns the matching base uid. If list of groups is empty, return None.

        Parameters
        ----------
        groups: Union[List[WsiDicomGroup], List[WsiDicomLevel]]
            List of groups or levels to check

        Returns
        ----------
        Optional[BaseUids]:
            Matching uids
        """
        WsiDataset.check_duplicate_dataset(self.datasets, self)
        WsiInstance.check_duplicate_instance(self.instances, self)

        try:
            base_group = groups[0]
            # print(base_group.wsi_type)
            # print(self.wsi_type)
            if base_group.wsi_type != self.wsi_type:
                raise WsiDicomMatchError(
                    str(base_group), str(self)
                )
            for group in groups[1:]:
                if not group.matches(base_group):
                    raise WsiDicomMatchError(
                        str(group), str(self)
                    )
            return base_group.uids
        except IndexError:
            return None

    def close(self) -> None:
        """Close all groups in the series."""
        for group in self.groups:
            group.close()


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
        labels = WsiDicomGroup.open(instances)
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
        overviews = WsiDicomGroup.open(instances)
        return WsiDicomOverviews(overviews)


class WsiDicomLevels(WsiDicomSeries):
    wsi_type = 'VOLUME'

    def __init__(self, levels: List[WsiDicomLevel]):
        """Holds a stack of levels.

        Parameters
        ----------
        levels: List[WsiDicomLevel]
            List of levels to include in series
        """
        self._levels = OrderedDict(
            (level.level, level)
            for level in sorted(levels, key=lambda level: level.level)
        )
        if len(self.groups) != 0 and self.groups[0].uids is not None:
            self._uids = self._validate_series(self.groups)
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
    def groups(self) -> List[WsiDicomGroup]:
        """Return contained groups"""
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
        closest_size = self.groups[0].size
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


class WsiDicom:
    def __init__(
        self,
        levels: WsiDicomLevels,
        labels: WsiDicomLabels,
        overviews: WsiDicomOverviews
    ):
        """Holds wsi dicom levels, labels and overviews.

        Parameters
        ----------
        levels: WsiDicomLevels
            Series of pyramidal levels.
        labels: WsiDicomLabels
            Series of label images.
        overviews: WsiDicomOverviews
            Series of overview images
        """

        self._levels = levels
        self._labels = labels
        self._overviews = overviews
        if self.levels.uids is not None:
            self.uids = self._validate_collection(
                [self.levels, self.labels, self.overviews],
            )
        else:
            self.uids = None

        self.optical = OpticalManager.open(
            levels.instances + labels.instances + overviews.instances
        )
        self.__enter__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
    def datasets(self) -> List[Dataset]:
        """Return contained datasets."""
        return (
            self.levels.datasets
            + self.labels.datasets
            + self.overviews.datasets
        )

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
            + list_pretty_str(self.levels.groups, indent, depth, 0, 2)
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

        base_dataset = cls._get_base_dataset(level_files)
        base_uids = base_dataset.base_uids
        base_tile_size = base_dataset.tile_size
        level_instances = WsiInstance.open(
            level_files,
            base_uids,
            base_tile_size
        )
        label_instances = WsiInstance.open(label_files, base_uids)
        overview_instances = WsiInstance.open(overview_files, base_uids)

        levels = WsiDicomLevels.open(level_instances)
        labels = WsiDicomLabels.open(label_instances)
        overviews = WsiDicomOverviews.open(overview_instances)

        return WsiDicom(levels, labels, overviews)

    @classmethod
    def open_tiler(
        cls,
        tiler: Tiler,
        base_dataset: Dataset,
        uid_generator: Callable[..., Uid] = pydicom.uid.generate_uid,
        transfer_syntax: Uid = pydicom.uid.JPEGBaseline8Bit,
        include_levels: List[int] = None,
        include_label: bool = True,
        include_overview: bool = True
    ) -> Tuple[List[WsiInstance], List[WsiInstance], List[WsiInstance]]:
        """Open tiler to produce WsiInstances.

        Parameters
        ----------
        tiler: Tiler
            Tiler that can produce WsiInstances.
        base_dataset: Dataset
            Base dataset to include in files.
        uid_generator: Callable[..., Uid] = pydicom.uid.generate_uid
            Function that can gernerate unique identifiers.
        transfer_syntax: Uid = pydicom.uid.JPEGBaseline8Bit
            Transfer syntax.
        include_levels: List[int] = None
            Optional list of levels to include. Include all levels if None.
        include_label: bool = True
            Include label(s), default true.
        include_overwiew: bool = True
            Include overview(s), default true.
        """
        level_instances = []
        if include_levels is None:
            include_levels = range(tiler.level_count)
        for level_index in include_levels:
            image_data = tiler.get_level(level_index)
            instance_dataset = WsiDataset.create_instance_dataset(
                base_dataset,
                'VOLUME',
                level_index,
                image_data.image_size,
                image_data.tile_size,
                image_data.pixel_spacing,
                uid_generator
            )
            instance = WsiInstance(
                WsiDataset(instance_dataset),
                image_data,
                transfer_syntax
            )
            level_instances.append(instance)

        label_instances = []
        if include_label:
            for label_index in range(tiler.label_count):
                image_data = tiler.get_label(label_index)
                instance_dataset = WsiDataset.create_instance_dataset(
                    base_dataset,
                    'LABEL',
                    level_index,
                    image_data.image_size,
                    image_data.tile_size,
                    image_data.pixel_spacing,
                    uid_generator
                )
                instance = WsiInstance(
                    WsiDataset(instance_dataset),
                    image_data,
                    transfer_syntax
                )
                label_instances.append(instance)

        overview_instances = []
        if include_overview:
            for overview_index in range(tiler.overview_count):
                image_data = tiler.get_overview(overview_index)
                image_data = tiler.get_label(label_index)
                instance_dataset = WsiDataset.create_instance_dataset(
                    base_dataset,
                    'OVERVIEW',
                    level_index,
                    image_data.image_size,
                    image_data.tile_size,
                    image_data.pixel_spacing,
                    uid_generator
                )
                instance = WsiInstance(
                    WsiDataset(instance_dataset),
                    image_data,
                    transfer_syntax
                )
                overview_instances.append(instance)

        return level_instances, label_instances, overview_instances

    @classmethod
    def import_wsi(
        cls,
        tiler: Tiler,
        base_dataset: Dataset = None,
    ) -> 'WsiDicom':
        """Open data in tiler as WsiDicom object.

        Parameters
        ----------
        tiler: Tiler
            Tiler that can produce WsiInstances.
        base_dataset: Dataset
            Base dataset to use in files. If none, use test dataset.
        """
        if base_dataset is None:
            base_dataset = WsiDataset.create_test_base_dataset()

        (
            level_instances,
            label_instances,
            overview_instances
        ) = cls.open_tiler(tiler, base_dataset)
        levels = WsiDicomLevels.open(level_instances)
        labels = WsiDicomLabels.open(label_instances)
        overviews = WsiDicomOverviews.open(overview_instances)
        return cls(levels, labels, overviews)

    @classmethod
    def convert(
        cls,
        output_path: Path,
        tiler: Tiler,
        base_dataset: Dataset,
        include_levels: List[int] = None,
        include_label: bool = True,
        include_overview: bool = True
    ) -> None:
        """Convert data in tiler to Dicom files in output path.

        Parameters
        ----------
        output_path: Path
            Folder path to save files to.
        tiler: Tiler
            Tiler that can produce WsiInstances.
        base_dataset: Dataset
            Base dataset to include in files.
        include_levels: List[int]
            Optional list of levels to include. Include all levels if None.
        include_label: bool
            Include label(s), default true.
        include_overwiew: bool
            Include overview(s), default true.
        """
        level_instances, label_instances, overview_instances = (
            cls.open_tiler(
                tiler,
                base_dataset,
                include_levels=include_levels,
                include_label=include_label,
                include_overview=include_overview
            )
        )

        for instance in level_instances+label_instances+overview_instances:
            instance.save(output_path)

        tiler.close()

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
    def _get_base_dataset(
        files: List[WsiDicomFile]
    ) -> WsiDataset:
        """Return file with largest image (width) from list of files.

        Parameters
        ----------
        files: List[WsiDicomFile]
           List of files.

        Returns
        ----------
        WsiDataset
            Base layer dataset.
        """
        base_size = Size(0, 0)
        base_dataset: WsiDataset
        for file in files:
            if file.dataset.image_size.width > base_size.width:
                base_dataset = file.dataset
                base_size = file.dataset.image_size
        return base_dataset

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
        series have the same base uids. Raises WsiDicomMatchError otherwise.
        Returns base uid for collection.

        Parameters
        ----------
        series: List[WsiDicomSeries]
            List of series to check

        Returns
        ----------
        BaseUids
            Matching uids
        """
        WsiDataset.check_duplicate_dataset(self.datasets, self)
        WsiInstance.check_duplicate_instance(self.instances, self)

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
            Requested pixel spacing (um/pixel)
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
