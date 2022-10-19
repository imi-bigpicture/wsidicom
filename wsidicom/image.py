#    Copyright 2021, 2022 SECTRA AB
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

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import (Callable, DefaultDict, Dict, List, Optional, OrderedDict,
                    Sequence, Set, Tuple, Union, cast)

from PIL import Image
from pydicom.uid import UID

from wsidicom.image_data import WsiImageOrgin
from wsidicom.dataset import WsiDicomDataset
from wsidicom.errors import (WsiDicomError, WsiDicomMatchError,
                             WsiDicomNotFoundError, WsiDicomOutOfBoundsError,
                             WsiDicomUidDuplicateError)
from wsidicom.file import WsiDicomFile
from wsidicom.file_writer import WsiDicomFileWriter
from wsidicom.geometry import Point, Region, RegionMm, Size, SizeMm
from wsidicom.image_data import WsiImageData, WsiDicomImageData
from wsidicom.stringprinting import dict_pretty_str
from wsidicom.uid import SlideUids


class WsiImage:
    """Represents a level, label, or overview wsi image, containing image data
    and datasets with metadata."""
    def __init__(
        self,
        datasets: Union[WsiDicomDataset, Sequence[WsiDicomDataset]],
        image_data: WsiImageData
    ):
        """Create an image from datasets with metadata and image data.

        Parameters
        ----------
        datasets: Union[WsiDicomDataset, Sequence[WsiDicomDataset]]
            Single dataset or list of datasets.
        image_data: WsiImageData
            Image data.
        """
        if not isinstance(datasets, Sequence):
            datasets = [datasets]
        self._datasets = datasets
        self._image_data = image_data
        self._identifier, self._uids = self._validate_image(self.datasets)
        self._wsi_type = self.dataset.wsi_type

        if self.ext_depth_of_field:
            if self.ext_depth_of_field_planes is None:
                raise WsiDicomError("WsiImage Missing NumberOfFocalPlanes")
            if self.ext_depth_of_field_plane_distance is None:
                raise WsiDicomError(
                    "WsiImage Missing DistanceBetweenFocalPlanes"
                )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.dataset}, {self.image_data})"

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: Optional[int] = None
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
            ' WsiImageData ' + self.image_data.pretty_str(indent+1, depth)
        )
        return string

    @property
    def wsi_type(self) -> str:
        """Return wsi type."""
        return self._wsi_type

    @property
    def datasets(self) -> Sequence[WsiDicomDataset]:
        return self._datasets

    @property
    def dataset(self) -> WsiDicomDataset:
        return self.datasets[0]

    @property
    def image_data(self) -> WsiImageData:
        return self._image_data

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
    def mm_size(self) -> Optional[SizeMm]:
        """Return slide size in mm."""
        return self.dataset.mm_size

    @property
    def mm_depth(self) -> Optional[float]:
        """Return imaged depth in mm."""
        return self.dataset.mm_depth

    @property
    def slice_thickness(self) -> Optional[float]:
        """Return slice thickness."""
        return self.dataset.slice_thickness

    @property
    def slice_spacing(self) -> Optional[float]:
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
    def identifier(self) -> UID:
        """Return identifier (image uid for single file image or
        concatenation uid for multiple file image)."""
        return self._identifier

    @property
    def image_number(self) -> int:
        return int(self.dataset.image_number)

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
    def uids(self) -> SlideUids:
        """Return base uids"""
        return self._uids

    @property
    def image_origin(self) -> WsiImageOrgin:
        return self.dataset.image_origin

    @classmethod
    def open(
        cls,
        files: Sequence[WsiDicomFile],
        series_uids: SlideUids,
        series_tile_size: Optional[Size] = None
    ) -> List['WsiImage']:
        """Create images from Dicom files. Only files with matching series
        uid and tile size, if defined, are used. Other files are closed.

        Parameters
        ----------
        files: Sequence[WsiDicomFile]
            Files to create images from.
        series_uids: SlideUids
            Uid to match against.
        series_tile_size: Optional[Size]
            Tile size to match against (for level images).

        Returns
        ----------
        List[WsiImage]
            List of created images.
        """
        filtered_files = WsiDicomFile.filter_files(
            files,
            series_uids,
            series_tile_size
        )
        files_grouped_by_image = WsiDicomFile.group_files(filtered_files)
        return [
            cls(
                [file.dataset for file in image_files],
                WsiDicomImageData(image_files)
            )
            for image_files in files_grouped_by_image.values()
        ]

    @staticmethod
    def check_duplicate_image(
        images: Sequence['WsiImage'],
        self: object
    ) -> None:
        """Check for duplicates in list of images. Images are duplicate
        if image identifier (file image uid or concatenation uid) match.
        Stops at first found duplicate and raises WsiDicomUidDuplicateError.

        Parameters
        ----------
        images: Sequence['WsiImage']
            List of images to check.
        caller: Object
            Object that the images belongs to.
        """
        image_identifiers: List[str] = []
        for image in images:
            image_identifier = image.identifier
            if image_identifier not in image_identifiers:
                image_identifiers.append(image_identifier)
            else:
                raise WsiDicomUidDuplicateError(str(image), str(self))

    def _validate_image(
        self,
        datasets: Sequence[WsiDicomDataset]
    ) -> Tuple[UID, SlideUids]:
        """Check that no files in image are duplicate, that all files in
        image matches (uid, type and size).
        Raises WsiDicomMatchError otherwise.
        Returns the matching file uid.

        Returns
        ----------
        Tuple[UID, SlideUids]
            Image identifier uid and base uids
        """
        WsiDicomDataset.check_duplicate_dataset(datasets, self)

        base_dataset = datasets[0]
        for dataset in datasets[1:]:
            if not base_dataset.matches_image(dataset):
                raise WsiDicomError("Datasets in images does not match")
        return (
            base_dataset.uids.identifier,
            base_dataset.uids.slide,
        )

    def matches(self, other_image: 'WsiImage') -> bool:
        """Return true if other image is of the same group as self.

        Parameters
        ----------
        other_image: WsiImage
            Image to check.

        Returns
        ----------
        bool
            True if instanes are of same group.

        """
        return (
            self.uids.matches(other_image.uids) and
            self.size == other_image.size and
            self.tile_size == other_image.tile_size and
            self.wsi_type == other_image.wsi_type
        )

    def close(self) -> None:
        self._image_data.close()


class WsiImageGroup:
    """Represents a group of images having the same size, but possibly
    different z coordinate and/or optical path."""
    def __init__(
        self,
        images: Sequence[WsiImage]
    ):
        """Create a group of WsiImage. Images should match in the common
        uids, wsi type, and tile size.

        Parameters
        ----------
        images: Sequence[WsiImage]
            Images to build the group.
        """
        self._images = {  # key is identifier (Uid)
            image.identifier: image for image in images
        }
        self._validate_group()

        base_image = images[0]
        self._wsi_type = base_image.wsi_type
        self._uids = base_image.uids

        self._size = base_image.size
        self._pixel_spacing = base_image.pixel_spacing
        self._default_image_uid: UID = base_image.identifier

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.images.values()})"

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: Optional[int] = None
    ) -> str:
        string = (
            f'Image: size: {self.size} px, mpp: {self.mpp} um/px'
        )
        if depth is not None:
            depth -= 1
            if(depth < 0):
                return string
        string += (
            ' Images: ' + dict_pretty_str(self.images, indent, depth)
        )
        return string

    def __getitem__(self, index: UID) -> WsiImage:
        return self.images[index]

    @property
    def uids(self) -> SlideUids:
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
    def images(self) -> Dict[UID, WsiImage]:
        """Return contained images"""
        return self._images

    @property
    def default_image(self) -> WsiImage:
        """Return default image"""
        return self.images[self._default_image_uid]

    @property
    def files(self) -> List[Path]:
        """Return contained files"""
        image_files = [
            image.image_data.files for image in self.images.values()
        ]
        return [file for sublist in image_files for file in sublist]

    @property
    def datasets(self) -> List[WsiDicomDataset]:
        """Return contained datasets."""
        image_datasets = [
            image.datasets for image in self.images.values()
        ]
        return [
            dataset for sublist in image_datasets for dataset in sublist
        ]

    @property
    def optical_paths(self) -> List[str]:
        return list({
            path
            for image in self.images.values()
            for path in image.optical_paths
        })

    @property
    def focal_planes(self) -> List[float]:
        return list({
            focal_plane
            for innstance in self.images.values()
            for focal_plane in innstance.focal_planes
        })

    @property
    def image_origin(self) -> WsiImageOrgin:
        return self.default_image.image_origin

    @classmethod
    def open(
        cls,
        images: Sequence[WsiImage],
    ) -> List['WsiImageGroup']:
        """Return list of groups created from wsi images.

        Parameters
        ----------
        files: Sequence[WsiImage]
            Images to create groups from.

        Returns
        ----------
        List[WsiImageGroup]
            List of created groups.

        """
        groups: List['WsiImageGroup'] = []

        grouped_images = cls._group_images(images)

        for group in grouped_images.values():
            groups.append(cls(group))

        return groups

    def matches(self, other_group: 'WsiImageGroup') -> bool:
        """Check if group matches other group. If strict common Uids should
        match. Wsi type should always match.

        Parameters
        ----------
        other_group: WsiImageGroup
            Other group to match against.

        Returns
        ----------
        bool
            True if other group matches.
        """
        return (
            self.uids.matches(other_group.uids) and
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

    def get_image(
        self,
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> WsiImage:
        """Search for image fullfilling the parameters.
        The behavior when z and/or path is none could be made more
        clear.

        Parameters
        ----------
        z: Optional[float] = None
            Z coordinate of the searched image
        path: Optional[str] = None
            Optical path of the searched image

        Returns
        ----------
        WsiImage
            The image containing selected path and z coordinate
        """
        if z is None and path is None:
            image = self.default_image
            z = image.default_z
            path = image.default_path

            return self.default_image

        # Sort images by number of focal planes (prefer simplest image)
        sorted_images = sorted(
            list(self._images.values()),
            key=lambda i: len(i.focal_planes)
        )
        try:
            if z is None:
                # Select the image with selected optical path
                image = next(i for i in sorted_images if
                             path in i.optical_paths)
            elif path is None:
                # Select the image with selected z
                image = next(i for i in sorted_images
                             if z in i.focal_planes)
            else:
                # Select by both path and z
                image = next(i for i in sorted_images
                             if (z in i.focal_planes and
                                 path in i.optical_paths))
        except StopIteration:
            raise WsiDicomNotFoundError(
                f"WsiImage for path: {path}, z: {z}", "group"
            )
        if z is None:
            z = image.default_z
        if path is None:
            path = image.default_path
        return image

    def get_default_full(self) -> Image.Image:
        """Read full image using default z coordinate and path.

        Returns
        ----------
        Image.Image
            Full image of the group.
        """
        image = self.default_image
        z = image.default_z
        path = image.default_path
        region = Region(position=Point(x=0, y=0), size=self.size)
        image = self.get_region(region, z, path)
        return image

    def get_region(
        self,
        region: Region,
        z: Optional[float] = None,
        path: Optional[str] = None,
    ) -> Image.Image:
        """Read region defined by pixels.

        Parameters
        ----------
        location: int, int
            Upper left corner of region in pixels
        size: int
            Size of region in pixels
        z: Optional[float] = None
            Z coordinate, optional
        path: Optional[str] = None
            optical path, optional

        Returns
        ----------
        Image.Image
            Region as image
        """

        image = self.get_image(z, path)
        if z is None:
            z = image.default_z
        if path is None:
            path = image.default_path
        region_image = image.image_data.stitch_tiles(region, path, z)
        return region_image

    def get_region_mm(
        self,
        region: RegionMm,
        z: Optional[float] = None,
        path: Optional[str] = None,
        slide_origin: bool = False
    ) -> Image.Image:
        """Read region defined by mm.

        Parameters
        ----------
        region: RegionMm
            Region defining upper left corner and size in mm.
        z: Optional[float] = None
            Z coordinate, optional.
        path: Optional[str] = None
            optical path, optional.
        slide_origin: bool = False.
            If to use the slide origin instead of image origin.

        Returns
        ----------
        Image.Image
            Region as image
        """
        if slide_origin:
            region = self.image_origin.transform_region(region)
        pixel_region = self.mm_to_pixel(region)
        image = self.get_region(pixel_region, z, path)
        if slide_origin:
            image = image.rotate(
                self.image_origin.rotation,
                resample=Image.Resampling.BILINEAR,
                expand=True
            )
        return image

    def get_tile(
        self,
        tile: Point,
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> Image.Image:
        """Return tile at tile coordinate x, y as image.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate
        z: Optional[float] = None
            Z coordinate
        path: Optional[str] = None
            Optical path

        Returns
        ----------
        Image.Image
            The tile as image
        """

        image = self.get_image(z, path)
        if z is None:
            z = image.default_z
        if path is None:
            path = image.default_path
        return image.image_data.get_tile(tile, z, path)

    def get_encoded_tile(
        self,
        tile: Point,
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> bytes:
        """Return tile at tile coordinate x, y as bytes.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate
        z: Optional[float] = None
            Z coordinate
        path: Optional[str] = None
            Optical path

        Returns
        ----------
        bytes
            The tile as bytes
        """
        image = self.get_image(z, path)
        if z is None:
            z = image.default_z
        if path is None:
            path = image.default_path
        return image.image_data.get_encoded_tile(tile, z, path)

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
            raise WsiDicomOutOfBoundsError(
                f"Region {region}", f"level size {self.size}"
            )
        return pixel_region

    def close(self) -> None:
        """Close all images on the group."""
        for image in self._images.values():
            image.close()

    def _validate_group(self):
        """Check that no file or image in group is duplicate, and if strict
        images in group matches. Raises WsiDicomMatchError otherwise.
        """
        images = list(self.images.values())
        base_image = images[0]
        for image in images[1:]:
            if not base_image.matches(image):
                raise WsiDicomMatchError(str(image), str(self))

        WsiDicomDataset.check_duplicate_dataset(self.datasets, self)
        WsiImage.check_duplicate_image(images, self)

    @classmethod
    def _group_images(
        cls,
        images: Sequence[WsiImage]
    ) -> OrderedDict[Size, List[WsiImage]]:
        """Return images grouped and sorted by image size.

        Parameters
        ----------
        images: Sequence[WsiImage]
            Images to group by image size.

        Returns
        ----------
        OrderedDict[Size, List[WsiImage]]:
            Images grouped by size, with size as key.

        """
        grouped_images: Dict[Size, List[WsiImage]] = {}
        for image in images:
            try:
                grouped_images[image.size].append(image)
            except KeyError:
                grouped_images[image.size] = [image]
        return OrderedDict(sorted(
            grouped_images.items(),
            key=lambda item: item[0].width,
            reverse=True)
        )

    def _group_images_to_file(
        self,
    ) -> List[List[WsiImage]]:
        """Group images by properties that can't differ in a DICOM-file,
        i.e. the images are grouped by output file.

        Returns
        ----------
        List[List[WsiImage]]
            Images grouped by common properties.
        """
        groups: DefaultDict[
            Tuple[str, UID, bool, Optional[int], Optional[float], str],
            List[WsiImage]
        ] = defaultdict(list)

        for image in self.images.values():
            groups[
                image.image_data.photometric_interpretation,
                image.image_data.transfer_syntax,
                image.ext_depth_of_field,
                image.ext_depth_of_field_planes,
                image.ext_depth_of_field_plane_distance,
                image.focus_method
            ].append(
                image
            )
        return list(groups.values())

    @staticmethod
    def _list_image_data(
        images: Sequence[WsiImage]
    ) -> Dict[Tuple[str, float], WsiImageData]:
        """Sort WsiImageData in images by optical path and focal
        plane.

        Parameters
        ----------
        images: Sequence[WsiImage]
            List of images with optical paths and focal planes to list and
            sort.

        Returns
        ----------
        Dict[Tuple[str, float], WsiImageData]:
            WsiImageData sorted by optical path and focal plane.
        """
        output: Dict[Tuple[str, float], WsiImageData] = {}
        for image in images:
            for optical_path in image.optical_paths:
                for z in sorted(image.focal_planes):
                    if (optical_path, z) not in output:
                        output[optical_path, z] = image.image_data
        return output

    @staticmethod
    def _get_frame_information(
        data: Dict[Tuple[str, float], WsiImageData]
    ) -> Tuple[List[float], List[str], Size]:
        """Return optical_paths, focal planes, and tiled size.
        """
        focal_planes_by_optical_path: Dict[str, Set[float]] = (
            defaultdict(set)
        )
        all_focal_planes: Set[float] = set()
        tiled_sizes: Set[Size] = set()
        for (optical_path, focal_plane), image_data in data.items():
            focal_planes_by_optical_path[optical_path].add(focal_plane)
            all_focal_planes.add(focal_plane)
            tiled_sizes.add(image_data.tiled_size)

        focal_planes_sparse_by_optical_path = any(
            optical_path_focal_planes != all_focal_planes
            for optical_path_focal_planes
            in focal_planes_by_optical_path.values()
        )
        if focal_planes_sparse_by_optical_path:
            raise ValueError(
                'Each optical path must have the same focal planes.'
            )

        if len(tiled_sizes) != 1:
            raise ValueError('Expected only one tiled size')
        tiled_size = list(tiled_sizes)[0]
        return (
            sorted(list(all_focal_planes)),
            sorted(list(focal_planes_by_optical_path.keys())),
            tiled_size
        )

    def save(
        self,
        output_path: str,
        uid_generator: Callable[..., UID],
        workers: int,
        chunk_size: int,
        offset_table: Optional[str]
    ) -> List[Path]:
        """Save a WsiImageGroup to files in output_path. Images are
        grouped by properties that cant differ in the same file:
            - photometric interpretation
            - transfer syntax
            - extended depth of field (and planes and distance)
            - focus method
        Other properties are assumed to be equal or to be updated.

        Parameters
        ----------
        output_path: str
            Folder path to save files to.
        uid_generator: Callable[..., UID]
            Uid generator to use.
        workers: int
            Maximum number of thread workers to use.
        chunk_size: int
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.
        offset_table: Optional[str]
            Offset table to use, 'bot' basic offset table, 'eot' extended
            offset table, None - no offset table.

        Returns
        ----------
        List[str]
            List of paths of created files.
        """
        filepaths: List[Path] = []
        for images in self._group_images_to_file():
            uid = uid_generator()
            filepath = Path(os.path.join(output_path, uid + '.dcm'))
            transfer_syntax = images[0].image_data.transfer_syntax
            image_data_list = self._list_image_data(images)
            focal_planes, optical_paths, tiled_size = (
                self._get_frame_information(image_data_list)
            )
            dataset = images[0].dataset.as_tiled_full(
                focal_planes,
                optical_paths,
                tiled_size
            )
            with WsiDicomFileWriter(filepath) as wsi_file:
                wsi_file.write(
                    uid,
                    transfer_syntax,
                    dataset,
                    image_data_list,
                    workers,
                    chunk_size,
                    offset_table
                )
            filepaths.append(filepath)
        return filepaths


class WsiLevel(WsiImageGroup):
    """Represents a level in the pyramid and contains one or more images
    having the same pyramid level index, pixel spacing, and size but possibly
    different focal planes and/or optical paths.
    """
    def __init__(
        self,
        images: Sequence[WsiImage],
        base_pixel_spacing: SizeMm
    ):
        """Create a level from list of WsiImages. Asign the pyramid level
        index from pixel spacing of base level.

        Parameters
        ----------
        images: Sequence[WsiImage]
            Images to build the level.
        base_pixel_spacing: SizeMm
            Pixel spacing of base level.
        """
        super().__init__(images)
        self._base_pixel_spacing = base_pixel_spacing
        self._level = self._assign_level(self._base_pixel_spacing)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.images}, "
            f"{self._base_pixel_spacing})"
        )

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: Optional[int] = None
    ) -> str:
        string = (
            f'Level: {self.level}, size: {self.size} px, mpp: {self.mpp} um/px'
        )
        if depth is not None:
            depth -= 1
            if(depth < 0):
                return string
        string += (
            ' WsiImages: ' + dict_pretty_str(self.images, indent, depth)
        )
        return string

    @property
    def pyramid(self) -> str:
        """Return string representation of the level"""
        return (
            f'Level [{self.level}]'
            f' tiles: {self.default_image.tiled_size},'
            f' size: {self.size}, mpp: {self.mpp} um/px'
        )

    @property
    def tile_size(self) -> Size:
        return self.default_image.tile_size

    @property
    def level(self) -> int:
        """Return pyramid level"""
        return self._level

    @classmethod
    def open(
        cls,
        images: Sequence[WsiImage],
    ) -> List['WsiLevel']:
        """Return list of levels created wsi files.

        Parameters
        ----------
        files: Sequence[WsiImage]
            Images to create levels from.

        Returns
        ----------
        List[WsiLevel]
            List of created levels.

        """
        levels: List['WsiLevel'] = []
        images_grouped_by_level = cls._group_images(images)
        base_group = list(images_grouped_by_level.values())[0]
        base_pixel_spacing = base_group[0].pixel_spacing
        for level in images_grouped_by_level.values():
            levels.append(cls(level, base_pixel_spacing))
        return levels

    def matches(self, other_level: 'WsiImageGroup') -> bool:
        """Check if level matches other level. If strict common Uids should
        match. Wsi type and tile size should always match.

        Parameters
        ----------
        other_level: WsiImageGroup
            Other level to match against.

        Returns
        ----------
        bool
            True if other level matches.
        """
        other_level = cast(WsiLevel, other_level)
        return (
            self.uids.matches(other_level.uids) and
            other_level.wsi_type == self.wsi_type and
            other_level.tile_size == self.tile_size
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
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> Image.Image:
        """Return tile in another level by scaling a region.
        If the tile is an edge tile, the resulting tile is croped
        to remove part outside of the image (as defiend by level size).

        Parameters
        ----------
        tile: Point
            Non scaled tile coordinate
        level: int
            Level to scale from
        z: Optional[float] = None
            Z coordinate
        path: Optional[str] = None
            Optical path

        Returns
        ----------
        Image.Image
            A tile image
        """
        scale = self.calculate_scale(level)
        image = self.get_image(z, path)
        scaled_region = Region.from_tile(tile, image.tile_size) * scale
        cropped_region = scaled_region.crop(image.image_data.image_size)
        if not self.valid_pixels(cropped_region):
            raise WsiDicomOutOfBoundsError(
                f"Region {cropped_region}", f"level size {self.size}"
            )
        tile_image = self.get_region(cropped_region, z, path)
        tile_size = cropped_region.size.ceil_div(scale)
        tile_image = tile_image.resize(
            tile_size.to_tuple(),
            resample=Image.Resampling.BILINEAR
        )
        return tile_image

    def get_scaled_encoded_tile(
        self,
        tile: Point,
        scale: int,
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> bytes:
        """Return encoded tile in another level by scaling a region.

        Parameters
        ----------
        tile: Point
            Non scaled tile coordinate
        level: int
           Level to scale from
        z: Optional[float] = None
            Z coordinate
        path: Optional[str] = None
            Optical path

        Returns
        ----------
        bytes
            A transfer syntax encoded tile
        """
        tile_image = self.get_scaled_tile(tile, scale, z, path)
        image = self.get_image(z, path)
        return image.image_data.encode(tile_image)

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
            raise NotImplementedError("Levels needs to be integer.")
        return level

    def create_child(
        self,
        scale: int,
        output_path: Path,
        uid_generator: Callable[..., UID],
        workers: int,
        chunk_size: int,
        offset_table: Optional[str]
    ) -> 'WsiLevel':
        """Creates a new WsiLevel from this level by scaling the image
        data.

        Parameters
        ----------
        scale: int
            Scale factor.
        output_path: Path
            The path to write child to.
        uid_generator: Callable[..., UID]
            Uid generator to use.
        workers: int
            Maximum number of thread workers to use.
        chunk_size: int
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.
        offset_table: Optional[str]
            Offset table to use, 'bot' basic offset table, 'eot' extended
            offset table, None - no offset table.

        Returns
        ----------
        'WsiLevel'
            Created scaled level.
        """
        filepaths: List[Path] = []
        if not isinstance(scale, int) or scale < 2:
            raise ValueError(
                "Scale must be integer and larger than 2"
            )
        if not isinstance(
            self.default_image.image_data,
            WsiDicomImageData
        ):
            raise NotImplementedError(
                "Can only construct pyramid from DICOM WSI files"
            )

        for images in self._group_images_to_file():
            uid = uid_generator()
            filepath = Path(os.path.join(output_path, uid + '.dcm'))
            transfer_syntax = images[0].image_data.transfer_syntax
            image_data_list = self._list_image_data(images)
            focal_planes, optical_paths, tiled_size = (
                self._get_frame_information(image_data_list)
            )
            dataset = images[0].dataset.as_tiled_full(
                focal_planes,
                optical_paths,
                tiled_size,
                scale
            )

            with WsiDicomFileWriter(filepath) as wsi_file:
                wsi_file.write(
                    uid,
                    transfer_syntax,
                    dataset,
                    image_data_list,
                    workers,
                    chunk_size,
                    offset_table,
                    scale
                )
            filepaths.append(filepath)

        created_images = WsiImage.open(
            [WsiDicomFile(filepath) for filepath in filepaths],
            self.uids,
            self.tile_size
        )
        return WsiLevel(created_images, self._base_pixel_spacing)
