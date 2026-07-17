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
from pathlib import Path

import numpy as np
from PIL.Image import Image
from pydicom import Dataset
from pydicom.uid import UID
from upath import UPath

from wsidicom.downsampler import Downsampler
from wsidicom.errors import (
    WsiDicomError,
    WsiDicomOutOfBoundsError,
    WsiDicomUidDuplicateError,
)
from wsidicom.geometry import Point, Region, Size, SizeMm
from wsidicom.instance.dataset import WsiDataset
from wsidicom.instance.image_data import ImageData
from wsidicom.instance.numpy_image_data import NumpyImageData
from wsidicom.metadata.image import ImageCoordinateSystem, ImageType
from wsidicom.stitcher import NumpyStitcher
from wsidicom.thread import ReadExecutor
from wsidicom.uid import SlideUids


class WsiInstance:
    """Represents a level, label, or overview wsi image, containing image data
    and datasets with metadata."""

    def __init__(
        self, datasets: WsiDataset | Sequence[WsiDataset], image_data: ImageData
    ):
        """Create a WsiInstance from datasets with metadata and image data.

        Parameters
        ----------
        datasets: WsiDataset | Sequence[WsiDataset]
            Single dataset or list of datasets.
        image_data: ImageData
        """
        if not isinstance(datasets, Sequence):
            datasets = [datasets]
        self._datasets = datasets
        self._image_data = image_data
        self._identifier, self._uids = self._validate_instance(self.datasets)
        self._image_type = self.dataset.image_type
        self._stitcher = NumpyStitcher()
        self._downsampler: Downsampler = Downsampler.create()

        if self.ext_depth_of_field:
            if self.ext_depth_of_field_planes is None:
                raise WsiDicomError("Instance Missing NumberOfFocalPlanes.")
            if self.ext_depth_of_field_plane_distance is None:
                raise WsiDicomError("Instance Missing DistanceBetweenFocalPlanes.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.dataset}, {self.image_data})"

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(self, indent: int = 0, depth: int | None = None) -> str:
        string = f"default z: {self.default_z} default path: {self.default_path}"
        if depth is not None:
            depth -= 1
            if depth < 0:
                return string
        string += " ImageData " + self.image_data.pretty_str(indent + 1, depth)
        return string

    @property
    def image_type(self) -> ImageType:
        """Return wsi type."""
        return self._image_type

    @property
    def datasets(self) -> Sequence[WsiDataset]:
        return self._datasets

    @property
    def dataset(self) -> WsiDataset:
        return self.datasets[0]

    @property
    def image_data(self) -> ImageData:
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
    def mpp(self) -> SizeMm | None:
        """Return pixel spacing in um/pixel."""
        if self.pixel_spacing is None:
            return None
        return self.pixel_spacing * 1000.0

    @property
    def pixel_spacing(self) -> SizeMm | None:
        """Return pixel spacing in mm/pixel."""
        return self._image_data.pixel_spacing

    @property
    def mm_size(self) -> SizeMm | None:
        """Return slide size in mm."""
        return self.dataset.mm_size

    @property
    def mm_depth(self) -> float | None:
        """Return imaged depth in mm."""
        return self.dataset.mm_depth

    @property
    def slice_thickness(self) -> float | None:
        """Return slice thickness."""
        return self.dataset.slice_thickness

    @property
    def slice_spacing(self) -> float | None:
        """Return slice spacing."""
        return self.dataset.spacing_between_slices

    @property
    def focus_method(self) -> str:
        return self.dataset.focus_method

    @property
    def ext_depth_of_field(self) -> bool:
        return self.dataset.ext_depth_of_field

    @property
    def ext_depth_of_field_planes(self) -> int | None:
        return self.dataset.ext_depth_of_field_planes

    @property
    def ext_depth_of_field_plane_distance(self) -> float | None:
        return self.dataset.ext_depth_of_field_plane_distance

    @property
    def identifier(self) -> UID:
        """Return identifier (instance uid for single file instance or
        concatenation uid for multiple file instance)."""
        return self._identifier

    @property
    def default_z(self) -> float:
        return self._image_data.default_z

    @property
    def default_path(self) -> str:
        return self._image_data.default_path

    @property
    def focal_planes(self) -> list[float]:
        return self._image_data.focal_planes

    @property
    def optical_paths(self) -> list[str]:
        return self._image_data.optical_paths

    @property
    def tiled_size(self) -> Size:
        return self._image_data.tiled_size

    @property
    def uids(self) -> SlideUids:
        """Return base uids"""
        return self._uids

    @property
    def image_coordinate_system(self) -> ImageCoordinateSystem | None:
        return self.image_data.image_coordinate_system

    def get_tile(
        self,
        tile_point: Point,
        z: float,
        path: str,
        crop_to_image_boundary: bool = True,
    ) -> np.ndarray:
        """Get the pixels of a tile.

        Parameters
        ----------
        tile_point: Point
            Tile to get.
        z: float
            Z coordinate.
        path: str
            Optical path.
        crop_to_image_boundary: bool = True
            If True, an edge tile is cropped to remove the part outside the
            image.

        Returns
        -------
        np.ndarray
            The tile pixels.
        """
        array = self._image_data.get_decoded_tile(tile_point, z, path)
        if crop_to_image_boundary:
            array = self._crop_tile(tile_point, array)
        return array

    def get_encoded_tile(
        self,
        tile: Point,
        z: float,
        path: str,
        crop_to_image_boundary: bool = True,
    ) -> bytes:
        """Get a tile as encoded bytes.

        Parameters
        ----------
        tile: Point
            Tile to get.
        z: float
            Z coordinate.
        path: str
            Optical path.
        crop_to_image_boundary: bool = True
            If True, an edge tile is cropped to remove the part outside the
            image, which requires re-encoding the cropped tile.

        Returns
        -------
        bytes
            The tile as encoded bytes.
        """
        if not crop_to_image_boundary:
            return self._image_data.get_encoded_tile(tile, z, path)
        # Check if tile is an edge tile that should be cropped
        cropped_tile_region = self._image_data.image_region.inside_crop(
            tile, self._image_data.tile_size
        )
        if cropped_tile_region.size == self._image_data.tile_size:
            return self._image_data.get_encoded_tile(tile, z, path)
        array = self._image_data.get_decoded_tile(tile, z, path)
        left, upper, right, lower = cropped_tile_region.box_from_origin
        return self._image_data.encoder.encode(array[upper:lower, left:right])

    def get_region(
        self,
        region: Region,
        z: float,
        path: str,
        output_size: Size | None = None,
        *,
        executor: ReadExecutor,
    ) -> np.ndarray:
        """Read the pixels of ``region`` from this instance.

        Assembles the pixels covering ``region`` and, if ``output_size`` is
        given, downsamples the result to it. Pillow is derived by the caller
        (``WsiDicom``) at the outermost boundary.

        Parameters
        ----------
        region: Region
            Pixel region to read.
        z: float
            Z coordinate.
        path: str
            Optical path.
        output_size: Size | None = None
            If given and different from the region size, the assembled region is
            downsampled to this size.
        executor: ReadExecutor
            Executor that splits the read's tile fetches across worker threads.

        Returns
        -------
        np.ndarray
            The region as ``(rows, columns)`` or ``(rows, columns, samples)``.
        """
        array = self._assemble_region(region, z, path, executor=executor)
        if output_size is None or output_size == region.size:
            return array
        return self._downsampler.downsample(array, output_size)

    def _assemble_region(
        self,
        region: Region,
        z: float,
        path: str,
        *,
        executor: ReadExecutor,
    ) -> np.ndarray:
        """Assemble the full-resolution pixels covering ``region``."""
        image_data = self._image_data
        tile_region = self._get_tile_range(region, z, path)
        if tile_region.size.area == 1:
            # Only one tile, no need to stitch
            array = image_data.get_decoded_tile(tile_region.start, z, path)
            tile_crop = region.inside_crop(tile_region.start, image_data.tile_size)
            if tile_crop.size != image_data.tile_size:
                left, upper, right, lower = tile_crop.box
                array = array[upper:lower, left:right]
            return array
        return self._stitcher.stitch_parallel(
            region=region,
            tile_region=tile_region,
            fetch=lambda chunk: image_data.get_decoded_tiles(chunk, z, path),
            tile_size=image_data.tile_size,
            dtype=image_data.dtype,
            samples_per_pixel=image_data.samples_per_pixel,
            executor=executor,
        )

    def get_thumbnail(
        self,
        max_size: Size,
        z: float,
        path: str,
        *,
        executor: ReadExecutor,
    ) -> np.ndarray:
        """Read the full image, resampled to fit within ``max_size``.

        Reads the whole image region, then resamples (via the instance's
        downsampler) to fit within ``max_size`` preserving aspect ratio,
        without upscaling.

        Parameters
        ----------
        max_size: Size
            Upper size limit of the thumbnail in pixels.
        z: float
            Z coordinate.
        path: str
            Optical path.
        executor: ReadExecutor
            Executor that splits image data reads across worker threads.

        Returns
        -------
        np.ndarray
            The thumbnail pixels.
        """
        region = Region(position=Point(0, 0), size=self.size)
        array = self.get_region(region, z, path, executor=executor)
        return self._downsampler.thumbnail(array, max_size)

    def _crop_tile(self, tile_point: Point, tile: np.ndarray) -> np.ndarray:
        """Crop a tile to the image boundary if it is an edge tile.

        Parameters
        ----------
        tile_point: Point
            Tile coordinate of the tile.
        tile: np.ndarray
            Tile pixels to crop.

        Returns
        -------
        np.ndarray
            The tile cropped to the image boundary if it is an edge tile,
            otherwise the tile unmodified.
        """
        tile_crop = self._image_data.image_region.inside_crop(
            tile_point, self._image_data.tile_size
        )
        # Check if tile is an edge tile that should be cropped
        if tile_crop.size != self._image_data.tile_size:
            left, upper, right, lower = tile_crop.box
            return tile[upper:lower, left:right]
        return tile

    def _get_tile_range(self, pixel_region: Region, z: float, path: str) -> Region:
        """Return range of tiles to cover the pixel region.

        Parameters
        ----------
        pixel_region: Region
            Pixel region to cover.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        -------
        Region
            Tile region, in tile coordinates, covering the pixel region.

        Raises
        ------
        WsiDicomOutOfBoundsError
            If the tile region falls outside the tiled image.
        """
        image_data = self._image_data
        tile_region = Region.from_points(
            pixel_region.start // image_data.tile_size,
            (pixel_region.end - 1) // image_data.tile_size + 1,
        )
        if not image_data.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBoundsError(
                f"Tile region {tile_region}", f"tiled size {image_data.tiled_size}"
            )
        return tile_region

    @classmethod
    def create_label(
        cls, image: Image | str | Path | UPath, base_dataset: Dataset
    ) -> "WsiInstance":
        """Create a label WsiInstance.

        Parameters
        ----------
        image: Image | str | Path | UPath
            Image or path to image.
        base_dataset: Dataset
            Base dataset to include.

        Returns
        -------
        WsiInstance
            Created label WsiInstance.
        """
        if isinstance(image, Image):
            image_data = NumpyImageData.from_image(image)
        else:
            image_data = NumpyImageData.from_file(image)
        instance_dataset = WsiDataset.create_instance_dataset(
            base_dataset, ImageType.LABEL, image_data
        )
        return cls(instance_dataset, image_data)

    @classmethod
    def create_instance(
        cls,
        image_data: ImageData,
        base_dataset: Dataset,
        image_type: ImageType,
        pyramid_index: int | None = None,
    ) -> "WsiInstance":
        """Create a WsiInstance from image data.

        Builds the instance dataset from the image data's metadata and composes
        the instance.

        Parameters
        ----------
        image_data: ImageData
            Format-specific image data for the instance.
        base_dataset: Dataset
            Base dataset to include.
        image_type: ImageType
            Type of instance to create.
        pyramid_index: int | None = None
            Pyramid index of image data, if volume image.

        Returns
        -------
        WsiInstance
            Created WsiInstance.
        """
        instance_dataset = WsiDataset.create_instance_dataset(
            base_dataset, image_type, image_data, pyramid_index
        )
        return cls(instance_dataset, image_data)

    @staticmethod
    def check_duplicate_instance(
        instances: Sequence["WsiInstance"], self: object
    ) -> None:
        """Check for duplicates in list of instances. Instances are duplicate
        if instance identifier (file instance uid or concatenation uid) match.
        Stops at first found duplicate and raises WsiDicomUidDuplicateError.

        Parameters
        ----------
        instances: Sequence['WsiInstance']
            List of instances to check.
        caller: Object
            Object that the instances belongs to.
        """
        instance_identifiers: list[str] = []
        for instance in instances:
            instance_identifier = instance.identifier
            if instance_identifier not in instance_identifiers:
                instance_identifiers.append(instance_identifier)
            else:
                raise WsiDicomUidDuplicateError(str(instance), str(self))

    def _validate_instance(
        self, datasets: Sequence[WsiDataset]
    ) -> tuple[UID, SlideUids]:
        """Check that no files in instance are duplicate, that all files in
        instance matches (uid, type and size).
        Raises WsiDicomMatchError otherwise.
        Returns the matching file uid.

        Returns
        -------
        tuple[UID, SlideUids]
            Instance identifier uid and base uids
        """
        WsiDataset.check_duplicate_dataset(datasets, self)

        base_dataset = datasets[0]
        for dataset in datasets[1:]:
            if not base_dataset.matches_instance(dataset):
                raise WsiDicomError("Datasets in instances does not match")
        return (
            base_dataset.uids.identifier,
            base_dataset.uids.slide,
        )

    def matches(self, other_instance: "WsiInstance") -> bool:
        """Return true if other instance is of the same group as self.

        Parameters
        ----------
        other_instance: WsiInstance
            Instance to check.

        Returns
        -------
        bool
            True if instances are of same group.

        """
        return (
            self.uids.matches(other_instance.uids)
            and self.size == other_instance.size
            and self.tile_size == other_instance.tile_size
            and self.image_type == other_instance.image_type
        )
