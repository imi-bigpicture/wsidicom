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
from collections.abc import Iterable, Iterator, Sequence
from functools import cached_property

import numpy as np
from pydicom.uid import UID

from wsidicom.errors import (
    WsiDicomMatchError,
    WsiDicomNoResolutionError,
    WsiDicomNotFoundError,
    WsiDicomOutOfBoundsError,
)
from wsidicom.geometry import Point, Region, RegionMm, Size, SizeMm
from wsidicom.instance import (
    ImageData,
    WsiDataset,
    WsiInstance,
)
from wsidicom.metadata import ImageCoordinateSystem, ImageType
from wsidicom.stringprinting import dict_pretty_str
from wsidicom.thread import ReadExecutor
from wsidicom.uid import SlideUids


class Instances:
    """Represents a group of instances having the same size, but possibly
    different z coordinate and/or optical path."""

    def __init__(self, instances: Iterable[WsiInstance]):
        """Create a group of WsiInstances. Instances should match in the common
        uids, wsi type, and tile size.

        Parameters
        ----------
        instances: Sequence[WsiInstance]
            Instances to build the group.
        """
        self._instances = {  # key is identifier (Uid)
            instance.identifier: instance for instance in instances
        }
        self._validate_group()

        base_instance = next(iter(self._instances.values()))
        self._image_type = base_instance.image_type
        self._uids = base_instance.uids

        self._size = base_instance.size
        self._pixel_spacing = base_instance.pixel_spacing
        self._default_instance_uid = base_instance.identifier

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.instances.values()})"

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(self, indent: int = 0, depth: int | None = None) -> str:
        string = f"Image: size: {self.size} px, mpp: {self.mpp} um/px"
        if depth is not None:
            depth -= 1
            if depth < 0:
                return string
        string += " Instances: " + dict_pretty_str(self.instances, indent, depth)
        return string

    def __getitem__(self, index: UID) -> WsiInstance:
        return self.instances[index]

    @property
    def uids(self) -> SlideUids:
        """Return uids"""
        return self._uids

    @property
    def image_type(self) -> ImageType:
        """Return wsi type"""
        return self._image_type

    @property
    def size(self) -> Size:
        """Return image size in pixels"""
        return self._size

    @property
    def mpp(self) -> SizeMm | None:
        """Return pixel spacing in um/pixel"""
        if self.pixel_spacing is None:
            return None
        return self.pixel_spacing * 1000.0

    @property
    def pixel_spacing(self) -> SizeMm | None:
        """Return pixel spacing in mm/pixel"""
        return self._pixel_spacing

    @property
    def instances(self) -> dict[UID, WsiInstance]:
        """Return contained instances"""
        return self._instances

    @property
    def default_instance(self) -> WsiInstance:
        """Return default instance"""
        return self.instances[self._default_instance_uid]

    @cached_property
    def image_data_map(self) -> dict[tuple[str, float], ImageData]:
        """Return mapping from (optical_path, focal_plane) to ImageData."""
        return {
            (optical_path, z): instance.image_data
            for instance in self._instances.values()
            for optical_path in instance.optical_paths
            for z in sorted(instance.focal_planes)
        }

    @property
    def tiled_size(self) -> Size:
        """Return tiled size (number of tiles in each dimension)."""
        return self.default_instance.image_data.tiled_size

    @property
    def datasets(self) -> list[WsiDataset]:
        """Return contained datasets."""
        instance_datasets = [instance.datasets for instance in self.instances.values()]
        return [dataset for sublist in instance_datasets for dataset in sublist]

    @property
    def optical_paths(self) -> list[str]:
        return list(
            {
                path
                for instance in self.instances.values()
                for path in instance.optical_paths
            }
        )

    @property
    def focal_planes(self) -> list[float]:
        return sorted(
            {
                focal_plane
                for innstance in self.instances.values()
                for focal_plane in innstance.focal_planes
            }
        )

    @property
    def focal_planes_by_optical_path(self) -> dict[str, list[float]]:
        """Return the focal planes present for each optical path.

        The (optical path, focal plane) grid may be sparse, i.e. an optical path
        is not required to have every focal plane.
        """
        focal_planes_by_optical_path: dict[str, list[float]] = defaultdict(list)
        for optical_path, focal_plane in self.image_data_map:
            focal_planes_by_optical_path[optical_path].append(focal_plane)
        return {
            optical_path: sorted(focal_planes)
            for optical_path, focal_planes in focal_planes_by_optical_path.items()
        }

    @cached_property
    def _instance_map(self) -> dict[tuple[str, float], WsiInstance]:
        """Return mapping from (optical_path, focal_plane) to its instance."""
        return {
            (optical_path, z): instance
            for instance in self._instances.values()
            for optical_path in instance.optical_paths
            for z in instance.focal_planes
        }

    def instance_at(self, optical_path: str, focal_plane: float) -> WsiInstance:
        """Return the instance providing the given optical path and focal plane."""
        return self._instance_map[(optical_path, focal_plane)]

    @property
    def image_coordinate_system(self) -> ImageCoordinateSystem | None:
        return self.default_instance.image_coordinate_system

    def matches(self, other_group: "Instances") -> bool:
        """Check if group matches other group. If strict common Uids should
        match. Wsi type should always match.

        Parameters
        ----------
        other_group: Group
            Other group to match against.

        Returns
        -------
        bool
            True if other group matches.
        """
        return (
            self.uids.matches(other_group.uids)
            and other_group.image_type == self.image_type
        )

    def valid_pixels(self, region: Region) -> bool:
        """Check if pixel region is within the size of the group image size.

        Parameters
        ----------
        region: Region
            Pixel region to check

        Returns
        -------
        bool
            True if pixel position and size is within image
        """
        # Return true if inside pixel plane.
        image_region = Region(Point(0, 0), self.size)
        return region.is_inside(image_region)

    def get_instance(
        self, z: float | None = None, path: str | None = None
    ) -> WsiInstance:
        """Search for instance fulfilling the parameters.
        The behavior when z and/or path is none could be made more
        clear.

        Parameters
        ----------
        z: float | None = None
            Z coordinate of the searched instance
        path: str | None = None
            Optical path of the searched instance

        Returns
        -------
        WsiInstance
            The instance containing selected path and z coordinate
        """
        if z is None and path is None:
            return self.default_instance

        # Sort instances by number of focal planes (prefer simplest instance)
        sorted_instances = sorted(
            list(self._instances.values()), key=lambda i: len(i.focal_planes)
        )
        try:
            if z is None:
                # Select the instance with selected optical path
                return next(
                    instance
                    for instance in sorted_instances
                    if path in instance.optical_paths
                )
            if path is None:
                # Select the instance with selected z
                return next(
                    instance
                    for instance in sorted_instances
                    if z in instance.focal_planes
                )

            # Select by both path and z
            return next(
                instance
                for instance in sorted_instances
                if (z in instance.focal_planes and path in instance.optical_paths)
            )
        except StopIteration as exception:
            raise WsiDicomNotFoundError(
                f"Instance for path: {path}, z: {z}", "group"
            ) from exception

    def get_default_full(self, *, executor: ReadExecutor) -> np.ndarray:
        """Read full image using default z coordinate and path.

        Parameters
        ----------
        executor: ReadExecutor
            Executor that splits image data reads across worker threads.

        Returns
        -------
        Image
            Full image of the group.
        """
        instance = self.default_instance
        z = instance.default_z
        path = instance.default_path
        region = Region(position=Point(x=0, y=0), size=self.size)
        return self.get_region(region, z, path, executor=executor)

    def iter_encoded_tiles(
        self,
        focal_planes: Sequence[float] | None = None,
        optical_paths: Sequence[str] | None = None,
    ) -> Iterator[bytes]:
        """Iterate all tiles in raster order, yielding encoded tile data.

        Parameters
        ----------
        focal_planes: Sequence[float] | None = None
            Optional subset of focal planes to iterate. Defaults to all.
        optical_paths: Sequence[str] | None = None
            Optional subset of optical paths to iterate. Defaults to all.

        Yields
        ------
        bytes
            Encoded tile data.
        """
        for image_data, z, path, chunk in self._iter_tile_chunks(
            focal_planes, optical_paths
        ):
            yield from image_data.get_encoded_tiles(chunk, z, path)

    def iter_decoded_tiles(
        self,
        focal_planes: Sequence[float] | None = None,
        optical_paths: Sequence[str] | None = None,
    ) -> Iterator[np.ndarray]:
        """Iterate all tiles in raster order, yielding tile pixels.

        Parameters
        ----------
        focal_planes: Sequence[float] | None = None
            Optional subset of focal planes to iterate. Defaults to all.
        optical_paths: Sequence[str] | None = None
            Optional subset of optical paths to iterate. Defaults to all.

        Yields
        ------
        np.ndarray
            Tile pixels.
        """
        for image_data, z, path, chunk in self._iter_tile_chunks(
            focal_planes, optical_paths
        ):
            yield from image_data.get_decoded_tiles(chunk, z, path)

    def _iter_tile_chunks(
        self,
        focal_planes: Sequence[float] | None = None,
        optical_paths: Sequence[str] | None = None,
    ) -> Iterator[tuple[ImageData, float, str, list[Point]]]:
        """Iterate tile position chunks in raster order.

        Chunk size is determined per image data from its
        suggested_minimum_chunk_size.

        Parameters
        ----------
        focal_planes: Sequence[float] | None = None
            Optional subset of focal planes to iterate. Defaults to all.
        optical_paths: Sequence[str] | None = None
            Optional subset of optical paths to iterate. Defaults to all.

        Yields
        ------
        Tuple[ImageData, float, str, List[Point]]
            Image data, focal plane, optical path, and chunk of tile positions.
        """
        for path in optical_paths if optical_paths is not None else self.optical_paths:
            for z in focal_planes if focal_planes is not None else self.focal_planes:
                image_data = self.image_data_map[(path, z)]
                chunk_size = image_data.suggested_minimum_chunk_size
                for y in range(self.tiled_size.height):
                    for x in range(0, self.tiled_size.width, chunk_size):
                        end = min(x + chunk_size, self.tiled_size.width)
                        chunk = [Point(cx, y) for cx in range(x, end)]
                        yield image_data, z, path, chunk

    def get_region(
        self,
        region: Region,
        z: float | None = None,
        path: str | None = None,
        output_size: Size | None = None,
        *,
        executor: ReadExecutor,
    ) -> np.ndarray:
        """Read region defined by pixels.

        Parameters
        ----------
        region: Region
            Pixel region to read.
        z: float | None = None
            Z coordinate, optional
        path: str | None = None
            optical path, optional
        output_size: Size | None = None
            If given and different from the region size, the read region is
            downsampled to this size.
        executor: ReadExecutor
            Executor that splits image data reads across worker threads.

        Returns
        -------
        Image
            Region as image
        """

        instance = self.get_instance(z, path)
        if z is None:
            z = instance.default_z
        if path is None:
            path = instance.default_path
        return instance.get_region(region, z, path, output_size, executor=executor)

    def _resolve_slide_origin(
        self, region: RegionMm, slide_origin: bool
    ) -> tuple[RegionMm, ImageCoordinateSystem | None]:
        """Map a mm region to image origin if reading from the slide origin.

        Parameters
        ----------
        region: RegionMm
            Region in mm, defined relative to the slide origin if
            ``slide_origin`` is True, otherwise the image origin.
        slide_origin: bool
            If the region is defined relative to the slide origin.

        Returns
        -------
        tuple[RegionMm, ImageCoordinateSystem | None]
            The (possibly mapped) region together with the coordinate system
            that must rotate the read image back to the slide orientation, or
            ``None`` when reading from the image origin (no rotation needed).
        """
        if not slide_origin:
            return region, None
        if self.image_coordinate_system is None:
            raise ValueError(
                "Can't map to slide region as image coordinate system is not defined."
            )
        to_coordinate_system = self.image_coordinate_system
        return to_coordinate_system.slide_to_image(region), to_coordinate_system

    def _read_and_rotate(
        self,
        pixel_region: Region,
        z: float | None,
        path: str | None,
        output_size: Size | None,
        to_coordinate_system: ImageCoordinateSystem | None,
        executor: ReadExecutor,
    ) -> np.ndarray:
        """Read a pixel region (downsampling to ``output_size``), then rotate.

        The downsample is applied pre-rotate at the single downsampling site
        (``get_region``); both an integer downsample and an absolute resample
        commute with the orthogonal slide-origin rotation, so the result is
        identical to resampling after the rotation.

        Parameters
        ----------
        pixel_region: Region
            Pixel region to read, in image orientation.
        z: float | None
            Z coordinate, optional.
        path: str | None
            Optical path, optional.
        output_size: Size | None
            If given and different from the region size, the read region is
            downsampled to this size.
        to_coordinate_system: ImageCoordinateSystem | None
            Coordinate system to rotate the read image back to the slide
            orientation, or ``None`` to skip rotation.
        executor: ReadExecutor
            Executor that splits image data reads across worker threads.

        Returns
        -------
        np.ndarray
            Region pixels, rotated to the slide orientation if a coordinate
            system was given.
        """
        array = self.get_region(
            pixel_region,
            z,
            path,
            output_size=output_size,
            executor=executor,
        )
        if to_coordinate_system is not None:
            array = self._rotate(array, to_coordinate_system.rotation)
        return array

    def _rotate(self, array: np.ndarray, rotation: float) -> np.ndarray:
        """Rotate an assembled region back to the slide orientation.

        Slide-origin rotations are 90-degree multiples, for which ``np.rot90``
        is a bit-exact, in-numpy rotation (matching Pillow's counter-clockwise
        rotate with ``expand``), keeping the read path free of a Pillow
        round-trip.

        Parameters
        ----------
        array: np.ndarray
            Region to rotate, in image orientation.
        rotation: float
            Rotation in degrees to apply to reach the slide orientation. Must be
            a multiple of 90 degrees.

        Returns
        -------
        np.ndarray
            The rotated region.
        """
        if rotation % 90 != 0:
            raise NotImplementedError(
                f"Only 90-degree slide rotations are supported, got {rotation}."
            )
        return np.rot90(array, k=int(rotation // 90) % 4)

    def get_region_mm(
        self,
        region: RegionMm,
        z: float | None = None,
        path: str | None = None,
        slide_origin: bool = False,
        scale: int = 1,
        *,
        executor: ReadExecutor,
    ) -> np.ndarray:
        """Read region defined by mm.

        Parameters
        ----------
        region: RegionMm
            Region defining upper left corner and size in mm.
        z: float | None = None
            Z coordinate, optional.
        path: str | None = None
            optical path, optional.
        slide_origin: bool = False.
            If to use the slide origin instead of image origin.
        scale: int = 1
            Integer factor by which the returned image is downsampled.
            ``scale=1`` returns full resolution, ``scale=2`` halves each
            dimension.
        executor: ReadExecutor
            Executor that splits image data reads across worker threads.

        Returns
        -------
        Image
            Region as image
        """
        image_region, to_coordinate_system = self._resolve_slide_origin(
            region, slide_origin
        )
        pixel_region = self._mm_to_pixel(image_region)
        output_size = pixel_region.size // scale if scale != 1 else None
        return self._read_and_rotate(
            pixel_region,
            z,
            path,
            output_size,
            to_coordinate_system,
            executor,
        )

    def get_region_mpp(
        self,
        region: RegionMm,
        pixel_spacing: float,
        z: float | None = None,
        path: str | None = None,
        slide_origin: bool = False,
        *,
        executor: ReadExecutor,
    ) -> np.ndarray:
        """Read region defined by mm, resampled to a requested pixel spacing.

        Like ``get_region_mm`` but resampling to an absolute pixel spacing
        rather than an integer level scale: the returned image has the
        requested ``pixel_spacing``.

        Parameters
        ----------
        region: RegionMm
            Region defining upper left corner and size in mm.
        pixel_spacing: float
            Requested pixel spacing (mm/pixel) of the returned image.
        z: float | None = None
            Z coordinate, optional.
        path: str | None = None
            optical path, optional.
        slide_origin: bool = False.
            If to use the slide origin instead of image origin.
        executor: ReadExecutor
            Executor that splits image data reads across worker threads.

        Returns
        -------
        Image
            Region as image
        """
        image_region, to_coordinate_system = self._resolve_slide_origin(
            region, slide_origin
        )
        pixel_region = self._mm_to_pixel(image_region)
        output_size = image_region.size // pixel_spacing
        return self._read_and_rotate(
            pixel_region,
            z,
            path,
            output_size,
            to_coordinate_system,
            executor,
        )

    def get_thumbnail(
        self,
        max_size: Size,
        z: float | None = None,
        path: str | None = None,
        *,
        executor: ReadExecutor,
    ) -> np.ndarray:
        """Read the full image, resampled to fit within ``max_size``.

        Parameters
        ----------
        max_size: Size
            Upper size limit of the thumbnail in pixels.
        z: float | None = None
            Z coordinate, optional.
        path: str | None = None
            optical path, optional.
        executor: ReadExecutor
            Executor that splits image data reads across worker threads.

        Returns
        -------
        Image
            The thumbnail image.
        """
        instance = self.get_instance(z, path)
        if z is None:
            z = instance.default_z
        if path is None:
            path = instance.default_path
        return instance.get_thumbnail(max_size, z, path, executor=executor)

    def get_tile(
        self,
        tile: Point,
        z: float | None = None,
        path: str | None = None,
        crop_to_image_boundary: bool = True,
    ) -> np.ndarray:
        """Return tile at tile coordinate x, y as image.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate
        z: float | None = None
            Z coordinate
        path: str | None = None
            Optical path
        crop_to_image_boundary: bool = True
            Whether to crop tiles that exceed image boundary.

        Returns
        -------
        Image
            The tile as image
        """

        instance = self.get_instance(z, path)
        if z is None:
            z = instance.default_z
        if path is None:
            path = instance.default_path
        return instance.get_tile(tile, z, path, crop_to_image_boundary)

    def get_encoded_tile(
        self,
        tile: Point,
        z: float | None = None,
        path: str | None = None,
        crop_to_image_boundary: bool = True,
    ) -> bytes:
        """Return tile at tile coordinate x, y as bytes.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate
        z: float | None = None
            Z coordinate
        path: str | None = None
            Optical path
        crop_to_image_boundary: bool = True
            Whether to crop tiles that exceed image boundary.

        Returns
        -------
        bytes
            The tile as bytes
        """
        instance = self.get_instance(z, path)
        if z is None:
            z = instance.default_z
        if path is None:
            path = instance.default_path
        return instance.get_encoded_tile(tile, z, path, crop_to_image_boundary)

    def _mm_to_pixel(self, region: RegionMm) -> Region:
        """Convert region in mm to pixel region.

        Parameters
        ----------
        region: RegionMm
            Region in mm

        Returns
        -------
        Region
            Region in pixels
        """
        if self.pixel_spacing is None:
            raise WsiDicomNoResolutionError()
        pixel_region = Region(
            position=region.position // self.pixel_spacing,
            size=region.size // self.pixel_spacing,
        )
        if not self.valid_pixels(pixel_region):
            raise WsiDicomOutOfBoundsError(
                f"Region {region}", f"level size {self.size}"
            )
        return pixel_region

    def _validate_group(self):
        """Check that no file or instance in group is duplicate, and if strict
        instances in group matches. Raises WsiDicomMatchError otherwise.
        """
        instances = list(self.instances.values())
        base_instance = instances[0]
        for instance in instances[1:]:
            if not base_instance.matches(instance):
                raise WsiDicomMatchError(str(instance), str(self))

        WsiDataset.check_duplicate_dataset(self.datasets, self)
        WsiInstance.check_duplicate_instance(instances, self)


class Overview(Instances):
    pass


class Label(Instances):
    pass


class Thumbnail(Instances):
    @property
    def mpp(self) -> SizeMm:
        if self.pixel_spacing is None:
            raise WsiDicomNoResolutionError()
        return self.pixel_spacing * 1000.0

    @property
    def pixel_spacing(self) -> SizeMm:
        if self._pixel_spacing is None:
            raise WsiDicomNoResolutionError()
        return self._pixel_spacing
