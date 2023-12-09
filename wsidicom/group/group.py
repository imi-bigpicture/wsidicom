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
from typing import Dict, Iterable, List, Optional, OrderedDict, Sequence, Set, Tuple

from PIL.Image import Image
from pydicom.uid import UID

from wsidicom.errors import (
    WsiDicomMatchError,
    WsiDicomNoResolutionError,
    WsiDicomNotFoundError,
    WsiDicomOutOfBoundsError,
)
from wsidicom.geometry import Point, Region, RegionMm, Size, SizeMm
from wsidicom.instance import (
    ImageCoordinateSystem,
    ImageData,
    ImageType,
    WsiDataset,
    WsiInstance,
)
from wsidicom.stringprinting import dict_pretty_str
from wsidicom.uid import SlideUids
from wsidicom.config import settings


class Group:
    """Represents a group of instances having the same size, but possibly
    different z coordinate and/or optical path."""

    def __init__(self, instances: Sequence[WsiInstance]):
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

        base_instance = instances[0]
        self._image_type = base_instance.image_type
        self._uids = base_instance.uids

        self._size = base_instance.size
        self._pixel_spacing = base_instance.pixel_spacing
        self._default_instance_uid: UID = base_instance.identifier

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.instances.values()})"

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
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
    def mpp(self) -> Optional[SizeMm]:
        """Return pixel spacing in um/pixel"""
        if self.pixel_spacing is None:
            return None
        return self.pixel_spacing * 1000.0

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        """Return pixel spacing in mm/pixel"""
        return self._pixel_spacing

    @property
    def instances(self) -> Dict[UID, WsiInstance]:
        """Return contained instances"""
        return self._instances

    @property
    def default_instance(self) -> WsiInstance:
        """Return default instance"""
        return self.instances[self._default_instance_uid]

    @property
    def datasets(self) -> List[WsiDataset]:
        """Return contained datasets."""
        instance_datasets = [instance.datasets for instance in self.instances.values()]
        return [dataset for sublist in instance_datasets for dataset in sublist]

    @property
    def optical_paths(self) -> List[str]:
        return list(
            {
                path
                for instance in self.instances.values()
                for path in instance.optical_paths
            }
        )

    @property
    def focal_planes(self) -> List[float]:
        return list(
            {
                focal_plane
                for innstance in self.instances.values()
                for focal_plane in innstance.focal_planes
            }
        )

    @property
    def image_coordinate_system(self) -> Optional[ImageCoordinateSystem]:
        return self.default_instance.image_coordinate_system

    @classmethod
    def open(
        cls,
        instances: Iterable[WsiInstance],
    ) -> List["Group"]:
        """Return list of groups created from wsi instances.

        Parameters
        ----------
        files: Iterable[WsiInstance]
            Instances to create groups from.

        Returns
        ----------
        List[Group]
            List of created groups.

        """
        groups: List["Group"] = []

        grouped_instances = cls._group_instances(instances)

        for group in grouped_instances.values():
            groups.append(cls(group))

        return groups

    def matches(self, other_group: "Group") -> bool:
        """Check if group matches other group. If strict common Uids should
        match. Wsi type should always match.

        Parameters
        ----------
        other_group: Group
            Other group to match against.

        Returns
        ----------
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
        ----------
        bool
            True if pixel position and size is within image
        """
        # Return true if inside pixel plane.
        image_region = Region(Point(0, 0), self.size)
        return region.is_inside(image_region)

    def get_instance(
        self, z: Optional[float] = None, path: Optional[str] = None
    ) -> WsiInstance:
        """Search for instance fulfilling the parameters.
        The behavior when z and/or path is none could be made more
        clear.

        Parameters
        ----------
        z: Optional[float] = None
            Z coordinate of the searched instance
        path: Optional[str] = None
            Optical path of the searched instance

        Returns
        ----------
        WsiInstance
            The instance containing selected path and z coordinate
        """
        if z is None and path is None:
            instance = self.default_instance
            z = instance.default_z
            path = instance.default_path

            return self.default_instance

        # Sort instances by number of focal planes (prefer simplest instance)
        sorted_instances = sorted(
            list(self._instances.values()), key=lambda i: len(i.focal_planes)
        )
        try:
            if z is None:
                # Select the instance with selected optical path
                instance = next(i for i in sorted_instances if path in i.optical_paths)
            elif path is None:
                # Select the instance with selected z
                instance = next(i for i in sorted_instances if z in i.focal_planes)
            else:
                # Select by both path and z
                instance = next(
                    i
                    for i in sorted_instances
                    if (z in i.focal_planes and path in i.optical_paths)
                )
        except StopIteration as exception:
            raise WsiDicomNotFoundError(
                f"Instance for path: {path}, z: {z}", "group"
            ) from exception
        if z is None:
            z = instance.default_z
        if path is None:
            path = instance.default_path
        return instance

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
        z: Optional[float] = None,
        path: Optional[str] = None,
        threads: int = 1,
    ) -> Image:
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
        threads: int = 1
            Number of threads to use for read.

        Returns
        ----------
        Image
            Region as image
        """

        instance = self.get_instance(z, path)
        if z is None:
            z = instance.default_z
        if path is None:
            path = instance.default_path
        image = instance.image_data.stitch_tiles(region, path, z, threads)
        return image

    def get_region_mm(
        self,
        region: RegionMm,
        z: Optional[float] = None,
        path: Optional[str] = None,
        slide_origin: bool = False,
        threads: int = 1,
    ) -> Image:
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
        threads: int = 1
            Number of threads to use for read.

        Returns
        ----------
        Image
            Region as image
        """
        to_coordinate_system = None
        if slide_origin:
            if self.image_coordinate_system is None:
                raise ValueError(
                    "Can't map to slide region as image coordinate system is not defined."
                )
            to_coordinate_system = self.image_coordinate_system
            region = to_coordinate_system.slide_to_image(region)
        pixel_region = self.mm_to_pixel(region)
        image = self.get_region(pixel_region, z, path, threads)
        if to_coordinate_system:
            image = image.rotate(
                to_coordinate_system.rotation,
                resample=settings.pillow_resampling_filter,
                expand=True,
            )
        return image

    def get_tile(
        self, tile: Point, z: Optional[float] = None, path: Optional[str] = None
    ) -> Image:
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
        Image
            The tile as image
        """

        instance = self.get_instance(z, path)
        if z is None:
            z = instance.default_z
        if path is None:
            path = instance.default_path
        return instance.image_data.get_tile(tile, z, path)

    def get_encoded_tile(
        self, tile: Point, z: Optional[float] = None, path: Optional[str] = None
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
        instance = self.get_instance(z, path)
        if z is None:
            z = instance.default_z
        if path is None:
            path = instance.default_path
        return instance.image_data.get_encoded_tile(tile, z, path)

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

    @classmethod
    def _group_instances(
        cls, instances: Iterable[WsiInstance]
    ) -> OrderedDict[Size, List[WsiInstance]]:
        """Return instances grouped and sorted by image size.

        Parameters
        ----------
        instances: Iterable[WsiInstance]
            Instances to group by image size.

        Returns
        ----------
        OrderedDict[Size, List[WsiInstance]]:
            Instances grouped by size, with size as key.

        """
        grouped_instances: Dict[Size, List[WsiInstance]] = {}
        for instance in instances:
            try:
                grouped_instances[instance.size].append(instance)
            except KeyError:
                grouped_instances[instance.size] = [instance]
        return OrderedDict(
            sorted(
                grouped_instances.items(), key=lambda item: item[0].width, reverse=True
            )
        )

    @staticmethod
    def _get_frame_information(
        data: Dict[Tuple[str, float], ImageData]
    ) -> Tuple[List[float], List[str], Size]:
        """Return optical_paths, focal planes, and tiled size."""
        focal_planes_by_optical_path: Dict[str, Set[float]] = defaultdict(set)
        all_focal_planes: Set[float] = set()
        tiled_sizes: Set[Size] = set()
        for (optical_path, focal_plane), image_data in data.items():
            focal_planes_by_optical_path[optical_path].add(focal_plane)
            all_focal_planes.add(focal_plane)
            tiled_sizes.add(image_data.tiled_size)

        focal_planes_sparse_by_optical_path = any(
            optical_path_focal_planes != all_focal_planes
            for optical_path_focal_planes in focal_planes_by_optical_path.values()
        )
        if focal_planes_sparse_by_optical_path:
            raise ValueError("Each optical path must have the same focal planes.")

        if len(tiled_sizes) != 1:
            raise ValueError("Expected only one tiled size")
        tiled_size = list(tiled_sizes)[0]
        return (
            sorted(list(all_focal_planes)),
            sorted(list(focal_planes_by_optical_path.keys())),
            tiled_size,
        )
