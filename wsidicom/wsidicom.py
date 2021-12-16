#    Copyright 2021 SECTRA AB
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
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import (Any, Callable, DefaultDict, Dict, List, Optional,
                    OrderedDict, Tuple, Union)

from PIL import Image
from pydicom.dataset import FileMetaDataset
from pydicom.filereader import read_file_meta_info
from pydicom.misc import is_dicom
from pydicom.uid import UID, generate_uid

from wsidicom.errors import (WsiDicomMatchError, WsiDicomNotFoundError,
                             WsiDicomOutOfBoundsError)
from wsidicom.geometry import Point, PointMm, Region, RegionMm, Size, SizeMm
from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.instance import (ImageData, WsiDataset, WsiDicomFile,
                               WsiDicomFileWriter, WsiDicomImageData,
                               WsiInstance)
from wsidicom.optical import OpticalManager
from wsidicom.stringprinting import (dict_pretty_str, list_pretty_str,
                                     str_indent)
from wsidicom.uid import ANN_SOP_CLASS_UID, WSI_SOP_CLASS_UID, BaseUids


class WsiDicomGroup:
    """Represents a group of instances having the same size, but possibly
    different z coordinate and/or optical path."""
    def __init__(
        self,
        instances: List[WsiInstance]
    ):
        """Create a group of WsiInstances. Instances should match in the common
        uids, wsi type, and tile size.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to build the group.
        """
        self._instances = {  # key is identifier (Uid)
            instance.identifier: instance for instance in instances
        }
        self._validate_group()

        base_instance = instances[0]
        self._wsi_type = base_instance.wsi_type
        self._uids = base_instance.uids

        self._size = base_instance.size
        self._pixel_spacing = base_instance.pixel_spacing
        self._default_instance_uid: UID = base_instance.identifier

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.instances.values()})"

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
            ' Instances: ' + dict_pretty_str(self.instances, indent, depth)
        )
        return string

    def __getitem__(self, index: UID) -> WsiInstance:
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
    def instances(self) -> Dict[UID, WsiInstance]:
        """Return contained instances"""
        return self._instances

    @property
    def default_instance(self) -> WsiInstance:
        """Return default instance"""
        return self.instances[self._default_instance_uid]

    @property
    def files(self) -> List[Path]:
        """Return contained files"""
        instance_files = [
            instance.image_data.files for instance in self.instances.values()
        ]
        return [file for sublist in instance_files for file in sublist]

    @property
    def datasets(self) -> List[WsiDataset]:
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
        groups: List['WsiDicomGroup'] = []

        grouped_instances = cls._group_instances(instances)

        for group in grouped_instances.values():
            groups.append(cls(group))

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
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> WsiInstance:
        """Search for instance fullfilling the parameters.
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
        return instance

    def get_default_full(self) -> Image.Image:
        """Read full image using default z coordinate and path.

        Returns
        ----------
        Image.Image
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

        instance = self.get_instance(z, path)
        if z is None:
            z = instance.default_z
        if path is None:
            path = instance.default_path
        image = instance.image_data.stitch_tiles(region, path, z)
        return image

    def get_region_mm(
        self,
        region: RegionMm,
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> Image.Image:
        """Read region defined by mm.

        Parameters
        ----------
        location: float, float
            Upper left corner of region in mm
        size: float
            Size of region in mm
        z: Optional[float] = None
            Z coordinate, optional
        path: Optional[str] = None
            optical path, optional

        Returns
        ----------
        Image.Image
            Region as image
        """
        pixel_region = self.mm_to_pixel(region)
        image = self.get_region(pixel_region, z, path)
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

        instance = self.get_instance(z, path)
        if z is None:
            z = instance.default_z
        if path is None:
            path = instance.default_path
        return instance.image_data.get_tile(tile, z, path)

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
    ) -> OrderedDict[Size, List[WsiInstance]]:
        """Return instances grouped and sorted by image size.

        Parameters
        ----------
        instances: List[WsiInstance]
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
        return OrderedDict(sorted(
            grouped_instances.items(),
            key=lambda item: item[0].width,
            reverse=True)
        )

    def _group_instances_to_file(
        self,
    ) -> List[List[WsiInstance]]:
        """Group instances by properties that can't differ in a DICOM-file,
        i.e. the instances are grouped by output file.

        Returns
        ----------
        List[List[WsiInstance]]
            Instances grouped by common properties.
        """
        groups: DefaultDict[Any, List[WsiInstance]] = (
            defaultdict(list)
        )
        for instance in self.instances.values():
            groups[
                instance.image_data.photometric_interpretation,
                instance.image_data.transfer_syntax,
                instance.ext_depth_of_field,
                instance.ext_depth_of_field_planes,
                instance.ext_depth_of_field_plane_distance,
                instance.focus_method,
                instance.slice_spacing
            ].append(
                instance
            )
        return list(groups.values())

    @staticmethod
    def _list_image_data(
        instances: List[WsiInstance]
    ) -> List[Tuple[Tuple[str, float], ImageData]]:
        """Sort ImageData in instances by optical path and focal
        plane.

        Parameters
        ----------
        instances: List[WsiInstance]
            List of instances with optical paths and focal planes to list and
            sort.

        Returns
        ----------
        Tuple[Tuple[str, float], ImageData]
            ImageData sorted by optical path and focal plane.
        """
        output: Dict[Tuple[str, float], ImageData] = {}
        for instance in instances:
            for optical_path in instance.optical_paths:
                for z in instance.focal_planes:
                    if (optical_path, z) not in output:
                        output[optical_path, z] = instance.image_data
        return list(OrderedDict(output).items())

    @staticmethod
    def _get_number_of_frames(
        instances: List[WsiInstance]
    ) -> int:
        """

        Parameters
        ----------
        instances: List[WsiInstance]


        Returns
        ----------
        int
        """
        frames = 0
        for instance in instances:
            tiles = instance.image_data.tiled_size.area
            optical_paths = len(instance.optical_paths)
            focal_planes = len(instance.focal_planes)
            frames += tiles*optical_paths*focal_planes
        return frames

    def save(
        self,
        output_path: str,
        uid_generator: Callable[..., UID],
        workers: int,
        chunk_size: int
    ) -> List[Path]:
        """Save a WsiDicomGroup to files in output_path. Instances are grouped
        by properties that cant differ in the same file:
            - photometric interpretation
            - transfer syntax
            - extended depth of field (and planes and distance)
            - focus method
            - spacing between slices
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

        Returns
        ----------
        List[str]
            List of paths of created files.
        """
        filepaths: List[Path] = []
        for instances in self._group_instances_to_file():
            uid = uid_generator()
            if output_path == os.devnull:
                filepath = Path(output_path)
            else:
                filepath = Path(os.path.join(output_path, uid + '.dcm'))

            transfer_syntax = instances[0].image_data.transfer_syntax
            dataset = deepcopy(instances[0].dataset)
            frames = self._get_number_of_frames(instances)
            dataset.NumberOfFrames = frames
            self._write_file(
                filepath,
                uid,
                transfer_syntax,
                dataset,
                instances,
                workers,
                chunk_size
            )
            filepaths.append(filepath)
        return filepaths

    @classmethod
    def _write_file(
        cls,
        filepath: Path,
        uid: UID,
        transfer_syntax: UID,
        dataset: WsiDataset,
        instances: List[WsiInstance],
        workers: int,
        chunk_size: int
    ) -> None:
        with WsiDicomFileWriter(filepath) as wsi_file:
            wsi_file.write_preamble()
            wsi_file.write_file_meta(uid, transfer_syntax)
            dataset.SOPInstanceUID = uid
            wsi_file.write_base(dataset)
            frames = dataset.NumberOfFrames
            bot_start, bot_end = wsi_file.write_pixel_data_start(frames)
            frame_positions: List[int] = []
            for (path, z), image_data in cls._list_image_data(instances):
                frame_positions += wsi_file.write_pixel_data(
                    image_data,
                    z,
                    path,
                    workers,
                    chunk_size
                )
            wsi_file.write_pixel_data_end()
            if filepath != Path(os.devnull):
                wsi_file.write_bot(bot_start, bot_end, frame_positions)
            wsi_file.close()


class WsiDicomLevel(WsiDicomGroup):
    """Represents a level in the pyramid and contains one or more instances
    having the same pyramid level index, pixel spacing, and size but possibly
    different focal planes and/or optical paths.
    """
    def __init__(
        self,
        instances: List[WsiInstance],
        base_pixel_spacing: SizeMm
    ):
        """Create a level from list of WsiInstances. Asign the pyramid level
        index from pixel spacing of base level.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to build the level.
        base_pixel_spacing: SizeMm
            Pixel spacing of base level.
        """
        super().__init__(instances)
        self._base_pixel_spacing = base_pixel_spacing
        self._level = self._assign_level(self._base_pixel_spacing)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.instances}, "
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
            ' Instances: ' + dict_pretty_str(self.instances, indent, depth)
        )
        return string

    @property
    def pyramid(self) -> str:
        """Return string representation of the level"""
        return (
            f'Level [{self.level}]'
            f' tiles: {self.default_instance.tiled_size},'
            f' size: {self.size}, mpp: {self.mpp} um/px'
        )

    @property
    def tile_size(self) -> Size:
        return self.default_instance.tile_size

    @property
    def level(self) -> int:
        """Return pyramid level"""
        return self._level

    @classmethod
    def open(
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
        levels: List['WsiDicomLevel'] = []
        instances_grouped_by_level = cls._group_instances(instances)

        base_group = list(instances_grouped_by_level.values())[0]
        base_pixel_spacing = base_group[0].pixel_spacing
        for level in instances_grouped_by_level.values():
            levels.append(cls(level, base_pixel_spacing))

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
        instance = self.get_instance(z, path)
        scaled_region = Region.from_tile(tile, instance.tile_size) * scale
        cropped_region = scaled_region.crop(instance.image_data.image_size)
        if not self.valid_pixels(cropped_region):
            raise WsiDicomOutOfBoundsError(
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
        image = self.get_scaled_tile(tile, scale, z, path)
        instance = self.get_instance(z, path)
        return instance.image_data.encode(image)

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
            raise NotImplementedError(f"Levels needs to be integer")
        return level

    def create_child(
        self,
        scale: int,
        uid_generator: Callable[..., UID],
        workers: int,
        chunk_size: int
    ) -> 'WsiDicomLevel':
        """Creates a new WsiDicomLevel from this level by scaling the image
        data. File is saved in same folder as parent.

        Parameters
        ----------
        scale: int
            Scale factor.
        uid_generator: Callable[..., UID]
            Uid generator to use.
        workers: int
            Maximum number of thread workers to use.
        chunk_size: int
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.

        Returns
        ----------
        'WsiDicomLevel'
            Created scaled level.
        """
        filepaths: List[Path] = []
        if not isinstance(scale, int) or scale < 2:
            raise ValueError(
                "Scale must be integer and larger than 2"
            )
        if not isinstance(
            self.default_instance.image_data,
            WsiDicomImageData
        ):
            raise NotImplementedError(
                "Can only construct pyramid from DICOM WSI files"
            )
        output_path = (
            self.default_instance.image_data._files[0].filepath.parent
        )
        new_image_size = self.default_instance.size / scale
        for instances in self._group_instances_to_file():
            uid = uid_generator()
            filepath = Path(os.path.join(output_path, uid + '.dcm'))
            transfer_syntax = instances[0].image_data.transfer_syntax
            dataset = deepcopy(instances[0].dataset)
            frames = self._get_number_of_frames(instances)
            # Modify dataset to reflect scaled data
            frames = frames // (scale*scale)
            dataset.NumberOfFrames = frames
            dataset.TotalPixelMatrixColumns = new_image_size.width
            dataset.TotalPixelMatrixRows = new_image_size.height

            new_pixel_spacing = self.pixel_spacing * scale
            (
                dataset.SharedFunctionalGroupsSequence[0].
                PixelMeasuresSequence[0].PixelSpacing
            ) = list(new_pixel_spacing.to_tuple())
            self._write_file(
                filepath,
                uid,
                transfer_syntax,
                dataset,
                instances,
                workers,
                chunk_size
            )
            filepaths.append(filepath)

        created_instances = WsiInstance.open(
            [WsiDicomFile(filepath) for filepath in filepaths],
            self.uids,
            self.tile_size
        )
        return WsiDicomLevel(created_instances, self._base_pixel_spacing)


class WsiDicomSeries(metaclass=ABCMeta):
    """Represents a series of WsiDicomGroups with the same image flavor, e.g.
    pyramidal levels, lables, or overviews.
    """

    def __init__(self, groups: List[WsiDicomGroup]):
        """Create a WsiDicomSeries from list of WsiDicomGroups.

        Parameters
        ----------
        groups: List[WsiDicomGroup]
            List of groups to include in the series.
        """
        self._groups: List[WsiDicomGroup] = groups

        if len(self.groups) != 0 and self.groups[0].uids is not None:
            self._uids = self._validate_series(self.groups)
        else:
            self._uids = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.groups})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of groups {self.groups}"

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

    def __len__(self) -> int:
        return len(self.groups)

    @property
    @abstractmethod
    def wsi_type(self) -> str:
        """Should return the wsi type of the series ('VOLUME', 'LABEL', or
        'OVERVIEW'"""
        raise NotImplementedError

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
    def files(self) -> List[Path]:
        """Return contained files."""
        series_files = [series.files for series in self.groups]
        return [file for sublist in series_files for file in sublist]

    @property
    def datasets(self) -> List[WsiDataset]:
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

    @classmethod
    @abstractmethod
    def open(cls, instances: List[WsiInstance]) -> 'WsiDicomSeries':
        raise NotImplementedError

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

    def save(
        self,
        output_path: str,
        uid_generator: Callable[..., UID],
        workers: int,
        chunk_size: int
    ) -> List[Path]:
        """Save WsiDicomSeries as DICOM-files in path.

        Parameters
        ----------
        output_path: str
        uid_generator: Callable[..., UID]
             Function that can gernerate unique identifiers.
        workers: int
            Maximum number of thread workers to use.
        chunk_size:
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.

        Returns
        ----------
        List[Path]
            List of paths of created files.
        """
        filepaths: List[Path] = []
        for group in self.groups:
            group_file_paths = group.save(
                output_path,
                uid_generator,
                workers,
                chunk_size
            )
            filepaths.extend(group_file_paths)
        return filepaths


class WsiDicomLabels(WsiDicomSeries):
    """Represents a series of WsiDicomGroups of the label wsi flavor."""
    @property
    def wsi_type(self) -> str:
        return 'LABEL'

    @classmethod
    def open(cls, instances: List[WsiInstance]) -> 'WsiDicomLabels':
        """Return labels created from wsi files.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to create labels from.

        Returns
        ----------
        WsiDicomOverviews
            Created labels.
        """
        labels = WsiDicomGroup.open(instances)
        return cls(labels)


class WsiDicomOverviews(WsiDicomSeries):
    """Represents a series of WsiDicomGroups of the overview wsi flavor."""
    @property
    def wsi_type(self) -> str:
        return 'OVERVIEW'

    @classmethod
    def open(cls, instances: List[WsiInstance]) -> 'WsiDicomOverviews':
        """Return overviews created from wsi files.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to create overviews from.

        Returns
        ----------
        WsiDicomOverviews
            Created overviews.
        """
        overviews = WsiDicomGroup.open(instances)
        return cls(overviews)


class WsiDicomLevels(WsiDicomSeries):
    """Represents a series of WsiDicomGroups of the volume (e.g. pyramidal
    level) wsi flavor."""
    @property
    def wsi_type(self) -> str:
        return 'VOLUME'

    @classmethod
    def open(cls, instances: List[WsiInstance]) -> 'WsiDicomLevels':
        """Return overviews created from wsi files.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to create levels from.

        Returns
        ----------
        WsiDicomOverviews
            Created levels.
        """
        levels = WsiDicomLevel.open(instances)
        return cls(levels)

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
            raise WsiDicomOutOfBoundsError(
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

    def construct_pyramid(
        self,
        highest_level: int,
        uid_generator: Callable[..., UID] = generate_uid,
        workers: Optional[int] = None,
        chunk_size: int = 100
    ) -> List[Path]:
        """Construct missing pyramid levels from excisting levels.

        Parameters
        ----------
        highest_level: int
        uid_generator: Callable[..., UID] = pydicom.uid.generate_uid
             Function that can gernerate unique identifiers.
        workers: Optional[int] = None
            Maximum number of thread workers to use.
        chunk_size: int = 100
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.

        Returns
        ----------
        List[Path]
            List of paths of created files.
        """
        if workers is None:
            cpus = os.cpu_count()
            if cpus is None:
                workers = 1
            else:
                workers = cpus

        filepaths: List[Path] = []

        for pyramid_level in range(highest_level):
            if pyramid_level not in self._levels.keys():
                # Find the closest larger level for missing level
                closest_level = self.get_closest_by_level(pyramid_level)
                # Create scaled level
                new_level = closest_level.create_child(
                    scale=2,
                    uid_generator=uid_generator,
                    workers=workers,
                    chunk_size=chunk_size
                )
                # Add level to available levels
                self._levels[new_level.level] = new_level
                filepaths += new_level.files
        return filepaths


class WsiDicom:
    """Represent a wsi slide containing pyramidal levels and optionally
    labels and/or overviews."""
    def __init__(
        self,
        levels: WsiDicomLevels,
        labels: WsiDicomLabels,
        overviews: WsiDicomOverviews,
        annotations: List[AnnotationInstance] = []

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
        annotations: List[AnnotationInstance] = []
            Sup-222 annotation instances.
        """

        self._levels = levels
        self._labels = labels
        self._overviews = overviews
        self.annotations = annotations

        self.uids = self._validate_collection(
            [self.levels, self.labels, self.overviews],
        )

        self.optical = OpticalManager.open(
            levels.instances + labels.instances + overviews.instances
        )

        if self.annotations != []:
            for annotation in self.annotations:
                if annotation.base_uids != self.uids:
                    warnings.warn("Annotations uids does not match")

        self.__enter__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.levels}, {self.labels}"
            f"{self.overviews}, {self.annotations})"
        )

    def __str__(self) -> str:
        return self.pretty_str()

    @property
    def base_level(self) -> WsiDicomLevel:
        return self.levels.base_level

    @property
    def size(self) -> Size:
        """Return pixel size of base level."""
        return self.base_level.size

    @property
    def tile_size(self) -> Size:
        """Return tile size of levels."""
        return self.base_level.tile_size

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
    def files(self) -> List[Path]:
        """Return contained files"""
        return self.levels.files + self.labels.files + self.overviews.files

    @property
    def datasets(self) -> List[WsiDataset]:
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

    @property
    def frame_of_reference(self) -> Optional[UID]:
        return self.uids.frame_of_reference

    def pretty_str(
        self,
        indent: int = 0,
        depth: Optional[int] = None
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
    def open(
        cls,
        path: Union[str, List[str], Path, List[Path]]
    ) -> 'WsiDicom':
        """Open valid wsi dicom files in path and return a WsiDicom object.
        Non-valid files are ignored.

        Parameters
        ----------
        path: Union[str, List[str], Path, List[Path]]
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
        annotation_files: List[Path] = []

        for filepath in cls._filter_paths(filepaths):
            sop_class_uid = cls._get_sop_class_uid(filepath)
            if sop_class_uid == WSI_SOP_CLASS_UID:
                wsi_file = WsiDicomFile(filepath)
                if(wsi_file.wsi_type == 'VOLUME'):
                    level_files.append(wsi_file)
                elif(wsi_file.wsi_type == 'LABEL'):
                    label_files.append(wsi_file)
                elif(wsi_file.wsi_type == 'OVERVIEW'):
                    overview_files.append(wsi_file)
                else:
                    wsi_file.close()
            elif sop_class_uid == ANN_SOP_CLASS_UID:
                annotation_files.append(filepath)

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
        annotations = AnnotationInstance.open(annotation_files)

        return cls(levels, labels, overviews, annotations)

    def read_label(self, index: int = 0) -> Image.Image:
        """Read label image of the whole slide. If several label
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int
            Index of the label image to read

        Returns
        ----------
        Image.Image
            label as image
        """
        try:
            label = self.labels[index]
            return label.get_default_full()
        except IndexError:
            raise WsiDicomNotFoundError("label", "series")

    def read_overview(self, index: int = 0) -> Image.Image:
        """Read overview image of the whole slide. If several overview
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int
            Index of the overview image to read

        Returns
        ----------
        Image.Image
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
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> Image.Image:
        """Read thumbnail image of the whole slide with dimensions
        no larger than given size.

        Parameters
        ----------
        size: int, int
            Upper size limit for thumbnail
        z: Optional[float] = None
            Z coordinate, optional
        path: Optional[str] = None
            optical path, optional

        Returns
        ----------
        Image.Image
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
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> Image.Image:
        """Read region defined by pixels.

        Parameters
        ----------
        location: int, int
            Upper left corner of region in pixels
        level: int
            Level in pyramid
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
        wsi_level = self.levels.get_closest_by_level(level)
        scale_factor = wsi_level.calculate_scale(level)
        scaled_region = Region(
            position=Point.from_tuple(location),
            size=Size.from_tuple(size)
        ) * scale_factor

        if not wsi_level.valid_pixels(scaled_region):
            raise WsiDicomOutOfBoundsError(
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
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> Image.Image:
        """Read image from region defined in mm.

        Parameters
        ----------
        location: float, float
            Upper left corner of region in mm
        level: int
            Level in pyramid
        size: float
            Size of region in mm
        z: Optional[float] = None
            Z coordinate, optional
        path: Optional[str] = None
            optical path, optional

        Returns
        ----------
        Image.Image
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
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> Image.Image:
        """Read image from region defined in mm with set pixel spacing.

        Parameters
        ----------
        location: float, float
            Upper left corner of region in mm
        mpp: float
            Requested pixel spacing (um/pixel)
        size: float
            Size of region in mm
        z: Optional[float] = None
            Z coordinate, optional
        path: Optional[str] = None
            optical path, optional

        Returns
        ----------
        Image.Image
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
        return image.resize(image_size.to_tuple(), resample=Image.BILINEAR)

    def read_tile(
        self,
        level: int,
        tile: Tuple[int, int],
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> Image.Image:
        """Read tile in pyramid level as image.

        Parameters
        ----------
        level: int
            Pyramid level
        tile: int, int
            tile xy coordinate
        z: Optional[float] = None
            Z coordinate, optional
        path: Optional[str] = None
            optical path, optional

        Returns
        ----------
        Image.Image
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
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> bytes:
        """Read tile in pyramid level as encoded bytes. For non-existing levels
        the tile is scaled down from a lower level, using the similar encoding.

        Parameters
        ----------
        level: int
            Pyramid level
        tile: int, int
            tile xy coordinate
        z: Optional[float] = None
            Z coordinate, optional
        path: Optional[str] = None
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
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> WsiInstance:

        """Return instance fullfilling level, z and/or path.

        Parameters
        ----------
        level: int
            Pyramid level
        z: Optional[float] = None
            Z coordinate, optional
        path: Optional[str] = None
            optical path, optional

        Returns
        ----------
        WsiInstance:
            Instance
        """
        wsi_level = self.levels.get_level(level)
        return wsi_level.get_instance(z, path)

    def close(self) -> None:
        """Close all files."""
        for series in [self.levels, self.overviews, self.labels]:
            series.close()

    def save(
        self,
        output_path: str,
        uid_generator: Callable[..., UID] = generate_uid,
        workers: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> List[Path]:
        """Save wsi as DICOM-files in path. Instances for the same pyramid
        level will be combined when possible to one file (e.g. not split
        for optical paths or focal planes). If instances are sparse tiled they
        will be converted to full tiled by inserting blank tiles. The PixelData
        will contain a basic offset table. All instance uids will be changed.

        Parameters
        ----------
        output_path: str
        uid_generator: Callable[..., UID] = pydicom.uid.generate_uid
             Function that can gernerate unique identifiers.
        workers: Optional[int] = None
            Maximum number of thread workers to use.
        chunk_size: Optional[int] = None
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.

        Returns
        ----------
        List[Path]
            List of paths of created files.
        """
        if workers is None:
            cpus = os.cpu_count()
            if cpus is None:
                workers = 1
            else:
                workers = cpus
        if chunk_size is None:
            chunk_size = 100

        collections: List[WsiDicomSeries] = [
            self.levels, self.labels, self.overviews
        ]

        filepaths: List[Path] = []
        for collection in collections:
            collection_filepaths = collection.save(
                output_path,
                uid_generator,
                workers,
                chunk_size
            )
            filepaths.extend(collection_filepaths)
        return filepaths

    @staticmethod
    def _get_sop_class_uid(path: Path) -> UID:
        metadata: FileMetaDataset = read_file_meta_info(path)
        return metadata.MediaStorageSOPClassUID

    @staticmethod
    def _get_filepaths(path: Union[str, List[str], Path, List[Path]]):
        """Return file paths to files in path.
        If path is folder, return list of folder files in path.
        If path is single file, return list of that path.
        If path is list, return list of paths that are files.
        Raises WsiDicomNotFoundError if no files found

        Parameters
        ----------
        path: path: Union[str, List[str], Path, List[Path]]
            Path to folder, file or list of files

        Returns
        ----------
        List[Path]
            List of found file paths
        """
        if isinstance(path, (str, Path)):
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
        base_dataset = files[0].dataset
        for file in files[1:]:
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
            path for path in filepaths if path.is_file() and is_dicom(path)
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
