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

import math
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, cast

from PIL import Image
from PIL.Image import Image as PILImage
from pydicom.uid import UID

from wsidicom.errors import WsiDicomNoResultionError, WsiDicomOutOfBoundsError
from wsidicom.file import OffsetTableType, WsiDicomFileWriter
from wsidicom.geometry import Point, Region, Size, SizeMm
from wsidicom.group.group import Group
from wsidicom.instance import WsiInstance
from wsidicom.stringprinting import dict_pretty_str
from wsidicom.file import (
    WsiDicomFile,
    WsiDicomFileImageData,
    WsiDicomFileSource,
)


class Level(Group):
    """Represents a level in the pyramid and contains one or more instances
    having the same pyramid level index, pixel spacing, and size but possibly
    different focal planes and/or optical paths.
    """

    def __init__(self, instances: Sequence[WsiInstance], base_pixel_spacing: SizeMm):
        """Create a level from list of WsiInstances. Asign the pyramid level
        index from pixel spacing of base level.

        Parameters
        ----------
        instances: Sequence[WsiInstance]
            Instances to build the level.
        base_pixel_spacing: SizeMm
            Pixel spacing of base level.
        """
        super().__init__(instances)
        self._base_pixel_spacing = base_pixel_spacing
        self._level = self._assign_level(self._base_pixel_spacing)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.instances}, " f"{self._base_pixel_spacing})"
        )

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
        string = f"Level: {self.level}, size: {self.size} px, mpp: {self.mpp} um/px"
        if depth is not None:
            depth -= 1
            if depth < 0:
                return string
        string += " Instances: " + dict_pretty_str(self.instances, indent, depth)
        return string

    @property
    def pyramid(self) -> str:
        """Return string representation of the level"""
        return (
            f"Level [{self.level}]"
            f" tiles: {self.default_instance.tiled_size},"
            f" size: {self.size}, mpp: {self.mpp} um/px"
        )

    @property
    def tile_size(self) -> Size:
        return self.default_instance.tile_size

    @property
    def level(self) -> int:
        """Return pyramid level"""
        return self._level

    @property
    def mpp(self) -> SizeMm:
        if self.pixel_spacing is None:
            raise WsiDicomNoResultionError()
        return self.pixel_spacing * 1000.0

    @property
    def pixel_spacing(self) -> SizeMm:
        if self._pixel_spacing is None:
            raise WsiDicomNoResultionError()
        return self._pixel_spacing

    @classmethod
    def open(
        cls,
        instances: Sequence[WsiInstance],
    ) -> List["Level"]:
        """Return list of levels created wsi files.

        Parameters
        ----------
        files: Sequence[WsiInstance]
            Instances to create levels from.

        Returns
        ----------
        List[Level]
            List of created levels.

        """
        levels: List["Level"] = []
        instances_grouped_by_level = cls._group_instances(instances)
        base_group = list(instances_grouped_by_level.values())[0]
        base_pixel_spacing = base_group[0].pixel_spacing
        if base_pixel_spacing is None:
            raise WsiDicomNoResultionError()
        for level in instances_grouped_by_level.values():
            levels.append(cls(level, base_pixel_spacing))
        return levels

    def matches(self, other_level: "Group") -> bool:
        """Check if level matches other level. If strict common Uids should
        match. Wsi type and tile size should always match.

        Parameters
        ----------
        other_level: Group
            Other level to match against.

        Returns
        ----------
        bool
            True if other level matches.
        """
        other_level = cast(Level, other_level)
        return (
            self.uids.matches(other_level.uids)
            and other_level.image_type == self.image_type
            and other_level.tile_size == self.tile_size
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
        path: Optional[str] = None,
    ) -> PILImage:
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
        PILImage
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
        tile_size = cropped_region.size.ceil_div(scale)
        image = image.resize(tile_size.to_tuple(), resample=Image.Resampling.BILINEAR)
        return image

    def get_scaled_encoded_tile(
        self,
        tile: Point,
        scale: int,
        z: Optional[float] = None,
        path: Optional[str] = None,
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
        float_level = math.log2(self.pixel_spacing.width / base_pixel_spacing.width)
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
        offset_table: Optional[str],
        instance_number: int,
    ) -> "Level":
        """Creates a new Level from this level by scaling the image
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
        'Level'
            Created scaled level.
        """

        filepaths: List[Path] = []
        if not isinstance(scale, int) or scale < 2:
            raise ValueError("Scale must be integer and larger than 2")
        if not isinstance(self.default_instance.image_data, WsiDicomFileImageData):
            raise NotImplementedError("Can only construct pyramid from DICOM WSI files")

        for instances in self._group_instances_to_file():
            uid = uid_generator()
            filepath = Path(os.path.join(output_path, uid + ".dcm"))
            transfer_syntax = instances[0].image_data.transfer_syntax
            image_data_list = self._list_image_data(instances)
            focal_planes, optical_paths, tiled_size = self._get_frame_information(
                image_data_list
            )
            dataset = instances[0].dataset.as_tiled_full(
                focal_planes, optical_paths, tiled_size, scale
            )

            with WsiDicomFileWriter(filepath) as wsi_file:
                wsi_file.write(
                    uid,
                    transfer_syntax,
                    dataset,
                    image_data_list,
                    workers,
                    chunk_size,
                    OffsetTableType.from_string(offset_table),
                    instance_number,
                    scale=scale,
                )
            filepaths.append(filepath)

        created_instances = WsiDicomFileSource.open_files(
            [WsiDicomFile(filepath) for filepath in filepaths],
            self.uids,
            self.tile_size,
        )
        return Level(created_instances, self._base_pixel_spacing)
