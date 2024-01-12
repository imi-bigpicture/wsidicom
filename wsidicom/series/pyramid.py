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

from typing import Iterable, List, Optional, OrderedDict

from wsidicom.errors import (
    WsiDicomNotFoundError,
    WsiDicomOutOfBoundsError,
)
from wsidicom.geometry import Size, SizeMm
from wsidicom.group import Level
from wsidicom.group.level import BaseLevel
from wsidicom.instance import ImageType, WsiInstance
from wsidicom.metadata import ImageCoordinateSystem
from wsidicom.series.series import Series
from wsidicom.stringprinting import list_pretty_str


class Pyramid(Series[Level]):
    """Represents a list of Levels of the volume (e.g. pyramidal level) wsi flavor
    forming a WSI pyramid. All levels in the pyramid must have the same image origin
    and extended depth of field."""

    def __init__(self, levels: Iterable[Level]):
        """Holds a stack of levels.

        Parameters
        ----------
        levels: Iterable[Level]
            List of levels to include in the pyramid.
        """
        self._levels = OrderedDict(
            (level.level, level)
            for level in sorted(levels, key=lambda level: level.level)
        )
        self._groups = list(self._levels.values())
        if len(self._levels) != 0 and self._levels[0].uids is not None:
            self._uids = self._validate_series(list(self._levels.values()))
        else:
            self._uids = None
        mm_size = next(
            level.default_instance.mm_size
            for level in self._levels.values()
            if level.default_instance.mm_size is not None
        )
        if mm_size is None:
            raise ValueError(
                "ImagedVolumeWidth and ImagedVolumeHeight must be set for "
                '"Volume" type'
            )
        self._mm_size = mm_size

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._levels})"

    def __str__(self) -> str:
        """Return string representation of pyramid."""
        return self.pretty_str()

    def __getitem__(self, index: int) -> Level:
        return list(self._levels.values())[index]

    def __len__(self) -> int:
        return len(self._levels)

    def get(self, index: int, pyramid_index: bool = True) -> Level:
        """Get level by index.

        Parameters
        ----------
        index: int
            Index in pyramid to get
        pyramid_index: bool = True
            If to get level by pyramid index (True) or list index (False).

        Returns
        -------
        Level
            The level at index in the pyramid
        """
        try:
            if pyramid_index:
                return self._levels[index]
            return self[index]
        except (KeyError, IndexError):
            raise WsiDicomNotFoundError(f"Level index {index}", "pyramid")

    @property
    def image_type(self) -> ImageType:
        return ImageType.VOLUME

    @property
    def size(self) -> Size:
        return self.base_level.size

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.base_level.pixel_spacing

    @property
    def mpp(self) -> SizeMm:
        return self.base_level.mpp

    @property
    def tile_size(self) -> Size:
        return self.base_level.tile_size

    @classmethod
    def open(cls, instances: Iterable[WsiInstance]) -> "Pyramid":
        """Return pyramid created from wsi instances.

        Parameters
        ----------
        instances: Iterable[WsiInstance]
            Instances to create pyramid from.

        Returns
        -------
        Pyramid
            Created pyramid.
        """
        instances_grouped_by_size = cls._group_instances_by_size(instances)
        base_level = BaseLevel(next(instances_grouped_by_size))
        levels: List[Level] = [base_level]
        levels.extend(
            [
                Level(instances, base_level.pixel_spacing)
                for instances in instances_grouped_by_size
            ]
        )
        return cls(levels)

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
        """Return string representation of pyramid."""
        string = self.__class__.__name__
        if depth is not None:
            depth -= 1
            if depth < 0:
                return string
        return (
            string
            + " of levels:\n"
            + list_pretty_str(self._levels.values(), indent + 2, depth)
        )

    @property
    def levels(self) -> List[Level]:
        """Return contained levels."""
        return list(self._levels.values())

    @property
    def pyramid_indices(self) -> List[int]:
        """Return contained levels."""
        return list(self._levels.keys())

    @property
    def highest_level(self) -> int:
        """Return highest valid pyramid level (which results in a 1x1 image)."""
        return self.base_level.get_highest_level()

    @property
    def lowest_single_tile_level(self) -> int:
        """Return lowest pyramid level that has consists of a single tile."""
        return self.base_level.get_lowest_single_tile_level()

    @property
    def base_level(self) -> Level:
        """Return the base level of the pyramid."""
        return self._levels[0]

    @property
    def mm_size(self) -> SizeMm:
        return self._mm_size

    @property
    def image_coordinate_system(self) -> Optional[ImageCoordinateSystem]:
        return self.base_level.image_coordinate_system

    def valid_level(self, level: int) -> bool:
        """Check that given level is less or equal to the highest level
        (1x1 pixel level).

        Parameters
        ----------
        level: int
            The level to check

        Returns
        -------
        bool
            True if level is valid
        """
        return level <= self.highest_level

    def get_closest_by_level(self, level: int) -> Level:
        """Search for level that is closest to and smaller than the given
        level.

        Parameters
        ----------
        level: int
            The level to search for

        Returns
        -------
        Level
            The level closest to searched level
        """
        if not self.valid_level(level):
            raise WsiDicomOutOfBoundsError(
                f"Level {level}", f"maximum level {self.highest_level}"
            )
        closest_level = 0
        closest = None
        for wsi_level in self._levels.values():
            if (level >= wsi_level.level) and (closest_level <= wsi_level.level):
                closest_level = wsi_level.level
                closest = wsi_level
        if closest is None:
            raise WsiDicomNotFoundError(f"Level for {level}", "level series")
        return closest

    def get_closest_by_size(self, size: Size) -> Level:
        """Search for level that by size is closest to and larger than the
        given size.

        Parameters
        ----------
        size: Size
            The size to search for

        Returns
        -------
        Level
            The level with size closest to searched size
        """
        closest_size = self._levels[0].size
        closest = None
        for wsi_level in self._levels.values():
            if (size.width <= wsi_level.size.width) and (
                wsi_level.size.width <= closest_size.width
            ):
                closest_size = wsi_level.size
                closest = wsi_level
        if closest is None:
            raise WsiDicomNotFoundError(f"Level for size {size}", "level series")
        return closest

    def get_closest_by_pixel_spacing(self, pixel_spacing: SizeMm) -> Level:
        """Search for level that by pixel spacing is closest to and smaller
        than the given pixel spacing. Only the spacing in x-axis is used.

        Parameters
        ----------
        pixel_spacing: SizeMm
            Pixel spacing to search for

        Returns
        -------
        Level
            The level with pixel spacing closest to searched spacing
        """
        closest_pixel_spacing: float = 0
        closest = None
        for wsi_level in self._levels.values():
            if (pixel_spacing.width >= wsi_level.pixel_spacing.width) and (
                closest_pixel_spacing <= wsi_level.pixel_spacing.width
            ):
                closest_pixel_spacing = wsi_level.pixel_spacing.width
                closest = wsi_level
        if closest is None:
            raise WsiDicomNotFoundError(
                f"Level for pixel spacing {pixel_spacing}", "level series"
            )
        return closest
