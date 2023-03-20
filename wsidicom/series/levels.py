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

from pathlib import Path
from typing import Callable, Iterable, List, Optional, OrderedDict, Sequence

from pydicom.uid import UID

from wsidicom.errors import WsiDicomNotFoundError, WsiDicomOutOfBoundsError
from wsidicom.geometry import Size, SizeMm
from wsidicom.group import Group, Level
from wsidicom.instance import ImageOrigin, ImageType, WsiInstance
from wsidicom.series.series import Series
from wsidicom.stringprinting import str_indent


class Levels(Series):
    """Represents a series of Groups of the volume (e.g. pyramidal
    level) wsi flavor."""

    @property
    def image_type(self) -> ImageType:
        return ImageType.VOLUME

    @classmethod
    def open(cls, instances: Iterable[WsiInstance]) -> "Levels":
        """Return overviews created from wsi files.

        Parameters
        ----------
        instances: Iterable[WsiInstance]
            Instances to create levels from.

        Returns
        ----------
        Overviews
            Created levels.
        """
        levels = Level.open(instances)
        return cls(levels)

    def __init__(self, levels: Sequence[Level]):
        """Holds a stack of levels.

        Parameters
        ----------
        levels: Sequence[Level]
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

        mm_size = self.base_level.default_instance.mm_size
        if mm_size is None:
            raise ValueError(
                "ImagedVolumeWidth and ImagedVolumeHeight must be set for "
                '"Volume" type'
            )
        self._mm_size = mm_size

    @property
    def pyramid(self) -> str:
        """Return string representation of pyramid"""
        return "Pyramid levels in file:\n" + "\n".join(
            [str_indent(2) + level.pyramid for level in self._levels.values()]
        )

    @property
    def groups(self) -> List[Group]:
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
    def lowest_single_tile_level(self) -> int:
        """Return lowest pyramid level that has consists of a single tile."""
        return self.base_level.get_lowest_single_tile_level()

    @property
    def base_level(self) -> Level:
        """Return the base level of the pyramid"""
        return self._levels[0]

    @property
    def mm_size(self) -> SizeMm:
        return self._mm_size

    @property
    def image_origin(self) -> ImageOrigin:
        return self.base_level.image_origin

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

    def get_level(self, level: int) -> Level:
        """Return wsi level.

        Parameters
        ----------
        level: int
            The level of the wsi level to return

        Returns
        ----------
        Level
            The searched level
        """
        try:
            return self._levels[level]
        except KeyError as exception:
            raise WsiDicomNotFoundError(
                f"Level of {level}", "level series"
            ) from exception

    def get_closest_by_level(self, level: int) -> Level:
        """Search for level that is closest to and smaller than the given
        level.

        Parameters
        ----------
        level: int
            The level to search for

        Returns
        ----------
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
        ----------
        Level
            The level with size closest to searched size
        """
        closest_size = self.groups[0].size
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
        ----------
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

    def save(
        self,
        output_path: Path,
        uid_generator: Callable[..., UID],
        workers: int,
        chunk_size: int,
        offset_table: Optional[str],
        instance_number: int,
        add_missing_levels: bool = False,
    ) -> List[Path]:
        """Save Series as DICOM-files in path.

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
        offset_table: Optional[str] = 'bot'
            Offset table to use, 'bot' basic offset table, 'eot' extended
            offset table, None - no offset table.
        instance_number: int
            Instance number for first instance in series.
        add_missing_levels: bool = False
            If to add missing dyadic levels up to the single tile level.

        Returns
        ----------
        List[Path]
            List of paths of created files.
        """
        filepaths: List[Path] = []
        for pyramid_level in range(self.lowest_single_tile_level):
            if pyramid_level in self._levels.keys():
                level = self._levels[pyramid_level]
                level_file_paths = level.save(
                    output_path,
                    uid_generator,
                    workers,
                    chunk_size,
                    offset_table,
                    instance_number,
                )
            elif add_missing_levels:
                # Create scaled level
                closest_level = self.get_closest_by_level(pyramid_level)
                level_file_paths = closest_level.create_child(
                    scale=2,
                    output_path=output_path,
                    uid_generator=uid_generator,
                    workers=workers,
                    chunk_size=chunk_size,
                    offset_table=offset_table,
                    instance_number=instance_number,
                )
            else:
                continue
            filepaths.extend(level_file_paths)
            instance_number += len(level_file_paths)

        return filepaths
