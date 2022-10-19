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

import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, OrderedDict, Sequence, Union

from pydicom.uid import UID, generate_uid

from wsidicom.dataset import WsiDicomDataset
from wsidicom.errors import (WsiDicomMatchError, WsiDicomNotFoundError,
                             WsiDicomOutOfBoundsError)
from wsidicom.geometry import Size, SizeMm
from wsidicom.image import WsiImage, WsiImageGroup, WsiLevel
from wsidicom.image_data import WsiImageOrgin
from wsidicom.stringprinting import str_indent
from wsidicom.uid import SlideUids


class WsiSeries(metaclass=ABCMeta):
    """Represents a series of WsiImageGroups with the same image flavor,
    e.g. pyramidal levels, lables, or overviews.
    """

    def __init__(self, groups: Sequence[WsiImageGroup]):
        """Create a WsiSeries from list of WsiImageGroups.

        Parameters
        ----------
        groups: Sequence[WsiImageGroup]
            List of groups to include in the series.
        """
        self._groups = groups

        if len(self.groups) != 0 and self.groups[0].uids is not None:
            self._uids = self._validate_series(self.groups)
        else:
            self._uids = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.groups})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of groups {self.groups}"

    def __getitem__(self, index: int) -> WsiImageGroup:
        """Get group by index.

        Parameters
        ----------
        index: int
            Index in series to get

        Returns
        ----------
        WsiImageGroup
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
        raise NotImplementedError()

    @property
    def groups(self) -> Sequence[WsiImageGroup]:
        """Return contained groups."""
        return self._groups

    @property
    def uids(self) -> Optional[SlideUids]:
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
    def datasets(self) -> List[WsiDicomDataset]:
        """Return contained datasets."""

        series_datasets = [
            series.datasets for series in self.groups
        ]
        return [
            dataset for sublist in series_datasets for dataset in sublist
        ]

    @property
    def images(self) -> List[WsiImage]:
        """Return contained images"""
        series_images = [
            series.images.values() for series in self.groups
        ]
        return [
            image for sublist in series_images for image in sublist
        ]

    @classmethod
    @abstractmethod
    def open(cls, images: Sequence[WsiImage]) -> 'WsiSeries':
        raise NotImplementedError

    def _validate_series(
            self,
            groups: Union[Sequence[WsiImageGroup], Sequence[WsiLevel]]
    ) -> Optional[SlideUids]:
        """Check that no files or images in series is duplicate and that
        all groups in series matches.
        Raises WsiDicomMatchError otherwise.
        Returns the matching base uid. If list of groups is empty, return None.

        Parameters
        ----------
        groups: Union[Sequence[WsiImageGroup], Sequence[WsiLevel]]
            List of groups or levels to check

        Returns
        ----------
        Optional[SlideUids]:
            Matching uids
        """
        WsiDicomDataset.check_duplicate_dataset(self.datasets, self)
        WsiImage.check_duplicate_image(self.images, self)

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
        chunk_size: int,
        offset_table: Optional[str]
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
                chunk_size,
                offset_table
            )
            filepaths.extend(group_file_paths)
        return filepaths


class WsiLabels(WsiSeries):
    """Represents a series of WsiImageGroups of the label wsi flavor."""
    WSI_TYPE = 'LABEL'

    @property
    def wsi_type(self) -> str:
        return self.WSI_TYPE

    @classmethod
    def open(
        cls,
        images: Sequence[WsiImage]
    ) -> 'WsiLabels':
        """Return labels created from wsi files.

        Parameters
        ----------
        images: Sequence[WsiImage]
            images to create labels from.

        Returns
        ----------
        Overviews
            Created labels.
        """
        labels = WsiImageGroup.open(images)
        return cls(labels)


class WsiOverviews(WsiSeries):
    """Represents a series of WsiImageGroups of the overview wsi flavor."""
    WSI_TYPE = 'OVERVIEW'

    @property
    def wsi_type(self) -> str:
        return self.WSI_TYPE

    @classmethod
    def open(
        cls,
        images: Sequence[WsiImage]
    ) -> 'WsiOverviews':
        """Return overviews created from wsi files.

        Parameters
        ----------
        images: Sequence[WsiImage]
            images to create overviews from.

        Returns
        ----------
        Overviews
            Created overviews.
        """
        overviews = WsiImageGroup.open(images)
        return cls(overviews)


class WsiLevels(WsiSeries):
    """Represents a series of WsiImageGroups of the volume (e.g. pyramidal
    level) wsi flavor."""
    WSI_TYPE = 'VOLUME'

    @property
    def wsi_type(self) -> str:
        return self.WSI_TYPE

    @classmethod
    def open(
        cls,
        images: Sequence[WsiImage]
    ) -> 'WsiLevels':
        """Return overviews created from wsi files.

        Parameters
        ----------
        images: Sequence[WsiImage]
            images to create levels from.

        Returns
        ----------
        Overviews
            Created levels.
        """
        levels = WsiLevel.open(images)
        return cls(levels)

    def __init__(self, levels: Sequence[WsiLevel]):
        """Holds a stack of levels.

        Parameters
        ----------
        levels: Sequence[WsiLevel]
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

        mm_size = self.base_level.default_image.mm_size
        if mm_size is None:
            raise ValueError(
                'ImagedVolumeWidth and ImagedVolumeHeight must be set for '
                '"Volume" type'
            )
        self._mm_size = mm_size

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
    def groups(self) -> List[WsiImageGroup]:
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
    def base_level(self) -> WsiLevel:
        """Return the base level of the pyramid"""
        return self._levels[0]

    @property
    def mm_size(self) -> SizeMm:
        return self._mm_size

    @property
    def image_origin(self) -> WsiImageOrgin:
        return self.base_level.default_image.image_origin

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

    def get_level(self, level: int) -> WsiLevel:
        """Return wsi level.

        Parameters
        ----------
        level: int
            The level of the wsi level to return

        Returns
        ----------
        WsiLevel
            The searched level
        """
        try:
            return self._levels[level]
        except KeyError:
            raise WsiDicomNotFoundError(
                f"WsiLevel of {level}", "level series"
            )

    def get_closest_by_level(self, level: int) -> WsiLevel:
        """Search for level that is closest to and smaller than the given
        level.

        Parameters
        ----------
        level: int
            The level to search for

        Returns
        ----------
        WsiLevel
            The level closest to searched level
        """
        if not self.valid_level(level):
            raise WsiDicomOutOfBoundsError(
                f"WsiLevel {level}", f"maximum level {self.highest_level}"
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
                f"WsiLevel for {level}", "level series"
            )
        return closest

    def get_closest_by_size(self, size: Size) -> WsiLevel:
        """Search for level that by size is closest to and larger than the
        given size.

        Parameters
        ----------
        size: Size
            The size to search for

        Returns
        ----------
        WsiLevel
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
                f"WsiLevel for size {size}", "level series"
            )
        return closest

    def get_closest_by_pixel_spacing(
        self,
        pixel_spacing: SizeMm
    ) -> WsiLevel:
        """Search for level that by pixel spacing is closest to and smaller
        than the given pixel spacing. Only the spacing in x-axis is used.

        Parameters
        ----------
        pixel_spacing: SizeMm
            Pixel spacing to search for

        Returns
        ----------
        WsiLevel
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
                f"WsiLevel for pixel spacing {pixel_spacing}", "level series")
        return closest

    def construct_pyramid(
        self,
        highest_level: int,
        uid_generator: Callable[..., UID] = generate_uid,
        workers: Optional[int] = None,
        chunk_size: int = 100,
        offset_table: Optional[str] = 'bot',
        add_to_excisting: bool = True
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
        offset_table: Optional[str] = 'bot'
            Offset table to use, 'bot' basic offset table, 'eot' extended
            offset table, None - no offset table.
        add_to_excisting: bool = True
            If to add the created levels to excisting levels.

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
                output_path = closest_level.files[0].parent
                new_level = closest_level.create_child(
                    scale=2,
                    output_path=output_path,
                    uid_generator=uid_generator,
                    workers=workers,
                    chunk_size=chunk_size,
                    offset_table=offset_table
                )
                # Add level to available levels
                if add_to_excisting:
                    self._levels[new_level.level] = new_level
                else:
                    new_level.close()
                filepaths += new_level.files
        return filepaths