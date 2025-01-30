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

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from functools import cached_property
from typing import (
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

from wsidicom.errors import WsiDicomMatchError, WsiDicomNotFoundError
from wsidicom.geometry import Size, SizeMm
from wsidicom.group import Group, Level
from wsidicom.instance import ImageType, WsiDataset, WsiInstance
from wsidicom.metadata.schema.dicom.wsi import WsiMetadataDicomSchema
from wsidicom.metadata.wsi import WsiMetadata
from wsidicom.uid import SlideUids

SeriesType = TypeVar("SeriesType", bound="Series")
GroupType = TypeVar("GroupType", Group, Level)


class Series(Generic[GroupType], metaclass=ABCMeta):
    """Represents a series of Groups with the same image flavor, e.g.
    pyramidal levels, labels, or overviews.
    """

    def __init__(self, groups: Iterable[GroupType]):
        """Create a Series from list of Groups.

        Parameters
        ----------
        groups: Iterable[Group]
            List of groups to include in the series.
        """
        self._groups = list(groups)
        if len(self._groups) != 0 and self._groups[0].uids is not None:
            self._uids = self._validate_series(self._groups)
        else:
            self._uids = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._groups})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of groups {self._groups}"

    def __getitem__(self, index: int) -> GroupType:
        return self._groups[index]

    def get(self, index: int) -> GroupType:
        """Get group by index.

        Parameters
        ----------
        index: int
            Index in series to get

        Returns
        -------
        Group
            The group at index in the series
        """
        try:
            return self[index]
        except IndexError:
            raise WsiDicomNotFoundError(f"Group index {index}", "series")

    def __len__(self) -> int:
        return len(self._groups)

    @property
    @abstractmethod
    def image_type(self) -> ImageType:
        """Should return the wsi type of the series ('VOLUME', 'LABEL', or
        'OVERVIEW'"""
        raise NotImplementedError()

    @property
    def groups(self) -> List[GroupType]:
        """Return contained groups."""
        return self._groups

    @property
    def uids(self) -> Optional[SlideUids]:
        """Return uids."""
        return self._uids

    @property
    def mpps(self) -> List[SizeMm]:
        """Return contained mpp (um/px)."""
        return [group.mpp for group in self if group.mpp is not None]

    @property
    def datasets(self) -> List[WsiDataset]:
        """Return contained datasets."""

        series_datasets = [series.datasets for series in self]
        return [dataset for sublist in series_datasets for dataset in sublist]

    @property
    def instances(self) -> List[WsiInstance]:
        """Return contained instances"""
        series_instances = [series.instances.values() for series in self]
        return [instance for sublist in series_instances for instance in sublist]

    @cached_property
    def metadata(self) -> WsiMetadata:
        return WsiMetadataDicomSchema().load(self.datasets[0])

    @classmethod
    def open(
        cls: Type[SeriesType], instances: Iterable[WsiInstance]
    ) -> Optional[SeriesType]:
        """Return series created from instances.

        Parameters
        ----------
        instances: Iterable[WsiInstance]
            Instances to create series from.

        Returns
        -------
        Optional[SeriesType]
            Created series.
        """
        instances_grouped_by_size = cls._group_instances_by_size(instances)
        groups = [Group(instances) for instances in instances_grouped_by_size]
        if len(groups) == 0:
            return None
        return cls(groups)

    def get_closest_by_size(self, size: Size) -> Optional[Group]:
        """Search for group that by size is closest to and larger than the
        given size.

        Parameters
        ----------
        size: Size
            The size to search for

        Returns
        -------
        Optional[Group]
            The group with size closest to searched size, or None if no group is
            larger than or equal to the searched size.
        """
        if len(self._groups) == 0:
            return None
        closest_size = self._groups[0].size
        closest = None
        for wsi_level in self._groups:
            if (size.width <= wsi_level.size.width) and (
                wsi_level.size.width <= closest_size.width
            ):
                closest_size = wsi_level.size
                closest = wsi_level
        return closest

    @classmethod
    def _group_instances_by_size(
        cls, instances: Iterable[WsiInstance]
    ) -> Iterator[List[WsiInstance]]:
        """Return instances grouped and sorted by image size.

        Parameters
        ----------
        instances: Iterable[WsiInstance]
            Instances to group by image size.

        Returns
        -------
        Iterator[List[WsiInstance]]:
            Instances grouped by size.

        """
        grouped_instances: Dict[Size, List[WsiInstance]] = defaultdict(list)
        for instance in instances:
            grouped_instances[instance.size].append(instance)
        return (
            grouped_instances[key]
            for key in sorted(
                grouped_instances.keys(), key=lambda size: size.width, reverse=True
            )
        )

    def _validate_series(self, groups: Sequence[GroupType]) -> Optional[SlideUids]:
        """Check that no files or instances in series is duplicate and that
        all groups in series matches.
        Raises WsiDicomMatchError otherwise.
        Returns the matching base uid. If list of groups is empty, return None.

        Parameters
        ----------
        groups: Union[Sequence[Group], Sequence[Level]]
            List of groups or levels to check

        Returns
        -------
        Optional[SlideUids]:
            Matching uids
        """
        WsiDataset.check_duplicate_dataset(self.datasets, self)
        WsiInstance.check_duplicate_instance(self.instances, self)

        try:
            base_group = groups[0]
            if base_group.image_type != self.image_type:
                raise WsiDicomMatchError(str(base_group), str(self))
            for group in groups[1:]:
                if not group.matches(base_group):
                    raise WsiDicomMatchError(str(group), str(self))
            return base_group.uids
        except IndexError:
            return None


class Overviews(Series[Group]):
    """Represents a series of Groups of the overview wsi flavor."""

    @property
    def image_type(self) -> ImageType:
        return ImageType.OVERVIEW


class Labels(Series[Group]):
    """Represents a series of Groups of the label wsi flavor."""

    @property
    def image_type(self) -> ImageType:
        return ImageType.LABEL


class Thumbnails(Series[Group]):
    """Represents a series of Groups of the thumbnail wsi flavor."""

    @property
    def image_type(self) -> ImageType:
        return ImageType.THUMBNAIL
