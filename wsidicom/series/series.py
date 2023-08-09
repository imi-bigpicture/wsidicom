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
from typing import (
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)


from wsidicom.errors import WsiDicomMatchError
from wsidicom.geometry import SizeMm
from wsidicom.group import Group, Level
from wsidicom.instance import ImageType, WsiDataset, WsiInstance
from wsidicom.uid import SlideUids

SeriesType = TypeVar("SeriesType")


class Series(metaclass=ABCMeta):
    """Represents a series of Groups with the same image flavor, e.g.
    pyramidal levels, labels, or overviews.
    """

    def __init__(self, groups: Iterable[Group]):
        """Create a Series from list of Groups.

        Parameters
        ----------
        groups: Iterable[Group]
            List of groups to include in the series.
        """
        self._groups = list(groups)
        self._group_type = Group
        if len(self.groups) != 0 and self.groups[0].uids is not None:
            self._uids = self._validate_series(self.groups)
        else:
            self._uids = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.groups})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of groups {self.groups}"

    def __getitem__(self, index: int) -> Group:
        """Get group by index.

        Parameters
        ----------
        index: int
            Index in series to get

        Returns
        ----------
        Group
            The group at index in the series
        """
        return self.groups[index]

    def __len__(self) -> int:
        return len(self.groups)

    @property
    @abstractmethod
    def image_type(self) -> ImageType:
        """Should return the wsi type of the series ('VOLUME', 'LABEL', or
        'OVERVIEW'"""
        raise NotImplementedError()

    @property
    def groups(self) -> List[Group]:
        """Return contained groups."""
        return self._groups

    @property
    def uids(self) -> Optional[SlideUids]:
        """Return uids."""
        return self._uids

    @property
    def mpps(self) -> List[SizeMm]:
        """Return contained mpp (um/px)."""
        return [group.mpp for group in self.groups if group.mpp is not None]

    @property
    def datasets(self) -> List[WsiDataset]:
        """Return contained datasets."""

        series_datasets = [series.datasets for series in self.groups]
        return [dataset for sublist in series_datasets for dataset in sublist]

    @property
    def instances(self) -> List[WsiInstance]:
        """Return contained instances"""
        series_instances = [series.instances.values() for series in self.groups]
        return [instance for sublist in series_instances for instance in sublist]

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
        ----------
        Optional[SeriesType]
            Created series.
        """
        groups = Group.open(instances)
        if len(groups) == 0:
            return None
        return cls(groups)

    def _validate_series(
        self, groups: Union[Sequence[Group], Sequence[Level]]
    ) -> Optional[SlideUids]:
        """Check that no files or instances in series is duplicate and that
        all groups in series matches.
        Raises WsiDicomMatchError otherwise.
        Returns the matching base uid. If list of groups is empty, return None.

        Parameters
        ----------
        groups: Union[Sequence[Group], Sequence[Level]]
            List of groups or levels to check

        Returns
        ----------
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


class Overviews(Series):
    """Represents a series of Groups of the overview wsi flavor."""

    @property
    def image_type(self) -> ImageType:
        return ImageType.OVERVIEW


class Labels(Series):
    """Represents a series of Groups of the label wsi flavor."""

    @property
    def image_type(self) -> ImageType:
        return ImageType.LABEL
