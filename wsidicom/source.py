#    Copyright 2023 SECTRA AB
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
from typing import Iterable

from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.instance import WsiDataset, WsiInstance

"""A Source provides image instances that can be opened by WsiDicom. Implement this
class to extend support to other DICOM sources or to read other WSI formats.
"""


class Source(metaclass=ABCMeta):
    """A source providing DICOM WSI instances to open.

    A source should be initiated with a path or similar, and parse the content into
    instances.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    @abstractmethod
    def base_dataset(self) -> WsiDataset:
        """Return a representative dataset for the source content."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def level_instances(self) -> Iterable[WsiInstance]:
        """Return all level instances from the source."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def label_instances(self) -> Iterable[WsiInstance]:
        """Return all label instances from the source."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def overview_instances(self) -> Iterable[WsiInstance]:
        """Return all overview instances from the source."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def annotation_instances(self) -> Iterable[AnnotationInstance]:
        """Return all annotation instances from the source."""
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Close any opened resources (such as files)."""
        raise NotImplementedError()
