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
from typing import List
from pydicom import Dataset

from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.instance import WsiInstance


class Source(metaclass=ABCMeta):
    @property
    @abstractmethod
    def base_dataset(self) -> Dataset:
        raise NotImplementedError()

    @property
    @abstractmethod
    def level_instances(self) -> List[WsiInstance]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def label_instances(self) -> List[WsiInstance]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def overview_instances(self) -> List[WsiInstance]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def annotation_instances(self) -> List[AnnotationInstance]:
        raise NotImplementedError()
