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


from typing import Optional, Sequence

from wsidicom.dataset import ImageType
from wsidicom.group import Group
from wsidicom.instance import WsiInstance
from wsidicom.series.series import Series


class Labels(Series):
    """Represents a series of Groups of the label wsi flavor."""

    @property
    def image_type(self) -> ImageType:
        return ImageType.LABEL

    @classmethod
    def open(cls, instances: Sequence[WsiInstance]) -> Optional["Labels"]:
        """Return labels created from wsi files.

        Parameters
        ----------
        instances: Sequence[WsiInstance]
            Instances to create labels from.

        Returns
        ----------
        Optional['Labels']
            Created labels.
        """
        labels = Group.open(instances)
        if len(labels) == 0:
            return None
        return cls(labels)
