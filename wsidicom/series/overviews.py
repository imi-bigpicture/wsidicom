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


class Overviews(Series):
    """Represents a series of Groups of the overview wsi flavor."""

    @property
    def image_type(self) -> ImageType:
        return ImageType.OVERVIEW

    @classmethod
    def open(cls, instances: Sequence[WsiInstance]) -> Optional["Overviews"]:
        """Return overviews created from wsi files.

        Parameters
        ----------
        instances: Sequence[WsiInstance]
            Instances to create overviews from.

        Returns
        ----------
        Optional[Overviews]
            Created overviews.
        """
        overviews = Group.open(instances)
        if len(overviews) == 0:
            return None
        return cls(overviews)
