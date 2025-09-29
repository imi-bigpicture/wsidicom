#    Copyright 2025 SECTRA AB
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

"""Overview model."""

from dataclasses import dataclass, replace
from typing import Optional, Sequence

from wsidicom.metadata.image import Image
from wsidicom.metadata.optical_path import OpticalPath


@dataclass(frozen=True)
class Overview:
    """
    Overview metadata.

    Parameters
    ----------
    image: Image
        Technical metadata of the overview image.
    optical_paths: Sequence[OpticalPath]
        Metadata of the optical paths used to acquire the overview image.
    contains_label: bool = True
        Whether the specimen label is present in the overview image.
    contains_phi: bool = True
        Whether the overview image contains personal health information.
    comments: Optional[str] = None
        Comments related to the overview image.
    """

    image: Image
    optical_paths: Sequence[OpticalPath]
    contains_phi: bool = False
    contains_label: bool = True
    comments: Optional[str] = None

    def remove_confidential(self) -> "Overview":
        return replace(self, image=self.image.remove_confidential(), comments=None)
