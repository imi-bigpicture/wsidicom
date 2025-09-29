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

"""Pyramid model."""

from dataclasses import dataclass, replace
from typing import Optional, Sequence

from wsidicom.metadata.image import Image
from wsidicom.metadata.optical_path import OpticalPath


@dataclass(frozen=True)
class Pyramid:
    """
    Pyramid metadata.

    Parameters
    ----------
    image: Image
        Technical metadata of the pyramid image.
    optical_paths: Sequence[OpticalPath]
        Metadata of the optical paths used to acquire the pyramid image.
    uid: Optional[str] = None
        The unique identifier of the pyramid image.
    description: Optional[str] = None
        Description of the pyramid image.
    label: Optional[str] = None
        User-defind label of the pyramid image.
    contains_phi : bool = False
        Whether the pyramid image contains personal health information.
    comments: Optional[str] = None
        Comments related to the pyramid image.
    """

    image: Image
    optical_paths: Sequence[OpticalPath]
    uid: Optional[str] = None
    description: Optional[str] = None
    label: Optional[str] = None
    contains_phi: bool = False
    comments: Optional[str] = None

    def remove_confidential(self) -> "Pyramid":
        return replace(self, image=self.image.remove_confidential(), comments=None)
