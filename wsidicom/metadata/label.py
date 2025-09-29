#    Copyright 2023, 2025 SECTRA AB
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

"""Label model."""

from dataclasses import dataclass, replace
from typing import Optional, Sequence

from wsidicom.metadata.image import Image
from wsidicom.metadata.optical_path import OpticalPath


@dataclass(frozen=True)
class Label:
    """
    Label metadata.

    Parameters
    ----------
    text : Optional[str] = None
        The label text.
    barcode : Optional[str] = None
        The label barcode.
    image : Optional[Image] = None
        Technical metadata of the label image.
    optical_paths : Optional[Sequence[OpticalPath]] = None
        Metadata of the optical paths used to acquire the label image.
    contains_phi : bool = True
        Whether the label image contains personal health information.
    comments: Optional[str] = None
        Comments related to the label image.
    """

    text: Optional[str] = None
    barcode: Optional[str] = None
    image: Optional[Image] = None
    optical_paths: Optional[Sequence[OpticalPath]] = None
    contains_phi: bool = True
    comments: Optional[str] = None

    def remove_confidential(self) -> "Label":
        return replace(
            self,
            text=None,
            barcode=None,
            image=self.image.remove_confidential() if self.image else None,
            comments=None,
        )
