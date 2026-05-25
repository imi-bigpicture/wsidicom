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

from collections.abc import Sequence
from dataclasses import dataclass, replace

from wsidicom.metadata.image import Image
from wsidicom.metadata.optical_path import OpticalPath


@dataclass(frozen=True)
class Label:
    """
    Label metadata.

    Parameters
    ----------
    text : str | None = None
        The label text.
    barcode : str | None = None
        The label barcode.
    image : Image | None = None
        Technical metadata of the label image.
    optical_paths : Sequence[OpticalPath] | None = None
        Metadata of the optical paths used to acquire the label image.
    contains_phi : bool = True
        Whether the label image contains personal health information.
    comments: str | None = None
        Comments related to the label image.
    """

    text: str | None = None
    barcode: str | None = None
    image: Image | None = None
    optical_paths: Sequence[OpticalPath] | None = None
    contains_phi: bool = True
    comments: str | None = None

    def remove_confidential(self) -> "Label":
        return replace(
            self,
            text=None,
            barcode=None,
            image=self.image.remove_confidential() if self.image else None,
            comments=None,
        )
