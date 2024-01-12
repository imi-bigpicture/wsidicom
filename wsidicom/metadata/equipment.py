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

"""Equipment model."""

from dataclasses import dataclass, replace
from typing import Optional, Sequence


@dataclass(frozen=True)
class Equipment:
    """
    Equipment used to produce the slide.

    Corresponds to the `Required` attributes in the Enhanced General Equipment Module:
    https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.5.2.html

    Parameters
    ----------
    manufacturer : Optional[str] = None
        The scanner manufacturer.
    model_name : Optional[str] = None
        The scanner model name.
    device_serial_number : Optional[str] = None
        The scanner device serial number.
    software_versions : Optional[Sequence[str]] = None
        The scanner software versions.
    """

    manufacturer: Optional[str] = None
    model_name: Optional[str] = None
    device_serial_number: Optional[str] = None
    software_versions: Optional[Sequence[str]] = None

    def remove_confidential(self) -> "Equipment":
        return replace(
            self,
            device_serial_number=None,
        )
