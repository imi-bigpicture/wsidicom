#    Copyright 2026 SECTRA AB
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

"""Contributing equipment model."""

import datetime
from collections.abc import Sequence
from dataclasses import dataclass

from wsidicom.conceptcode import ContributingEquipmentPurposeCode


@dataclass(frozen=True)
class ContributingEquipment:
    """
    Equipment that contributed to the creation of the instance, other than the
    acquisition equipment described by `Equipment`.

    Corresponds to an item of the Contributing Equipment Sequence:
    https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.12.html

    Parameters
    ----------
    purpose : ContributingEquipmentPurposeCode
        The purpose of the equipment's contribution, e.g.
        `ContributingEquipmentPurposeCode("Processing Equipment")`.
    manufacturer : str | None = None
        The equipment manufacturer.
    model_name : str | None = None
        The equipment model name.
    software_versions : Sequence[str] | None = None
        The equipment software versions.
    description : str | None = None
        Free-text description of the contribution.
    contribution_datetime : datetime.datetime | None = None
        The date and time of the contribution.
    """

    purpose: ContributingEquipmentPurposeCode
    manufacturer: str | None = None
    model_name: str | None = None
    software_versions: Sequence[str] | None = None
    description: str | None = None
    contribution_datetime: datetime.datetime | None = None
