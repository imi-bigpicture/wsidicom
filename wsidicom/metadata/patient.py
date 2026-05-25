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

"""Patient model."""

import datetime
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

from pydicom.sr.coding import Code


class PatientSex(Enum):
    """Patient sex."""

    F = "female"
    M = "male"
    O = "other"  # noqa: E741


@dataclass(frozen=True)
class PatientDeIdentification:
    """Patient de-identification.

    Parameters
    ----------
    identity_removed : bool
        Whether the patient identity has been removed.
    methods : Sequence[str | Code] | None = None
        The methods used to de-identify the patient.
    """

    identity_removed: bool
    methods: Sequence[str | Code] | None = None


@dataclass(frozen=True)
class Patient:
    """
    Patient metadata.

    Corresponds to the `Required` and `Required, Empty if Unknown` attributes in the
    Patient Module:
    https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.html

    Parameters
    ----------
    name : str | None = None
        The patient name.
    identifier : str | None = None
        The patient identifier.
    birth_date : datetime.date | None = None
        The patient birth date.
    sex : PatientSex | None = None
        The sex of the patient.
    species_description : str | Code | None = None
        The species description of the patient.
    de_identification : PatientDeIdentification | None = None
        The de-identification of the patient.
    comments : str | None = None
        Comments about the patient.
    """

    name: str | None = None
    identifier: str | None = None
    birth_date: datetime.date | None = None
    sex: PatientSex | None = None
    species_description: str | Code | None = None
    de_identification: PatientDeIdentification | None = None
    comments: str | None = None

    def remove_confidential(self) -> "Patient":
        return replace(
            self,
            name=None,
            identifier=None,
            birth_date=None,
            sex=None,
            de_identification=(
                replace(self.de_identification, identity_removed=True)
                if self.de_identification
                else None
            ),
            comments=None,
        )
