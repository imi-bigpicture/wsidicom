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
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Union

from pydicom.sr.coding import Code


class PatientSex(Enum):
    """Patient sex."""

    F = "female"
    M = "male"
    O = "other"


@dataclass
class PatientDeIdentification:
    """Patient de-identification.

    Parameters
    ----------
    identity_removed : bool
        Whether the patient identity has been removed.
    methods : Optional[Sequence[Union[str, Code]]] = None
        The methods used to de-identify the patient.
    """

    identity_removed: bool
    methods: Optional[Sequence[Union[str, Code]]] = None


@dataclass
class Patient:
    """
    Patient metadata.

    Corresponds to the `Required` and `Required, Empty if Unknown` attributes in the
    Patient Module:
    https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.html

    Parameters
    ----------
    name : Optional[str] = None
        The patient name.
    identifier : Optional[str] = None
        The patient identifier.
    birth_date : Optional[datetime.date] = None
        The patient birth date.
    sex : Optional[PatientSex] = None
        The sex of the patient.
    species_description : Optional[Union[str, Code]] = None
        The species description of the patient.
    de_identification : Optional[PatientDeIdentification] = None
        The de-identification of the patient.
    """

    name: Optional[str] = None
    identifier: Optional[str] = None
    birth_date: Optional[datetime.date] = None
    sex: Optional[PatientSex] = None
    species_description: Optional[Union[str, Code]] = None
    de_identification: Optional[PatientDeIdentification] = None
