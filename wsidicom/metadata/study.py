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

"""Study model."""

import datetime
from dataclasses import dataclass
from functools import cached_property

from pydicom.uid import UID, generate_uid


@dataclass(frozen=True)
class Study:
    """
    Study metadata.

    Corresponds to the `Required` and `Required, Empty if Unknown` attributes in the
    General Study Module:
    https://dicom.nema.org/medical/Dicom/current/output/chtml/part03/sect_C.7.2.html

    Parameters
    ----------
    uid : UID | None = None
        The study instance UID.
    identifier : str | None = None
        The study identifier (study ID).
    date : datetime.date | None = None
        The date the study was performed.
    time : datetime.time | None = None
        The time the study was performed.
    accession_number : str | None = None
        The accession number of the study.
    referring_physician_name : str | None = None
        The name of the referring physician.
    description : str | None = None
        The description of the study.
    """

    uid: UID | None = None
    identifier: str | None = None
    date: datetime.date | None = None
    time: datetime.time | None = None
    accession_number: str | None = None
    referring_physician_name: str | None = None
    description: str | None = None

    @cached_property
    def default_uid(self) -> UID:
        if self.uid is not None:
            return self.uid
        return generate_uid()
