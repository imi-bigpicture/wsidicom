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

"""DICOM schema for Study model."""

from typing import Type

from pydicom.valuerep import VR

from wsidicom.metadata.schema.dicom.fields import (
    DateDicomField,
    DefaultingNoneDicomField,
    DefaultingTagDicomField,
    PatientNameDicomField,
    StringDicomField,
    TimeDicomField,
    UidDicomField,
)
from wsidicom.metadata.schema.dicom.schema import ModuleDicomSchema
from wsidicom.metadata.study import Study


class StudyDicomSchema(ModuleDicomSchema[Study]):
    uid = DefaultingTagDicomField(
        UidDicomField(), tag="default_uid", data_key="StudyInstanceUID", allow_none=True
    )
    identifier = DefaultingNoneDicomField(
        StringDicomField(value_representation=VR.ST),
        data_key="StudyID",
        allow_none=True,
    )
    date = DefaultingNoneDicomField(
        DateDicomField(), data_key="StudyDate", allow_none=True
    )
    time = DefaultingNoneDicomField(
        TimeDicomField(), data_key="StudyTime", allow_none=True
    )
    accession_number = DefaultingNoneDicomField(
        StringDicomField(value_representation=VR.ST),
        data_key="AccessionNumber",
        allow_none=True,
    )
    referring_physician_name = DefaultingNoneDicomField(
        PatientNameDicomField(), data_key="ReferringPhysicianName", allow_none=True
    )

    @property
    def load_type(self) -> Type[Study]:
        return Study

    @property
    def module_name(self) -> str:
        return "study"
