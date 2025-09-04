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

"""DICOM schema for Series model."""

from typing import Type

from marshmallow import fields
from pydicom.valuerep import VR

from wsidicom.metadata.schema.dicom.fields import (
    DefaultingDicomField,
    DefaultingTagDicomField,
    StringDicomField,
    UidDicomField,
)
from wsidicom.metadata.schema.dicom.schema import ModuleDicomSchema
from wsidicom.metadata.series import Series


class SeriesDicomSchema(ModuleDicomSchema[Series]):
    uid = DefaultingTagDicomField(
        UidDicomField(), tag="default_uid", data_key="SeriesInstanceUID"
    )
    number = DefaultingDicomField(
        fields.Integer(), dump_default=1, data_key="SeriesNumber", allow_none=True
    )
    description = StringDicomField(
        value_representation=VR.LT, data_key="SeriesDescription", allow_none=True
    )
    body_part_examined = StringDicomField(
        value_representation=VR.CS, data_key="BodyPartExamined", allow_none=True
    )

    @property
    def load_type(self) -> Type[Series]:
        return Series

    @property
    def module_name(self) -> str:
        return "series"
