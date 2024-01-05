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

from typing import Type

from marshmallow import fields, pre_load
from pydicom import Dataset

from wsidicom.metadata.schema.dicom.fields import FlattenOnDumpNestedDicomField
from wsidicom.metadata.schema.dicom.schema import DicomSchema


class PropertySchema(DicomSchema):
    value = fields.String(data_key="PatientID")

    @property
    def load_type(self) -> Type[dict]:
        return dict

    @pre_load
    def pre_load(self, dataset: Dataset, many: bool, **kwargs):
        return super().pre_load(dataset, many, **kwargs)


class ChildSchema(DicomSchema):
    nested = FlattenOnDumpNestedDicomField(PropertySchema())

    @property
    def load_type(self) -> Type[dict]:
        return dict

    @pre_load
    def pre_load(self, dataset: Dataset, many: bool, **kwargs):
        return super().pre_load(dataset, many, **kwargs)


class ParentSchema(DicomSchema):
    nested = FlattenOnDumpNestedDicomField(ChildSchema())

    @property
    def load_type(self) -> Type[dict]:
        return dict

    @pre_load
    def pre_load(self, dataset: Dataset, many: bool, **kwargs):
        return super().pre_load(dataset, many, **kwargs)


class TestNesting:
    def test_nesting(self):
        # Arrange
        schema = ParentSchema()

        dataset = Dataset()
        dataset.PatientID = "patient id"

        deserialized = schema.load(dataset)

        assert deserialized["nested"]["nested"]["value"] == "patient id"
