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

import datetime
from typing import Optional, Type

import pytest
from marshmallow import fields
from pydicom import Dataset
from pydicom import config as pydicom_config
from pydicom.sr.coding import Code
from pydicom.valuerep import MAX_VALUE_LEN, STR_VR, VR, DSfloat

from tests.metadata.dicom_schema.helpers import (
    assert_dicom_issuer_of_identifier_equals_issuer_of_identifier,
    create_code_dataset,
)
from wsidicom import config
from wsidicom.conceptcode import UnitCode
from wsidicom.metadata.sample import (
    IssuerOfIdentifier,
    LocalIssuerOfIdentifier,
    Measurement,
    UniversalIssuerOfIdentifier,
    UniversalIssuerType,
)
from wsidicom.metadata.schema.dicom.fields import (
    CodeDicomField,
    CodeItemDicomField,
    DateTimeItemDicomField,
    FlattenOnDumpNestedDicomField,
    IssuerOfIdentifierDicomField,
    MeasurementtemDicomField,
    StringDicomField,
    StringItemDicomField,
)
from wsidicom.metadata.schema.dicom.schema import DicomSchema


class ChildFlattenOnDumpSchema(DicomSchema):
    value = fields.String(data_key="PatientID")

    @property
    def load_type(self) -> Type[dict]:
        return dict


class ParentFlattenOnDumpSchema(DicomSchema):
    nested = FlattenOnDumpNestedDicomField(ChildFlattenOnDumpSchema())

    @property
    def load_type(self) -> Type[dict]:
        return dict


class TestDicomFields:
    def test_code_content_item_dicom_field_serialize(self):
        # Arrange
        code = Code("1234", "DCM", "Test Code")
        field = CodeItemDicomField(Code)

        # Act
        serialized = field.serialize("attribute", {"attribute": code})

        # Assert
        assert isinstance(serialized, Dataset)
        assert "ValueType" in serialized
        assert serialized.ValueType == "CODE"
        assert "ConceptCodeSequence" in serialized
        assert serialized.ConceptCodeSequence[0].CodeValue == code.value
        assert (
            serialized.ConceptCodeSequence[0].CodingSchemeDesignator
            == code.scheme_designator
        )
        assert serialized.ConceptCodeSequence[0].CodeMeaning == code.meaning

    def test_code_content_item_dicom_field_deserialize(self):
        # Arrange
        dataset = Dataset()
        dataset.ValueType = "CODE"
        code_dataset = Dataset()
        code_dataset.CodeValue = "1234"
        code_dataset.CodingSchemeDesignator = "DCM"
        code_dataset.CodeMeaning = "Test Code"
        dataset.ConceptCodeSequence = [code_dataset]
        field = CodeItemDicomField(Code)

        # Act
        deserialized = field.deserialize(dataset, "attribute")

        # Assert
        assert isinstance(deserialized, Code)
        assert deserialized.value == "1234"
        assert deserialized.scheme_designator == "DCM"
        assert deserialized.meaning == "Test Code"

    def test_text_content_item_dicom_field_serialize(self):
        # Arrange
        value = "test"
        field = StringItemDicomField()

        # Act
        serialized = field.serialize("attribute", {"attribute": value})

        # Assert
        assert isinstance(serialized, Dataset)
        assert "ValueType" in serialized
        assert serialized.ValueType == "TEXT"
        assert "TextValue" in serialized
        assert serialized.TextValue == value

    def test_text_content_item_dicom_field_deserialize(self):
        # Arrange
        dataset = Dataset()
        dataset.ValueType = "TEXT"
        dataset.TextValue = "test"
        field = StringItemDicomField()

        # Act
        deserialized = field.deserialize(dataset, "attribute")

        # Assert
        assert isinstance(deserialized, str)
        assert deserialized == "test"

    def test_datetime_content_item_dicom_field_serialize(self):
        # Arrange
        value = datetime.datetime(2021, 1, 1)
        field = DateTimeItemDicomField()

        # Act
        serialized = field.serialize("attribute", {"attribute": value})

        # Assert
        assert isinstance(serialized, Dataset)
        assert "ValueType" in serialized
        assert serialized.ValueType == "DATETIME"
        assert "DateTime" in serialized
        assert serialized.DateTime == value

    def test_datetime_content_item_dicom_field_deserialize(self):
        # Arrange
        dataset = Dataset()
        dataset.ValueType = "DATETIME"
        dataset.DateTime = "20210101"
        field = DateTimeItemDicomField()

        # Act
        deserialized = field.deserialize(dataset, "attribute")

        # Assert
        assert isinstance(deserialized, datetime.datetime)
        assert deserialized == datetime.datetime(2021, 1, 1)

    def test_float_content_item_dicom_field_serialize(self):
        # Arrange
        value = 1.0
        unit = UnitCode("mm", "UCUM", "mm")
        measurement = Measurement(value, unit)
        field = MeasurementtemDicomField()

        # Act
        serialized = field.serialize("attribute", {"attribute": measurement})

        # Assert
        assert isinstance(serialized, Dataset)
        assert "ValueType" in serialized
        assert serialized.ValueType == "NUMERIC"
        assert "NumericValue" in serialized
        assert serialized.NumericValue == DSfloat(measurement.value)
        assert "FloatingPointValue" in serialized
        assert serialized.FloatingPointValue == measurement.value
        assert "MeasurementUnitsCodeSequence" in serialized
        assert serialized.MeasurementUnitsCodeSequence[0].CodeValue == unit.value
        assert (
            serialized.MeasurementUnitsCodeSequence[0].CodingSchemeDesignator
            == unit.scheme_designator
        )
        assert serialized.MeasurementUnitsCodeSequence[0].CodeMeaning == unit.meaning

    def test_float_content_item_dicom_field_deserialize(self):
        # Arrange
        value = 1.0
        unit = Code("1234", "DCM", "Test Unit")
        dataset = Dataset()
        dataset.ValueType = "NUMERIC"
        dataset.NumericValue = DSfloat(1.0)
        dataset.FloatingPointValue = 1.0
        unit_dataset = create_code_dataset(unit)
        dataset.MeasurementUnitsCodeSequence = [unit_dataset]

        field = MeasurementtemDicomField()

        # Act
        deserialized = field.deserialize(dataset, "attribute")

        # Assert
        assert isinstance(deserialized, Measurement)
        assert deserialized.value == value
        assert deserialized.unit == unit

    @pytest.mark.parametrize(
        "issuer",
        [
            None,
            LocalIssuerOfIdentifier("issuer"),
            UniversalIssuerOfIdentifier("issuer", UniversalIssuerType.UUID),
            UniversalIssuerOfIdentifier("issuer", UniversalIssuerType.UUID, "local"),
        ],
    )
    def test_issuer_of_identifier_field_serialize(
        self, issuer: Optional[IssuerOfIdentifier]
    ):
        # Arrange
        field = IssuerOfIdentifierDicomField()

        # Act
        serialized = field.serialize("issuer", {"issuer": issuer})

        # Assert
        assert isinstance(serialized, list)
        if issuer is None:
            assert len(serialized) == 0
        else:
            assert isinstance(serialized, list)
            assert len(serialized) == 1
            item = serialized[0]
            assert isinstance(item, Dataset)
            assert_dicom_issuer_of_identifier_equals_issuer_of_identifier(item, issuer)

    @pytest.mark.parametrize("value_representation", STR_VR)
    def test_string_field_serialize(self, value_representation: VR):
        # Arrange
        length = MAX_VALUE_LEN.get(value_representation.name, 10)
        value = "A" * length
        field = StringDicomField(value_representation)

        # Act

        serialized = field.serialize("attribute", {"attribute": value})

        # Assert
        assert serialized == value

    @pytest.mark.parametrize("value_representation", (VR(vr) for vr in MAX_VALUE_LEN))
    def test_string_field_serialize_truncate_long_strings(
        self, value_representation: VR
    ):
        # Arrange
        pydicom_config.settings.writing_validation_mode = pydicom_config.RAISE
        config.settings.truncate_long_dicom_strings_on_validation_error = True
        maximum_allowed_length = MAX_VALUE_LEN[value_representation.name]
        length = maximum_allowed_length + 1
        value = "A" * length
        field = StringDicomField(value_representation)

        # Act
        serialized = field.serialize("attribute", {"attribute": value})

        # Assert
        assert serialized == value[:maximum_allowed_length]

    @pytest.mark.parametrize("code_version", [None, "version"])
    def test_code_field_serialize(self, code_version: Optional[str]):
        # Arrange
        code = Code("code", "scheme", "meaning", code_version)
        field = CodeDicomField(Code)

        # Act
        serialized = field.serialize("attribute", {"attribute": code})

        # Assert
        assert isinstance(serialized, Dataset)
        assert serialized["CodeValue"].value == code.value
        assert serialized["CodingSchemeDesignator"].value == code.scheme_designator
        assert serialized["CodeMeaning"].value == code.meaning
        if code.scheme_version is not None:
            assert serialized["CodingSchemeVersion"].value == code.scheme_version
        else:
            assert "CodingSchemeVersion" not in serialized

    @pytest.mark.parametrize("code_version", [None, "version"])
    def test_code_field_deserialize(self, code_version: Optional[str]):
        # Arrange
        code = Code("code", "scheme", "meaning", code_version)
        dataset = Dataset()
        dataset.CodeValue = code.value
        dataset.CodingSchemeDesignator = code.scheme_designator
        dataset.CodeMeaning = code.meaning
        if code_version is not None:
            dataset.CodingSchemeVersion = code_version
        field = CodeDicomField(Code)

        # Act
        deserialized = field.deserialize(dataset)

        # Assert
        assert isinstance(deserialized, Code)
        assert deserialized == code

    def test_flatten_on_dump_nested_dicom_field_load(self):
        # Arrange
        schema = ParentFlattenOnDumpSchema()

        dataset = Dataset()
        dataset.PatientID = "patient id"

        # Act
        deserialized = schema.load(dataset)

        # Assert
        assert deserialized["nested"]["value"] == "patient id"

    def test_flatten_on_dump_nested_dicom_field_dump(self):
        # Arrange
        schema = ParentFlattenOnDumpSchema()

        data = {"nested": {"value": "patient id"}}

        # Act
        serialized = schema.dump(data)

        # Assert
        assert serialized["PatientID"].value == "patient id"
