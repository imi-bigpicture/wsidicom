import datetime
from typing import Union
from pydicom import Dataset
from pydicom.sr.coding import Code
from pydicom.valuerep import DSfloat
import pytest

from wsidicom.metadata.dicom_schema.fields import (
    CodeItemDicomField,
    DateTimeItemDicomField,
    FloatContentItemDicomField,
    StringItemDicomField,
    StringOrCodeItemDicomField,
)


class TestDicomFields:
    def test_code_content_item_dicom_field_serialize(self):
        # Arrange
        code = Code("1234", "DCM", "Test Code")
        field = CodeItemDicomField(Code)

        # Act
        serialized = field.serialize("attribute", {"attribute": code})

        # Assert
        assert isinstance(serialized, Dataset)
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

    def test_test_code_content_item_dicom_field_serialize(self):
        # Arrange
        value = "test"
        field = StringItemDicomField()

        # Act
        serialized = field.serialize("attribute", {"attribute": value})

        # Assert
        assert isinstance(serialized, Dataset)
        assert "TextValue" in serialized
        assert serialized.TextValue == value

    def test_text_content_item_dicom_field_deserialize(self):
        # Arrange
        dataset = Dataset()
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
        assert "DateTime" in serialized
        assert serialized.DateTime == value

    def test_datetime_content_item_dicom_field_deserialize(self):
        # Arrange
        dataset = Dataset()
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
        field = FloatContentItemDicomField(Code("1234", "DCM", "Test Unit"))

        # Act
        serialized = field.serialize("attribute", {"attribute": value})

        # Assert
        assert isinstance(serialized, Dataset)
        assert "NumericValue" in serialized
        assert serialized.NumericValue == DSfloat(value)
        assert "FloatingPointValue" in serialized
        assert serialized.FloatingPointValue == value

    def test_float_content_item_dicom_field_deserialize(self):
        # Arrange
        dataset = Dataset()
        dataset.NumericValue = DSfloat(1.0)
        dataset.FloatingPointValue = 1.0
        field = FloatContentItemDicomField(Code("1234", "DCM", "Test Unit"))

        # Act
        deserialized = field.deserialize(dataset, "attribute")

        # Assert
        assert isinstance(deserialized, float)
        assert deserialized == 1.0

    @pytest.mark.parametrize("value", ["test", Code("1234", "DCM", "Test Code")])
    def test_string_or_code_item_dicom_field_serialize(self, value: Union[str, Code]):
        # Arrange
        field = StringOrCodeItemDicomField(Code)

        # Act
        serialized = field.serialize("attribute", {"attribute": value})

        # Assert
        assert isinstance(serialized, Dataset)
        if isinstance(value, str):
            assert "TextValue" in serialized
            assert serialized.TextValue == value
        else:
            assert "ConceptCodeSequence" in serialized
            assert serialized.ConceptCodeSequence[0].CodeValue == value.value
            assert (
                serialized.ConceptCodeSequence[0].CodingSchemeDesignator
                == value.scheme_designator
            )
            assert serialized.ConceptCodeSequence[0].CodeMeaning == value.meaning

    @pytest.mark.parametrize("value", ["test", Code("1234", "DCM", "Test Code")])
    def test_string_or_code_item_dicom_field_deserialize(self, value: Union[str, Code]):
        # Arrange
        dataset = Dataset()
        if isinstance(value, str):
            dataset.TextValue = value
        else:
            code_dataset = Dataset()
            code_dataset.CodeValue = value.value
            code_dataset.CodingSchemeDesignator = value.scheme_designator
            code_dataset.CodeMeaning = value.meaning
            dataset.ConceptCodeSequence = [code_dataset]
        field = StringOrCodeItemDicomField(Code)

        # Act
        deserialized = field.deserialize(dataset, "attribute")

        # Assert
        if isinstance(value, str):
            assert isinstance(deserialized, str)
        else:
            assert isinstance(deserialized, Code)
        assert deserialized == value
