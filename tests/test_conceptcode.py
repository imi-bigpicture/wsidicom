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

import pytest
from pydicom import Dataset

from wsidicom.conceptcode import CodeValueAttribute, SpecimenStainsCode, UnitCode


@pytest.mark.unittest
class TestConceptCode:
    def test_to_ds_writes_the_code(self) -> None:
        """Value, scheme designator and meaning are written as their attributes."""
        # Arrange
        code = UnitCode(value="mm", scheme_designator="UCUM", meaning="millimeter")

        # Act
        dataset = code.to_ds()

        # Assert
        assert dataset.CodeValue == "mm"
        assert dataset.CodingSchemeDesignator == "UCUM"
        assert dataset.CodeMeaning == "millimeter"

    def test_from_ds_reads_the_code(self) -> None:
        # Arrange
        code = UnitCode(
            value="mm",
            scheme_designator="UCUM",
            meaning="millimeter",
            scheme_version="1.4",
        )
        dataset = Dataset()
        dataset.MeasurementUnitsCodeSequence = [code.to_ds()]

        # Act
        read = UnitCode.from_ds(dataset)

        # Assert
        assert read == code

    def test_from_ds_without_a_scheme_version_reads_none(self) -> None:
        # Arrange
        code = UnitCode(value="mm", scheme_designator="UCUM", meaning="millimeter")
        dataset = Dataset()
        dataset.MeasurementUnitsCodeSequence = [code.to_ds()]

        # Act
        read = UnitCode.from_ds(dataset)

        # Assert
        assert read is not None
        assert read.scheme_version is None

    def test_to_ds_writes_the_scheme_version(self) -> None:
        """The version is written as Coding Scheme Version, the attribute for it."""
        # Arrange
        code = UnitCode(
            value="mm",
            scheme_designator="UCUM",
            meaning="millimeter",
            scheme_version="1.4",
        )

        # Act
        dataset = code.to_ds()

        # Assert
        assert dataset.CodingSchemeVersion == "1.4"

    def test_to_ds_without_a_scheme_version_writes_none(self) -> None:
        # Arrange
        code = UnitCode(value="mm", scheme_designator="UCUM", meaning="millimeter")

        # Act
        dataset = code.to_ds()

        # Assert
        assert "CodingSchemeVersion" not in dataset


@pytest.mark.unittest
class TestCodeValueAttribute:
    @pytest.mark.parametrize(
        ["value", "expected"],
        [
            ("85756007", "CodeValue"),
            ("1234567890123456", "CodeValue"),
            ("abc:def", "CodeValue"),
            ("12345678901234567", "LongCodeValue"),
            ("999000011000000103", "LongCodeValue"),
            ("urn:oid:1.2.3", "URNCodeValue"),
            ("URN:oid:1.2.3", "URNCodeValue"),
            ("http://example.org/codes/1", "URNCodeValue"),
            ("https://example.org/codes/a-much-longer-code", "URNCodeValue"),
        ],
    )
    def test_keyword_for(self, value: str, expected: str) -> None:
        """Up to 16 characters is Code Value, longer is Long Code Value.

        A URN
        or URL is URN Code Value whatever its length.
        """
        # Arrange

        # Act
        keyword = CodeValueAttribute.keyword_for(value)

        # Assert
        assert keyword == expected

    @pytest.mark.parametrize("keyword", ["CodeValue", "LongCodeValue", "URNCodeValue"])
    def test_read_reads_whichever_attribute_holds_the_value(self, keyword: str) -> None:
        # Arrange
        dataset = Dataset()
        setattr(dataset, keyword, "value")

        # Act
        value = CodeValueAttribute.read(dataset)

        # Assert
        assert value == "value"

    def test_read_skips_an_empty_attribute(self) -> None:
        # Arrange
        dataset = Dataset()
        dataset.CodeValue = ""
        dataset.LongCodeValue = "999000011000000103"

        # Act
        value = CodeValueAttribute.read(dataset)

        # Assert
        assert value == "999000011000000103"

    def test_read_without_a_value_raises(self) -> None:
        # Arrange
        dataset = Dataset()
        dataset.CodingSchemeDesignator = "SCT"

        # Act & Assert
        with pytest.raises(ValueError, match="holds no value"):
            CodeValueAttribute.read(dataset)


@pytest.mark.unittest
class TestConceptCodeValueAttribute:
    @pytest.mark.parametrize(
        ["value", "keyword"],
        [
            ("999000011000000103", "LongCodeValue"),
            ("urn:oid:1.2.3", "URNCodeValue"),
        ],
    )
    def test_to_ds_writes_a_value_code_value_cannot_hold_elsewhere(
        self, value: str, keyword: str
    ) -> None:
        # Arrange
        code = SpecimenStainsCode(value=value, scheme_designator="SCT", meaning="x")

        # Act
        dataset = code.to_ds()

        # Assert
        assert dataset[keyword].value == value
        assert "CodeValue" not in dataset

    @pytest.mark.parametrize("value", ["999000011000000103", "urn:oid:1.2.3"])
    def test_from_ds_reads_a_value_code_value_cannot_hold(self, value: str) -> None:
        # Arrange
        code = SpecimenStainsCode(value=value, scheme_designator="SCT", meaning="x")
        dataset = Dataset()
        code.insert_into_ds(dataset)

        # Act
        read = SpecimenStainsCode.from_ds(dataset)

        # Assert
        assert read == code
