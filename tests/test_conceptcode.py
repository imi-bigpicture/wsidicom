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

from wsidicom.conceptcode import UnitCode


@pytest.mark.unittest
class TestConceptCode:
    def test_to_ds_writes_the_code(self):
        """Value, scheme designator and meaning are written as their attributes."""
        # Arrange
        code = UnitCode(value="mm", scheme_designator="UCUM", meaning="millimeter")

        # Act
        dataset = code.to_ds()

        # Assert
        assert dataset.CodeValue == "mm"
        assert dataset.CodingSchemeDesignator == "UCUM"
        assert dataset.CodeMeaning == "millimeter"

    def test_from_ds_reads_the_code(self):
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

    def test_from_ds_without_a_scheme_version_reads_none(self):
        # Arrange
        code = UnitCode(value="mm", scheme_designator="UCUM", meaning="millimeter")
        dataset = Dataset()
        dataset.MeasurementUnitsCodeSequence = [code.to_ds()]

        # Act
        read = UnitCode.from_ds(dataset)

        # Assert
        assert read is not None
        assert read.scheme_version is None

    def test_to_ds_writes_the_scheme_version(self):
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

    def test_to_ds_without_a_scheme_version_writes_none(self):
        # Arrange
        code = UnitCode(value="mm", scheme_designator="UCUM", meaning="millimeter")

        # Act
        dataset = code.to_ds()

        # Assert
        assert "CodingSchemeVersion" not in dataset
