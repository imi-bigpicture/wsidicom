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
from pydicom.uid import generate_uid

from wsidicom.conceptcode import UnitCode
from wsidicom.config import Settings, use_settings
from wsidicom.metadata import Patient, Series
from wsidicom.metadata.sample import Measurement
from wsidicom.metadata.schema.dicom import PatientDicomSchema, SeriesDicomSchema
from wsidicom.metadata.schema.dicom.fields import MeasurementItemDicomField
from wsidicom.options import DicomValueValidationOption


@pytest.mark.unittest
class TestDicomValueValidation:
    """Values are checked as wsidicom writes them, with the mode passed per value."""

    def test_written_refuses_a_value_too_long_for_its_vr(self):
        # Arrange
        written = Settings(dicom_value_validation=DicomValueValidationOption.WRITTEN)
        # Series Description is LO, which allows 64
        series = Series(uid=generate_uid(), description="X" * 100)

        # Act & Assert
        with use_settings(written), pytest.raises(ValueError):
            SeriesDicomSchema().dump(series)

    def test_none_lets_it_through(self):
        # Arrange
        none = Settings(dicom_value_validation=DicomValueValidationOption.NONE)
        series = Series(uid=generate_uid(), description="X" * 100)

        # Act
        with use_settings(none):
            dataset = SeriesDicomSchema().dump(series)

        # Assert
        assert dataset.SeriesDescription == "X" * 100

    def test_a_conformant_value_passes(self):
        # Arrange
        series = Series(uid=generate_uid(), description="a description")

        # Act
        dataset = SeriesDicomSchema().dump(series)

        # Assert
        assert dataset.SeriesDescription == "a description"

    def test_a_value_in_a_sequence_item_is_checked(self):
        """A field building a sequence item checks what it puts in it."""
        # Arrange
        written = Settings(dicom_value_validation=DicomValueValidationOption.WRITTEN)
        field = MeasurementItemDicomField()
        # Code Meaning is LO, which allows 64
        measurement = Measurement(
            1.0, UnitCode(value="mm", scheme_designator="UCUM", meaning="M" * 100)
        )

        # Act & Assert
        with use_settings(written), pytest.raises(ValueError):
            field._serialize(measurement, "attribute", None)

    def test_a_value_from_a_defaulting_field_is_checked(self):
        """A field that has a default checks its value as any other field does.

        Having a value to fall back on says nothing about how the attribute is
        written, so the value the object does have is checked all the same.
        """
        # Arrange
        written = Settings(dicom_value_validation=DicomValueValidationOption.WRITTEN)
        # Patient Name is PN, which allows 64 characters per component
        patient = Patient(name="X" * 100)

        # Act & Assert
        with use_settings(written), pytest.raises(ValueError):
            PatientDicomSchema().dump(patient)
