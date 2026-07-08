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

from datetime import datetime

import pytest
from pydicom import Dataset

from wsidicom.conceptcode import ContributingEquipmentPurposeCode
from wsidicom.metadata.contributing_equipment import ContributingEquipment
from wsidicom.metadata.schema.dicom.contributing_equipment import (
    ContributingEquipmentDicomSchema,
)
from wsidicom.metadata.schema.dicom.wsi import BaseWsiMetadataDicomSchema


@pytest.fixture
def equipment() -> ContributingEquipment:
    return ContributingEquipment(
        purpose=ContributingEquipmentPurposeCode("Modifying Equipment"),
        manufacturer="wsidicom",
        model_name="wsidicomizer",
        software_versions=["1.2.3"],
        description="Converted to DICOM WSI by wsidicomizer",
        contribution_datetime=datetime(2026, 7, 8, 12, 30, 15),
    )


class TestContributingEquipmentDicomSchema:
    def test_serialize(self, equipment: ContributingEquipment):
        # Arrange
        schema = ContributingEquipmentDicomSchema()

        # Act
        serialized = schema.dump(equipment)

        # Assert
        assert isinstance(serialized, Dataset)
        purpose = serialized.PurposeOfReferenceCodeSequence[0]
        assert purpose.CodeValue == "109103"
        assert purpose.CodingSchemeDesignator == "DCM"
        assert serialized.Manufacturer == "wsidicom"
        assert serialized.ManufacturerModelName == "wsidicomizer"
        assert serialized.SoftwareVersions == "1.2.3"
        assert serialized.ContributionDescription == (
            "Converted to DICOM WSI by wsidicomizer"
        )
        assert str(serialized.ContributionDateTime).startswith("20260708123015")

    def test_round_trip(self, equipment: ContributingEquipment):
        # Arrange
        schema = ContributingEquipmentDicomSchema()

        # Act
        deserialized = schema.load(schema.dump(equipment))

        # Assert
        assert deserialized == equipment

    def test_loaded_from_metadata_sequence(self):
        # Arrange
        equipment = ContributingEquipment(
            purpose=ContributingEquipmentPurposeCode("Modifying Equipment"),
            model_name="wsidicomizer",
        )
        dataset = Dataset()
        dataset.ContributingEquipmentSequence = [
            ContributingEquipmentDicomSchema().dump(equipment)
        ]

        # Act
        metadata = BaseWsiMetadataDicomSchema().load(dataset)

        # Assert
        assert list(metadata.contributing_equipment) == [equipment]

    def test_absent_sequence_loads_empty(self):
        # Arrange
        dataset = Dataset()

        # Act
        metadata = BaseWsiMetadataDicomSchema().load(dataset)

        # Assert
        assert list(metadata.contributing_equipment) == []
