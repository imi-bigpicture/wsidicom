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

"""DICOM schema for ContributingEquipment model."""

from pydicom.valuerep import VR

from wsidicom.conceptcode import ContributingEquipmentPurposeCode
from wsidicom.metadata.contributing_equipment import ContributingEquipment
from wsidicom.metadata.schema.dicom.fields import (
    DateTimeDicomField,
    ListDicomField,
    SingleCodeSequenceField,
    StringDicomField,
)
from wsidicom.metadata.schema.dicom.schema import DicomSchema


class ContributingEquipmentDicomSchema(DicomSchema[ContributingEquipment]):
    purpose = SingleCodeSequenceField(
        ContributingEquipmentPurposeCode, data_key="PurposeOfReferenceCodeSequence"
    )
    manufacturer = StringDicomField(
        VR.LO, data_key="Manufacturer", allow_none=True, load_default=None
    )
    model_name = StringDicomField(
        VR.LO, data_key="ManufacturerModelName", allow_none=True, load_default=None
    )
    software_versions = ListDicomField(
        StringDicomField(VR.LO),
        dump_none_if_empty=True,
        data_key="SoftwareVersions",
        allow_none=True,
        load_default=None,
    )
    description = StringDicomField(
        VR.ST, data_key="ContributionDescription", allow_none=True, load_default=None
    )
    contribution_datetime = DateTimeDicomField(
        data_key="ContributionDateTime", allow_none=True, load_default=None
    )

    @property
    def load_type(self) -> type[ContributingEquipment]:
        return ContributingEquipment
