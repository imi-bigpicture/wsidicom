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

"""DICOM schema for Equipment model."""

from typing import Type

from pydicom.valuerep import VR

from wsidicom.metadata.equipment import Equipment
from wsidicom.metadata.schema.dicom.defaults import defaults
from wsidicom.metadata.schema.dicom.fields import (
    DefaultingDicomField,
    ListDicomField,
    StringDicomField,
)
from wsidicom.metadata.schema.dicom.schema import ModuleDicomSchema


class EquipmentDicomSchema(ModuleDicomSchema[Equipment]):
    manufacturer = DefaultingDicomField(
        StringDicomField(VR.LO),
        dump_default=defaults.string,
        load_default=None,
        data_key="Manufacturer",
    )
    model_name = DefaultingDicomField(
        StringDicomField(VR.LO),
        dump_default=defaults.string,
        load_default=None,
        data_key="ManufacturerModelName",
    )
    device_serial_number = DefaultingDicomField(
        StringDicomField(VR.LO),
        dump_default=defaults.string,
        load_default=None,
        data_key="DeviceSerialNumber",
    )
    software_versions = DefaultingDicomField(
        ListDicomField(StringDicomField(VR.LO)),
        dump_default=[defaults.string],
        load_default=None,
        data_key="SoftwareVersions",
    )

    @property
    def load_type(self) -> Type[Equipment]:
        return Equipment

    @property
    def module_name(self) -> str:
        return "equipment"
