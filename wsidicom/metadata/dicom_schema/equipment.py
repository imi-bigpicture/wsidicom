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

from wsidicom.metadata.defaults import Defaults
from wsidicom.metadata.dicom_schema.fields import (
    DefaultingDicomField,
    ListDicomField,
    StringDicomField,
)
from wsidicom.metadata.dicom_schema.schema import DicomSchema
from wsidicom.metadata.equipment import Equipment


class EquipmentDicomSchema(DicomSchema[Equipment]):
    manufacturer = DefaultingDicomField(
        StringDicomField(),
        dump_default=Defaults.string,
        load_default=None,
        data_key="Manufacturer",
    )
    model_name = DefaultingDicomField(
        StringDicomField(),
        dump_default=Defaults.string,
        load_default=None,
        data_key="ManufacturerModelName",
    )
    device_serial_number = DefaultingDicomField(
        StringDicomField(),
        dump_default=Defaults.string,
        load_default=None,
        data_key="DeviceSerialNumber",
    )
    software_versions = DefaultingDicomField(
        ListDicomField(StringDicomField()),
        dump_default=[Defaults.string],
        load_default=None,
        data_key="SoftwareVersions",
    )

    @property
    def load_type(self) -> Type[Equipment]:
        return Equipment
