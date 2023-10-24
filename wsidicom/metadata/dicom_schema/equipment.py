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
    """
    Type 1:
    - manufacturer
    - model_name
    - device_serial_number
    - software_versions
    """

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
