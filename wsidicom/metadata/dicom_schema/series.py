from typing import Type
from wsidicom.metadata.dicom_schema.base_dicom_schema import DicomSchema
from wsidicom.metadata.dicom_schema.dicom_fields import (
    DefaultingDicomField,
    DefaultingTagDicomField,
    UidDicomField,
)

from marshmallow import fields

from wsidicom.metadata.series import Series


class SeriesDicomSchema(DicomSchema[Series]):
    """
    Type 1
    - uid
    - number
    """

    uid = DefaultingTagDicomField(
        UidDicomField(), tag="_uid", data_key="SeriesInstanceUID"
    )
    number = DefaultingDicomField(
        fields.Integer(), dump_default=1, data_key="SeriesNumber", allow_none=True
    )

    @property
    def load_type(self) -> Type[Series]:
        return Series
