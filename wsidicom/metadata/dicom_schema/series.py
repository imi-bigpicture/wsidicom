from typing import Type
from wsidicom.metadata.dicom_schema.schema import DicomSchema
from wsidicom.metadata.dicom_schema.fields import (
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
        UidDicomField(), tag="default_uid", data_key="SeriesInstanceUID"
    )
    number = DefaultingDicomField(
        fields.Integer(), dump_default=1, data_key="SeriesNumber", allow_none=True
    )

    @property
    def load_type(self) -> Type[Series]:
        return Series
