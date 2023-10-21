from typing import Type
from wsidicom.metadata.dicom_schema.dicom_schema import DicomSchema
from wsidicom.metadata.dicom_schema.dicom_fields import (
    DateDicomField,
    DefaultingTagDicomField,
    PatientNameDicomField,
    TimeDicomField,
    UidDicomField,
)

from marshmallow import fields

from wsidicom.metadata.series import Series
from wsidicom.metadata.study import Study


class StudyDicomSchema(DicomSchema[Study]):
    """
    Type 1
    - uid
    Type 2
    - identifier
    - date
    - time
    - accession number
    - referring_physician_name
    """

    uid = DefaultingTagDicomField(
        UidDicomField(), tag="_uid", data_key="StudyInstanceUID", allow_none=True
    )
    identifier = fields.String(data_key="StudyID", allow_none=True)
    date = DateDicomField(data_key="StudyDate", allow_none=True)
    time = TimeDicomField(data_key="StudyTime", allow_none=True)
    accession_number = fields.String(data_key="AccessionNumber", allow_none=True)
    referring_physician_name = PatientNameDicomField(
        data_key="ReferringPhysicianName", allow_none=True
    )

    @property
    def load_type(self) -> Type[Study]:
        return Study
