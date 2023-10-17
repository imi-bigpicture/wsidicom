from typing import Type

from marshmallow import fields
from pydicom.uid import VLWholeSlideMicroscopyImageStorage

from wsidicom.metadata.dicom_schema.base_dicom_schema import DicomSchema
from wsidicom.metadata.dicom_schema.dicom_fields import (
    DefaultingTagDicomField,
    FlatteningNestedField,
    SequenceWrappingField,
    UidDicomField,
)
from wsidicom.metadata.dicom_schema.equipment import EquipmentDicomSchema
from wsidicom.metadata.dicom_schema.image import ImageDicomSchema
from wsidicom.metadata.dicom_schema.label import LabelDicomSchema
from wsidicom.metadata.dicom_schema.optical_path import OpticalPathDicomSchema
from wsidicom.metadata.dicom_schema.patient import PatientDicomSchema
from wsidicom.metadata.dicom_schema.series import SeriesDicomSchema
from wsidicom.metadata.dicom_schema.slide import SlideDicomSchema
from wsidicom.metadata.dicom_schema.study import StudyDicomSchema
from wsidicom.metadata.equipment import Equipment
from wsidicom.metadata.image import Image
from wsidicom.metadata.label import Label
from wsidicom.metadata.optical_path import OpticalPath
from wsidicom.metadata.patient import Patient
from wsidicom.metadata.series import Series
from wsidicom.metadata.slide import Slide
from wsidicom.metadata.study import Study
from wsidicom.metadata.wsi import WsiMetadata


class WsiMetadataDicomSchema(DicomSchema[WsiMetadata]):
    study = FlatteningNestedField(StudyDicomSchema(), dump_default=Study())
    series = FlatteningNestedField(SeriesDicomSchema(), dump_default=Series())
    patient = FlatteningNestedField(
        PatientDicomSchema(), allow_none=True, dump_default=Patient()
    )
    equipment = FlatteningNestedField(EquipmentDicomSchema(), dump_default=Equipment())
    optical_paths = fields.List(
        FlatteningNestedField(OpticalPathDicomSchema(), dump_default=OpticalPath()),
        data_key="OpticalPathSequence",
    )
    slide = FlatteningNestedField(SlideDicomSchema(), dump_default=Slide())
    label = FlatteningNestedField(LabelDicomSchema(), dump_default=Label())
    image = FlatteningNestedField(ImageDicomSchema(), dump_default=Image())
    frame_of_reference_uid = DefaultingTagDicomField(
        UidDicomField(),
        allow_none=True,
        data_key="FrameOfReferenceUID",
        tag="_frame_of_reference_uid",
    )
    dimension_organization_uid = SequenceWrappingField(
        DefaultingTagDicomField(
            UidDicomField(),
            allow_none=True,
            tag="_dimension_organization_uid",
            data_key="DimensionOrganizationUID",
        ),
        data_key="DimensionOrganizationSequence",
    )
    sop_class_uid = fields.Constant(
        VLWholeSlideMicroscopyImageStorage, dump_only=True, data_key="SOPClassUID"
    )
    modality = fields.Constant("SM", dump_only=True, data_key="Modality")
    positiion_reference_indicator = fields.Constant(
        "SLIDE_CORNER", dump_only=True, data_key="PositionReferenceIndicator"
    )
    volumetric_properties = fields.Constant(
        "VOLUME", dump_only=True, data_key="VolumetricProperties"
    )
    acquisition_context = fields.Constant(
        [], data_key="AcquisitionContextSequence", dump_only=True
    )

    @property
    def load_type(self) -> Type[WsiMetadata]:
        return WsiMetadata
