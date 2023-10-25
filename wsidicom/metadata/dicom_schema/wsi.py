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

from typing import Sequence, Type

from marshmallow import fields
from pydicom import Dataset
from pydicom.uid import VLWholeSlideMicroscopyImageStorage
from wsidicom.instance.dataset import ImageType

from wsidicom.metadata.dicom_schema.schema import DicomSchema
from wsidicom.metadata.dicom_schema.fields import (
    DefaultingTagDicomField,
    FlatteningNestedField,
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
    study = FlatteningNestedField(
        StudyDicomSchema(), dump_default=Study(), load_default=Study()
    )
    series = FlatteningNestedField(
        SeriesDicomSchema(), dump_default=Series(), load_default=Series()
    )
    patient = FlatteningNestedField(
        PatientDicomSchema(), dump_default=Patient(), load_default=Patient()
    )
    equipment = FlatteningNestedField(
        EquipmentDicomSchema(), dump_default=Equipment(), load_default=Equipment()
    )
    optical_paths = fields.List(
        FlatteningNestedField(OpticalPathDicomSchema()),
        data_key="OpticalPathSequence",
        dump_default=[OpticalPath()],
        load_default=[],
    )
    slide = FlatteningNestedField(
        SlideDicomSchema(), dump_default=Slide(), load_default=Slide()
    )
    label = FlatteningNestedField(
        LabelDicomSchema(), dump_default=Label(), load_default=Label()
    )
    image = FlatteningNestedField(
        ImageDicomSchema(), dump_default=Image(), load_default=Image()
    )
    frame_of_reference_uid = DefaultingTagDicomField(
        UidDicomField(),
        allow_none=True,
        data_key="FrameOfReferenceUID",
        tag="default_frame_of_reference_uid",
    )
    dimension_organization_uids = fields.List(
        DefaultingTagDicomField(
            UidDicomField(),
            tag="default_dimension_organization_uids",
            data_key="DimensionOrganizationUID",
        ),
        data_key="DimensionOrganizationSequence",
    )
    sop_class_uid = fields.Constant(
        VLWholeSlideMicroscopyImageStorage, dump_only=True, data_key="SOPClassUID"
    )
    modality = fields.Constant("SM", dump_only=True, data_key="Modality")
    position_reference_indicator = fields.Constant(
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

    @classmethod
    def from_datasets(cls, datasets: Sequence[Dataset]) -> WsiMetadata:
        label_dataset = next(
            (
                dataset
                for dataset in datasets
                if dataset.ImageType[2] == ImageType.LABEL.value
            ),
            None,
        )
        overview_dataset = next(
            (
                dataset
                for dataset in datasets
                if dataset.ImageType[2] == ImageType.OVERVIEW.value
            ),
            None,
        )
        volume_dataset = next(
            dataset
            for dataset in datasets
            if dataset.ImageType[2] == ImageType.VOLUME.value
        )
        label_dicom_schema = LabelDicomSchema()
        metadata = WsiMetadataDicomSchema().load(volume_dataset)
        if label_dataset is None:
            label_label = None
        else:
            label_label = label_dicom_schema.load(label_dataset)
        if overview_dataset is None:
            overview_label = None
        else:
            overview_label = label_dicom_schema.load(overview_dataset)
        assert metadata.label is not None
        merged_label = Label.merge_image_types(
            metadata.label, label_label, overview_label
        )
        metadata.label = merged_label
        return metadata
