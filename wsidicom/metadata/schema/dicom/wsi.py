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

"""DICOM schema for complete WsiMetadata model."""

from dataclasses import replace
from typing import Sequence, Type

from marshmallow import fields
from pydicom import Dataset
from pydicom.uid import VLWholeSlideMicroscopyImageStorage

from wsidicom.instance.dataset import ImageType
from wsidicom.metadata.equipment import Equipment
from wsidicom.metadata.image import Image
from wsidicom.metadata.label import Label
from wsidicom.metadata.optical_path import OpticalPath
from wsidicom.metadata.patient import Patient
from wsidicom.metadata.schema.dicom.equipment import EquipmentDicomSchema
from wsidicom.metadata.schema.dicom.fields import (
    DefaultingListDicomField,
    DefaultingListTagDicomField,
    DefaultingTagDicomField,
    FlattenOnDumpNestedDicomField,
    UidDatasetDicomField,
    UidDicomField,
)
from wsidicom.metadata.schema.dicom.image import ImageDicomSchema
from wsidicom.metadata.schema.dicom.label import LabelDicomSchema
from wsidicom.metadata.schema.dicom.optical_path import OpticalPathDicomSchema
from wsidicom.metadata.schema.dicom.patient import PatientDicomSchema
from wsidicom.metadata.schema.dicom.schema import DicomSchema
from wsidicom.metadata.schema.dicom.series import SeriesDicomSchema
from wsidicom.metadata.schema.dicom.slide import SlideDicomSchema
from wsidicom.metadata.schema.dicom.study import StudyDicomSchema
from wsidicom.metadata.series import Series
from wsidicom.metadata.slide import Slide
from wsidicom.metadata.study import Study
from wsidicom.metadata.wsi import WsiMetadata


class WsiMetadataDicomSchema(DicomSchema[WsiMetadata]):
    study = FlattenOnDumpNestedDicomField(
        StudyDicomSchema(), dump_default=Study(), load_default=Study()
    )
    series = FlattenOnDumpNestedDicomField(
        SeriesDicomSchema(), dump_default=Series(), load_default=Series()
    )
    patient = FlattenOnDumpNestedDicomField(
        PatientDicomSchema(), dump_default=Patient(), load_default=Patient()
    )
    equipment = FlattenOnDumpNestedDicomField(
        EquipmentDicomSchema(), dump_default=Equipment(), load_default=Equipment()
    )
    optical_paths = DefaultingListDicomField(
        FlattenOnDumpNestedDicomField(OpticalPathDicomSchema()),
        data_key="OpticalPathSequence",
        dump_default=[OpticalPath()],
        load_default=[],
    )
    slide = FlattenOnDumpNestedDicomField(
        SlideDicomSchema(), dump_default=Slide(), load_default=Slide()
    )
    label = FlattenOnDumpNestedDicomField(
        LabelDicomSchema(), dump_default=Label(), load_default=Label()
    )
    image = FlattenOnDumpNestedDicomField(
        ImageDicomSchema(), dump_default=Image(), load_default=Image()
    )
    frame_of_reference_uid = DefaultingTagDicomField(
        UidDicomField(),
        allow_none=True,
        data_key="FrameOfReferenceUID",
        tag="default_frame_of_reference_uid",
    )
    dimension_organization_uids = DefaultingListTagDicomField(
        UidDatasetDicomField(data_key="DimensionOrganizationUID"),
        tag="default_dimension_organization_uids",
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
        return replace(metadata, label=merged_label)
