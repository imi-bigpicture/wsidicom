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

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TypeVar

from marshmallow import fields
from PIL import ImageCms
from pydicom import Dataset
from pydicom.uid import UID, VLWholeSlideMicroscopyImageStorage

from wsidicom.metadata.contributing_equipment import ContributingEquipment
from wsidicom.metadata.equipment import Equipment
from wsidicom.metadata.image import Image, ImageCoordinateSystem, ImageType
from wsidicom.metadata.label import Label
from wsidicom.metadata.optical_path import OpticalPath
from wsidicom.metadata.overview import Overview
from wsidicom.metadata.patient import Patient
from wsidicom.metadata.pyramid import Pyramid
from wsidicom.metadata.schema.dicom.contributing_equipment import (
    ContributingEquipmentDicomSchema,
)
from wsidicom.metadata.schema.dicom.defaults import defaults
from wsidicom.metadata.schema.dicom.equipment import EquipmentDicomSchema
from wsidicom.metadata.schema.dicom.fields import (
    FlattenOnDumpNestedDicomField,
    ListDicomField,
    UidDatasetDicomField,
    UidDicomField,
)
from wsidicom.metadata.schema.dicom.label import (
    LabelDicomSchema,
    LabelOnlyDicomSchema,
)
from wsidicom.metadata.schema.dicom.overview import OverviewDicomSchema
from wsidicom.metadata.schema.dicom.patient import PatientDicomSchema
from wsidicom.metadata.schema.dicom.pyramid import PyramidDicomSchema
from wsidicom.metadata.schema.dicom.schema import DicomSchema
from wsidicom.metadata.schema.dicom.series import SeriesDicomSchema
from wsidicom.metadata.schema.dicom.slide import SlideDicomSchema
from wsidicom.metadata.schema.dicom.study import StudyDicomSchema
from wsidicom.metadata.series import Series
from wsidicom.metadata.slide import Slide
from wsidicom.metadata.study import Study
from wsidicom.metadata.wsi import WsiMetadata


@dataclass(frozen=True)
class BaseWsiMetadata:
    """Base metadata without pyramid, label, and series."""

    study: Study
    series: Series
    patient: Patient
    equipment: Equipment
    slide: Slide
    frame_of_reference_uid: UID | None = None
    dimension_organization_uids: Sequence[UID] | None = None
    contributing_equipment: Sequence[ContributingEquipment] = ()


class BaseWsiMetadataDicomSchema(DicomSchema[BaseWsiMetadata]):
    """Schema to load and dump BaseWsiMetadata to and from DICOM dataset."""

    study = FlattenOnDumpNestedDicomField(StudyDicomSchema(), load_default=Study())
    series = FlattenOnDumpNestedDicomField(SeriesDicomSchema(), load_default=Series())
    patient = FlattenOnDumpNestedDicomField(
        PatientDicomSchema(), load_default=Patient()
    )
    equipment = FlattenOnDumpNestedDicomField(
        EquipmentDicomSchema(), load_default=Equipment()
    )
    slide = FlattenOnDumpNestedDicomField(SlideDicomSchema(), load_default=Slide())
    frame_of_reference_uid = UidDicomField(
        data_key="FrameOfReferenceUID", load_default=None, dump_required=True
    )
    dimension_organization_uids = ListDicomField(
        UidDatasetDicomField(data_key="DimensionOrganizationUID"),
        data_key="DimensionOrganizationSequence",
        load_default=None,
        dump_required=True,
    )
    contributing_equipment = ListDicomField(
        fields.Nested(ContributingEquipmentDicomSchema()),
        data_key="ContributingEquipmentSequence",
        dump_none_if_empty=True,
        load_default=(),
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
    def load_type(self) -> type[BaseWsiMetadata]:
        return BaseWsiMetadata


ModuleWithImage = TypeVar("ModuleWithImage", Pyramid, Label, Overview)


class WsiMetadataDicomSchema:
    """Schema to load and dump WsiMetadata to and from DICOM dataset."""

    def dump(
        self,
        metadata: WsiMetadata,
        image_type: ImageType,
        require_icc_profile: bool = False,
        **kwargs,
    ) -> Dataset:
        base_dataset = BaseWsiMetadataDicomSchema().dump(
            BaseWsiMetadata(
                study=metadata.study,
                series=metadata.series,
                patient=metadata.patient,
                equipment=metadata.equipment,
                slide=metadata.slide,
                frame_of_reference_uid=metadata.frame_of_reference_uid,
                dimension_organization_uids=metadata.dimension_organization_uids,
                contributing_equipment=metadata.contributing_equipment,
            )
        )
        # Default text to UTF-8 so non-ASCII metadata (e.g. ideographic/phonetic
        # PatientName groups) is encoded correctly; merged into every image type
        # below. setdefault so a caller-supplied SpecificCharacterSet is respected.
        base_dataset.setdefault("SpecificCharacterSet", "ISO_IR 192")
        if image_type == ImageType.VOLUME or image_type == ImageType.THUMBNAIL:
            pyramid = metadata.pyramid
            if metadata.pyramid.image.image_coordinate_system is None:
                pyramid = replace(
                    pyramid,
                    image=WsiMetadataDicomSchema._insert_default_image_coordinate_system(
                        pyramid.image, ImageType.VOLUME
                    ),
                )
            if require_icc_profile:
                pyramid = self._insert_default_icc_profile(pyramid)
            pyramid_dataset = PyramidDicomSchema().dump(pyramid)
            label_dataset = LabelOnlyDicomSchema().dump(metadata.label)
            pyramid_dataset.update(base_dataset)
            pyramid_dataset.update(label_dataset)
            return pyramid_dataset
        if image_type == ImageType.LABEL:
            label = metadata.label
            if label.image is None:
                label = replace(label, image=Image())
            if label.optical_paths is None:
                label = replace(label, optical_paths=[])
            if label.image is not None and label.image.image_coordinate_system is None:
                label = replace(
                    label,
                    image=self._insert_default_image_coordinate_system(
                        label.image, ImageType.LABEL
                    ),
                )

            if require_icc_profile:
                label = self._insert_default_icc_profile(label)
            label_dataset = LabelDicomSchema().dump(label)
            label_dataset.update(base_dataset)
            return label_dataset
        if image_type == ImageType.OVERVIEW:
            if metadata.overview is not None:
                overview = metadata.overview
            else:
                overview = Overview(image=Image(), optical_paths=[])
            if overview.image.image_coordinate_system is None:
                overview = replace(
                    overview,
                    image=WsiMetadataDicomSchema._insert_default_image_coordinate_system(
                        overview.image, ImageType.OVERVIEW
                    ),
                )
            if require_icc_profile:
                overview = self._insert_default_icc_profile(overview)
            overview_dataset = OverviewDicomSchema().dump(overview)
            label_dataset = LabelOnlyDicomSchema().dump(metadata.label)
            overview_dataset.update(base_dataset)
            overview_dataset.update(label_dataset)
            return overview_dataset
        raise ValueError(f"Unsupported image type {image_type}")

    def load(
        self,
        pyramid_dataset: Dataset,
        label_dataset: Dataset | None = None,
        overview_dataset: Dataset | None = None,
        **kwargs,
    ) -> WsiMetadata:
        metadata = BaseWsiMetadataDicomSchema().load(pyramid_dataset)
        pyramid = PyramidDicomSchema().load(pyramid_dataset)
        if label_dataset is not None:
            label = LabelDicomSchema().load(label_dataset)
        else:
            label = LabelOnlyDicomSchema().load(pyramid_dataset)
        if overview_dataset is not None:
            overview = OverviewDicomSchema().load(overview_dataset)
        else:
            overview = None

        return WsiMetadata(
            study=metadata.study,
            series=metadata.series,
            patient=metadata.patient,
            equipment=metadata.equipment,
            slide=metadata.slide,
            pyramid=pyramid,
            label=label,
            overview=overview,
            frame_of_reference_uid=metadata.frame_of_reference_uid,
            dimension_organization_uids=metadata.dimension_organization_uids,
            contributing_equipment=metadata.contributing_equipment,
        )

    @classmethod
    def _insert_default_icc_profile(cls, module: ModuleWithImage) -> ModuleWithImage:
        if module.optical_paths is None:
            return module
        if len(module.optical_paths) == 0:
            # No optical paths defined, add one with icc profile
            optical_paths = [
                OpticalPath(
                    icc_profile=cls._create_default_icc_profile(),
                    color_space="SRGB",
                )
            ]
        else:
            # Optical paths defined, add icc profile if missing
            optical_paths = [
                (
                    replace(
                        optical_path,
                        icc_profile=cls._create_default_icc_profile(),
                        color_space="SRGB",
                    )
                    if optical_path.icc_profile is None
                    else optical_path
                )
                for optical_path in module.optical_paths
            ]
        return replace(
            module,
            optical_paths=optical_paths,
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def _create_default_icc_profile() -> bytes:
        # createProfile stamps the current time into the ICC header (bytes 24:36),
        # making output non-reproducible. Zero the date and cache so the default
        # profile is byte-identical across runs and generated only once.
        profile = bytearray(
            ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        )
        profile[24:36] = bytes(12)
        return bytes(profile)

    @staticmethod
    def _insert_default_image_coordinate_system(
        image: Image, image_type: ImageType
    ) -> Image:
        return replace(
            image,
            image_coordinate_system=ImageCoordinateSystem.default_for(
                defaults.image_coordinate_system_rotation,
                image_type,
                slide_size_without_label=defaults.slide_size_without_label,
                slide_size_with_label=defaults.slide_size_with_label,
            ),
        )
