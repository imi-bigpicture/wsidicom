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

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Type, TypeVar

from marshmallow import fields
from PIL import ImageCms
from pydicom import Dataset
from pydicom.uid import UID, VLWholeSlideMicroscopyImageStorage

from wsidicom.geometry import PointMm
from wsidicom.instance.dataset import ImageType
from wsidicom.metadata.equipment import Equipment
from wsidicom.metadata.image import Image, ImageCoordinateSystem
from wsidicom.metadata.label import Label
from wsidicom.metadata.optical_path import OpticalPath
from wsidicom.metadata.overview import Overview
from wsidicom.metadata.patient import Patient
from wsidicom.metadata.pyramid import Pyramid
from wsidicom.metadata.schema.dicom.defaults import defaults
from wsidicom.metadata.schema.dicom.equipment import EquipmentDicomSchema
from wsidicom.metadata.schema.dicom.fields import (
    FlattenOnDumpNestedDicomField,
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
    frame_of_reference_uid: Optional[UID] = None
    dimension_organization_uids: Optional[Sequence[UID]] = None


class BaseWsiMetadataDicomSchema(DicomSchema[BaseWsiMetadata]):
    """Schema to load and dump BaseWsiMetadata to and from DICOM dataset."""

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
    slide = FlattenOnDumpNestedDicomField(
        SlideDicomSchema(), dump_default=Slide(), load_default=Slide()
    )
    frame_of_reference_uid = UidDicomField(
        data_key="FrameOfReferenceUID", load_default=None
    )
    dimension_organization_uids = fields.List(
        UidDatasetDicomField(data_key="DimensionOrganizationUID"),
        data_key="DimensionOrganizationSequence",
        load_default=None,
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
    def load_type(self) -> Type[BaseWsiMetadata]:
        return BaseWsiMetadata


ModuleWithImage = TypeVar("ModuleWithImage", Pyramid, Label, Overview)


class WsiMetadataDicomSchema:
    """Schema to load and dump WsiMetadata to and from DICOM dataset."""

    def dump(
        self,
        metadata: WsiMetadata,
        image_type: ImageType,
        require_icc_profile,
        **kwargs,
    ) -> Dataset:
        base_dataset = BaseWsiMetadataDicomSchema().dump(
            BaseWsiMetadata(
                study=metadata.study,
                series=metadata.series,
                patient=metadata.patient,
                equipment=metadata.equipment,
                slide=metadata.slide,
                frame_of_reference_uid=(
                    metadata.frame_of_reference_uid
                    if metadata.frame_of_reference_uid
                    else metadata.default_frame_of_reference_uid
                ),
                dimension_organization_uids=(
                    metadata.dimension_organization_uids
                    if metadata.dimension_organization_uids
                    else metadata.default_dimension_organization_uids
                ),
            )
        )
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
        label_dataset: Optional[Dataset] = None,
        overview_dataset: Optional[Dataset] = None,
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
        )

    @classmethod
    def _insert_default_icc_profile(cls, module: ModuleWithImage) -> ModuleWithImage:
        if module.optical_paths is None:
            return module
        if len(module.optical_paths) == 0:
            # No optical paths defined, add one with icc profile
            optical_paths = [OpticalPath(icc_profile=cls._create_default_icc_profile())]
        else:
            # Optical paths defined, add icc profile if missing
            optical_paths = [
                (
                    replace(
                        optical_path,
                        icc_profile=cls._create_default_icc_profile(),
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
    def _create_default_icc_profile() -> bytes:
        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    @staticmethod
    def _insert_default_image_coordinate_system(
        image: Image, image_type: ImageType
    ) -> Image:
        if image_type == ImageType.VOLUME:
            default_size = defaults.slide_size_without_label
        else:
            default_size = defaults.slide_size_with_label
        if defaults.image_coordinate_system_rotation == 0:
            x = 0
            y = 0
        elif defaults.image_coordinate_system_rotation == 90:
            x = default_size.width
            y = 0
        elif defaults.image_coordinate_system_rotation == 180:
            x = default_size.width
            y = default_size.height
        elif defaults.image_coordinate_system_rotation == 270:
            x = 0
            y = default_size.height
        else:
            raise ValueError(
                f"Unsupported default image coordinate system rotation {defaults.image_coordinate_system_rotation}"
            )
        return replace(
            image,
            image_coordinate_system=ImageCoordinateSystem(
                origin=PointMm(x, y),
                rotation=defaults.image_coordinate_system_rotation,
                z_offset=None,
            ),
        )
