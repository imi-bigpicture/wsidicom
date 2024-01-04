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

"""DICOM schema for Image model."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from marshmallow import fields, post_load, pre_dump

from wsidicom.geometry import SizeMm
from wsidicom.metadata.image import (
    ExtendedDepthOfField,
    FocusMethod,
    Image,
    ImageCoordinateSystem,
)
from wsidicom.metadata.schema.dicom.defaults import Defaults
from wsidicom.metadata.schema.dicom.fields import (
    BooleanDicomField,
    DateTimeDicomField,
    DefaultingDicomField,
    FlattenOnDumpNestedDicomField,
    FloatDicomField,
    ImageOrientationSlideField,
    NestedDatasetDicomField,
    OffsetInSlideCoordinateSystemField,
    PixelSpacingDicomField,
)
from wsidicom.metadata.schema.dicom.schema import (
    DicomSchema,
    ModuleDicomSchema,
)


class ExtendedDepthOfFieldDicomSchema(DicomSchema[ExtendedDepthOfField]):
    number_of_focal_planes = fields.Integer(
        data_key="NumberOfFocalPlanes", allow_none=False
    )
    distance_between_focal_planes = fields.Float(
        data_key="DistanceBetweenFocalPlanes", allow_none=False
    )

    @property
    def load_type(self) -> Type[ExtendedDepthOfField]:
        return ExtendedDepthOfField


class ImageCoordinateSystemDicomSchema(DicomSchema[ImageCoordinateSystem]):
    origin = OffsetInSlideCoordinateSystemField(
        data_key="TotalPixelMatrixOriginSequence",
        allow_none=False,
        dump_default=Defaults.image_coordinate_system_origin,
    )
    rotation = ImageOrientationSlideField(
        data_key="ImageOrientationSlide",
        allow_none=False,
        dump_default=Defaults.image_coordinate_system_rotation,
    )

    @property
    def load_type(self) -> Type[ImageCoordinateSystem]:
        return ImageCoordinateSystem


@dataclass
class PixelMeasureDicomModel:
    pixel_spacing: Optional[SizeMm] = None
    focal_plane_spacing: Optional[float] = None
    depth_of_field: Optional[float] = None


class PixelMeasureDicomSchema(DicomSchema[PixelMeasureDicomModel]):
    pixel_spacing = PixelSpacingDicomField(data_key="PixelSpacing", allow_none=True)
    focal_plane_spacing = FloatDicomField(
        data_key="SpacingBetweenSlices", allow_none=True
    )
    depth_of_field = FloatDicomField(data_key="SliceThickness", allow_none=True)

    @property
    def load_type(self) -> Type[PixelMeasureDicomModel]:
        return PixelMeasureDicomModel


class ImageDicomSchema(ModuleDicomSchema[Image]):
    acquisition_datetime = DefaultingDicomField(
        DateTimeDicomField(),
        data_key="AcquisitionDateTime",
        dump_default=Defaults.date_time,
        load_default=None,
    )
    focus_method = DefaultingDicomField(
        fields.Enum(FocusMethod),
        data_key="FocusMethod",
        dump_default=Defaults.focus_method,
        load_default=None,
    )
    extended_depth_of_field_bool = BooleanDicomField(data_key="ExtendedDepthOfField")
    extended_depth_of_field = FlattenOnDumpNestedDicomField(
        ExtendedDepthOfFieldDicomSchema(),
        allow_none=True,
        load_default=None,
    )
    image_coordinate_system = FlattenOnDumpNestedDicomField(
        ImageCoordinateSystemDicomSchema(),
        allow_none=True,
        load_default=None,
    )
    pixel_measure = NestedDatasetDicomField(
        PixelMeasureDicomSchema(),
        data_key="SharedFunctionalGroupsSequence",
        nested_data_key="PixelMeasuresSequence",
    )

    @property
    def load_type(self) -> Type[Image]:
        return Image

    @pre_dump
    def pre_dump(self, image: Image, **kwargs):
        return {
            "acquisition_datetime": image.acquisition_datetime,
            "focus_method": image.focus_method,
            "extended_depth_of_field_bool": image.extended_depth_of_field is not None,
            "extended_depth_of_field": image.extended_depth_of_field,
            "image_coordinate_system": image.image_coordinate_system,
            "pixel_measure": PixelMeasureDicomModel(
                pixel_spacing=image.pixel_spacing,
                focal_plane_spacing=image.focal_plane_spacing,
                depth_of_field=image.depth_of_field,
            ),
        }

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs):
        extended_depth_of_field_bool = data.pop("extended_depth_of_field_bool")
        extended_depth_of_field = data.get("extended_depth_of_field", None)
        if (extended_depth_of_field_bool) != (extended_depth_of_field is not None):
            raise ValueError(
                (
                    f"Extended depth of field bool {extended_depth_of_field_bool} did ",
                    f"not match depth of field data {extended_depth_of_field}.",
                )
            )
        pixel_measure: Optional[PixelMeasureDicomModel] = data.pop(
            "pixel_measure", None
        )
        if pixel_measure is not None:
            data["pixel_spacing"] = pixel_measure.pixel_spacing
            data["focal_plane_spacing"] = pixel_measure.focal_plane_spacing
            data["depth_of_field"] = pixel_measure.depth_of_field
        return super().post_load(data, **kwargs)

    @property
    def module_name(self) -> str:
        return "image"
