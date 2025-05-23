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
from typing import Any, Dict, Optional, Sequence, Tuple, Type

from marshmallow import fields, post_load, pre_dump
from pydicom.dataset import Dataset
from pydicom.valuerep import VR

from wsidicom.codec import LossyCompressionIsoStandard
from wsidicom.geometry import PointMm, SizeMm
from wsidicom.metadata.image import (
    ExtendedDepthOfField,
    FocusMethod,
    Image,
    ImageCoordinateSystem,
    LossyCompression,
)
from wsidicom.metadata.schema.dicom.defaults import defaults
from wsidicom.metadata.schema.dicom.fields import (
    BooleanDicomField,
    DateTimeDicomField,
    DefaultingDicomField,
    EnumDicomField,
    FlattenOnDumpNestedDicomField,
    FloatDicomField,
    ImageOrientationSlideField,
    ListDicomField,
    NestedDatasetDicomField,
    OffsetInSlideCoordinateSystemField,
    PixelSpacingDicomField,
    StringDicomField,
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


class ImageCoordinateSystemDicomSchema(DicomSchema[Optional[ImageCoordinateSystem]]):
    origin = OffsetInSlideCoordinateSystemField(
        data_key="TotalPixelMatrixOriginSequence", allow_none=False
    )
    rotation = ImageOrientationSlideField(
        data_key="ImageOrientationSlide", allow_none=False
    )

    @property
    def load_type(self) -> Type[ImageCoordinateSystem]:
        return ImageCoordinateSystem

    def load(self, dataset: Dataset, **kwargs) -> Optional[ImageCoordinateSystem]:
        try:
            return super().load(dataset, **kwargs)
        except (TypeError, AttributeError, KeyError, IndexError):
            return None

    @post_load
    def post_load(
        self, data: Dict[str, Any], **kwargs
    ) -> Optional[ImageCoordinateSystem]:
        """Post load hook to handle separation of xy and z offset."""
        origin: Tuple[ImageCoordinateSystem, float] = data.pop("origin")
        return super().post_load(
            {"origin": (origin[0]), "rotation": data["rotation"], "z_offset": origin[1]}
        )

    @pre_dump
    def pre_dump(
        self, image_coordinate_system: Optional[ImageCoordinateSystem], **kwargs
    ) -> Dict[str, Any]:
        """Pre dump hook to handle default dump value if value is None."""
        if image_coordinate_system is None:
            image_size = self.context.get("image_size", None)
            if image_size is not None:
                # Based on image size place image in the middle of the slide
                assert isinstance(image_size, SizeMm)
                image_coordinate_system = ImageCoordinateSystem.from_middle_of_slide(
                    slide_middle=PointMm(
                        defaults.slide_size_without_label.width / 2,
                        defaults.slide_size_without_label.height / 2,
                    ),
                    image_size=image_size,
                    rotation=defaults.image_coordinate_system_rotation,
                    z_offset=None,
                )
            else:
                # Place the image with the origin right below the the label.
                image_coordinate_system = ImageCoordinateSystem(
                    origin=PointMm(
                        defaults.slide_size_without_label.width,
                        defaults.slide_size_without_label.height,
                    ),
                    rotation=180,
                )
        return {
            "origin": (
                image_coordinate_system.origin,
                image_coordinate_system.z_offset,
            ),
            "rotation": image_coordinate_system.rotation,
        }


@dataclass(frozen=True)
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


class LossyCompressionDicomSchema:
    method = StringDicomField(VR.CS)
    ratio = FloatDicomField()


class LossyCompressionsDicomSchema(DicomSchema[Sequence[LossyCompression]]):
    methods = ListDicomField(
        EnumDicomField(LossyCompressionIsoStandard, by_value=True),
        data_key="LossyImageCompressionMethod",
        dump_none_if_empty=True,
    )
    ratios = ListDicomField(
        FloatDicomField(),
        data_key="LossyImageCompressionRatio",
        dump_none_if_empty=True,
    )
    lossy_compressed = BooleanDicomField(
        data_key="LossyImageCompression", dump_only=True, truthy="01", falsy="00"
    )

    @property
    def load_type(self) -> Type[Sequence[LossyCompression]]:
        return list

    @pre_dump
    def pre_dump(
        self, lossy_compressions: Sequence[LossyCompression], **kwargs
    ) -> Dict[str, Any]:
        return {
            "methods": [compression.method for compression in lossy_compressions],
            "ratios": [compression.ratio for compression in lossy_compressions],
            "lossy_compressed": len(lossy_compressions) > 0,
        }

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs) -> Sequence[LossyCompression]:
        methods = data.pop("methods", [])
        ratios = data.pop("ratios", [])
        if len(methods) != len(ratios):
            raise ValueError(
                (
                    f"Number of lossy compression methods {len(methods)} did not match "
                    f"number of ratios {len(ratios)}."
                )
            )
        return [
            LossyCompression(method, ratio) for method, ratio in zip(methods, ratios)
        ]


class ImageDicomSchema(ModuleDicomSchema[Image]):
    acquisition_datetime = DefaultingDicomField(
        DateTimeDicomField(),
        data_key="AcquisitionDateTime",
        dump_default=defaults.date_time,
        load_default=None,
    )
    focus_method = DefaultingDicomField(
        fields.Enum(FocusMethod),
        data_key="FocusMethod",
        dump_default=defaults.focus_method,
        load_default=None,
    )
    extended_depth_of_field_bool = BooleanDicomField(
        data_key="ExtendedDepthOfField", load_default=False
    )
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
    lossy_compressions = FlattenOnDumpNestedDicomField(
        LossyCompressionsDicomSchema(),
        allow_none=True,
        load_default=None,
    )
    comments = StringDicomField(
        value_representation=VR.LO, data_key="ImageComments", allow_none=True
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
            "lossy_compressions": (
                image.lossy_compressions if image.lossy_compressions else []
            ),
            "comments": image.comments,
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
