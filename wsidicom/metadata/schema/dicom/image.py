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

import datetime
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from marshmallow import post_load, pre_dump
from pydicom.dataset import Dataset
from pydicom.valuerep import VR

from wsidicom.codec import LossyCompressionIsoStandard
from wsidicom.geometry import SizeMm
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
    DateDicomField,
    DateTimeDicomField,
    EnumDicomField,
    FlattenOnDumpNestedDicomField,
    FloatDicomField,
    ImageOrientationSlideDicomField,
    IntegerDicomField,
    ListDicomField,
    NestedDatasetDicomField,
    OffsetInSlideCoordinateSystemDicomField,
    PixelSpacingDicomField,
    StringDicomField,
    TimeDicomField,
)
from wsidicom.metadata.schema.dicom.schema import (
    DicomSchema,
    ModuleDicomSchema,
)

logger = logging.getLogger(__name__)


class ExtendedDepthOfFieldDicomSchema(DicomSchema[ExtendedDepthOfField]):
    number_of_focal_planes = IntegerDicomField(
        data_key="NumberOfFocalPlanes", allow_none=False
    )
    distance_between_focal_planes = FloatDicomField(
        data_key="DistanceBetweenFocalPlanes", allow_none=False
    )

    @property
    def load_type(self) -> type[ExtendedDepthOfField]:
        return ExtendedDepthOfField


class ImageCoordinateSystemDicomSchema(DicomSchema[ImageCoordinateSystem | None]):
    origin = OffsetInSlideCoordinateSystemDicomField(
        data_key="TotalPixelMatrixOriginSequence", allow_none=False
    )
    rotation = ImageOrientationSlideDicomField(
        data_key="ImageOrientationSlide", allow_none=False
    )

    @property
    def load_type(self) -> type[ImageCoordinateSystem]:
        return ImageCoordinateSystem

    def load(self, data: Dataset, **kwargs) -> ImageCoordinateSystem | None:
        try:
            return super().load(data, **kwargs)
        except (TypeError, AttributeError, KeyError, IndexError):
            return None

    @post_load
    def post_load(self, data: dict[str, Any], **kwargs) -> ImageCoordinateSystem | None:
        """Post load hook to handle separation of xy and z offset."""
        origin: tuple[ImageCoordinateSystem, float] = data.pop("origin")
        return super().post_load(
            {"origin": (origin[0]), "rotation": data["rotation"], "z_offset": origin[1]}
        )

    @pre_dump
    def pre_dump(
        self, image_coordinate_system: ImageCoordinateSystem | None, **kwargs
    ) -> dict[str, Any]:
        """Pre dump hook to handle default dump value if value is None."""
        if image_coordinate_system is None:
            raise ValueError("Image coordinate system is None.")
        return {
            "origin": (
                image_coordinate_system.origin,
                image_coordinate_system.z_offset,
            ),
            "rotation": image_coordinate_system.rotation,
        }


@dataclass(frozen=True)
class PixelMeasureDicomModel:
    pixel_spacing: SizeMm | None = None
    focal_plane_spacing: float | None = None
    depth_of_field: float | None = None


class PixelMeasureDicomSchema(DicomSchema[PixelMeasureDicomModel]):
    pixel_spacing = PixelSpacingDicomField(data_key="PixelSpacing", allow_none=True)
    focal_plane_spacing = FloatDicomField(
        data_key="SpacingBetweenSlices", allow_none=True
    )
    depth_of_field = FloatDicomField(data_key="SliceThickness", allow_none=True)

    @property
    def load_type(self) -> type[PixelMeasureDicomModel]:
        return PixelMeasureDicomModel


class LossyCompressionDicomSchema:
    method = StringDicomField(VR.CS)
    ratio = FloatDicomField(value_representation=VR.DS)


class LossyCompressionsDicomSchema(DicomSchema[Sequence[LossyCompression]]):
    methods = ListDicomField(
        EnumDicomField(
            LossyCompressionIsoStandard, by_value=True, value_representation=VR.CS
        ),
        data_key="LossyImageCompressionMethod",
        dump_none_if_empty=True,
    )
    ratios = ListDicomField(
        FloatDicomField(value_representation=VR.DS),
        data_key="LossyImageCompressionRatio",
        dump_none_if_empty=True,
    )
    lossy_compressed = BooleanDicomField(
        data_key="LossyImageCompression", dump_only=True, truthy="01", falsy="00"
    )

    @property
    def load_type(self) -> type[Sequence[LossyCompression]]:
        return list

    @pre_dump
    def pre_dump(
        self, lossy_compressions: Sequence[LossyCompression], **kwargs
    ) -> dict[str, Any]:
        return {
            "methods": [compression.method for compression in lossy_compressions],
            "ratios": [compression.ratio for compression in lossy_compressions],
            "lossy_compressed": len(lossy_compressions) > 0,
        }

    @post_load
    def post_load(self, data: dict[str, Any], **kwargs) -> Sequence[LossyCompression]:
        methods = data.pop("methods", [])
        ratios = data.pop("ratios", [])
        if len(methods) != len(ratios):
            raise ValueError(
                f"Number of lossy compression methods {len(methods)} did not match "
                f"number of ratios {len(ratios)}."
            )
        return [
            LossyCompression(method, ratio)
            for method, ratio in zip(methods, ratios, strict=True)
        ]


class ImageDicomSchema(ModuleDicomSchema[Image]):
    acquisition_datetime = DateTimeDicomField(
        data_key="AcquisitionDateTime",
        default_if_none=defaults.date_time,
        load_default=None,
    )
    content_date = DateDicomField(
        data_key="ContentDate",
        load_default=None,
    )
    content_time = TimeDicomField(
        data_key="ContentTime",
        load_default=None,
    )
    focus_method = EnumDicomField(
        FocusMethod,
        data_key="FocusMethod",
        default_if_none=defaults.focus_method,
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
        allow_none=False,
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

    @property
    def load_type(self) -> type[Image]:
        return Image

    @pre_dump
    def pre_dump(self, image: Image, **kwargs):
        content_datetime = image.content_datetime
        if content_datetime is None:
            content_datetime = image.acquisition_datetime
        if content_datetime is None:
            content_datetime = defaults.date_time
        content_date, content_time = self._split_datetime(content_datetime)
        return {
            "acquisition_datetime": image.acquisition_datetime,
            "content_date": content_date,
            "content_time": content_time,
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
        }

    @staticmethod
    def _split_datetime(
        datetime_value: datetime.datetime | datetime.date | None,
    ) -> tuple[datetime.date | None, datetime.time | None]:
        """Split a datetime into the date and the time it holds.

        The time is None when the value is a date alone, and both are None when
        there is no value, so that a time that was never given is not made up.
        """
        # A datetime is a date, so it has to be the one checked for first.
        if isinstance(datetime_value, datetime.datetime):
            return datetime_value.date(), datetime_value.time()
        if isinstance(datetime_value, datetime.date):
            return datetime_value, None
        return None, None

    @staticmethod
    def _join_datetime(
        date_value: datetime.date | None, time_value: datetime.time | None
    ) -> datetime.datetime | datetime.date | None:
        """Join a date and a time into the value the two of them hold.

        Either may be missing on its own. A date with no time is kept as a date
        rather than made into a datetime at a time that was never given, and a
        time with no date is dropped, as there is no date to place it on.

        What is returned is remade as a plain `datetime.date` or
        `datetime.datetime`, so that a value read as a subclass of one of them
        is not passed on as that subclass.
        """
        if date_value is None:
            if time_value is not None:
                logger.warning(
                    "Dataset holds a time but no date. The time is dropped, as "
                    "there is no date to place it on."
                )
            return None
        if time_value is None:
            return datetime.date(date_value.year, date_value.month, date_value.day)
        return datetime.datetime.combine(date_value, time_value)

    @post_load
    def post_load(self, data: dict[str, Any], **kwargs):
        data["content_datetime"] = self._join_datetime(
            data.pop("content_date", None), data.pop("content_time", None)
        )
        extended_depth_of_field_bool = data.pop("extended_depth_of_field_bool")
        extended_depth_of_field = data.get("extended_depth_of_field")
        if (extended_depth_of_field_bool) != (extended_depth_of_field is not None):
            raise ValueError(
                (
                    f"Extended depth of field bool {extended_depth_of_field_bool} did ",
                    f"not match depth of field data {extended_depth_of_field}.",
                )
            )
        pixel_measure: PixelMeasureDicomModel | None = data.pop("pixel_measure", None)
        if pixel_measure is not None:
            data["pixel_spacing"] = pixel_measure.pixel_spacing
            data["focal_plane_spacing"] = pixel_measure.focal_plane_spacing
            data["depth_of_field"] = pixel_measure.depth_of_field

        return super().post_load(data, **kwargs)

    @property
    def module_name(self) -> str:
        return "image"
