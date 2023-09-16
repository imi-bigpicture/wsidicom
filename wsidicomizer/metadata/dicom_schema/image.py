from typing import Any, Dict, Type

from pydicom import Dataset
from wsidicomizer.metadata.defaults import Defaults
from wsidicomizer.metadata.dicom_schema.base_dicom_schema import DicomSchema
from marshmallow import fields, post_dump, post_load, pre_dump, pre_load, types
from wsidicomizer.metadata.dicom_schema.dicom_fields import (
    BooleanDicomField,
    DateTimeDicomField,
    DefaultingDicomField,
    FlatteningNestedField,
    ImageOrientationSlideField,
    OffsetInSlideCoordinateSystemField,
)

from wsidicomizer.metadata.image import (
    ExtendedDepthOfField,
    FocusMethod,
    Image,
    ImageCoordinateSystem,
)


class ExtendedDepthOfFieldDicomSchema(DicomSchema):
    number_of_focal_planes = fields.Integer(
        data_key="NumberOfFocalPlanes", allow_none=False
    )
    distance_between_focal_planes = fields.Float(
        data_key="DistanceBetweenFocalPlanes", allow_none=False
    )

    @property
    def load_type(self) -> Type[ExtendedDepthOfField]:
        return ExtendedDepthOfField


class ImageCoordinateSystemDicomSchema(DicomSchema):
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


class ImageDicomSchema(DicomSchema):
    """
    Type 1
    acquisition_datetime
    focus_method
    extended_depth_of_field_bool
    image_coordinate_system
    """

    acquisition_datetime = DefaultingDicomField(
        DateTimeDicomField(),
        data_key="AcquisitionDateTime",
        dump_default=Defaults.date_time,
    )
    focus_method = DefaultingDicomField(
        fields.Enum(FocusMethod),
        data_key="FocusMethod",
        dump_default=Defaults.focus_method,
    )
    extended_depth_of_field_bool = BooleanDicomField(data_key="ExtendedDepthOfField")
    extended_depth_of_field = FlatteningNestedField(
        ExtendedDepthOfFieldDicomSchema(), allow_none=True
    )
    image_coordinate_system = FlatteningNestedField(
        ImageCoordinateSystemDicomSchema(), allow_none=True
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
        }

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs):
        extended_depth_of_field_bool = data.pop("extended_depth_of_field_bool")
        if (extended_depth_of_field_bool) != ("extended_depth_of_field" in data):
            raise ValueError(
                (
                    f"Extended depth of field bool {extended_depth_of_field_bool} did ",
                    f"not match presence of depth of field data.",
                )
            )
        return super().post_load(data, **kwargs)