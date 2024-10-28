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

"""Json schema for Image model."""

from marshmallow import fields

from wsidicom.codec.encoder import LossyCompressionIsoStandard
from wsidicom.metadata.image import (
    ExtendedDepthOfField,
    FocusMethod,
    Image,
    ImageCoordinateSystem,
    LossyCompression,
)
from wsidicom.metadata.schema.common import LoadingSchema
from wsidicom.metadata.schema.json.fields import PointMmJsonField, SizeMmJsonField


class ExtendedDepthOfFieldJsonSchema(LoadingSchema[ExtendedDepthOfField]):
    number_of_focal_planes = fields.Integer()
    distance_between_focal_planes = fields.Float()

    @property
    def load_type(self):
        return ExtendedDepthOfField


class ImageCoordinateSystemJsonSchema(LoadingSchema[ImageCoordinateSystem]):
    origin = PointMmJsonField()
    rotation = fields.Float()
    z_offset = fields.Float(allow_none=True)

    @property
    def load_type(self):
        return ImageCoordinateSystem


class LossyCompressionJsonSchema(LoadingSchema[LossyCompression]):
    method = fields.Enum(LossyCompressionIsoStandard, by_value=True)
    ratio = fields.Float()

    @property
    def load_type(self):
        return LossyCompression


class ImageJsonSchema(LoadingSchema[Image]):
    acquisition_datetime = fields.DateTime(allow_none=True)
    focus_method = fields.Enum(FocusMethod, by_value=True, allow_none=True)
    extended_depth_of_field = fields.Nested(
        ExtendedDepthOfFieldJsonSchema(), allow_none=True
    )
    image_coordinate_system = fields.Nested(
        ImageCoordinateSystemJsonSchema(), allow_none=True
    )
    pixel_spacing = SizeMmJsonField(allow_none=True)
    focal_plane_spacing = fields.Float(allow_none=True)
    depth_of_field = fields.Float(allow_none=True)
    lossy_compressions = fields.List(
        fields.Nested(LossyCompressionJsonSchema()), allow_none=True
    )

    @property
    def load_type(self):
        return Image
