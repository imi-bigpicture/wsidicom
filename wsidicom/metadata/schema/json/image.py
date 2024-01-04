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

from marshmallow import fields

from wsidicom.metadata.image import (
    ExtendedDepthOfField,
    FocusMethod,
    Image,
    ImageCoordinateSystem,
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

    @property
    def load_type(self):
        return ImageCoordinateSystem


class ImageJsonSchema(LoadingSchema[Image]):
    acquisition_datetime = fields.DateTime(allow_none=True)
    focus_method = fields.Enum(FocusMethod, by_value=True, allow_none=True)
    extended_depth_of_field = fields.Nested(
        ExtendedDepthOfFieldJsonSchema(), allow_none=True
    )
    image_coordinate_system = fields.Nested(
        ImageCoordinateSystemJsonSchema(), allow_none=True
    )
    pixel_spacing = SizeMmJsonField()
    focal_plane_spacing = fields.Float()
    depth_of_field = fields.Float()

    @property
    def load_type(self):
        return Image
