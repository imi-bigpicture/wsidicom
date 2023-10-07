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

from marshmallow import Schema, fields, post_load

from wsidicom.metadata.image import (
    ExtendedDepthOfField,
    FocusMethod,
    Image,
    ImageCoordinateSystem,
)
from wsidicom.metadata.json_schema.fields import PointMmJsonField


class ExtendedDepthOfFieldJsonSchema(Schema):
    number_of_focal_planes = fields.Integer()
    distance_between_focal_planes = fields.Float()

    @post_load
    def load_to_object(self, data, **kwargs):
        return ExtendedDepthOfField(**data)


class ImageCoordinateSystemJsonSchema(Schema):
    origin = PointMmJsonField()
    rotation = fields.Float()

    @post_load
    def load_to_object(self, data, **kwargs):
        return ImageCoordinateSystem(**data)


class ImageJsonSchema(Schema):
    acquisition_datetime = fields.DateTime(allow_none=True)
    focus_method = fields.Enum(FocusMethod, by_value=True, allow_none=True)
    extended_depth_of_field = fields.Nested(
        ExtendedDepthOfFieldJsonSchema(), allow_none=True
    )
    image_coordinate_system = fields.Nested(
        ImageCoordinateSystemJsonSchema(), allow_none=True
    )

    @post_load
    def load_to_object(self, data, **kwargs):
        return Image(**data)