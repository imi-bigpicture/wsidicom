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

"""Json schema for Optical path model."""

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

from marshmallow import Schema, fields, post_load

from wsidicom.conceptcode import (
    IlluminationCode,
    IlluminationColorCode,
    ImagePathFilterCode,
    LenseCode,
    LightPathFilterCode,
)
from wsidicom.metadata.optical_path import (
    ConstantLutSegment,
    DiscreteLutSegment,
    ImagePathFilter,
    LightPathFilter,
    LinearLutSegment,
    Lut,
    LutSegment,
    Objectives,
    OpticalPath,
)
from wsidicom.metadata.schema.common import LoadingSchema
from wsidicom.metadata.schema.json.fields import (
    FileLoadingField,
    JsonFieldFactory,
    NpUIntDTypeField,
)


class BaseLutSegmentJsonSchema(Schema):
    _load_class: type[LinearLutSegment | ConstantLutSegment | DiscreteLutSegment]

    @post_load
    def post_load(self, data: dict[str, Any], **kwargs) -> LutSegment:
        """Return a object of given load class using the defined dataclass fields."""
        return self._load_class(
            **{
                field.name: data[field.name]
                for field in dataclasses.fields(self._load_class)
                if field.name in data
            }
        )


class DiscreteLutSegmentJsonSchema(BaseLutSegmentJsonSchema):
    values = fields.List(fields.Integer())
    _load_class = DiscreteLutSegment


class LinearLutSegmentJsonSchema(BaseLutSegmentJsonSchema):
    start_value = fields.Integer()
    end_value = fields.Integer()
    length = fields.Integer()
    _load_class = LinearLutSegment


class ConstantLutSegmentJsonSchema(BaseLutSegmentJsonSchema):
    value = fields.Integer()
    length = fields.Integer()
    _load_class = ConstantLutSegment


class LutSegmentJsonSchema(Schema):
    """Mapping segment type to schema."""

    _type_to_schema_mapping: dict[type[LutSegment], type[Schema]] = {
        DiscreteLutSegment: DiscreteLutSegmentJsonSchema,
        LinearLutSegment: LinearLutSegmentJsonSchema,
        ConstantLutSegment: ConstantLutSegmentJsonSchema,
    }

    """Mapping key in serialized segment to schema."""
    _key_to_schema_mapping: dict[str, type[Schema]] = {
        "values": DiscreteLutSegmentJsonSchema,
        "start_value": LinearLutSegmentJsonSchema,
        "value": ConstantLutSegmentJsonSchema,
    }

    def dump(
        self,
        obj: LutSegment | Iterable[LutSegment],
        **kwargs,
    ):
        if isinstance(obj, LutSegment):
            return self._subschema_dump(obj)
        return [self._subschema_dump(item) for item in obj]

    def load(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        data: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        **kwargs,
    ):
        if isinstance(data, Mapping):
            return self._subschema_load(data)
        return [self._subschema_load(step) for step in data]

    def _subschema_load(self, segment: Mapping) -> LutSegment:
        """Select a schema and load and return step using the schema."""
        try:
            schema = next(
                schema
                for key, schema in self._key_to_schema_mapping.items()
                if key in segment
            )
        except StopIteration:
            raise NotImplementedError() from None
        loaded = schema().load(segment, many=False)
        assert isinstance(loaded, LutSegment)
        return loaded

    def _subschema_dump(self, segment: LutSegment):
        """Select a schema and dump the step using the schema."""
        schema = self._type_to_schema_mapping[type(segment)]
        return schema().dump(segment, many=False)


class LutJsonSchema(LoadingSchema[Lut]):
    red = fields.List(fields.Nested(LutSegmentJsonSchema()))
    green = fields.List(fields.Nested(LutSegmentJsonSchema()))
    blue = fields.List(fields.Nested(LutSegmentJsonSchema()))
    data_type = NpUIntDTypeField(data_key="bits")

    @property
    def load_type(self):
        return Lut


class LightPathFilterJsonSchema(LoadingSchema[LightPathFilter]):
    filters = fields.List(JsonFieldFactory.concept_code(LightPathFilterCode)())
    nominal = fields.Float(allow_none=True)
    low_pass = fields.Float(allow_none=True)
    high_pass = fields.Float(allow_none=True)

    @property
    def load_type(self):
        return LightPathFilter


class ImagePathFilterJsonSchema(LoadingSchema[ImagePathFilter]):
    filters = fields.List(JsonFieldFactory.concept_code(ImagePathFilterCode)())
    nominal = fields.Float(allow_none=True)
    low_pass = fields.Float(allow_none=True)
    high_pass = fields.Float(allow_none=True)

    @property
    def load_type(self):
        return ImagePathFilter


class ObjectivesJsonSchema(LoadingSchema[Objectives]):
    """Set of lens conditions for optical path"""

    lenses = fields.List(JsonFieldFactory.concept_code(LenseCode), allow_none=True)
    condenser_power = fields.Float(allow_none=True)
    objective_power = fields.Float(allow_none=True)
    objective_numerical_aperture = fields.Float(allow_none=True)

    @property
    def load_type(self):
        return Objectives


class OpticalPathJsonSchema(LoadingSchema[OpticalPath]):
    """Optical path.

    Icc profile is not included but is loaded via the icc_profile field.
    """

    identifier = fields.String(allow_none=True)
    description = fields.String(allow_none=True)
    illumination_types = fields.List(
        JsonFieldFactory.concept_code(IlluminationCode)(), allow_none=True
    )
    illumination = JsonFieldFactory.float_or_concept_code(IlluminationColorCode)(
        allow_none=True
    )
    lut = fields.Nested(LutJsonSchema(), allow_none=True)
    icc_profile = FileLoadingField(load_only=True, load_default=None)
    color_space = fields.String(allow_none=True)
    light_path_filter = fields.Nested(LightPathFilterJsonSchema(), allow_none=True)
    image_path_filter = fields.Nested(ImagePathFilterJsonSchema(), allow_none=True)
    objective = fields.Nested(ObjectivesJsonSchema(), allow_none=True)

    @property
    def load_type(self) -> type[OpticalPath]:
        return OpticalPath
