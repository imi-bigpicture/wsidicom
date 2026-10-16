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

"""Json schema for Label model."""

from marshmallow import fields

from wsidicom.metadata.label import Label
from wsidicom.metadata.schema.common import LoadingSchema
from wsidicom.metadata.schema.json.image import ImageJsonSchema
from wsidicom.metadata.schema.json.optical_path import OpticalPathJsonSchema


class LabelJsonSchema(LoadingSchema[Label]):
    text = fields.String(allow_none=True)
    barcode = fields.String(allow_none=True)
    image = fields.Nested(ImageJsonSchema(), load_default=None)
    optical_paths = fields.List(
        fields.Nested(OpticalPathJsonSchema()), load_default=None
    )
    contains_phi = fields.Boolean(allow_none=True)
    comments = fields.String(allow_none=True)

    @property
    def load_type(self) -> type[Label]:
        return Label
