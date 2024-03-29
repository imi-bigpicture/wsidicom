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


"""Json schema for Equipment model."""

from marshmallow import fields

from wsidicom.metadata.equipment import Equipment
from wsidicom.metadata.schema.common import LoadingSchema


class EquipmentJsonSchema(LoadingSchema[Equipment]):
    manufacturer = fields.String(allow_none=True)
    model_name = fields.String(allow_none=True)
    device_serial_number = fields.String(allow_none=True)
    software_versions = fields.List(fields.String(), allow_none=True)

    @property
    def load_type(self):
        return Equipment
