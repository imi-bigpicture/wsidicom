#    Copyright 2026 SECTRA AB
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

"""Json schema for ContributingEquipment model."""

from marshmallow import fields

from wsidicom.conceptcode import ContributingEquipmentPurposeCode
from wsidicom.metadata.contributing_equipment import ContributingEquipment
from wsidicom.metadata.schema.common import LoadingSchema
from wsidicom.metadata.schema.json.fields import JsonFieldFactory


class ContributingEquipmentJsonSchema(LoadingSchema[ContributingEquipment]):
    purpose = JsonFieldFactory.concept_code(ContributingEquipmentPurposeCode)()
    manufacturer = fields.String(allow_none=True)
    model_name = fields.String(allow_none=True)
    software_versions = fields.List(fields.String(), allow_none=True)
    description = fields.String(allow_none=True)
    contribution_datetime = fields.DateTime(allow_none=True)

    @property
    def load_type(self):
        return ContributingEquipment
