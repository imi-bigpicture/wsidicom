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
from wsidicom.metadata.schema.common import LoadingSchema

from wsidicom.metadata.schema.json.fields import UidJsonField
from wsidicom.metadata.series import Series


class SeriesJsonSchema(LoadingSchema[Series]):
    uid = UidJsonField(allow_none=True)
    number = fields.Integer(allow_none=True)

    @property
    def load_type(self):
        return Series
