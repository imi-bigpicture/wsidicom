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

from wsidicom.metadata.equipment import Equipment
from wsidicom.metadata.image import Image
from wsidicom.metadata.schema.common import LoadingSchema
from wsidicom.metadata.schema.json.equipment import EquipmentJsonSchema
from wsidicom.metadata.schema.json.fields import UidJsonField
from wsidicom.metadata.schema.json.image import ImageJsonSchema
from wsidicom.metadata.schema.json.label import LabelJsonSchema
from wsidicom.metadata.schema.json.optical import OpticalPathJsonSchema
from wsidicom.metadata.schema.json.patient import PatientJsonSchema
from wsidicom.metadata.schema.json.series import SeriesJsonSchema
from wsidicom.metadata.schema.json.slide import SlideJsonSchema
from wsidicom.metadata.schema.json.study import StudyJsonSchema
from wsidicom.metadata.label import Label
from wsidicom.metadata.patient import Patient
from wsidicom.metadata.series import Series
from wsidicom.metadata.slide import Slide
from wsidicom.metadata.study import Study
from wsidicom.metadata.wsi import WsiMetadata


class WsiMetadataJsonSchema(LoadingSchema[WsiMetadata]):
    study = fields.Nested(StudyJsonSchema(), load_default=Study())
    series = fields.Nested(SeriesJsonSchema(), load_default=Series())
    patient = fields.Nested(PatientJsonSchema(), load_default=Patient())
    equipment = fields.Nested(EquipmentJsonSchema(), load_default=Equipment())
    optical_paths = fields.List(fields.Nested(OpticalPathJsonSchema()), load_default=[])
    slide = fields.Nested(SlideJsonSchema(), load_default=Slide())
    label = fields.Nested(LabelJsonSchema(), load_default=Label())
    image = fields.Nested(ImageJsonSchema(), load_default=Image())
    frame_of_reference_uid = UidJsonField(allow_none=True)
    dimension_organization_uids = fields.List(UidJsonField(), allow_none=True)

    @property
    def load_type(self):
        return WsiMetadata
