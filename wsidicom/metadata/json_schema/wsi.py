from marshmallow import Schema, fields, post_load

from wsidicom.metadata.equipment import Equipment
from wsidicom.metadata.image import Image
from wsidicom.metadata.json_schema.equipment import EquipmentJsonSchema
from wsidicom.metadata.json_schema.fields import UidJsonField
from wsidicom.metadata.json_schema.image import ImageJsonSchema
from wsidicom.metadata.json_schema.label import LabelJsonSchema
from wsidicom.metadata.json_schema.optical import OpticalPathJsonSchema
from wsidicom.metadata.json_schema.patient import PatientJsonSchema
from wsidicom.metadata.json_schema.series import SeriesJsonSchema
from wsidicom.metadata.json_schema.slide import SlideJsonSchema
from wsidicom.metadata.json_schema.study import StudyJsonSchema
from wsidicom.metadata.label import Label
from wsidicom.metadata.patient import Patient
from wsidicom.metadata.series import Series
from wsidicom.metadata.slide import Slide
from wsidicom.metadata.study import Study
from wsidicom.metadata.wsi import WsiMetadata


class WsiMetadataJsonSchema(Schema):
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

    @post_load
    def load_to_object(self, data, **kwargs):
        return WsiMetadata(**data)
