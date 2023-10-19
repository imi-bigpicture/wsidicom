from marshmallow import Schema, fields, post_load

from wsidicom.metadata.json_schema.equipment import EquipmentJsonSchema
from wsidicom.metadata.json_schema.fields import UidJsonField
from wsidicom.metadata.json_schema.image import ImageJsonSchema
from wsidicom.metadata.json_schema.label import LabelJsonSchema
from wsidicom.metadata.json_schema.optical import OpticalPathJsonSchema
from wsidicom.metadata.json_schema.patient import PatientJsonSchema
from wsidicom.metadata.json_schema.series import SeriesJsonSchema
from wsidicom.metadata.json_schema.slide import SlideJsonSchema
from wsidicom.metadata.json_schema.study import StudyJsonSchema
from wsidicom.metadata.wsi import WsiMetadata


class WsiMetadataJsonSchema(Schema):
    study = fields.Nested(StudyJsonSchema(), allow_none=True)
    series = fields.Nested(SeriesJsonSchema(), allow_none=True)
    patient = fields.Nested(PatientJsonSchema(), allow_none=True)
    equipment = fields.Nested(EquipmentJsonSchema(), allow_none=True)
    optical_paths = fields.List(fields.Nested(OpticalPathJsonSchema()))
    slide = fields.Nested(SlideJsonSchema(), allow_none=True)
    label = fields.Nested(LabelJsonSchema(), allow_none=True)
    image = fields.Nested(ImageJsonSchema(), allow_none=True)
    frame_of_reference_uid = UidJsonField(allow_none=True)
    dimension_organization_uids = UidJsonField(allow_none=True)

    @post_load
    def load_to_object(self, data, **kwargs):
        return WsiMetadata(**data)
