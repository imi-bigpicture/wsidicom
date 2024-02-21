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

"""Schemas for serializing metadata models to and from json."""

from wsidicom.metadata.schema.json.equipment import EquipmentJsonSchema
from wsidicom.metadata.schema.json.image import ImageJsonSchema
from wsidicom.metadata.schema.json.label import LabelJsonSchema
from wsidicom.metadata.schema.json.optical_path import OpticalPathJsonSchema
from wsidicom.metadata.schema.json.patient import PatientJsonSchema
from wsidicom.metadata.schema.json.series import SeriesJsonSchema
from wsidicom.metadata.schema.json.slide import SlideJsonSchema
from wsidicom.metadata.schema.json.study import StudyJsonSchema
from wsidicom.metadata.schema.json.wsi import WsiMetadataJsonSchema

__all__ = [
    "EquipmentJsonSchema",
    "ImageJsonSchema",
    "LabelJsonSchema",
    "OpticalPathJsonSchema",
    "PatientJsonSchema",
    "SeriesJsonSchema",
    "SlideJsonSchema",
    "StudyJsonSchema",
    "WsiMetadataJsonSchema",
]
