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

"""Schemas for serializing metadata models to and from DICOM."""

from wsidicom.metadata.schema.dicom.equipment import EquipmentDicomSchema
from wsidicom.metadata.schema.dicom.image import ImageDicomSchema
from wsidicom.metadata.schema.dicom.label import LabelDicomSchema
from wsidicom.metadata.schema.dicom.optical_path import OpticalPathDicomSchema
from wsidicom.metadata.schema.dicom.patient import PatientDicomSchema
from wsidicom.metadata.schema.dicom.series import SeriesDicomSchema
from wsidicom.metadata.schema.dicom.slide import SlideDicomSchema
from wsidicom.metadata.schema.dicom.study import StudyDicomSchema
from wsidicom.metadata.schema.dicom.wsi import WsiMetadataDicomSchema

__all__ = [
    "EquipmentDicomSchema",
    "ImageDicomSchema",
    "LabelDicomSchema",
    "OpticalPathDicomSchema",
    "PatientDicomSchema",
    "SeriesDicomSchema",
    "SlideDicomSchema",
    "StudyDicomSchema",
    "WsiMetadataDicomSchema",
]
