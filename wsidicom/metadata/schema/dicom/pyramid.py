#    Copyright 2025 SECTRA AB
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

"""DICOM schema for Pyramid model."""

from typing import Type

from pydicom.valuerep import VR

from wsidicom.metadata.image import Image
from wsidicom.metadata.optical_path import OpticalPath
from wsidicom.metadata.pyramid import Pyramid
from wsidicom.metadata.schema.dicom.fields import (
    BooleanDicomField,
    DefaultingListDicomField,
    FlattenOnDumpNestedDicomField,
    StringDicomField,
    UidDicomField,
)
from wsidicom.metadata.schema.dicom.image import ImageDicomSchema
from wsidicom.metadata.schema.dicom.optical_path import OpticalPathDicomSchema
from wsidicom.metadata.schema.dicom.schema import DicomSchema


class PyramidDicomSchema(DicomSchema[Pyramid]):
    image = FlattenOnDumpNestedDicomField(
        ImageDicomSchema(), dump_default=Image(), load_default=Image()
    )
    optical_paths = DefaultingListDicomField(
        FlattenOnDumpNestedDicomField(OpticalPathDicomSchema()),
        data_key="OpticalPathSequence",
        dump_default=[OpticalPath()],
        load_default=[],
    )
    uid = UidDicomField(data_key="PyramidUID", allow_none=True)
    description = StringDicomField(
        value_representation=VR.LO, data_key="PyramidDescription", allow_none=True
    )
    label = StringDicomField(
        value_representation=VR.LO, data_key="PyramidLabel", allow_none=True
    )
    contains_phi = BooleanDicomField(data_key="BurnedInAnnotation", allow_none=True)
    comments = StringDicomField(
        value_representation=VR.LO, data_key="ImageComments", allow_none=True
    )
    specimen_label_in_image = StringDicomField(
        VR.CS, data_key="SpecimenLabelInImage", dump_only=True, dump_default="NO"
    )

    @property
    def load_type(self) -> Type[Pyramid]:
        return Pyramid
