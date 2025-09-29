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

"""DICOM schema for Label model."""

from typing import Any, Dict, Type

from marshmallow import post_dump, pre_dump
from pydicom.valuerep import VR

from wsidicom.metadata.image import Image
from wsidicom.metadata.label import Label
from wsidicom.metadata.optical_path import OpticalPath
from wsidicom.metadata.schema.dicom.fields import (
    BooleanDicomField,
    DefaultingDicomField,
    DefaultingListDicomField,
    FlattenOnDumpNestedDicomField,
    StringDicomField,
)
from wsidicom.metadata.schema.dicom.image import (
    ImageDicomSchema,
)
from wsidicom.metadata.schema.dicom.optical_path import OpticalPathDicomSchema
from wsidicom.metadata.schema.dicom.schema import ModuleDicomSchema


class LabelBaseDicomSchema(ModuleDicomSchema[Label]):
    text = DefaultingDicomField(
        StringDicomField(VR.UT),
        data_key="LabelText",
        allow_none=True,
        dump_default=None,
    )
    barcode = DefaultingDicomField(
        StringDicomField(VR.LO),
        data_key="BarcodeValue",
        allow_none=True,
        dump_default=None,
    )

    @property
    def load_type(self) -> Type[Label]:
        return Label

    @property
    def module_name(self) -> str:
        return "label"


class LabelOnlyDicomSchema(LabelBaseDicomSchema):
    """Schema to use for loading label metadata from non-label instances."""

    @pre_dump
    def pre_dump(self, label: Label, **kwargs):
        return {
            "text": label.text,
            "barcode": label.barcode,
        }

    @post_dump
    def post_dump(self, data: Dict[str, Any], **kwargs):
        # Remove text and barcode if empty
        if data["LabelText"] is None:
            data.pop("LabelText")
        if data["BarcodeValue"] is None:
            data.pop("BarcodeValue")
        return super().post_dump(data, **kwargs)


class LabelDicomSchema(LabelBaseDicomSchema):
    """Schema to use for label instance."""

    image = FlattenOnDumpNestedDicomField(
        ImageDicomSchema(), dump_default=Image(), load_default=Image()
    )
    optical_paths = DefaultingListDicomField(
        FlattenOnDumpNestedDicomField(OpticalPathDicomSchema()),
        data_key="OpticalPathSequence",
        dump_default=[OpticalPath()],
        load_default=[],
    )
    contains_phi = BooleanDicomField(data_key="BurnedInAnnotation", allow_none=True)
    comments = StringDicomField(
        value_representation=VR.LO, data_key="ImageComments", allow_none=True
    )
    specimen_label_in_image = StringDicomField(
        VR.CS, data_key="SpecimenLabelInImage", dump_only=True, dump_default="YES"
    )
