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

from marshmallow import fields, post_dump, post_load, pre_dump
from pydicom.valuerep import VR

from wsidicom.instance import ImageType
from wsidicom.metadata.label import Label
from wsidicom.metadata.schema.dicom.fields import (
    BooleanDicomField,
    DefaultingDicomField,
    StringDicomField,
)
from wsidicom.metadata.schema.dicom.schema import ModuleDicomSchema


class LabelDicomSchema(ModuleDicomSchema[Label]):
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

    label_in_volume_image = BooleanDicomField(load_only=True, allow_none=True)
    label_in_overview_image = BooleanDicomField(load_only=True, allow_none=True)
    label_is_phi = BooleanDicomField(load_only=True, allow_none=True)
    burned_in_annotation = BooleanDicomField(
        data_key="BurnedInAnnotation", load_default=False
    )
    specimen_label_in_image = BooleanDicomField(
        data_key="SpecimenLabelInImage", load_default=False
    )
    image_type = fields.List(
        StringDicomField(VR.CS), load_only=True, data_key="ImageType"
    )

    @property
    def load_type(self) -> Type[Label]:
        return Label

    @pre_dump
    def pre_dump(self, label: Label, **kwargs):
        image_type = self.context.get("image_type", None)
        label_in_image = False
        contains_phi = False
        if (
            (image_type == ImageType.VOLUME and label.label_in_volume_image)
            or (image_type == ImageType.OVERVIEW and label.label_in_overview_image)
            or image_type == ImageType.LABEL
        ):
            label_in_image = "True"
            contains_phi = label.label_is_phi
        attributes = {
            "text": label.text,
            "barcode": label.barcode,
            "burned_in_annotation": contains_phi,
            "specimen_label_in_image": label_in_image,
        }

        return attributes

    @post_dump
    def post_dump(self, data: Dict[str, Any], **kwargs):
        # Remove text and barcode if empty and not label image type
        image_type = self.context.get("image_type", None)
        if image_type != ImageType.LABEL:
            if data["LabelText"] is None:
                data.pop("LabelText")
            if data["BarcodeValue"] is None:
                data.pop("BarcodeValue")

        return super().post_dump(data, **kwargs)

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs):
        image_type = ImageType(data.pop("image_type")[2])
        burned_in_annotation = data.pop("burned_in_annotation")
        specimen_label_in_image = data.pop("specimen_label_in_image")
        label_is_phi = False
        label_in_volume_image = False
        label_in_overview_image = False
        if image_type == ImageType.LABEL:
            label_is_phi = burned_in_annotation
        elif image_type == ImageType.VOLUME and specimen_label_in_image:
            label_is_phi = burned_in_annotation
            label_in_volume_image = True
        elif image_type == ImageType.OVERVIEW and specimen_label_in_image:
            label_is_phi = burned_in_annotation
            label_in_overview_image = True
        data["label_is_phi"] = label_is_phi
        data["label_in_volume_image"] = label_in_volume_image
        data["label_in_overview_image"] = label_in_overview_image
        return super().post_load(data, **kwargs)

    @property
    def module_name(self) -> str:
        return "label"
