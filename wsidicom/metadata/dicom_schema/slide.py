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

from typing import Any, Dict, Type

from marshmallow import fields, post_dump, pre_dump, pre_load
from pydicom import Dataset
from wsidicom.conceptcode import ContainerTypeCode
from wsidicom.metadata.defaults import defaults

from wsidicom.metadata.defaults import Defaults
from wsidicom.metadata.dicom_schema.schema import DicomSchema
from wsidicom.metadata.dicom_schema.fields import (
    DefaultingDicomField,
    SingleCodeDicomField,
    StringDicomField,
)
from wsidicom.metadata.dicom_schema.sample import SlideSampleDicom
from wsidicom.metadata.sample import SlideSample
from wsidicom.metadata.slide import Slide


class SlideDicomSchema(DicomSchema[Slide]):
    """
    IssuerOfTheContainerIdentifierSequence
    ContainerComponentSequence:
        ContainerComponentTypeCodeSequence
        ContainerComponentMaterial
    """

    identifier = DefaultingDicomField(
        StringDicomField(),
        dump_default=Defaults.string,
        data_key="ContainerIdentifier",
        allow_none=True,
    )
    stainings = fields.List(fields.Field, load_only=True, allow_none=True)
    samples = fields.List(
        fields.Field, data_key="SpecimenDescriptionSequence", allow_none=True
    )
    container_type = SingleCodeDicomField(
        ContainerTypeCode,
        data_key="ContainerTypeCodeSequence",
        dump_only=True,
        dump_default=Defaults.slide_container_type,
    )

    @property
    def load_type(self) -> Type[Slide]:
        return Slide

    @pre_dump
    def pre_dump(self, slide: Slide, **kwargs):
        # move staining to samples so that sample field can serialize both
        if slide.samples is None:
            samples = [SlideSample(identifier=defaults.string)]
        else:
            samples = slide.samples
        serialied_samples = [
            SlideSampleDicom.to_dataset(slide_sample, slide.stainings)
            for slide_sample in samples
        ]

        return {"identifier": slide.identifier, "samples": serialied_samples}

    @post_dump
    def post_dump(self, data: Dict[str, Any], **kwargs):
        data["IssuerOfTheContainerIdentifierSequence"] = []
        data["ContainerComponentSequence"] = []
        return super().post_dump(data, **kwargs)

    @pre_load
    def pre_load(self, dataset: Dataset, **kwargs):
        # move staining steps from SpecimenDescriptionSequence to staining
        data = super().pre_load(dataset, **kwargs)
        specimen_description_sequence = data.get("SpecimenDescriptionSequence", None)
        if specimen_description_sequence is not None:
            samples, stainings = SlideSampleDicom().from_dataset(
                specimen_description_sequence
            )
            data["SpecimenDescriptionSequence"] = samples
            data["stainings"] = stainings
        else:
            data["SpecimenDescriptionSequence"] = None
            data["stainings"] = None
        return data
