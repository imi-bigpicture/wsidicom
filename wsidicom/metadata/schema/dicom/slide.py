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

"""DICOM schema for Slide model."""

import logging
from typing import Any

from marshmallow import fields, post_load, pre_dump
from pydicom.valuerep import VR

from wsidicom.conceptcode import ContainerTypeCode
from wsidicom.metadata.sample import SpecimenIdentifier
from wsidicom.metadata.schema.dicom.defaults import defaults
from wsidicom.metadata.schema.dicom.fields import (
    DefaultingDicomField,
    IssuerOfIdentifierDicomField,
    ListDicomField,
    SingleCodeSequenceField,
    StringDicomField,
)
from wsidicom.metadata.schema.dicom.sample import (
    SpecimenDescriptionDicomSchema,
)
from wsidicom.metadata.schema.dicom.sample.formatter import SpecimenDicomFormatter
from wsidicom.metadata.schema.dicom.sample.parser import SpecimenDicomParser
from wsidicom.metadata.schema.dicom.schema import ModuleDicomSchema
from wsidicom.metadata.slide import Slide


class SlideDicomSchema(ModuleDicomSchema[Slide]):
    identifier = DefaultingDicomField(
        StringDicomField(value_representation=VR.LO),
        dump_default=defaults.string,
        data_key="ContainerIdentifier",
        allow_none=True,
    )
    issuer_of_identifier = IssuerOfIdentifierDicomField(
        dump_default=[],
        data_key="IssuerOfTheContainerIdentifierSequence",
        allow_none=True,
    )
    samples = ListDicomField(
        fields.Nested(SpecimenDescriptionDicomSchema()),
        data_key="SpecimenDescriptionSequence",
        allow_none=True,
        dump_required=True,
    )
    container_type = SingleCodeSequenceField(
        ContainerTypeCode,
        data_key="ContainerTypeCodeSequence",
        dump_only=True,
        dump_default=defaults.slide_container_type,
    )

    @property
    def load_type(self) -> type[Slide]:
        return Slide

    @pre_dump
    def pre_dump(self, slide: Slide, **kwargs):
        # Move staining onto samples so that the sample field can serialize both.
        dicom_samples = [
            SpecimenDicomFormatter.to_dicom(slide_sample, slide.stainings)
            for slide_sample in (slide.samples or [])
        ]
        if isinstance(slide.identifier, SpecimenIdentifier):
            identifier = slide.identifier.value
            issuer_of_identifier = slide.identifier.issuer
        else:
            identifier = slide.identifier
            issuer_of_identifier = None
        return {
            "identifier": identifier,
            "issuer_of_identifier": issuer_of_identifier,
            "samples": dicom_samples,
        }

    @post_load
    def post_load(self, data: dict[str, Any], **kwargs):
        dicom_samples = data.pop("samples", None)
        if dicom_samples is not None:
            try:
                samples, stainings = SpecimenDicomParser().parse_descriptions(
                    dicom_samples
                )

            except (AttributeError, ValueError):
                logging.warning(
                    "Failed to parse SpecimenDescriptionSequence", exc_info=True
                )
                samples = None
                stainings = None
        else:
            samples = None
            stainings = None
        data["samples"] = samples
        data["stainings"] = stainings
        issuer_of_identifier = data.pop("issuer_of_identifier", None)
        if issuer_of_identifier is not None:
            data["identifier"] = SpecimenIdentifier(
                value=data["identifier"], issuer=issuer_of_identifier
            )
        return super().post_load(data, **kwargs)

    @property
    def module_name(self) -> str:
        return "slide"
