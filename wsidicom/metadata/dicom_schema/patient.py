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

from collections import defaultdict
from typing import Any, Dict, Type
from wsidicom.metadata.dicom_schema.schema import (
    DefaultIfValidationFailedDicomSchema,
    DicomSchema,
)
from wsidicom.metadata.dicom_schema.fields import (
    BooleanDicomField,
    CodeDicomField,
    DateDicomField,
    EnumDicomField,
    FlatteningNestedField,
    ListDicomField,
    PatientNameDicomField,
    SingleCodeSequenceField,
    StringDicomField,
)
from marshmallow import fields, pre_dump, post_load
from pydicom.sr.coding import Code
from wsidicom.metadata.patient import Patient, PatientDeIdentification, PatientSex


class PatientDeIdentificationDicomSchema(DicomSchema[PatientDeIdentification]):
    identity_removed = BooleanDicomField(data_key="PatientIdentityRemoved")
    method_strings = ListDicomField(fields.String(), data_key="DeidentificationMethod")
    method_codes = fields.List(
        CodeDicomField(Code), data_key="DeidentificationMethodCodeSequence"
    )

    @property
    def load_type(self) -> Type[PatientDeIdentification]:
        return PatientDeIdentification

    @pre_dump
    def pre_dump(self, de_identification: PatientDeIdentification, **kwargs):
        fields = {"identity_removed": de_identification.identity_removed}
        if de_identification.methods is not None:
            de_identification_fields = defaultdict(list)
            for method in de_identification.methods:
                if isinstance(method, str):
                    de_identification_fields["method_strings"].append(method)
                else:
                    de_identification_fields["method_codes"].append(method)
            fields.update(de_identification_fields)
        return fields

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs):
        method_strings = data.pop("method_strings", [])
        method_codes = data.pop("method_codes", [])
        methods = method_strings + method_codes
        if len(methods) > 0:
            data["methods"] = methods
        return super().post_load(data, **kwargs)


class PatientDicomSchema(DefaultIfValidationFailedDicomSchema[Patient]):
    name = PatientNameDicomField(data_key="PatientName", load_default=None)
    identifier = StringDicomField(data_key="PatientID", load_default=None)
    birth_date = DateDicomField(data_key="PatientBirthDate", load_default=None)
    sex = EnumDicomField(PatientSex, data_key="PatientSex", load_default=None)
    species_description_string = StringDicomField(
        data_key="PatientSpeciesDescription", allow_none=True
    )
    species_description_code = SingleCodeSequenceField(
        Code, data_key="PatientSpeciesCodeSequence", allow_none=True
    )
    de_identification = FlatteningNestedField(
        PatientDeIdentificationDicomSchema(), allow_none=True
    )

    @property
    def load_type(self) -> Type[Patient]:
        return Patient

    @pre_dump
    def pre_dump(self, patient: Patient, **kwargs):
        fields = {
            "name": patient.name,
            "identifier": patient.identifier,
            "birth_date": patient.birth_date,
            "sex": patient.sex,
            "de_identification": patient.de_identification,
        }

        if isinstance(patient.species_description, str):
            fields["species_description_string"] = patient.species_description
        elif isinstance(patient.species_description, Code):
            fields["species_description_code"] = patient.species_description
        return fields

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs):
        species_description_string = data.pop("species_description_string", None)
        species_description_code = data.pop("species_description_code", None)
        if species_description_code is not None:
            data["species_description"] = species_description_code
        elif species_description_string is not None:
            data["species_description"] = species_description_string
        return super().post_load(data, **kwargs)
