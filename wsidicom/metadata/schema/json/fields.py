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

"""Json fields for serializing values."""

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from marshmallow import ValidationError, fields
from pydicom.sr.coding import Code
from pydicom.uid import UID

from wsidicom.conceptcode import CidConceptCode, CidConceptCodeType, UnitCode
from wsidicom.geometry import PointMm, SizeMm
from wsidicom.metadata.optical_path import LutDataType
from wsidicom.metadata.sample import (
    IssuerOfIdentifier,
    LocalIssuerOfIdentifier,
    Measurement,
    SpecimenIdentifier,
    UniversalIssuerOfIdentifier,
    UniversalIssuerType,
)


class IssuerOfIdentifierJsonField(fields.Field):
    def _serialize(
        self, value: IssuerOfIdentifier | None, attr, obj, **kwargs
    ) -> str | dict | None:
        if value is None:
            return None
        if isinstance(value, LocalIssuerOfIdentifier):
            return {
                "identifier": value.identifier,
            }
        elif isinstance(value, UniversalIssuerOfIdentifier):
            serialized = {
                "identifier": value.identifier,
                "issuer_type": value.issuer_type.name,
            }
            if value.local_identifier is not None:
                serialized["local_identifier"] = value.local_identifier
            return serialized
        raise NotImplementedError(f"Serialization of {type(value)} is not implemented.")

    def _deserialize(
        self,
        value: dict[str, Any],
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ) -> IssuerOfIdentifier:
        try:
            identifier = value["identifier"]
            if "issuer_type" in value:
                issuer_type = UniversalIssuerType(value["issuer_type"])
                local_identifier = value.get("local_identifier")
                return UniversalIssuerOfIdentifier(
                    identifier, issuer_type, local_identifier
                )
            return LocalIssuerOfIdentifier(identifier)
        except ValueError as error:
            raise ValidationError(
                "Could not deserialize issuer of identifier."
            ) from error


class SpecimenIdentifierJsonField(fields.Field):
    _issuer_of_identifier_field = IssuerOfIdentifierJsonField()

    def _serialize(
        self,
        value: str | SpecimenIdentifier | None,
        attr,
        obj,
        **kwargs,
    ) -> str | dict | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if value.issuer is None:
            return {
                "value": value.value,
            }

        return {
            "value": value.value,
            "issuer": self._issuer_of_identifier_field._serialize(
                value.issuer, attr, obj, **kwargs
            ),
        }

    def _deserialize(
        self,
        value: str | dict,
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ) -> str | SpecimenIdentifier:
        try:
            if isinstance(value, str):
                return value
            if "issuer" not in value:
                return SpecimenIdentifier(value["value"])
            issuer = self._issuer_of_identifier_field._deserialize(
                value["issuer"], attr, data, **kwargs
            )
            return SpecimenIdentifier(value["value"], issuer)

        except ValueError as error:
            raise ValidationError(
                "Could not deserialize specimen identifier."
            ) from error


class PointMmJsonField(fields.Field):
    def _serialize(self, value: PointMm | None, attr, obj, **kwargs) -> dict | None:
        if value is None:
            return None
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }

    def _deserialize(
        self,
        value: dict,
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ) -> PointMm:
        try:
            return PointMm(**value)
        except ValueError as error:
            raise ValidationError("Could not deserialize PointMm.") from error


class SizeMmJsonField(fields.Field):
    def _serialize(self, value: SizeMm | None, attr, obj, **kwargs) -> dict | None:
        if value is None:
            return None
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }

    def _deserialize(
        self,
        value: dict,
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ) -> SizeMm:
        try:
            return SizeMm(**value)
        except ValueError as error:
            raise ValidationError("Could not deserialize SizeMm.") from error


class UidJsonField(fields.Field):
    def _serialize(self, value: UID | None, attr, obj, **kwargs) -> str | None:
        if value is None:
            return None
        return str(value)

    def _deserialize(self, value: str, attr, data, **kwargs) -> UID:
        try:
            return UID(value)
        except ValueError as error:
            raise ValidationError("Could not deserialize UID.") from error


class CodeJsonField(fields.Field):
    def _serialize(self, value: Code | None, attr, obj, **kwargs) -> dict | None:
        if value is None:
            return None
        return JsonFieldFactory._serialize_code(value)

    def _deserialize(self, value: dict, attr, data, **kwargs) -> Code:
        try:
            return Code(**value)
        except ValueError as error:
            raise ValidationError("Could not deserialize Code.") from error


class StringOrCodeJsonField(fields.Field):
    def _serialize(
        self, value: str | Code | None, attr, obj, **kwargs
    ) -> str | dict | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        code = {
            "value": value.value,
            "scheme_designator": value.scheme_designator,
            "meaning": value.meaning,
        }
        if value.scheme_version is not None:
            code["scheme_version"] = value.scheme_version
        return code

    def _deserialize(self, value: str | dict, attr, data, **kwargs) -> str | Code:
        if isinstance(value, str):
            return value
        try:
            return Code(**value)
        except ValueError as error:
            raise ValidationError("Could not deserialize Code.") from error


class JsonFieldFactory:
    @classmethod
    def float_or_concept_code(
        cls, concept_code_type: type[CidConceptCodeType], many=False, **metadata
    ) -> type[fields.Field]:
        def serialize(
            self, value: float | CidConceptCodeType | None, attr, obj, **kwargs
        ) -> float | dict | None:
            if isinstance(value, float):
                return value
            if isinstance(value, (Code, CidConceptCode)):
                return cls._serialize_code(value)
            return None

        def deserialize(
            self,
            value: float | dict[str, Any],
            attr: str | None,
            data: Mapping[str, Any] | None,
            **kwargs,
        ) -> float | CidConceptCodeType:
            if isinstance(value, (int, float)):
                return value
            try:
                return concept_code_type(**value)
            except (ValueError, KeyError) as error:
                raise ValidationError("Could not deserialize Code.") from error

        return type(
            f"FloatOr{CidConceptCodeType}Field",
            (fields.Field,),
            {"_serialize": serialize, "_deserialize": deserialize},
        )

    @classmethod
    def str_or_concept_code(
        cls, concept_code_type: type[CidConceptCodeType], many=False, **metadata
    ) -> type[fields.Field]:
        def serialize(
            self, value: str | CidConceptCodeType | None, attr, obj, **kwargs
        ) -> str | dict | None:
            if value is None:
                return None
            if isinstance(value, str):
                return value

            return cls._serialize_code(value)

        def deserialize(
            self,
            value: str | dict,
            attr: str | None,
            data: Mapping[str, Any] | None,
            **kwargs,
        ) -> str | CidConceptCodeType:
            if isinstance(value, str):
                return value
            try:
                return concept_code_type(**value)
            except ValueError as error:
                raise ValidationError("Could not deserialize Code.") from error

        return type(
            f"StringOr{CidConceptCodeType}Field",
            (fields.Field,),
            {"_serialize": serialize, "_deserialize": deserialize},
        )

    @classmethod
    def concept_code(
        cls,
        concept_code_type: type[CidConceptCodeType],
    ) -> type[fields.Field]:
        def serialize(
            self, value: CidConceptCodeType | None, attr, obj, **kwargs
        ) -> dict | None:
            if value is None:
                return None
            return cls._serialize_code(value)

        deserialize = cls._concept_code_deserializer_factory(concept_code_type)

        return type(
            f"{CidConceptCodeType}Field",
            (fields.Field,),
            {"_serialize": serialize, "_deserialize": deserialize},
        )

    @staticmethod
    def _concept_code_deserializer_factory(
        concept_code_type: type[CidConceptCodeType],
    ):
        def _deserialize(
            self,
            value: dict,
            attr: str | None,
            data: Mapping[str, Any] | None,
            **kwargs,
        ) -> CidConceptCodeType:
            try:
                return concept_code_type(**value)
            except ValueError as error:
                raise ValidationError("Could not deserialize Code.") from error

        return _deserialize

    @staticmethod
    def _serialize_code(code: Code | CidConceptCode) -> dict[str, str]:
        try:
            result = {
                "value": code.value,
                "scheme_designator": code.scheme_designator,
                "meaning": code.meaning,
            }
            if code.scheme_version is not None:
                result["scheme_version"] = code.scheme_version
            return result

        except Exception as exception:
            raise ValueError(f"Failed to serialize code {code}") from exception


class NpUIntDTypeField(fields.Field):
    def _serialize(
        self, value: LutDataType, attr: str | None, obj: Any, **kwargs
    ) -> int:
        return 8 * value().itemsize

    def _deserialize(
        self,
        value: int,
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ) -> LutDataType:
        if value == 8:
            return np.uint8
        if value == 16:
            return np.uint16
        raise NotImplementedError(f"Not-implemented bit count {value}.")


class FileLoadingField(fields.Field):
    def _serialize(self, value: bytes, attr: str | None, obj: Any, **kwargs):
        raise NotImplementedError("Dumping bytes to file not implemented.")

    def _deserialize(
        self,
        value: str,
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ):
        path = Path(value)
        if not path.exists() or not path.is_file():
            raise ValidationError(f"File {path} does not exist or is not a file.")
        with open(path, "rb") as file:
            return file.read()


class MeasurementJsonField(fields.Field):
    def _serialize(
        self, value: Measurement | None, attr, obj, **kwargs
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return {"value": value.value, "unit": value.unit.value}

    def _deserialize(
        self,
        value: dict[str, Any],
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ) -> Measurement:
        try:
            return Measurement(value["value"], UnitCode.from_unit(value["unit"]))
        except ValueError as error:
            raise ValidationError("Could not deserialize measurement.") from error
