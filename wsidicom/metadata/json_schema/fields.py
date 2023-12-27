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

import dataclasses
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, Union

from marshmallow import ValidationError, fields
import numpy as np
from pydicom.sr.coding import Code
from pydicom.uid import UID
from wsidicom.conceptcode import CidConceptCode, CidConceptCodeType
from wsidicom.geometry import PointMm, SizeMm
from wsidicom.metadata.optical_path import LutDataType

from wsidicom.metadata.sample import (
    IssuerOfIdentifier,
    LocalIssuerOfIdentifier,
    SlideSamplePosition,
    Specimen,
    SpecimenIdentifier,
    UniversalIssuerOfIdentifier,
    UniversalIssuerType,
)


class SlideSamplePositionJsonField(fields.Field):
    def _serialize(
        self, value: Optional[Union[str, SlideSamplePosition]], attr, obj, **kwargs
    ) -> Optional[Union[str, Dict]]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return {
            "x": value.x,
            "y": value.y,
            "z": value.z,
        }

    def _deserialize(
        self,
        value: Union[str, Dict],
        attr: Optional[str],
        data: Optional[Mapping[str, Any]],
        **kwargs,
    ) -> Union[str, SlideSamplePosition]:
        try:
            if isinstance(value, str):
                return value
            return SlideSamplePosition(value["x"], value["y"], value["z"])
        except ValueError as error:
            raise ValidationError(
                "Could not deserialize slide sample position."
            ) from error


class IssuerOfIdentifierJsonField(fields.Field):
    def _serialize(
        self, value: Optional[IssuerOfIdentifier], attr, obj, **kwargs
    ) -> Optional[Union[str, Dict]]:
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
        value: Dict[str, Any],
        attr: Optional[str],
        data: Optional[Mapping[str, Any]],
        **kwargs,
    ) -> IssuerOfIdentifier:
        try:
            identifier = value["identifier"]
            if "issuer_type" in value:
                print(value["issuer_type"], type(value["issuer_type"]))
                issuer_type = UniversalIssuerType(value["issuer_type"])
                local_identifier = value.get("local_identifier", None)
                return UniversalIssuerOfIdentifier(
                    identifier, issuer_type, local_identifier
                )
            return LocalIssuerOfIdentifier(identifier)
        except ValueError as error:
            print(error)
            raise ValidationError(
                "Could not deserialize issuer of identifier."
            ) from error


class SpecimenIdentifierJsonField(fields.Field):
    _issuer_of_identifier_field = IssuerOfIdentifierJsonField()

    def _serialize(
        self,
        value: Optional[Union[str, SpecimenIdentifier]],
        attr,
        obj,
        **kwargs,
    ) -> Optional[Union[str, Dict]]:
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
        value: Union[str, Dict],
        attr: Optional[str],
        data: Optional[Mapping[str, Any]],
        **kwargs,
    ) -> Union[str, SpecimenIdentifier]:
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
    def _serialize(
        self, value: Optional[PointMm], attr, obj, **kwargs
    ) -> Optional[Dict]:
        if value is None:
            return None
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }

    def _deserialize(
        self,
        value: Dict,
        attr: Optional[str],
        data: Optional[Mapping[str, Any]],
        **kwargs,
    ) -> PointMm:
        try:
            return PointMm(**value)
        except ValueError as error:
            raise ValidationError("Could not deserialize PointMm.") from error


class SizeMmJsonField(fields.Field):
    def _serialize(
        self, value: Optional[SizeMm], attr, obj, **kwargs
    ) -> Optional[Dict]:
        if value is None:
            return None
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }

    def _deserialize(
        self,
        value: Dict,
        attr: Optional[str],
        data: Optional[Mapping[str, Any]],
        **kwargs,
    ) -> SizeMm:
        try:
            return SizeMm(**value)
        except ValueError as error:
            raise ValidationError("Could not deserialize PointMm.") from error


class UidJsonField(fields.Field):
    def _serialize(self, value: Optional[UID], attr, obj, **kwargs) -> Optional[str]:
        if value is None:
            return None
        return str(value)

    def _deserialize(self, value: str, attr, data, **kwargs) -> UID:
        try:
            return UID(value)
        except ValueError as error:
            raise ValidationError("Could not deserialize UID.") from error


class CodeJsonField(fields.Field):
    def _serialize(self, value: Optional[Code], attr, obj, **kwargs) -> Optional[Dict]:
        if value is None:
            return None
        return JsonFieldFactory._serialize_code(value)

    def _deserialize(self, value: Dict, attr, data, **kwargs) -> Code:
        try:
            return Code(**value)
        except ValueError as error:
            raise ValidationError("Could not deserialize Code.") from error


class StringOrCodeJsonField(fields.Field):
    def _serialize(
        self, value: Optional[Union[str, Code]], attr, obj, **kwargs
    ) -> Optional[Union[str, Dict]]:
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

    def _deserialize(
        self, value: Union[str, Dict], attr, data, **kwargs
    ) -> Union[str, Code]:
        if isinstance(value, str):
            return value
        try:
            return Code(**value)
        except ValueError as error:
            raise ValidationError("Could not deserialize Code.") from error


class JsonFieldFactory:
    @classmethod
    def float_or_concept_code(
        cls, concept_code_type: Type[CidConceptCodeType], many=False, **metadata
    ) -> Type[fields.Field]:
        def serialize(
            self, value: Optional[Union[float, CidConceptCodeType]], attr, obj, **kwargs
        ) -> Optional[Union[float, Dict]]:
            if isinstance(value, float):
                return value
            if isinstance(value, (Code, CidConceptCode)):
                return cls._serialize_code(value)

        def deserialize(
            self,
            value: Union[float, Dict[str, Any]],
            attr: Optional[str],
            data: Optional[Mapping[str, Any]],
            **kwargs,
        ) -> Union[float, CidConceptCodeType]:
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
        cls, concept_code_type: Type[CidConceptCodeType], many=False, **metadata
    ) -> Type[fields.Field]:
        def serialize(
            self, value: Optional[Union[str, CidConceptCodeType]], attr, obj, **kwargs
        ) -> Optional[Union[str, Dict]]:
            if value is None:
                return None
            if isinstance(value, str):
                return value

            return cls._serialize_code(value)

        def deserialize(
            self,
            value: Union[str, Dict],
            attr: Optional[str],
            data: Optional[Mapping[str, Any]],
            **kwargs,
        ) -> Union[str, CidConceptCodeType]:
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
        concept_code_type: Type[CidConceptCodeType],
    ) -> Type[fields.Field]:
        def serialize(
            self, value: Optional[CidConceptCodeType], attr, obj, **kwargs
        ) -> Optional[Dict]:
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
        concept_code_type: Type[CidConceptCodeType],
    ):
        def _deserialize(
            self,
            value: Dict,
            attr: Optional[str],
            data: Optional[Mapping[str, Any]],
            **kwargs,
        ) -> CidConceptCodeType:
            try:
                return concept_code_type(**value)
            except ValueError as error:
                raise ValidationError("Could not deserialize Code.") from error

        return _deserialize

    @staticmethod
    def _serialize_code(code: Union[Code, CidConceptCode]) -> Dict[str, str]:
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
        self, value: LutDataType, attr: Optional[str], obj: Any, **kwargs
    ) -> int:
        return 8 * value().itemsize

    def _deserialize(
        self,
        value: int,
        attr: Optional[str],
        data: Optional[Mapping[str, Any]],
        **kwargs,
    ) -> LutDataType:
        if value == 8:
            return np.uint8
        if value == 16:
            return np.uint16
        raise NotImplementedError(f"Not-implemented bit count {value}.")


class FileLoadingField(fields.Field):
    def _serialize(self, value: bytes, attr: Optional[str], obj: Any, **kwargs):
        raise NotImplementedError("Dumping bytes to file not implemented.")

    def _deserialize(
        self,
        value: str,
        attr: Optional[str],
        data: Optional[Mapping[str, Any]],
        **kwargs,
    ):
        path = Path(value)
        if not path.exists() or not path.is_file():
            raise ValidationError(f"File {path} does not exist or is not a file.")
        with open(path, "rb") as file:
            return file.read()
