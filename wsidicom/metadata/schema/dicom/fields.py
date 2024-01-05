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

"""DICOM fields for attribute serialization."""

import datetime
import math
from abc import abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from marshmallow import Schema, fields
from marshmallow.fields import Field
from marshmallow.utils import missing
from pydicom import DataElement, Dataset
from pydicom.multival import MultiValue
from pydicom.sr.coding import Code
from pydicom.valuerep import DA, DT, TM, DSfloat, PersonName
from pydicom.uid import UID

from wsidicom.conceptcode import ConceptCode, UnitCode
from wsidicom.geometry import Orientation, PointMm, SizeMm
from wsidicom.metadata.sample import (
    IssuerOfIdentifier,
    LocalIssuerOfIdentifier,
    Measurement,
    UniversalIssuerOfIdentifier,
    UniversalIssuerType,
)


class StringLikeDicomField(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs) -> Any:
        """By default pydicom returns empty string for empty string-like elements."""
        deserialized = super()._deserialize(value, attr, data, **kwargs)
        if deserialized == "":
            return None
        return deserialized


class StringDicomField(StringLikeDicomField):
    pass


class EnumDicomField(fields.Enum):
    def _deserialize(self, value, attr, data, **kwargs):
        if value == "":
            return None
        return super()._deserialize(value, attr, data, **kwargs)


class DateTimeDicomField(StringLikeDicomField):
    def _serialize(
        self,
        value: Optional[datetime.datetime],
        attr: Optional[str],
        obj: Any,
        **kwargs,
    ):
        if value is None:
            return None
        return DT(value)

    def _deserialize(self, value, attr, data, **kwargs) -> Any:
        if value is None:
            return None
        try:
            return DT(value)
        except ValueError:
            return None


class DateDicomField(StringLikeDicomField):
    def _serialize(
        self, value: Optional[datetime.date], attr: Optional[str], obj: Any, **kwargs
    ):
        if value is None:
            return None
        return DA(value)

    def _deserialize(self, value, attr, data, **kwargs) -> Any:
        if value is None:
            return None
        try:
            return DA(value)
        except ValueError:
            return None


class TimeDicomField(StringLikeDicomField):
    def _serialize(
        self, value: Optional[datetime.time], attr: Optional[str], obj: Any, **kwargs
    ):
        if value is None:
            return None
        return TM(value)

    def _deserialize(self, value, attr, data, **kwargs) -> Any:
        if value is None:
            return None
        try:
            return TM(value)
        except ValueError:
            return None


class BooleanDicomField(fields.Boolean):
    def __init__(self, **kwargs):
        super().__init__(truthy=set(["YES"]), falsy=set(["NO"]), **kwargs)

    def _serialize(self, value: bool, attr: Optional[str], obj: Any, **kwargs):
        if value:
            string_value = self.truthy
        else:
            string_value = self.falsy
        return list(string_value)[0]

    def _deserialize(self, value, attr, data, **kwargs) -> Any:
        deserialized = super()._deserialize(value, attr, data, **kwargs)
        if deserialized == "":
            return None
        return deserialized


class OffsetInSlideCoordinateSystemField(fields.Field):
    def _serialize(
        self, origin: Optional[PointMm], attr: Optional[str], obj: Any, **kwargs
    ):
        if origin is None:
            if self.dump_default is None:
                return None
            assert isinstance(self.dump_default, PointMm)
            origin = self.dump_default
        origin_element = Dataset()
        origin_element.XOffsetInSlideCoordinateSystem = DSfloat(origin.x, True)
        origin_element.YOffsetInSlideCoordinateSystem = DSfloat(origin.y, True)
        return [origin_element]

    def _deserialize(
        self,
        value: DataElement,
        attr: Optional[str],
        data: Optional[Dict[str, Any]],
        **kwargs,
    ) -> PointMm:
        return PointMm(
            x=value[0].XOffsetInSlideCoordinateSystem,
            y=value[0].YOffsetInSlideCoordinateSystem,
        )


class ImageOrientationSlideField(fields.Field):
    def _serialize(
        self, rotation: Optional[float], attr: Optional[str], obj: Any, **kwargs
    ):
        if rotation is None:
            if self.dump_default is None:
                return None
            assert isinstance(self.dump_default, float)
            rotation = self.dump_default
        x = round(math.sin(rotation * math.pi / 180), 8)
        y = round(math.cos(rotation * math.pi / 180), 8)
        return [-x, y, 0, y, x, 0]

    def _deserialize(
        self,
        value: Tuple[float, float, float, float, float, float],
        attr: Optional[str],
        data: Optional[Dict[str, Any]],
        **kwargs,
    ) -> float:
        orientation = Orientation(value)
        return orientation.rotation


class ListDicomField(fields.List):
    """Wrapper around normal list that handles single-valued lists from pydicom."""

    def _deserialize(
        self, value: Union[Any, List[Any]], attr, data, **kwargs
    ) -> List[Any]:
        if not isinstance(value, MultiValue):
            value = [value]
        return super()._deserialize(value, attr, data, **kwargs)


class FlattenOnDumpNestedDicomField(fields.Nested):
    """Field that flattens the nested dataset into the parent dataset on dump.

    On load the nested fields are deflatten from the parent dataset to a nested dataset.

    The flatten/deflatten is done by the parent schema.
    """

    def __init__(self, nested: Schema, **kwargs):
        self._nested = nested
        super().__init__(nested=nested, **kwargs)

    @property
    def nested_schema(self) -> Schema:
        return self._nested

    def de_flatten(self, dataset: Dataset) -> Optional[Dataset]:
        """Create new dataset containing the attributes defined in nested schema."""
        nested = Dataset()
        for nested_field in self.nested_schema.fields.values():
            if nested_field.dump_only:
                continue
            if isinstance(nested_field, FlattenOnDumpNestedDicomField):
                de_flatten_nested_field = nested_field.de_flatten(dataset)
                if de_flatten_nested_field is not None:
                    nested.update(de_flatten_nested_field)
            elif nested_field.data_key is not None and nested_field.data_key in dataset:
                nested_value = dataset.get(nested_field.data_key)
                setattr(nested, nested_field.data_key, nested_value)
        if len(nested) == 0:
            return None
        return nested

    def flatten(self, data: Dict[str, Any]):
        """Insert attributes from nested dataset into data."""
        key = self.name
        if self.data_key is not None:
            key = self.data_key
        nested = data.pop(key, None)
        if isinstance(nested, Dataset):
            for nested_key, nested_value in nested.items():
                data[nested_key] = nested_value  # type: ignore

    def _serialize(self, nested_obj, attr: Optional[str], obj: Any, **kwargs):
        if nested_obj is None and self.dump_default != missing:
            nested_obj = self.dump_default
        return super()._serialize(nested_obj, attr, obj, **kwargs)


class FlattenOnLoadNestedDicomField(fields.Nested):
    """Field that flattens the nested loaded item into the parent item on load.

    On dump the nested fields are deflatten from the parent dataset to a nested dataset.
    """

    def __init__(self, nested: Schema, **kwargs):
        self._nested = nested
        super().__init__(nested=nested, **kwargs)

    @property
    def nested_schema(self) -> Schema:
        return self._nested

    def de_flatten(self, dataset: Dataset) -> Optional[Dataset]:
        """Create new dataset containing the attributes defined in nested schema."""
        nested = Dataset()
        for nested_field in self.nested_schema.fields.values():
            if nested_field.dump_only:
                continue
            if isinstance(nested_field, FlattenOnDumpNestedDicomField):
                de_flatten_nested_field = nested_field.de_flatten(dataset)
                if de_flatten_nested_field is not None:
                    nested.update(de_flatten_nested_field)
            elif nested_field.data_key is not None and nested_field.data_key in dataset:
                nested_value = dataset.get(nested_field.data_key)
                setattr(nested, nested_field.data_key, nested_value)
        if len(nested) == 0:
            return None
        return nested

    def flatten(self, data: Dict[str, Any]):
        """Insert attributes from nested dataset into data."""
        key = self.name
        if self.data_key is not None:
            key = self.data_key
        nested = data.pop(key, None)
        if isinstance(nested, Dataset):
            for nested_key, nested_value in nested.items():
                data[nested_key] = nested_value  # type: ignore


CodeType = TypeVar("CodeType", Code, ConceptCode)


class FloatDicomField(fields.Float):
    def _serialize(self, value: float, attr: Optional[str], obj: Any, **kwargs):
        return DSfloat(value)


class CodeDicomField(fields.Field, Generic[CodeType]):
    def __init__(self, load_type: Type[CodeType], **kwargs) -> None:
        self._load_type = load_type
        super().__init__(**kwargs)

    def _serialize(
        self, value: Optional[CodeType], attr: Optional[str], obj: Any, **kwargs
    ):
        if value is None:
            return self.dump_default
        dataset = Dataset()
        dataset.CodeValue = value.value
        dataset.CodingSchemeDesignator = value.scheme_designator
        dataset.CodeMeaning = value.meaning
        dataset.CodingSchemeVersion = value.scheme_version

        return dataset

    def _deserialize(
        self,
        value: Dataset,
        attr: Optional[str],
        data: Optional[Dict[str, Any]],
        **kwargs,
    ):
        return self._load_type(
            value=value.CodeValue,
            scheme_designator=value.CodingSchemeDesignator,
            meaning=value.CodeMeaning,
            scheme_version=value.get("CodingSchemeVersion", None),
        )


class SingleCodeSequenceField(CodeDicomField):
    """Field for a DICOM `code sequence` that can only contain one code."""

    def _serialize(self, value: CodeType, attr: Optional[str], obj: Any, **kwargs):
        return [super()._serialize(value, attr, obj, **kwargs)]

    def _deserialize(
        self,
        value: Sequence[Dataset],
        attr: Optional[str],
        data: Optional[Dict[str, Any]],
        **kwargs,
    ):
        return super()._deserialize(value[0], attr, data, **kwargs)


class FloatOrCodeDicomField(fields.Field, Generic[CodeType]):
    def __init__(self, load_type: Type[CodeType], **kwargs) -> None:
        self._float_field = FloatDicomField()
        self._code_field = CodeDicomField(load_type)
        super().__init__(**kwargs)

    def _serialize(
        self, value: Union[float, CodeType], attr: Optional[str], obj: Any, **kwargs
    ):
        assert attr is not None
        if isinstance(value, float):
            return self._float_field.serialize(attr, obj, **kwargs)
        return self._code_field.serialize(attr, obj, **kwargs)

    def _deserialize(self, value: Union[DSfloat, Dataset], attr, data, **kwargs):
        if isinstance(value, DSfloat):
            return self._float_field.deserialize(value, attr, data, **kwargs)
        return self._code_field.deserialize(value, attr, data, **kwargs)


class UidDicomField(fields.Field):
    def _deserialize(self, value: Any, attr, data, **kwargs):
        if value is None or value == "":
            return None
        if isinstance(value, UID):
            return value
        return UID(value)


class UidDatasetDicomField(UidDicomField):
    def __init__(self, data_key: str, **kwargs):
        self._data_key = data_key
        super().__init__(data_key=data_key, **kwargs)

    def _serialize(
        self, value: Optional[UID], attr: Optional[str], obj: Any, **kwargs
    ) -> Optional[Dataset]:
        if value is None:
            return None
        dataset = Dataset()
        setattr(dataset, self._data_key, value)
        return dataset

    def _deserialize(self, value: Dataset, attr, data, **kwargs) -> Optional[UID]:
        nested_value = getattr(value, self._data_key, None)
        if nested_value is None:
            return None
        return self._deserialize(nested_value, attr, data, **kwargs)


class PatientNameDicomField(fields.String):
    def _deserialize(self, value: PersonName, attr, data, **kwargs) -> str:
        return str(value)

    def _serialize(self, value: str, attr, obj, **kwargs) -> Optional[PersonName]:
        return PersonName(value)


class IssuerOfIdentifierDicomField(fields.Field):
    def _deserialize(
        self, value: Optional[Sequence[Dataset]], attr, data, **kwargs
    ) -> Optional[IssuerOfIdentifier]:
        if value is None or len(value) == 0:
            return None
        dataset = value[0]
        if "UniversalEntityIDType" in dataset:
            return UniversalIssuerOfIdentifier(
                dataset.UniversalEntityID,
                UniversalIssuerType(dataset.UniversalEntityIDType),
                dataset.get("LocalNamespaceEntityID", None),
            )
        if "LocalNamespaceEntityID" in dataset:
            return LocalIssuerOfIdentifier(dataset.LocalNamespaceEntityID)
        return None

    def _serialize(
        self, value: Optional[IssuerOfIdentifier], attr, obj, **kwargs
    ) -> Optional[Sequence[Dataset]]:
        if value is None:
            return []
        dataset = Dataset()
        if isinstance(value, UniversalIssuerOfIdentifier):
            dataset.UniversalEntityID = value.identifier
            dataset.UniversalEntityIDType = value.issuer_type.name
            if value.local_identifier is not None:
                dataset.LocalNamespaceEntityID = value.local_identifier
        elif isinstance(value, LocalIssuerOfIdentifier):
            dataset.LocalNamespaceEntityID = value.identifier
        else:
            raise NotImplementedError()
        return [dataset]


class PixelSpacingDicomField(fields.Field):
    def _serialize(
        self, value: Optional[SizeMm], attr: Optional[str], obj: Any, **kwargs
    ):
        if value is None:
            return None
        return [DSfloat(value.width, True), DSfloat(value.height, True)]

    def _deserialize(
        self, value: Optional[Sequence[DSfloat]], attr, data, **kwargs
    ) -> Optional[SizeMm]:
        if value is None or len(value) == 0:
            return None
        return SizeMm(value[0], value[1])


ValueType = TypeVar("ValueType")


class TypeDicomField(fields.Field, Generic[ValueType]):
    def __init__(self, nested: Field, **kwargs):
        self._nested = nested
        super().__init__(**kwargs)

    def _serialize(
        self, value: Optional[ValueType], attr: Optional[str], obj: Any, **kwargs
    ):
        return self._nested._serialize(value, attr, obj, **kwargs)

    def _deserialize(self, value: Any, attr, data, **kwargs):
        de_serialized = self._nested._deserialize(value, attr, data, **kwargs)
        return de_serialized


class DefaultingDicomField(TypeDicomField[ValueType]):
    """Wrapper around a field that should always be present and have a value. Default
    value is constant."""

    def __init__(self, nested: Field, dump_default: ValueType, **kwargs):
        self._dump_default = dump_default
        super().__init__(nested=nested, dump_default=dump_default, **kwargs)

    def _serialize(
        self, value: Optional[ValueType], attr: Optional[str], obj: Any, **kwargs
    ):
        if value is None:
            value = self._dump_default
        return super()._serialize(value, attr, obj, **kwargs)


class DefaultingNoneDicomField(DefaultingDicomField[Optional[ValueType]]):
    """Wrapper around a field that should always be present but can be None."""

    def __init__(self, nested: Field, **kwargs):
        super().__init__(nested=nested, dump_default=None, **kwargs)

    def _serialize(
        self, value: Optional[ValueType], attr: Optional[str], obj: Any, **kwargs
    ):
        return super()._serialize(value, attr, obj, **kwargs)


class DefaultingTagDicomField(TypeDicomField[ValueType]):
    """Wrapper around a field that should always be present and have a value. Default
    value is taken from object by attribute defined by tag."""

    def __init__(self, nested: Field, tag: str, **kwargs):
        self._tag = tag
        super().__init__(nested=nested, **kwargs)

    def _serialize(
        self, value: Optional[ValueType], attr: Optional[str], obj: Any, **kwargs
    ):
        if value is None:
            value = getattr(obj, self._tag)
        return super()._serialize(value, attr, obj, **kwargs)


class DefaultingListDicomField(fields.List):
    def __init__(self, nested: Field, dump_default: List, **kwargs):
        self._dump_default = dump_default
        super().__init__(cls_or_instance=nested, dump_default=dump_default, **kwargs)

    def _serialize(self, value, attr, obj, **kwargs) -> Any:
        if value is None or len(value) == 0:
            value = self.dump_default
        return super()._serialize(value, attr, obj, **kwargs)


class DefaultingListTagDicomField(fields.List):
    def __init__(self, nested: Field, tag: str, **kwargs):
        self._tag = tag
        super().__init__(cls_or_instance=nested, **kwargs)

    def _serialize(self, value, attr, obj, **kwargs) -> Any:
        if value is None or len(value) == 0:
            value = getattr(obj, self._tag)
        return super()._serialize(value, attr, obj, **kwargs)


class NestedDatasetDicomField(fields.Nested, Generic[ValueType]):
    """Field for attribute of a single-item dataset sequence with a nested
    sing-item dataset sequence with the item the nested schema should handle."""

    def __init__(self, nested: Schema, data_key: str, nested_data_key: str, **kwargs):
        self._nested = nested
        self._data_key = data_key
        self._nested_data_key = nested_data_key
        super().__init__(nested=nested, data_key=data_key, **kwargs)

    def _serialize(
        self, value: Optional[ValueType], attr: Optional[str], obj: Any, **kwargs
    ) -> Optional[List[Dataset]]:
        nested_value = super()._serialize(value, attr, obj, **kwargs)
        if nested_value is None:
            return None
        dataset = Dataset()
        setattr(dataset, self._nested_data_key, [nested_value])
        return [dataset]

    def _deserialize(
        self, value: Sequence[Dataset], attr, data, **kwargs
    ) -> Optional[ValueType]:
        nested_value = getattr(value[0], self._nested_data_key, None)
        if nested_value is None:
            return None
        return super()._deserialize(nested_value[0], attr, data, **kwargs)  # type: ignore


class ContentItemDicomField(fields.Field, Generic[ValueType]):
    @abstractmethod
    def _serialize(
        self, value: Optional[ValueType], attr: Optional[str], obj: Any, **kwargs
    ) -> Optional[Dataset]:
        raise NotImplementedError()

    @abstractmethod
    def _deserialize(
        self, value: Dataset, attr: Optional[str], obj: Any, **kwargs
    ) -> Optional[ValueType]:
        raise NotImplementedError()


class CodeItemDicomField(ContentItemDicomField[Code]):
    def __init__(self, load_type: Type[CodeType], **kwargs) -> None:
        self._load_type = load_type
        super().__init__(**kwargs)

    def _serialize(
        self, value: Optional[Code], attr: Optional[str], obj: Any, **kwargs
    ) -> Optional[Dataset]:
        if value is None:
            return None
        code_dataset = Dataset()
        code_dataset.CodeValue = value.value
        code_dataset.CodingSchemeDesignator = value.scheme_designator
        code_dataset.CodeMeaning = value.meaning
        code_dataset.CodingSchemeVersion = value.scheme_version
        dataset = Dataset()
        dataset.ConceptCodeSequence = [code_dataset]
        return dataset

    def _deserialize(self, dataset: Dataset, attr: Optional[str], obj: Any, **kwargs):
        return self._load_type(
            value=dataset.ConceptCodeSequence[0].CodeValue,
            scheme_designator=dataset.ConceptCodeSequence[0].CodingSchemeDesignator,
            meaning=dataset.ConceptCodeSequence[0].CodeMeaning,
            scheme_version=dataset.ConceptCodeSequence[0].get(
                "CodingSchemeVersion", None
            ),
        )


class StringItemDicomField(ContentItemDicomField[str]):
    def _serialize(
        self, value: Optional[str], attr: Optional[str], obj: Any, **kwargs
    ) -> Optional[Dataset]:
        if value is None:
            return None
        dataset = Dataset()
        dataset.TextValue = value
        return dataset

    def _deserialize(self, dataset: Dataset, attr: Optional[str], obj: Any, **kwargs):
        return dataset.TextValue


class StringOrCodeItemDicomField(ContentItemDicomField[Union[str, Code]]):
    def __init__(self, load_type: Type[CodeType], **kwargs) -> None:
        self._string_field = StringItemDicomField()
        self._code_field = CodeItemDicomField(load_type)
        super().__init__(**kwargs)

    def _serialize(
        self, value: Optional[Union[str, Code]], attr: str, obj: Any, **kwargs
    ) -> Optional[Dataset]:
        if value is None:
            return None
        if isinstance(value, str):
            return self._string_field._serialize(value, attr, obj, **kwargs)
        return self._code_field._serialize(value, attr, obj, **kwargs)

    def _deserialize(self, dataset: Dataset, attr: str, obj: Any, **kwargs):
        if hasattr(dataset, "TextValue"):
            return self._string_field.deserialize(dataset, attr, obj, **kwargs)
        return self._code_field.deserialize(dataset, attr, obj, **kwargs)


class DateTimeItemDicomField(ContentItemDicomField[datetime.datetime]):
    def _serialize(
        self,
        value: Optional[datetime.datetime],
        attr: Optional[str],
        obj: Any,
        **kwargs,
    ) -> Optional[Dataset]:
        if value is None:
            return None
        dataset = Dataset()
        dataset.DateTime = DT(value)
        return dataset

    def _deserialize(self, dataset: Dataset, attr: Optional[str], obj: Any, **kwargs):
        try:
            return DT(dataset.DateTime)
        except ValueError:
            return None


class MeasurementtemDicomField(ContentItemDicomField[Measurement]):
    def _serialize(
        self, value: Optional[Measurement], attr: Optional[str], obj: Any, **kwargs
    ) -> Optional[Dataset]:
        if value is None:
            return None
        dataset = Dataset()
        dataset.NumericValue = DSfloat(value.value)
        dataset.FloatingPointValue = value.value
        unit_dataset = Dataset()
        unit_dataset.CodeValue = value.unit.value
        unit_dataset.CodingSchemeDesignator = value.unit.scheme_designator
        unit_dataset.CodeMeaning = value.unit.meaning
        if value.unit.scheme_version is not None:
            unit_dataset.CodingSchemeVersion = value.unit.scheme_version
        dataset.MeasurementUnitsCodeSequence = [unit_dataset]
        return dataset

    def _deserialize(self, dataset: Dataset, attr: Optional[str], obj: Any, **kwargs):
        if hasattr(dataset, "FloatingPointValue"):
            value = dataset.FloatingPointValue
        else:
            value = DSfloat(dataset.NumericValue)
            assert isinstance(value, float)
        unit_dataset = dataset.MeasurementUnitsCodeSequence[0]
        unit = UnitCode(
            value=unit_dataset.CodeValue,
            scheme_designator=unit_dataset.CodingSchemeDesignator,
            meaning=unit_dataset.CodeMeaning,
            scheme_version=unit_dataset.get("CodingSchemeVersion", None),
        )
        return Measurement(value=value, unit=unit)
