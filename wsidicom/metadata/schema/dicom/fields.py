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
import logging
import math
import re
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from enum import Enum, auto
from typing import (
    Any,
    Final,
    Generic,
    TypeVar,
)

from marshmallow import (
    Schema,
    ValidationError,
    fields,
    missing,
)
from marshmallow.types import StrSequenceOrSet
from pydicom import DataElement, Dataset
from pydicom import Sequence as DicomSequence
from pydicom.config import RAISE
from pydicom.datadict import (
    dictionary_VR,
    get_entry,
    keyword_for_tag,
    tag_for_keyword,
)
from pydicom.multival import MultiValue
from pydicom.sr.coding import Code
from pydicom.tag import BaseTag, Tag
from pydicom.uid import UID
from pydicom.valuerep import (
    DA,
    DT,
    MAX_VALUE_LEN,
    STR_VR,
    TM,
    VR,
    DSfloat,
    PersonName,
    validate_value,
    validate_vr_length,
)

from wsidicom.conceptcode import CodeType, UnitCode
from wsidicom.config import dicom_validation_mode, get_settings
from wsidicom.geometry import Orientation, PointMm, SizeMm
from wsidicom.metadata.sample import (
    IssuerOfIdentifier,
    LocalIssuerOfIdentifier,
    Measurement,
    UniversalIssuerOfIdentifier,
    UniversalIssuerType,
)
from wsidicom.options import DicomValueValidationOption
from wsidicom.tags import (
    CodeMeaningTag,
    CodeValueTag,
    CodingSchemeDesignatorTag,
    CodingSchemeVersionTag,
    ConceptCodeSequenceTag,
    ConceptNameCodeSequenceTag,
    DateTimeTag,
    FloatingPointValueTag,
    LocalNamespaceEntityIDTag,
    MeasurementUnitsCodeSequenceTag,
    NumericValueTag,
    TextValueTag,
    UniversalEntityIDTag,
    UniversalEntityIDTypeTag,
    ValueTypeTag,
    XOffsetInSlideCoordinateSystemTag,
    YOffsetInSlideCoordinateSystemTag,
    ZOffsetInSlideCoordinateSystemTag,
)

logger = logging.getLogger(__name__)

ValueType = TypeVar("ValueType")


class _Empty(Enum):
    """The type of `EMPTY`, which has the one value and stands for no value.

    A type of its own rather than a value of the type an attribute holds, so
    that asking for an attribute with no value cannot be mistaken for asking
    for one holding a value that happens to look like it.
    """

    EMPTY = auto()


EMPTY: Final = _Empty.EMPTY
"""Written as the value of an attribute that is to be there holding nothing."""


class AttributeDicomField(fields.Field[ValueType], Generic[ValueType]):
    """Base for a field that writes one DICOM attribute.

    What a field writes is the field's to get right, so it checks the value
    against how the attribute is written rather than leaving it to the dataset
    the value ends up in. How it is written is the attribute's own when the
    field writes a named one, and stated by the field when it does not: a field
    making part of what another field writes has no attribute of its own.

    An attribute that has to be present is written with the value given as
    `default_if_none` when the object has none of its own, and written empty
    when that is `EMPTY`. A field given no default leaves the attribute out of
    the dataset when the object has no value for it. The default takes the
    place of the value as the value is read, and not as it is written, so that
    it also reaches a field that makes nothing of a value it does not have.
    """

    def __init__(
        self,
        *args,
        value_representation: str | None = None,
        default_if_none: ValueType | _Empty | None = None,
        **kwargs,
    ):
        """Make a field writing one DICOM attribute.

        Parameters
        ----------
        value_representation: str | None = None
            How what this writes is written. If not given derived from the data_key.
        default_if_none: ValueType | _Empty | None = None
            Value to write when the object has none of its own, or `EMPTY` to
            write the attribute with no value. None, as by default, leaves the
            attribute out of the dataset.
        """
        self._default_if_none = default_if_none
        super().__init__(*args, **kwargs)
        self._value_representation = self._resolve_value_representation(
            value_representation
        )
        self._value_multiplicity = self._resolve_value_multiplicity()

    def _resolve_value_multiplicity(self) -> tuple[str, int, int | None] | None:
        """How many values the attribute this writes holds.

        As the multiplicity states it, the fewest it holds and the most, which
        is None when it holds as many as are given. None altogether when this
        writes no named attribute, or when the multiplicity is given as
        multiples of a number, which is not something to check a count against.
        """
        if not isinstance(self.data_key, str):
            return None
        tag = tag_for_keyword(self.data_key)
        if tag is None:
            return None
        stated = get_entry(Tag(tag))[1]
        held = re.fullmatch(r"(\d+)(?:-(\d+|n))?", stated)
        if held is None:
            return None
        fewest = int(held.group(1))
        most = held.group(2)
        if most is None:
            return stated, fewest, fewest
        return stated, fewest, None if most == "n" else int(most)

    def _resolve_value_representation(self, stated_by_the_field: str | None) -> str:
        """How what this writes is written, settled as the field is made.

        The attribute's own when this writes a named one, and what the field
        was made with when it does not: a field making part of what another
        field writes has no attribute to take it from. A field that has neither
        cannot say how what it writes is written, and is not a field.

        Raises
        ------
        ValueError
            If neither is there, if the attribute has more than one value
            representation and none was stated, or if what was stated is not
            what the attribute is written as.
        """
        tag = tag_for_keyword(self.data_key) if isinstance(self.data_key, str) else None
        if tag is None:
            if stated_by_the_field is None:
                raise ValueError(
                    f"{type(self).__name__} writes neither a named DICOM "
                    "attribute nor a stated value representation, so how what "
                    "it writes is written cannot be known. Give it a data_key "
                    "or a value_representation."
                )
            return stated_by_the_field
        of_the_attribute = dictionary_VR(Tag(tag))
        if " or " in of_the_attribute:
            if stated_by_the_field is None:
                raise ValueError(
                    f"{self.data_key} is written as {of_the_attribute}, so "
                    "which of them has to be stated."
                )
            return stated_by_the_field
        if stated_by_the_field is not None and stated_by_the_field != of_the_attribute:
            raise ValueError(
                f"{self.data_key} is written as {of_the_attribute}, "
                f"but {stated_by_the_field} was stated."
            )
        return of_the_attribute

    @property
    def writes_when_empty(self) -> bool:
        """Whether the attribute is written even when there is no value."""
        return self._default_if_none is not None

    def _is_empty(self, value: Any) -> bool:
        """Whether a value counts as no value for the attribute."""
        return value is None or value is missing

    def get_value(
        self, obj: Any, attr: str, accessor: Any = None, default: Any = missing
    ) -> Any:
        """The value to write, which is the default when the object has none."""
        value = super().get_value(obj, attr, accessor=accessor, default=default)
        if not self.writes_when_empty or not self._is_empty(value):
            return value
        # An attribute written empty is written with no value at all, which is
        # what None is once the value is being written rather than given.
        return None if self._default_if_none is EMPTY else self._default_if_none

    @property
    def value_representation(self) -> str:
        """How what this writes is written."""
        return self._value_representation

    def _check_how_many(self, given: int) -> None:
        """Check how many values were given against how many the attribute holds.

        Raises
        ------
        ValueError
            If the attribute holds fewer or more values than were given.
        """
        if self._value_multiplicity is None:
            return
        stated, fewest, most = self._value_multiplicity
        if given < fewest or (most is not None and given > most):
            raise ValueError(
                f"{self.data_key} holds {stated} values, but was given {given}."
            )

    def _validate_written_value(self, value: Any) -> None:
        """Check a value against how what this writes is written.

        For a field to check what it writes itself, so that the field naming
        the attribute is the one that refuses the value. A field that writes no
        named attribute checks nothing: what it makes is checked by whatever
        writes it.

        Raises
        ------
        ValueError
            If the value does not conform to its value representation.
        """
        if get_settings().dicom_value_validation is DicomValueValidationOption.NONE:
            return
        if value is None or value is missing:
            return
        values = value if isinstance(value, (list, tuple, MultiValue)) else [value]
        self._check_how_many(len(values))
        # A validator takes one value at a time, so the values of an attribute
        # holding several are checked one by one: handing it all of them is
        # itself reported as not a value the attribute can hold.
        for one in values:
            # The validators take the value as it is given to an element, which
            # for a value representation stored as text is the text: `IS` and
            # `DS` are held as `IS` and `DSfloat`, which they do not take.
            if one is not None and self.value_representation in STR_VR:
                one = str(one)
            validate_value(self.value_representation, one, RAISE)


class DatasetDicomField(fields.Field[ValueType], Generic[ValueType]):
    """Base for a field that builds the dataset of a sequence item.

    Not an `AttributeDicomField`: what it makes is a dataset rather than the
    value of one attribute, so it writes no attribute of its own and has
    neither a tag nor a value representation.

    What each field makes of a value is what it writes into the dataset, which
    is made here: the attributes of it are set through `_set_attribute`, so
    that a value is checked as it is set. Setting an attribute of a dataset
    directly hands the value to pydicom, which makes the element with its
    global validation mode and gives no say in how the value is checked.
    """

    @staticmethod
    def _set_attribute(
        dataset: Dataset,
        keyword: str | BaseTag,
        value: Any,
        value_representation: str | None = None,
    ) -> None:
        """Set an attribute of a dataset, checking the value as it is set.

        Set as the element holding the value: setting it by attribute hands the
        value to pydicom, which makes the element itself with its global
        validation mode and gives no say in how the value is checked.

        Parameters
        ----------
        dataset: Dataset
            Dataset to set the attribute of.
        keyword: str | BaseTag
            Keyword or tag of the attribute to set.
        value: Any
            Value to set it to.
        value_representation: str | None = None
            Which value representation to write an attribute that has more than
            one as. Only for those: an attribute with one of its own is written
            as that one, and saying otherwise writes a value it cannot hold.

        Raises
        ------
        ValueError
            If the attribute has more than one value representation and none
            was given, if one was given for an attribute that has a single one,
            or if the value does not conform to the one it is set as.
        """
        tag = Tag(keyword)
        name = keyword_for_tag(tag)
        stated = dictionary_VR(tag)
        if " or " not in stated:
            if value_representation is not None:
                raise ValueError(
                    f"{name} is written as {stated}, which is not for the "
                    f"caller to decide, but {value_representation} was given."
                )
            value_representation = stated
        elif value_representation is None:
            raise ValueError(
                f"{name} is written as {stated}, so which of them has to be given."
            )
        dataset[tag] = DataElement(
            tag,
            value_representation,
            value,
            validation_mode=dicom_validation_mode(),
        )


class StringDicomField(AttributeDicomField[str]):
    def __init__(self, value_representation: VR, **kwargs):
        if value_representation not in STR_VR:
            raise ValueError(
                f"Value representation {value_representation} is not a string."
            )
        super().__init__(value_representation=value_representation, **kwargs)

    def _serialize(self, value: str | None, attr: str | None, obj: Any, **kwargs):
        if value is None:
            return None
        value_representation = self.value_representation
        if value_representation == VR.CS:
            value = re.sub(r"[^A-Z0-9 _]", "_", value.upper()).strip()
        valid, _ = validate_vr_length(value_representation, value)
        if not valid and get_settings().truncate_long_dicom_strings_on_validation_error:
            maximum_allowed_length = MAX_VALUE_LEN[value_representation]
            logger.warning(
                f"Truncating long DICOM string {value} of value representation "
                f"{value_representation} with maximum allowed length "
                f"{maximum_allowed_length} to {value[:maximum_allowed_length]}."
            )
            value = value[:maximum_allowed_length]
        self._validate_written_value(value)
        return super()._serialize(value, attr, obj, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs) -> Any:
        """By default pydicom returns empty string for empty string-like elements."""
        deserialized = super()._deserialize(value, attr, data, **kwargs)
        if deserialized == "":
            return None
        return deserialized


class EnumDicomField(AttributeDicomField, fields.Enum):
    def _deserialize(self, value, attr, data, **kwargs):
        if value == "":
            return None
        return super()._deserialize(value, attr, data, **kwargs)


class IntegerDicomField(AttributeDicomField[int], fields.Integer):
    """Field for an attribute holding an integer."""


class DateTimeDicomField(AttributeDicomField[datetime.datetime]):
    def _serialize(
        self,
        value: datetime.datetime | None,
        attr: str | None,
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


class DateDicomField(AttributeDicomField[datetime.date]):
    def _serialize(
        self, value: datetime.date | None, attr: str | None, obj: Any, **kwargs
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


class TimeDicomField(AttributeDicomField[datetime.time]):
    def _serialize(
        self, value: datetime.time | None, attr: str | None, obj: Any, **kwargs
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


class BooleanDicomField(AttributeDicomField[bool], fields.Boolean):
    def __init__(
        self,
        truthy: str | None = None,
        falsy: str | None = None,
        **kwargs,
    ):
        if truthy is None:
            truthy = "YES"
        if falsy is None:
            falsy = "NO"
        super().__init__(truthy=set([truthy]), falsy=set([falsy]), **kwargs)

    def _serialize(self, value: bool | None, attr: str | None, obj: Any, **kwargs):
        string_value = self.truthy if value else self.falsy
        return list(string_value)[0]

    def _deserialize(self, value, attr, data, **kwargs) -> Any:
        deserialized = super()._deserialize(value, attr, data, **kwargs)
        if deserialized == "":
            return None
        return deserialized


class OffsetInSlideCoordinateSystemDicomField(DatasetDicomField):
    def _serialize(
        self,
        value: tuple[PointMm, float | None] | None,
        attr: str | None,
        obj: Any,
        **kwargs,
    ):
        origin = value
        if origin is None:
            if self.dump_default is None:
                return None
            xy_origin = self.dump_default[0]
            z_origin = self.dump_default[1]
            assert isinstance(xy_origin, PointMm)
            assert z_origin is None or isinstance(z_origin, float)
        else:
            xy_origin = origin[0]
            z_origin = origin[1]
        origin_element = Dataset()
        self._set_attribute(
            origin_element,
            XOffsetInSlideCoordinateSystemTag,
            DSfloat(xy_origin.x, True),
        )
        self._set_attribute(
            origin_element,
            YOffsetInSlideCoordinateSystemTag,
            DSfloat(xy_origin.y, True),
        )
        if z_origin is not None:
            self._set_attribute(
                origin_element,
                ZOffsetInSlideCoordinateSystemTag,
                DSfloat(z_origin, True),
            )
        return [origin_element]

    def _deserialize(
        self,
        value: DataElement,
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ) -> tuple[PointMm, float | None]:
        z_offset = value[0].get(ZOffsetInSlideCoordinateSystemTag, None)
        return PointMm(
            x=value[0][XOffsetInSlideCoordinateSystemTag].value,
            y=value[0][YOffsetInSlideCoordinateSystemTag].value,
        ), None if z_offset is None else z_offset.value


class ImageOrientationSlideDicomField(AttributeDicomField[float]):
    def _serialize(self, value: float | None, attr: str | None, obj: Any, **kwargs):
        rotation = value
        if rotation is None:
            if self.dump_default is None:
                return None
            assert isinstance(self.dump_default, (int, float))
            rotation = self.dump_default
        x = round(math.sin(rotation * math.pi / 180), 8)
        y = round(math.cos(rotation * math.pi / 180), 8)
        orientation = [-x, y, 0, y, x, 0]
        self._validate_written_value(orientation)
        return orientation

    def _deserialize(
        self,
        value: tuple[float, float, float, float, float, float],
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ) -> float:
        orientation = Orientation(value)
        return orientation.rotation


class ListDicomField(AttributeDicomField, fields.List):
    """Wrapper around normal list that handles single-valued lists from pydicom.

    Set ``dump_required=True`` for Type-1 list attributes whose value
    multiplicity is at least 1: dumping a `None` or empty list then raises a
    `ValidationError` with a structural message. The load path is unaffected.
    """

    def __init__(
        self,
        cls_or_instance: fields.Field | type,
        dump_none_if_empty: bool = False,
        *,
        dump_required: bool = False,
        **kwargs,
    ):
        self._dump_none_if_empty = dump_none_if_empty
        self._dump_required = dump_required
        super().__init__(cls_or_instance, **kwargs)

    def _is_empty(self, value: Any) -> bool:
        """Whether a value counts as no value, which a list without items does."""
        return super()._is_empty(value) or len(value) == 0

    def _serialize(self, value, attr, obj, **kwargs) -> Any:
        if self._dump_required and not value:
            raise ValidationError(
                f"{self.data_key or attr} is a Type 1 DICOM attribute and "
                "must contain at least one item; populate the metadata "
                "field before dumping."
            )
        if self._dump_none_if_empty and value is None or len(value) == 0:
            return None
        return super()._serialize(value, attr, obj, **kwargs)

    def _deserialize(self, value: Any | list[Any], attr, data, **kwargs) -> list[Any]:
        if not isinstance(value, (MultiValue, DicomSequence)):
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

    def de_flatten(self, dataset: Dataset) -> Dataset | None:
        """Create new dataset containing the attributes defined in nested schema."""
        nested = Dataset()
        for nested_field in self.nested_schema.fields.values():
            if nested_field.dump_only:
                continue
            if isinstance(nested_field, FlattenOnDumpNestedDicomField):
                de_flatten_nested_field = nested_field.de_flatten(dataset)
                if de_flatten_nested_field is not None:
                    for element in de_flatten_nested_field:
                        nested.add(element)
            elif nested_field.data_key is not None and nested_field.data_key in dataset:
                # The element as it was read, not the value set again: what was
                # read is not wsidicom's to write, so it is neither remade nor
                # checked against how the attribute is written.
                tag = Tag(nested_field.data_key)
                nested[tag] = dataset[tag]
        if len(nested) == 0:
            return None
        return nested

    def flatten(self, data: dict[str, Any]):
        """Insert attributes from nested dataset into data."""
        key = self.name
        if self.data_key is not None:
            key = self.data_key
        assert key is not None
        nested = data.pop(key, None)
        if isinstance(nested, Dataset):
            for nested_key, nested_value in nested.items():
                data[nested_key] = nested_value  # type: ignore

    def _serialize(self, nested_obj, attr: str | None, obj: Any, **kwargs):
        if nested_obj is None and self.dump_default != missing:
            nested_obj = self.dump_default
        return super()._serialize(nested_obj, attr, obj, **kwargs)


class FloatDicomField(AttributeDicomField[float]):
    def _deserialize(self, value: Any, attr, data, **kwargs) -> float:
        return float(value)

    def _serialize(self, value: float | None, attr: str | None, obj: Any, **kwargs):
        if value is None:
            return None
        # A float of enough digits makes a decimal string too long to be one,
        # which `DSfloat` gives without complaint, so it is checked here.
        serialized = DSfloat(value)
        # A float of enough digits makes a decimal string too long to be one,
        # which raises OverflowError when the element is made, and not the
        # ValueError everything else raises.
        self._validate_written_value(serialized)
        return serialized


def _scheme_version_of(dataset: Dataset) -> str | None:
    """The coding scheme version a code dataset states, or None if it states none.

    A version that is there but empty is no version, the same as one that is
    not there at all.
    """
    version = dataset.get(CodingSchemeVersionTag, None)
    if version is None or version.value == "":
        return None
    return version.value


class CodeDicomField(DatasetDicomField[CodeType], Generic[CodeType]):
    """Field for a DICOM `code sequence`.

    The dataset of the code is built here rather than by a schema of its own,
    so that a code is written the one way wherever one is written.
    """

    _held = (
        ("value", CodeValueTag),
        ("scheme_designator", CodingSchemeDesignatorTag),
        ("meaning", CodeMeaningTag),
        ("scheme_version", CodingSchemeVersionTag),
    )

    def __init__(self, load_type: type[CodeType], **kwargs) -> None:
        self._load_type = load_type
        super().__init__(**kwargs)

    def _serialize(
        self, value: CodeType | None, attr: str | None, obj: Any, **kwargs
    ) -> Dataset | None:
        if value is None:
            return None
        dataset = Dataset()
        self._set_attribute(dataset, CodeValueTag, value.value)
        self._set_attribute(dataset, CodingSchemeDesignatorTag, value.scheme_designator)
        self._set_attribute(dataset, CodeMeaningTag, value.meaning)
        # A scheme that states no version says so by not being written, and an
        # empty version is no version.
        if value.scheme_version is not None and value.scheme_version != "":
            self._set_attribute(dataset, CodingSchemeVersionTag, value.scheme_version)
        return dataset

    def _deserialize(
        self, value: Dataset, attr: str | None, data: Any, **kwargs
    ) -> CodeType:
        held = {name: value.get(tag, None) for name, tag in self._held}
        return self._load_type(
            **{
                # An attribute that is there but empty holds nothing, which is
                # not something the code can be made with.
                name: element.value
                for name, element in held.items()
                if element is not None and element.value != ""
            }
        )


class SingleCodeSequenceDicomField(AttributeDicomField[CodeType], Generic[CodeType]):
    """Field for a DICOM `code sequence` that can only contain one code.

    Holds the field that makes the code rather than being one: what this writes
    is the sequence attribute, and what a `CodeDicomField` makes is the item in
    it. The same way a sequence of several codes is written, which is a list
    field holding one of these.
    """

    def __init__(self, load_type: type[CodeType], **kwargs) -> None:
        self._item = CodeDicomField(load_type)
        super().__init__(**kwargs)

    def _serialize(
        self, value: CodeType | None, attr: str | None, obj: Any, **kwargs
    ) -> list[Dataset] | None:
        """The sequence holding the one code, as the attribute holds a sequence."""
        item = self._item._serialize(value, attr, obj, **kwargs)
        if item is None:
            return None
        items = [item]
        self._validate_written_value(items)
        return items

    def _deserialize(
        self,
        value: Sequence[Dataset],
        attr: str | None,
        data: Mapping[str, Any] | None,
        **kwargs,
    ) -> CodeType:
        return self._item._deserialize(value[0], attr, data, **kwargs)


class UidDicomField(StringDicomField):
    """DICOM UI value-representation field.

    Set ``dump_required=True`` for Type-1 attributes: dumping a `None` value
    then raises a `ValidationError` with a structural message. The load path
    is unaffected — use ``allow_none=True`` independently to control whether
    a missing/None value is accepted at deserialization.
    """

    def __init__(self, *, dump_required: bool = False, **kwargs):
        super().__init__(value_representation=VR.UI, **kwargs)
        self._dump_required = dump_required

    def _serialize(self, value: Any, attr: str | None, obj: Any, **kwargs):
        if value is None and self._dump_required:
            raise ValidationError(
                f"{self.data_key or attr} is a Type 1 DICOM attribute and "
                "must be set; populate the metadata field before dumping."
            )
        return super()._serialize(value, attr, obj, **kwargs)

    def _deserialize(self, value: Any, attr, data, **kwargs):
        if value is None or value == "":
            return None
        if isinstance(value, UID):
            return value
        return UID(value)


class UidDatasetDicomField(DatasetDicomField[UID | None]):
    """Field for a sequence item holding nothing but a UID.

    The data key is the attribute of the item this makes, not one of the
    dataset the item is in: the field around this one writes the sequence the
    item belongs to.
    """

    def __init__(self, data_key: str, **kwargs):
        self._data_key = data_key
        super().__init__(data_key=data_key, **kwargs)

    def _serialize(
        self, value: UID | None, attr: str | None, obj: Any, **kwargs
    ) -> Dataset | None:
        if value is None:
            return None
        dataset = Dataset()
        self._set_attribute(dataset, self._data_key, value)
        return dataset

    def _deserialize(self, value: Dataset, attr, data, **kwargs) -> UID | None:
        held = value.get(self._data_key, None)
        if held is None or held == "":
            return None
        return held if isinstance(held, UID) else UID(held)


class PersonNameDicomField(AttributeDicomField[str]):
    def _deserialize(self, value: PersonName, attr, data, **kwargs) -> str:
        return str(value)

    def _serialize(self, value: str | None, attr, obj, **kwargs) -> PersonName | None:
        if value is None:
            return None
        # A component longer than a person name allows is only warned about
        # when the name is made, so it is checked here.
        serialized = PersonName(value)
        # A component longer than a person name allows is governed by pydicom's
        # writing validation mode, which making the element does not consult,
        # so making the element would let it through.
        self._validate_written_value(serialized)
        return serialized


class IssuerOfIdentifierDicomField(DatasetDicomField):
    def _deserialize(
        self, value: Sequence[Dataset] | None, attr, data, **kwargs
    ) -> IssuerOfIdentifier | None:
        if value is None or len(value) == 0:
            return None
        dataset = value[0]
        if UniversalEntityIDTypeTag in dataset:
            local_identifier = dataset.get(LocalNamespaceEntityIDTag, None)
            return UniversalIssuerOfIdentifier(
                dataset.UniversalEntityID,
                UniversalIssuerType(dataset.UniversalEntityIDType),
                None if local_identifier is None else local_identifier.value,
            )
        if LocalNamespaceEntityIDTag in dataset:
            return LocalIssuerOfIdentifier(dataset.LocalNamespaceEntityID)
        return None

    def _serialize(
        self, value: IssuerOfIdentifier | None, attr, obj, **kwargs
    ) -> Sequence[Dataset] | None:
        # An issuer that is not there is written as an empty sequence rather
        # than left out, so this cannot take the guard that gives nothing.
        if value is None:
            return []
        dataset = Dataset()
        if isinstance(value, UniversalIssuerOfIdentifier):
            self._set_attribute(dataset, UniversalEntityIDTag, value.identifier)
            self._set_attribute(
                dataset, UniversalEntityIDTypeTag, value.issuer_type.name
            )
            if value.local_identifier is not None:
                self._set_attribute(
                    dataset, LocalNamespaceEntityIDTag, value.local_identifier
                )
        elif isinstance(value, LocalIssuerOfIdentifier):
            self._set_attribute(dataset, LocalNamespaceEntityIDTag, value.identifier)
        else:
            raise NotImplementedError()
        return [dataset]


class PixelSpacingDicomField(AttributeDicomField[SizeMm | None]):
    def _serialize(self, value: SizeMm | None, attr: str | None, obj: Any, **kwargs):
        if value is None:
            return None
        spacing = [DSfloat(value.width, True), DSfloat(value.height, True)]
        self._validate_written_value(spacing)
        return spacing

    def _deserialize(
        self, value: Sequence[DSfloat] | None, attr, data, **kwargs
    ) -> SizeMm | None:
        if value is None or len(value) == 0:
            return None
        return SizeMm(value[0], value[1])


class NestedDatasetDicomField(DatasetDicomField[ValueType], fields.Nested):
    """fields.Field for attribute of a single-item dataset sequence with a nested
    sing-item dataset sequence with the item the nested schema should handle."""

    def __init__(self, nested: Schema, data_key: str, nested_data_key: str, **kwargs):
        self._nested = nested
        self._data_key = data_key
        self._nested_data_key = nested_data_key
        super().__init__(nested=nested, data_key=data_key, **kwargs)

    # Makes the sequence holding the one item it builds.
    # `Field` names this parameter `value` and `Nested` names it `nested_obj`,
    # and this is both, so whichever name is taken one of them disagrees.
    def _serialize(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, nested_obj: ValueType | None, attr: str | None, obj: Any, **kwargs
    ) -> list[Dataset] | None:
        nested_value = super()._serialize(nested_obj, attr, obj, **kwargs)
        if nested_value is None:
            return None
        dataset = Dataset()
        self._set_attribute(dataset, self._nested_data_key, [nested_value])
        return [dataset]

    def _deserialize(
        self,
        value: Sequence[Dataset],
        attr,
        data,
        partial: StrSequenceOrSet | bool | None = None,
        **kwargs,
    ) -> Any:
        nested_value = getattr(value[0], self._nested_data_key, None)
        if nested_value is None:
            return None
        return super()._deserialize(nested_value[0], attr, data, partial, **kwargs)


class ContentItemDicomField(DatasetDicomField[ValueType]):
    def name_item(self, item: Dataset, name: Code) -> Dataset:
        """Name an item this made, so what it holds is known by what names it."""
        name_dataset = Dataset()
        self._set_attribute(name_dataset, CodeValueTag, name.value)
        self._set_attribute(
            name_dataset, CodingSchemeDesignatorTag, name.scheme_designator
        )
        self._set_attribute(name_dataset, CodeMeaningTag, name.meaning)
        if name.scheme_version is not None and name.scheme_version != "":
            self._set_attribute(
                name_dataset, CodingSchemeVersionTag, name.scheme_version
            )
        self._set_attribute(item, ConceptNameCodeSequenceTag, [name_dataset])
        return item

    @abstractmethod
    def _deserialize(
        self, value: Dataset, attr: str | None, data: Any, **kwargs
    ) -> ValueType:
        """What the item holds. A field making several takes the several.

        An item that holds nothing gives None, so a field for one that need not
        be there is for a value that may be None.
        """
        raise NotImplementedError()


class CodeItemDicomField(ContentItemDicomField[CodeType], Generic[CodeType]):
    def __init__(self, load_type: type[CodeType], **kwargs) -> None:
        self._load_type = load_type
        super().__init__(**kwargs)

    def _serialize(
        self, value: CodeType | None, attr: str | None, obj: Any, **kwargs
    ) -> Dataset | None:
        if value is None:
            return None
        code_dataset = Dataset()
        self._set_attribute(code_dataset, CodeValueTag, value.value)
        self._set_attribute(
            code_dataset, CodingSchemeDesignatorTag, value.scheme_designator
        )
        self._set_attribute(code_dataset, CodeMeaningTag, value.meaning)
        if value.scheme_version is not None and value.scheme_version != "":
            self._set_attribute(
                code_dataset, CodingSchemeVersionTag, value.scheme_version
            )
        dataset = Dataset()
        self._set_attribute(dataset, ValueTypeTag, "CODE")
        self._set_attribute(dataset, ConceptCodeSequenceTag, [code_dataset])
        return dataset

    def _deserialize(
        self, value: Dataset, attr: str | None, data: Any, **kwargs
    ) -> CodeType:
        dataset = value
        version = _scheme_version_of(dataset.ConceptCodeSequence[0])
        return self._load_type(
            value=dataset.ConceptCodeSequence[0].CodeValue,
            scheme_designator=dataset.ConceptCodeSequence[0].CodingSchemeDesignator,
            meaning=dataset.ConceptCodeSequence[0].CodeMeaning,
            scheme_version=version,
        )


class StringItemDicomField(ContentItemDicomField[str | None]):
    def _serialize(
        self, value: str | None, attr: str | None, obj: Any, **kwargs
    ) -> Dataset | None:
        if value is None:
            return None
        dataset = Dataset()
        self._set_attribute(dataset, ValueTypeTag, "TEXT")
        self._set_attribute(dataset, TextValueTag, value)
        return dataset

    def _deserialize(self, value: Dataset, attr: str | None, data: Any, **kwargs):
        dataset = value
        return dataset.TextValue


class DateTimeItemDicomField(ContentItemDicomField[datetime.datetime | None]):
    def _serialize(
        self, value: datetime.datetime | None, attr: str | None, obj: Any, **kwargs
    ) -> Dataset | None:
        if value is None:
            return None
        dataset = Dataset()
        self._set_attribute(dataset, ValueTypeTag, "DATETIME")
        self._set_attribute(dataset, DateTimeTag, DT(value))
        return dataset

    def _deserialize(self, value: Dataset, attr: str | None, data: Any, **kwargs):
        dataset = value
        try:
            return DT(dataset.DateTime)
        except ValueError:
            return None


class MeasurementItemDicomField(ContentItemDicomField[Measurement | None]):
    def _serialize(
        self, value: Measurement | None, attr: str | None, obj: Any, **kwargs
    ) -> Dataset | None:
        if value is None:
            return None
        dataset = Dataset()
        self._set_attribute(dataset, ValueTypeTag, "NUMERIC")
        self._set_attribute(dataset, NumericValueTag, DSfloat(value.value))
        self._set_attribute(dataset, FloatingPointValueTag, value.value)
        unit_dataset = Dataset()
        self._set_attribute(unit_dataset, CodeValueTag, value.unit.value)
        self._set_attribute(
            unit_dataset, CodingSchemeDesignatorTag, value.unit.scheme_designator
        )
        self._set_attribute(unit_dataset, CodeMeaningTag, value.unit.meaning)
        if value.unit.scheme_version is not None and value.unit.scheme_version != "":
            self._set_attribute(
                unit_dataset, CodingSchemeVersionTag, value.unit.scheme_version
            )
        self._set_attribute(dataset, MeasurementUnitsCodeSequenceTag, [unit_dataset])
        return dataset

    def _deserialize(self, value: Dataset, attr: str | None, data: Any, **kwargs):
        dataset = value
        if FloatingPointValueTag in dataset:
            measurement_value = dataset.FloatingPointValue
        else:
            measurement_value = DSfloat(dataset.NumericValue)
            assert isinstance(measurement_value, float)
        unit_dataset = dataset.MeasurementUnitsCodeSequence[0]
        version = _scheme_version_of(unit_dataset)
        unit = UnitCode(
            value=unit_dataset.CodeValue,
            scheme_designator=unit_dataset.CodingSchemeDesignator,
            meaning=unit_dataset.CodeMeaning,
            scheme_version=version,
        )
        return Measurement(value=measurement_value, unit=unit)
