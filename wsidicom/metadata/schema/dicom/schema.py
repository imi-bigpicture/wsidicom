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

"""Base DICOM schemas."""

import datetime
import logging
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    TypeVar,
)

from marshmallow import ValidationError, post_dump, pre_load
from pydicom import Dataset
from pydicom.datadict import dictionary_VR
from pydicom.dataelem import DataElement
from pydicom.sr.coding import Code
from pydicom.tag import Tag

from wsidicom.conceptcode import dataset_to_code
from wsidicom.config import dicom_validation_mode
from wsidicom.metadata.sample import Measurement
from wsidicom.metadata.schema.common import LoadingSchema, LoadType
from wsidicom.metadata.schema.dicom.fields import (
    AttributeDicomField,
    ContentItemDicomField,
    FlattenOnDumpNestedDicomField,
)

logger = logging.getLogger(__name__)

DumpType = TypeVar("DumpType", Dataset, Iterable[Dataset])


class BaseDicomSchema(LoadingSchema[LoadType], Generic[LoadType, DumpType]):
    """Base class for DICOM schemas that dumps and loads to specified type."""

    @property
    @abstractmethod
    def dump_type(self) -> type[DumpType]:
        raise NotImplementedError()

    def load(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, data: DumpType, **kwargs
    ) -> LoadType:
        """Load object from DumpType."""
        item = super().load(data, **kwargs)  # type: ignore
        assert isinstance(item, self.load_type)
        return item

    def dump(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, obj: LoadType, **kwargs
    ) -> DumpType:
        """Dump object to DumpType."""
        dumped = super().dump(obj, **kwargs)
        assert isinstance(dumped, self.dump_type)
        return dumped


class DicomSchema(BaseDicomSchema[LoadType, Dataset]):
    """Base DICOM schema for attributes in a dataset."""

    @property
    def dump_type(self) -> type[Dataset]:
        return Dataset

    @post_dump
    def post_dump(self, data: dict[str, Any], many: bool, **kwargs) -> Dataset:
        """Create pydicom Dataset from attributes in dictionary."""
        for field in self.fields.values():
            if isinstance(field, FlattenOnDumpNestedDicomField):
                # Flatten nested fields into data
                field.flatten(data)
            if (
                field.data_key in data
                and data[field.data_key] is None
                and not (
                    isinstance(field, AttributeDicomField) and field.writes_when_empty
                )
            ):
                # Remove empty non-defaulting fields
                data.pop(field.data_key)
        dataset = Dataset()
        for key, value in data.items():
            try:
                if isinstance(value, DataElement):
                    # Flattened in from a nested schema, which made the element
                    # and checked the value with the value representation its
                    # own field states.
                    dataset[value.tag] = value
                else:
                    dataset[Tag(key)] = self._data_element(key, value)
            except ValueError as exception:
                # A value that does not conform to its value representation,
                # raised as what it is so it can be caught as such.
                raise ValueError(
                    f"Failed to set {key} of dataset to {value}."
                ) from exception
            except Exception as exception:
                raise Exception(
                    f"Failed to set {key} of dataset to {value}."
                ) from exception
        return dataset

    def _data_element(self, keyword: str, value: Any) -> DataElement:
        """Make the element holding a value, checking the value as it is made.

        For a value from a field that writes no element of its own: a field
        that does states how what it writes is written, and gives the element.
        The mode is passed for the value, so pydicom's global validation mode
        is never set and a process using wsidicom keeps the mode it chose.
        """
        tag = Tag(keyword)
        return DataElement(
            tag,
            dictionary_VR(tag),
            value,
            validation_mode=dicom_validation_mode(),
        )

    @pre_load
    def pre_load(self, dataset: Dataset, many: bool, **kwargs) -> dict[str, Any]:
        """Return dictionary of attributes from dataset."""
        attributes = {}
        for key, field in self.fields.items():
            if field.dump_only:
                continue
            if field.data_key is not None and field.data_key in dataset:
                attributes[field.data_key] = dataset.get(field.data_key)
            elif isinstance(field, FlattenOnDumpNestedDicomField):
                # De-flatten nested fields from dataset
                de_flattened = field.de_flatten(dataset)
                if de_flattened is not None:
                    attributes[key] = de_flattened
        return attributes


class ModuleDicomSchema(DicomSchema[LoadType]):
    """Base DICOM schema for a module, returning a default when failing to load."""

    @property
    @abstractmethod
    def module_name(self) -> str:
        raise NotImplementedError()

    def load(self, data: Dataset, **kwargs) -> LoadType:
        """Load dataset to LoadType. Return default LoadType if validation error."""
        try:
            return super().load(data, **kwargs)  # type: ignore
        except ValidationError:
            logger.warning(
                f"Failed to load module {self.module_name} with schema {self}.",
                exc_info=True,
            )
            return self.load_type()


@dataclass(frozen=True)
class ItemField:
    name: Code
    value_types: tuple[type, ...]
    many: bool


class ItemSequenceDicomSchema(BaseDicomSchema[LoadType, Iterable[Dataset]]):
    """Base DICOM schema for sequence of content items (each a dataset)."""

    _dump_only_fields: list[str] = []

    @property
    @abstractmethod
    def load_type(self) -> type[LoadType]:
        raise NotImplementedError()

    @property
    def dump_type(self) -> type[Iterable[Dataset]]:
        return list

    @property
    @abstractmethod
    def item_fields(self) -> dict[str, ItemField]:
        """Describe the fields in the schema.

        Fields should be ordered as in TID if applicable. The key is the python name of
        the field, and the value is a ItemField with the DICOM code name of the field,
        the allowed value types (tuple of one or more types), and if the field can
        hold multiple values (e.g. is a list)."""
        raise NotImplementedError()

    @post_dump
    def post_dump(
        self, data: dict[str, Dataset | Sequence[Dataset]], many: bool, **kwargs
    ) -> list[Dataset]:
        """Format content items into sequence in a dataset."""
        return [
            self._item_field(key).name_item(flatten_item, description.name)
            for key, description in self.item_fields.items()
            if (item := data[key]) is not None
            for flatten_item in ([item] if isinstance(item, Dataset) else item)
        ]

    def _item_field(self, key: str) -> ContentItemDicomField:
        """The field making the items of `key`.

        Every item of the sequence is made by a field that can also name it: a
        field that cannot is one this schema cannot write, rather than one
        whose items are left out.
        """
        field = self.fields.get(key)
        if not isinstance(field, ContentItemDicomField):
            raise TypeError(
                f"{type(self).__name__} writes {key} as items of a sequence, so "
                f"the field for it has to make items, and "
                f"{type(field).__name__} does not."
            )
        return field

    @pre_load
    def pre_load(
        self, sequence: Sequence[Dataset], many: bool, **kwargs
    ) -> dict[str, Any]:
        """Parse the sequence items from a dataset into a dictionary."""
        data = {
            key: self._get_item(sequence, description)
            for key, description in self.item_fields.items()
        }
        for field in self._dump_only_fields:
            data.pop(field)
        return data

    def _get_item(
        self, sequence: Iterable[Dataset], field: ItemField
    ) -> Dataset | list[Dataset] | None:
        """Get item dataset from dataset content item sequence.

        Parameters
        ----------
        dataset: Dataset
            Dataset to get item from.
        field:
            Description of the field to get.

        Returns
        -------
        Dataset | list[Dataset] | None
            Item dataset or datasets or None if not found.
        """
        items = (
            item
            for item in sequence
            if dataset_to_code(item.ConceptNameCodeSequence[0]) == field.name
            and self.dataset_to_type(item) in field.value_types
        )
        if field.many:
            return list(items)
        return next(items, None)

    @staticmethod
    def dataset_to_type(dataset: Dataset) -> type:
        value_type = dataset.ValueType
        if value_type == "CODE":
            return Code
        if value_type == "TEXT":
            return str
        if value_type == "DATETIME":
            return datetime.datetime
        if value_type == "NUMERIC":
            return Measurement
        raise NotImplementedError(
            f"Not implemented type-handling for dataset {dataset}."
        )
