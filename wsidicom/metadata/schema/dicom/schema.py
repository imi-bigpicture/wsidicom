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
from pydicom.config import IGNORE
from pydicom.datadict import dictionary_VR
from pydicom.dataelem import DataElement
from pydicom.sr.coding import Code
from pydicom.tag import BaseTag, Tag

from wsidicom.conceptcode import dataset_to_code
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
    def post_dump(
        self, data: dict[str | BaseTag, Any], many: bool, **kwargs
    ) -> Dataset:
        """Create pydicom Dataset from attributes in dictionary.

        Keyed by the keyword of the attribute for what a field made of a
        value, and by the tag for an element flattened in from a nested
        schema, which is already made.
        """
        for field in self.fields.values():
            if isinstance(field, FlattenOnDumpNestedDicomField):
                self._flatten(data, field)
            data_key = field.data_key
            if (
                data_key is not None
                and data_key in data
                and data[data_key] is None
                and not (
                    isinstance(field, AttributeDicomField) and field.writes_when_empty
                )
            ):
                # Remove empty non-defaulting fields
                data.pop(data_key)
        dataset = Dataset()
        for key, value in data.items():
            try:
                if not isinstance(value, DataElement):
                    tag = Tag(key)
                    value = DataElement(
                        tag,
                        dictionary_VR(tag),
                        value,
                        validation_mode=IGNORE,
                    )
                dataset[value.tag] = value
            except ValueError as exception:
                # A value the attribute cannot hold at all, such as one of a
                # type it is never written as. What does not conform to the
                # value representation was refused by the field.
                raise ValueError(
                    f"Failed to set {key} of dataset to {value}."
                ) from exception
            except Exception as exception:
                raise Exception(
                    f"Failed to set {key} of dataset to {value}."
                ) from exception
        return dataset

    @classmethod
    def _flatten(
        cls,
        data: dict[str | BaseTag, Any], field: FlattenOnDumpNestedDicomField
    ) -> None:
        """Put what a nested schema made in with what this schema made.

        The nested schema made elements, which go in under their own tags: they
        are made already, and the schema that made them settled how each is
        written. What the field made is taken out, as the dataset holding them
        is not itself an attribute.
        """
        key = field.data_key if field.data_key is not None else field.name
        if key is None:
            return
        nested = data.pop(key, None)
        if not isinstance(nested, Dataset):
            return
        for element in nested:
            data[element.tag] = element

    @classmethod
    def _de_flatten(
        cls,
        dataset: Dataset, field: FlattenOnDumpNestedDicomField
    ) -> Dataset | None:
        """Take back out of a dataset what a nested schema is to be loaded from.

        Undoes `_flatten`, gathering the attributes the nested schema names into
        a dataset of their own. None when the dataset holds none of them, which
        is a nested schema with nothing to load.
        """
        nested = Dataset()
        for nested_field in field.nested_schema.fields.values():
            if nested_field.dump_only:
                continue
            if isinstance(nested_field, FlattenOnDumpNestedDicomField):
                de_flattened = cls._de_flatten(dataset, nested_field)
                if de_flattened is not None:
                    for element in de_flattened:
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
                de_flattened = self._de_flatten(dataset, field)
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
