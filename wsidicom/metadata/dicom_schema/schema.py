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

from abc import abstractmethod
from dataclasses import dataclass
import datetime
import logging
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from marshmallow import Schema, ValidationError, post_dump, pre_load, post_load
from pydicom import Dataset
from pydicom.sr.coding import Code
from wsidicom.conceptcode import dataset_to_code

from wsidicom.metadata.dicom_schema.fields import (
    FlatteningNestedField,
)
from wsidicom.metadata.sample import Measurement

LoadType = TypeVar("LoadType")
DumpType = TypeVar("DumpType", Dataset, Iterable[Dataset])


class BaseDicomSchema(Schema, Generic[LoadType, DumpType]):
    """Base class for DICOM schemas that dumps and loads to specified type."""

    @property
    @abstractmethod
    def load_type(self) -> Type[LoadType]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def dump_type(self) -> Type[DumpType]:
        raise NotImplementedError()

    def load(self, dataset: DumpType, **kwargs) -> LoadType:
        """Load object from DumpType."""
        item = super().load(dataset, **kwargs)  # type: ignore
        assert isinstance(item, self.load_type)
        return item

    def dump(self, obj: LoadType, **kwargs) -> DumpType:
        """Dump object to DumpType."""
        dumped = super().dump(obj, **kwargs)
        assert isinstance(dumped, self.dump_type)
        return dumped

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs) -> LoadType:
        """Load object from dictionary."""
        return self.load_type(**data)


class DicomSchema(BaseDicomSchema[LoadType, Dataset]):
    """Base DICOM schema for attributes in a dataset."""

    @property
    def dump_type(self) -> Type[Dataset]:
        return Dataset

    @post_dump
    def post_dump(self, data: Dict[str, Any], many: bool, **kwargs) -> Dataset:
        """Create pydicom Dataset from attributes in dictionary."""
        for field in self.fields.values():
            if isinstance(field, FlatteningNestedField):
                field.flatten(data)
        dataset = Dataset()
        dataset.update(data)  # type: ignore
        return dataset

    @pre_load
    def pre_load(self, dataset: Dataset, many: bool, **kwargs) -> Dict[str, Any]:
        """Return dictionary of attributes from dataset."""
        attributes = {}
        for key, field in self.fields.items():
            if field.dump_only:
                continue
            if field.data_key is not None and field.data_key in dataset:
                attributes[field.data_key] = dataset.get(field.data_key)
            elif isinstance(field, FlatteningNestedField):
                de_flattened = field.de_flatten(dataset)
                if de_flattened is not None:
                    attributes[key] = de_flattened
        return attributes


class DefaultIfValidationFailedDicomSchema(DicomSchema[LoadType]):
    def load(self, dataset: Dataset, **kwargs) -> LoadType:
        """Load dataset to LoadType. Return default LoadType if validation error."""
        try:
            return super().load(dataset, **kwargs)  # type: ignore
        except ValidationError:
            logging.warning(f"Failed to load item with schema {self}", exc_info=True)
            return self.load_type()


@dataclass
class ItemField:
    name: Code
    value_types: Tuple[Type, ...]
    many: bool


class ItemSequenceDicomSchema(BaseDicomSchema[LoadType, Iterable[Dataset]]):
    """Base DICOM schema for sequence of content items (each a dataset)."""

    _dump_only_fields: List[str] = []

    @property
    @abstractmethod
    def load_type(self) -> Type[LoadType]:
        raise NotImplementedError()

    @property
    def dump_type(self) -> Type[Iterable[Dataset]]:
        return list

    @property
    @abstractmethod
    def item_fields(self) -> Dict[str, ItemField]:
        """Describe the fields in the schema.

        Fields should be ordered as in TID if applicable. The key is the python name of
        the field, and the value is a ItemField with the DICOM code name of the field,
        the allowed value types (tuple of one or more types), and if the field can
        hold multiple values (e.g. is a list)."""
        raise NotImplementedError()

    @post_dump
    def post_dump(
        self, data: Dict[str, Union[Dataset, Sequence[Dataset]]], many: bool, **kwargs
    ) -> List[Dataset]:
        """Format content items into sequence in a dataset."""
        return [
            self._name_item(flatten_item, name)
            for item, name in [
                (data[key], description.name)
                for key, description in self.item_fields.items()
            ]
            if item is not None
            for flatten_item in ([item] if isinstance(item, Dataset) else item)
        ]

    @pre_load
    def pre_load(
        self, sequence: Sequence[Dataset], many: bool, **kwargs
    ) -> Dict[str, Any]:
        """Parse the sequence items from a dataset into a dictionary."""
        data = {
            key: self._get_item(sequence, description)
            for key, description in self.item_fields.items()
        }
        for field in self._dump_only_fields:
            data.pop(field)
        return data

    @staticmethod
    def _name_item(item: Dataset, name: Code):
        """Add concept name code sequence to dataset."""
        name_dataset = Dataset()
        name_dataset.CodeValue = name.value
        name_dataset.CodingSchemeDesignator = name.scheme_designator
        name_dataset.CodeMeaning = name.meaning
        name_dataset.CodingSchemeVersion = name.scheme_version
        item.ConceptNameCodeSequence = [name_dataset]
        return item

    def _get_item(
        self, sequence: Iterable[Dataset], field: ItemField
    ) -> Optional[Union[Dataset, List[Dataset]]]:
        """Get item dataset from dataset content item sequence.

        Parameters
        ----------
        dataset: Dataset
            Dataset to get item from.
        field:
            Description of the field to get.

        Returns
        -------
        Optional[Union[Dataset, List[Dataset]]]
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
    def dataset_to_type(dataset: Dataset) -> Type:
        if "ConceptCodeSequence" in dataset:
            return Code
        if "TextValue" in dataset:
            return str
        if "DateTime" in dataset:
            return datetime.datetime
        if "NumericValue" in dataset:
            return Measurement
        raise NotImplementedError()
