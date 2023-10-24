from abc import abstractmethod
from typing import Any, Dict, Generic, Type, TypeVar
from marshmallow import Schema, post_dump, pre_load, post_load
from pydicom import Dataset


from wsidicom.metadata.dicom_schema.fields import (
    FlatteningNestedField,
)

LoadType = TypeVar("LoadType")


class DicomSchema(Schema, Generic[LoadType]):
    def dump(self, obj: LoadType, **kwargs) -> Dataset:
        """Dump object to pydicom Dataset."""
        dataset = super().dump(obj, **kwargs)
        assert isinstance(dataset, Dataset)
        return dataset

    def load(self, dataset: Dataset, **kwargs) -> LoadType:
        """Load object from pydicom Dataset."""
        item = super().load(dataset, **kwargs)  # type: ignore
        assert isinstance(item, self.load_type)
        return item

    @property
    @abstractmethod
    def load_type(self) -> Type[LoadType]:
        raise NotImplementedError()

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

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs) -> LoadType:
        """Load object from dictionary."""
        return self.load_type(**data)
