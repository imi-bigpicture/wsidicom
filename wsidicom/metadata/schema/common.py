from abc import abstractmethod
from typing import Any, Dict, Generic, Type, TypeVar
from marshmallow import Schema, post_load

LoadType = TypeVar("LoadType")


class LoadingSchema(Schema, Generic[LoadType]):
    """Schema for loading metadata"""

    @property
    @abstractmethod
    def load_type(self) -> Type[LoadType]:
        """Return the specimen type to use for deserialization."""
        raise NotImplementedError()

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs) -> LoadType:
        """Load object from dictionary."""
        return self.load_type(**data)
