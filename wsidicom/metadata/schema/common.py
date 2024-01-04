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

"""Module with base marshmallow schemas that loads objects."""

from abc import abstractmethod
import dataclasses
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


class DataclassLoadingSchema(LoadingSchema[LoadType]):
    """Schema for loading metadata"""

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs) -> LoadType:
        """Return a object of given load class using the defined dataclass fields."""
        assert dataclasses.is_dataclass(self.load_type)
        return self.load_type(
            **{
                field.name: data[field.name]
                for field in dataclasses.fields(self.load_type)
                if field.name in data
            }
        )
