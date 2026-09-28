#    Copyright 2026 SECTRA AB
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

"""Elements whose values were deferred when a dataset was read."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydicom.dataelem import RawDataElement
from pydicom.dataset import Dataset
from pydicom.tag import BaseTag

from wsidicom.file.io.constants import UNDEFINED_LENGTH


@dataclass(frozen=True)
class DeferredElement(ABC):
    """An element skipped while reading a dataset, to be read when it is used.

    Holds the position and encoding needed to read the element back.
    """

    tag: BaseTag
    """Tag of the element."""

    value_representation: str | None
    """Value representation of the element.

    None for implicit VR, in which case the value is made using the dictionary VR
    of the tag.
    """

    offset: int
    """Offset in the file the value starts at."""

    length: int
    """Length of the value in bytes."""

    is_implicit_value_representation: bool
    """Whether the value was written without its value representation stated."""

    is_little_endian: bool
    """Whether the value was written least significant byte first."""

    @property
    @abstractmethod
    def _stated_length(self) -> int:
        """Length to state in the element read back."""
        raise NotImplementedError

    def _as_raw(self, value: bytes) -> RawDataElement:
        """Create a raw element holding `value`.

        A raw element lets pydicom convert the value as if it had read it itself,
        which is needed for sequences and for implicit VR.

        Parameters
        ----------
        value: bytes
            Bytes read from `offset`, of `length` bytes.

        Returns
        -------
        RawDataElement
            Element holding `value`.
        """
        return RawDataElement(
            self.tag,
            self.value_representation,
            self._stated_length,
            value,
            self.offset,
            self.is_implicit_value_representation,
            self.is_little_endian,
        )


@dataclass(frozen=True)
class NestedDeferredElement(DeferredElement):
    """An element that belongs in a sequence item, e.g. an optical path ICC profile.

    Holds the dataset of the item, as the tag alone does not say which item the
    element belongs in.
    """

    dataset: Dataset
    """Dataset of the sequence item the element belongs in."""

    @property
    def _stated_length(self) -> int:
        """Length of the value, as deferred values have a defined length."""
        return self.length

    def set(self, value: bytes) -> None:
        """Put `value` into the dataset it belongs in.

        Parameters
        ----------
        value: bytes
            Bytes read from `offset`, of `length` bytes.
        """
        self.dataset[self.tag] = self._as_raw(value)


@dataclass(frozen=True)
class RootDeferredElement(DeferredElement):
    """An element that belongs in the root dataset, e.g. a stepped over sequence.

    The dataset is given when setting the value rather than held, as it is not
    created until all elements have been read.
    """

    @property
    def _stated_length(self) -> int:
        """Undefined length, as only undefined length sequences are stepped over.

        `length` is the measured length of the value, excluding the sequence
        delimiter, which pydicom writes itself for undefined length values.
        """
        return UNDEFINED_LENGTH

    def set(self, dataset: Dataset, value: bytes) -> None:
        """Put `value` into `dataset`.

        Parameters
        ----------
        dataset: Dataset
            Root dataset to put the element in.
        value: bytes
            Bytes read from `offset`, of `length` bytes.
        """
        dataset[self.tag] = self._as_raw(value)
