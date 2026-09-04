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

"""An element whose value was deferred when a dataset was read."""

from dataclasses import dataclass

from pydicom.dataelem import RawDataElement
from pydicom.dataset import Dataset
from pydicom.tag import BaseTag


@dataclass(frozen=True)
class DeferredElement:
    """A value deferred while reading a dataset, to be read when it is used.

    Carries the dataset the value belongs in rather than a path from the root to
    it, so that reading it is a matter of putting an element into that dataset,
    however deeply it sits.
    """

    dataset: Dataset
    """Dataset the value belongs in, which may be one nested below the one read
    from the file.

    A value worth deferring is one of the large ones, and those tend to sit in an
    item of a sequence rather than at the root: the ICC profile of an optical path
    is one. A tag alone would not say which item, and there is no telling how deep
    a sequence nests, so the dataset itself is held and the tag is read within it.
    """

    tag: BaseTag
    """Tag of the element the value belongs to."""

    value_representation: str | None
    """Value representation of the element the value belongs to.

    None where the transfer syntax states value representations implicitly, there
    being none written to read. The value is made from the one the dictionary
    gives for the tag, as it would have been had it not been deferred.
    """

    offset: int
    """Offset in the file the value starts at."""

    length: int
    """Length of the value in bytes."""

    is_implicit_value_representation: bool
    """Whether the value was written without its value representation stated."""

    is_little_endian: bool
    """Whether the value was written least significant byte first."""

    def set(self, value: bytes) -> None:
        """Put `value` into the dataset it belongs in.

        Put in raw, as pydicom would have left it had it read the value itself, so
        that the value is made from the bytes the way it would have been. Making it
        here instead would hold the bytes as the value, which is wrong for a
        sequence: what is deferred is chosen by length alone, so a sequence
        longer than that size is deferred whole and has to be read back as one.
        Making it here would also need a value representation, which an implicitly
        stated one does not have.

        Parameters
        ----------
        value: bytes
            Bytes read from `offset`, of `length` bytes.
        """
        self.dataset[self.tag] = RawDataElement(
            self.tag,
            self.value_representation,
            self.length,
            value,
            self.offset,
            self.is_implicit_value_representation,
            self.is_little_endian,
        )
