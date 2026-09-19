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

"""Constants of the DICOM encoding, shared by the readers and writers of a stream.

Each is named by what it is and in what shape, so the same thing held as bytes and as
a number gets two names, and two sizes that happen to be equal keep separate names.
"""

import struct
from typing import Final

from pydicom.tag import ItemTag, SequenceDelimiterTag

UNDEFINED_LENGTH: Final = 0xFFFFFFFF
"""Length stated by an element that ends at a delimiter instead."""

ITEM_TAG_BYTES: Final = struct.pack("<HH", ItemTag.group, ItemTag.element)
"""Tag of an item, as written."""

SEQUENCE_DELIMITER_TAG_BYTES: Final = struct.pack(
    "<HH", SequenceDelimiterTag.group, SequenceDelimiterTag.element
)
"""Tag of a sequence delimiter, as written."""

SEQUENCE_DELIMITER_BYTES: Final = struct.pack(
    "<HHI", SequenceDelimiterTag.group, SequenceDelimiterTag.element, 0
)
"""A sequence delimiter as written: its tag and zero length."""

TAG_AND_LENGTH_SIZE: Final = 8
"""Size of a tag and a four byte length: an item, a delimiter or an implicit VR
element header."""

SHORT_FORM_HEADER_SIZE: Final = 8
"""Size of a short form element header: tag, VR and two byte length."""

LONG_FORM_HEADER_SIZE: Final = 12
"""Size of a long form element header: tag, VR, reserved and four byte length."""

TAG_AND_VALUE_REPRESENTATION_SIZE: Final = 6
"""Size of a tag and VR, and offset of the length in a short form header."""

CHARACTER_SET_ESCAPE: Final = 0x1B
"""Byte starting a character set escape sequence."""
