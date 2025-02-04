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

from io import BytesIO
from typing import List, Tuple

import pytest
from pydicom.encaps import encapsulate, encapsulate_extended
from pydicom.tag import SequenceDelimiterTag
from pydicom.uid import (
    UID,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    JPEGBaseline8Bit,
)

from wsidicom.file.io.frame_index.basic import BasicOffsetTableFrameIndexParser
from wsidicom.file.io.frame_index.extended import ExtendedOffsetFrameIndexParser
from wsidicom.file.io.frame_index.native_pixel_data import (
    NativePixelDataFrameIndexParser,
)
from wsidicom.file.io.frame_index.pixel_data import PixelDataFrameIndexParser
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.geometry import Size
from wsidicom.tags import (
    ExtendedOffsetTableLengthsTag,
    ExtendedOffsetTableTag,
    PixelDataTag,
)


@pytest.fixture
def transfer_syntax():
    yield JPEGBaseline8Bit


@pytest.fixture
def buffer(transfer_syntax: UID):
    with WsiDicomIO(BytesIO(), transfer_syntax=transfer_syntax) as buffer:
        yield buffer


@pytest.fixture
def tiles(bits: int):
    yield [
        bytes([1 for _ in range(bits // 8)]),
        bytes([2 for _ in range(bits // 8)]),
        bytes([3 for _ in range(bits // 8)]),
        bytes([4 for _ in range(bits // 8)]),
    ]


class TestFrameIndexParser:
    @pytest.mark.parametrize(
        "transfer_syntax",
        [ImplicitVRLittleEndian, ExplicitVRLittleEndian, ExplicitVRBigEndian],
    )
    @pytest.mark.parametrize("bits", [8, 16])
    def test_read_native_pixel_data_offset_table_frame_positions(
        self, buffer: WsiDicomIO, tiles: List[bytes], transfer_syntax: UID, bits: int
    ):
        # Arrange
        buffer.write_tag_of_vr_and_length(PixelDataTag, "OB", len(tiles) * bits // 8)
        expected_frame_index: List[Tuple[int, int]] = []
        for tile in tiles:
            position = buffer.tell()
            length = len(tile)
            expected_frame_index.append((position, length))
            buffer.write(tile)

        # Act
        parser = NativePixelDataFrameIndexParser(
            buffer, 0, len(tiles), Size(1, 1), 1, bits
        )
        frame_index = parser.parse_frame_index()

        # Assert
        assert frame_index == expected_frame_index

    @pytest.mark.parametrize("bits", [8, 16])
    def test_pixel_data_offset_table(self, buffer: WsiDicomIO, tiles: List[bytes]):
        # Arrange
        EMPTY_BOT = 16
        ITEM_TAG_AND_LENGTH = 8
        FRAME_LENGTH = 2
        ITEM_LENGTH = ITEM_TAG_AND_LENGTH + FRAME_LENGTH

        buffer.write_tag_of_vr_and_length(PixelDataTag, "OB")
        encapsulated = encapsulate(tiles, has_bot=False)
        bot_start = buffer.tell()
        buffer.write(encapsulated)
        buffer.write_tag(SequenceDelimiterTag)
        buffer.write_UL(0)
        expected_frame_index: List[Tuple[int, int]] = [
            ((bot_start + EMPTY_BOT + index * ITEM_LENGTH), FRAME_LENGTH)
            for index in range(len(tiles))
        ]

        # Act
        parser = PixelDataFrameIndexParser(buffer, 0, len(tiles))
        frame_index = parser.parse_frame_index()

        # Assert
        assert frame_index == expected_frame_index

    @pytest.mark.parametrize("bits", [8, 16])
    def test_basic_offset_table(self, buffer: WsiDicomIO, tiles: List[bytes]):
        # Arrange
        BOT = 16 + len(tiles) * 4
        ITEM_TAG_AND_LENGTH = 8
        FRAME_LENGTH = 2
        ITEM_LENGTH = ITEM_TAG_AND_LENGTH + FRAME_LENGTH
        buffer.write_tag_of_vr_and_length(PixelDataTag, "OB")
        encapsulated = encapsulate(tiles, has_bot=True)
        bot_start = buffer.tell()
        buffer.write(encapsulated)
        buffer.write_tag(SequenceDelimiterTag)
        buffer.write_UL(0)
        expected_frame_index: List[Tuple[int, int]] = [
            ((bot_start + BOT + index * ITEM_LENGTH), FRAME_LENGTH)
            for index in range(len(tiles))
        ]

        # Act
        parser = BasicOffsetTableFrameIndexParser(buffer, 0, len(tiles))
        frame_index = parser.parse_frame_index()

        # Assert
        assert frame_index == expected_frame_index

    @pytest.mark.parametrize("bits", [8, 16])
    def test_extended_offset_table(self, buffer: WsiDicomIO, tiles: List[bytes]):
        # Arrange
        EMPTY_BOT = 16
        ITEM_TAG_AND_LENGTH = 8
        FRAME_LENGTH = 2
        ITEM_LENGTH = ITEM_TAG_AND_LENGTH + FRAME_LENGTH

        def ensure_tile_is_even_length(tile: bytes) -> bytes:
            if len(tile) % 2 != 0:
                return tile + bytes([0])
            return tile

        encapsulated, eot_positions, eot_length = encapsulate_extended(
            [ensure_tile_is_even_length(tile) for tile in tiles]
        )
        buffer.write_tag_of_vr_and_length(
            ExtendedOffsetTableTag, "OV", len(eot_positions)
        )
        buffer.write(eot_positions)

        buffer.write_tag_of_vr_and_length(
            ExtendedOffsetTableLengthsTag, "OV", len(eot_length)
        )
        buffer.write(eot_length)
        buffer.write_tag_of_vr_and_length(PixelDataTag, "OB")
        bot_start = buffer.tell()
        buffer.write(encapsulated)
        buffer.write_tag(SequenceDelimiterTag)
        buffer.write_UL(0)
        expected_frame_index: List[Tuple[int, int]] = [
            ((bot_start + EMPTY_BOT + index * ITEM_LENGTH), FRAME_LENGTH)
            for index in range(len(tiles))
        ]

        # Act
        parser = ExtendedOffsetFrameIndexParser(buffer, 0, len(tiles))
        frame_index = parser.parse_frame_index()

        # Assert
        assert frame_index == expected_frame_index
