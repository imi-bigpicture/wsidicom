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

import struct
from io import BytesIO
from typing import Sequence

import pytest
from pydicom.tag import ItemTag
from pydicom.uid import (
    UID,
    JPEGBaseline8Bit,
)

from wsidicom.file.io.frame_index import BotWriter, EotWriter
from wsidicom.file.io.frame_index.basic import BasicOffsetTableFrameIndexParser
from wsidicom.file.io.frame_index.extended import ExtendedOffsetFrameIndexParser
from wsidicom.file.io.wsidicom_io import WsiDicomIO
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
def positions():
    yield [0, 100, 200, 300, 400]


class TestOffsetTableWriter:
    @pytest.mark.parametrize("frame_count", [1, 8])
    def test_reserve_bot(self, buffer: WsiDicomIO, frame_count: int):
        # Arrange
        writer = BotWriter(buffer)

        # Act
        writer.reserve(frame_count)

        # Assert
        buffer.seek(0)
        tag = buffer.read_tag()
        assert tag == ItemTag
        BOT_ITEM_LENGTH = 4
        length = buffer.read_tag_length(True)
        assert length == BOT_ITEM_LENGTH * frame_count
        for frame in range(frame_count):
            assert buffer.read_UL() == 0
        self.assertEndOfFile(buffer)

    @pytest.mark.parametrize("frame_count", [1, 8])
    def test_reserve_eot(self, buffer: WsiDicomIO, frame_count: int):
        # Arrange
        writer = EotWriter(buffer)

        # Act
        writer.reserve(frame_count)

        # Assert
        buffer.seek(0)

        tag = buffer.read_tag()
        assert tag == ExtendedOffsetTableTag

        buffer.read_tag_vr()
        EOT_ITEM_LENGTH = 8
        length = buffer.read_tag_length(True)
        assert length == EOT_ITEM_LENGTH * frame_count
        for frame in range(frame_count):
            assert struct.unpack("<Q", buffer.read(EOT_ITEM_LENGTH))[0] == 0

        tag = buffer.read_tag()
        buffer.read_tag_vr()
        assert tag == ExtendedOffsetTableLengthsTag
        length = buffer.read_tag_length(True)
        EOT_ITEM_LENGTH = 8
        assert length == EOT_ITEM_LENGTH * frame_count
        for frame in range(frame_count):
            assert struct.unpack("<Q", buffer.read(EOT_ITEM_LENGTH))[0] == 0
        self.assertEndOfFile(buffer)

    def test_write_bot(self, buffer: WsiDicomIO, positions: Sequence[int]):
        # Arrange
        # Write pixel data tag and reserve bot
        writer = BotWriter(buffer)
        buffer.write_tag_of_vr_and_length(PixelDataTag, "OB")
        writer.reserve(len(positions))

        # Act
        writer.write(0, positions, 0)

        # Assert
        buffer.seek(0)
        BasicOffsetTableFrameIndexParser(buffer, 0, len(positions))

    def test_write_eot(self, buffer: WsiDicomIO, positions: Sequence[int]):
        # Arrange
        # Reserve Eot, write pixel data tag and empty bot
        writer = EotWriter(buffer)
        writer.reserve(len(positions))
        buffer.write_tag_of_vr_and_length(PixelDataTag, "OB")
        buffer.write_tag(ItemTag)
        buffer.write_UL(0)

        # Act
        writer.write(0, positions, 600)

        # Assert
        buffer.seek(0)
        ExtendedOffsetFrameIndexParser(buffer, 0, len(positions))

    @staticmethod
    def assertEndOfFile(file: WsiDicomIO):
        with pytest.raises(EOFError):
            file.read(1, need_exact_length=True)
