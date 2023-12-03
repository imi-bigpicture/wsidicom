import struct
from io import BytesIO

import pytest
from pydicom.tag import ItemTag
from pydicom.uid import (
    UID,
    JPEGBaseline8Bit,
)

from typing import Sequence
from wsidicom.file.io.frame_index import BotWriter, EotWriter
from wsidicom.file.io.frame_index.bot import Bot
from wsidicom.file.io.frame_index.eot import Eot
from wsidicom.file.io.tags import (
    ExtendedOffsetTableLengthsTag,
    ExtendedOffsetTableTag,
    PixelDataTag,
)
from wsidicom.file.io.wsidicom_io import WsiDicomIO


@pytest.fixture
def transfer_syntax():
    yield JPEGBaseline8Bit


@pytest.fixture
def buffer(transfer_syntax: UID):
    with WsiDicomIO(
        BytesIO(), transfer_syntax.is_little_endian, transfer_syntax.is_implicit_VR
    ) as buffer:
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
        length = buffer.read_tag_length()
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
        length = buffer.read_tag_length()
        assert length == EOT_ITEM_LENGTH * frame_count
        for frame in range(frame_count):
            assert struct.unpack("<Q", buffer.read(EOT_ITEM_LENGTH))[0] == 0

        tag = buffer.read_tag()
        buffer.read_tag_vr()
        assert tag == ExtendedOffsetTableLengthsTag
        length = buffer.read_tag_length()
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
        Bot(buffer, 0, len(positions))

    def test_write_eot(self, buffer: WsiDicomIO, positions: Sequence[int]):
        # Arrange
        # Reserve Eot, write pixel data tag and empty bot
        writer = EotWriter(buffer)
        writer.reserve(len(positions))
        buffer.write_tag_of_vr_and_length(PixelDataTag, "OB")
        buffer.write_tag(ItemTag)
        buffer.write_leUL(0)

        # Act
        writer.write(0, positions, 600)

        # Assert
        buffer.seek(0)
        Eot(buffer, 0, len(positions))

    @staticmethod
    def assertEndOfFile(file: WsiDicomIO):
        with pytest.raises(EOFError):
            file.read(1, need_exact_length=True)
