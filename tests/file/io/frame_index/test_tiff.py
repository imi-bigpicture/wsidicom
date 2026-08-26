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

from collections.abc import Callable, Sequence
from io import BytesIO
from struct import pack

import pytest
from PIL import Image as Pillow
from PIL.TiffImagePlugin import ImageFileDirectory_v2
from pydicom.encaps import encapsulate
from pydicom.tag import SequenceDelimiterTag
from pydicom.uid import JPEGBaseline8Bit
from upath import UPath

from wsidicom.file.io.frame_index.offset_table_type import OffsetTableType
from wsidicom.file.io.frame_index.tiff import (
    EmptyTiffFrameTagsException,
    TiffFrameIndexParser,
)
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.tags import PixelDataTag

TILE_OFFSETS_TAG = 324
TILE_BYTE_COUNTS_TAG = 325

# A slide-sized image, far larger than the default `PIL.Image.MAX_IMAGE_PIXELS`
# of about 89 million pixels.
SLIDE_SIZE = (200000, 100000)


@pytest.fixture
def buffer(placeholder_path: UPath):
    with WsiDicomIO(
        BytesIO(), filepath=placeholder_path, transfer_syntax=JPEGBaseline8Bit
    ) as buffer:
        yield buffer


@pytest.fixture
def write_dual_file(buffer: WsiDicomIO) -> Callable[..., int]:
    """Return a callable writing a tiff-and-dicom dual file into the buffer.

    The callable writes a tiff header and image file directory at the start of the
    buffer and dicom encapsulated pixel data after it, and returns the position the
    pixel data starts at.
    """

    def write(
        offsets: Sequence[int],
        lengths: Sequence[int],
        size: tuple[int, int] = SLIDE_SIZE,
        big_tiff: bool = False,
    ) -> int:
        if big_tiff:
            header = b"II" + pack("<HHH", 43, 8, 0) + pack("<Q", 16)
        else:
            header = b"II" + pack("<H", 42) + pack("<L", 8)
        directory = ImageFileDirectory_v2(header)
        directory[256] = size[0]  # ImageWidth
        directory[257] = size[1]  # ImageLength
        directory[258] = (8, 8, 8)  # BitsPerSample
        directory[259] = 7  # Compression, jpeg
        directory[262] = 2  # PhotometricInterpretation, rgb
        directory[277] = 3  # SamplesPerPixel
        directory[284] = 1  # PlanarConfiguration
        directory[322] = 256  # TileWidth
        directory[323] = 256  # TileLength
        if offsets:
            directory[TILE_OFFSETS_TAG] = tuple(offsets)
        if lengths:
            directory[TILE_BYTE_COUNTS_TAG] = tuple(lengths)
        buffer.write(header + directory.tobytes(len(header)))

        pixel_data_start = buffer.tell()
        buffer.write_tag_of_vr_and_length(PixelDataTag, "OB")
        buffer.write(encapsulate([b"\x00\x00" for _ in offsets], has_bot=False))
        buffer.write_tag(SequenceDelimiterTag)
        buffer.write_UL(0)
        return pixel_data_start

    return write


@pytest.mark.unittest
class TestTiffFrameIndexParser:
    @pytest.mark.parametrize("big_tiff", [False, True])
    def test_parse_frame_index_from_tiff_tags(
        self, buffer: WsiDicomIO, write_dual_file: Callable[..., int], big_tiff: bool
    ):
        # Arrange
        offsets = [1000, 2000, 3000, 4000]
        lengths = [110, 220, 330, 440]
        pixel_data_start = write_dual_file(offsets, lengths, big_tiff=big_tiff)

        # Act
        parser = TiffFrameIndexParser(buffer, pixel_data_start, len(offsets))
        frame_index = parser.parse_frame_index()

        # Assert
        assert frame_index == list(zip(offsets, lengths, strict=True))
        assert parser.offset_table_type == OffsetTableType.TIFF

    def test_parse_frame_index_does_not_open_file_as_image(
        self,
        buffer: WsiDicomIO,
        write_dual_file: Callable[..., int],
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Tags are read from the image file directory alone.

        Opening the file as an image is refused for one larger than
        `PIL.Image.MAX_IMAGE_PIXELS`, so the parser must not do so.
        """
        # Arrange
        offsets = [1000, 2000]
        lengths = [110, 220]
        pixel_data_start = write_dual_file(offsets, lengths)

        def fail_if_opened(*args: object, **kwargs: object):
            raise AssertionError("Tiff file should not be opened as an image.")

        monkeypatch.setattr(Pillow, "open", fail_if_opened)

        # Act
        parser = TiffFrameIndexParser(buffer, pixel_data_start, len(offsets))
        frame_index = parser.parse_frame_index()

        # Assert
        assert frame_index == list(zip(offsets, lengths, strict=True))

    def test_parse_frame_index_leaves_decompression_bomb_limit_unchanged(
        self, buffer: WsiDicomIO, write_dual_file: Callable[..., int]
    ):
        # Arrange
        offsets = [1000, 2000]
        lengths = [110, 220]
        pixel_data_start = write_dual_file(offsets, lengths)
        expected_limit = Pillow.MAX_IMAGE_PIXELS

        # Act
        TiffFrameIndexParser(buffer, pixel_data_start, len(offsets))

        # Assert
        assert expected_limit == Pillow.MAX_IMAGE_PIXELS

    @pytest.mark.parametrize("missing_tag", [TILE_OFFSETS_TAG, TILE_BYTE_COUNTS_TAG])
    def test_raises_empty_tiff_frame_tags_when_tag_is_missing(
        self,
        buffer: WsiDicomIO,
        write_dual_file: Callable[..., int],
        missing_tag: int,
    ):
        # Arrange
        offsets = [1000, 2000]
        lengths = [110, 220]
        pixel_data_start = write_dual_file(
            [] if missing_tag == TILE_OFFSETS_TAG else offsets,
            [] if missing_tag == TILE_BYTE_COUNTS_TAG else lengths,
        )

        # Act & Assert
        with pytest.raises(EmptyTiffFrameTagsException):
            TiffFrameIndexParser(buffer, pixel_data_start, len(offsets))

    def test_raises_empty_tiff_frame_tags_when_tag_lengths_differ(
        self, buffer: WsiDicomIO, write_dual_file: Callable[..., int]
    ):
        # Arrange
        offsets = [1000, 2000, 3000]
        lengths = [110, 220]
        pixel_data_start = write_dual_file(offsets, lengths)

        # Act & Assert
        with pytest.raises(EmptyTiffFrameTagsException):
            TiffFrameIndexParser(buffer, pixel_data_start, len(offsets))

    # A truncated directory makes Pillow warn and return the tags it did parse,
    # which then fails on the missing tag rather than on the short read.
    @pytest.mark.filterwarnings("ignore:Corrupt EXIF data:UserWarning")
    @pytest.mark.parametrize(
        "content",
        [
            b"",  # Empty file
            b"II",  # Truncated header
            b"NOTATIFF" + b"\x00" * 64,  # Not a tiff at all
            b"II" + pack("<H", 42) + pack("<L", 0) + b"\x00" * 64,  # No directory
            b"II" + pack("<H", 42) + pack("<L", 8) + b"\xff" * 8,  # Truncated directory
        ],
    )
    def test_raises_empty_tiff_frame_tags_when_not_a_valid_tiff(
        self, buffer: WsiDicomIO, content: bytes
    ):
        # Arrange
        buffer.write(content)
        pixel_data_start = buffer.tell()
        buffer.write_tag_of_vr_and_length(PixelDataTag, "OB")
        buffer.write(encapsulate([b"\x00\x00"], has_bot=False))
        buffer.write_tag(SequenceDelimiterTag)
        buffer.write_UL(0)

        # Act & Assert
        with pytest.raises(EmptyTiffFrameTagsException):
            TiffFrameIndexParser(buffer, pixel_data_start, 1)
