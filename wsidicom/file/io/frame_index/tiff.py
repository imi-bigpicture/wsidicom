from enum import Enum
from typing import Any

from PIL import Image as Pillow
from PIL import UnidentifiedImageError

from wsidicom.file.io.frame_index.offset_table_type import OffsetTableType
from wsidicom.file.io.frame_index.pixel_data import PixelDataFrameIndexParser
from wsidicom.file.io.wsidicom_io import WsiDicomIO


class EmptyTiffFrameTagsException(Exception):
    """Exception raised when file does not contain required tiff tags."""

    pass


class TiffTags(Enum):
    TILEOFFSETS = 324
    TILEBYTECOUNTS = 325


class TiffFrameIndexParser(PixelDataFrameIndexParser):
    """Frame index for TIFF, parsing the index from `TileOffsets`and TileByteCounts`
    if present. Only works with `DICOM-TIFF dual files."""

    def __init__(self, file: WsiDicomIO, pixel_data_start: int, frame_count: int):
        super().__init__(file, pixel_data_start, frame_count)
        self._offsets, self._lengths = self._get_tags()

    @property
    def offset_table_type(self):
        return OffsetTableType.TIFF

    def _get_index(self):
        return list(zip(self._offsets, self._lengths, strict=False))

    def _get_tags(self):
        """Return the tags used for the TIFF table."""
        # Large images will cause `DecompressionBombError` when opened if this is not
        # set to None. Restore it when we are done.
        max_image_pixels_restore = Pillow.MAX_IMAGE_PIXELS
        Pillow.MAX_IMAGE_PIXELS = None
        try:
            image = Pillow.open(self._file.stream)
            tags: dict[int, Any] | None = getattr(image, "tag_v2", None)
            if tags is None:
                raise EmptyTiffFrameTagsException("File does not contain tiff tags.")
            try:
                offsets: list[int] = tags[TiffTags.TILEOFFSETS.value]
                lengths: list[int] = tags[TiffTags.TILEBYTECOUNTS.value]
            except KeyError as exception:
                raise EmptyTiffFrameTagsException(
                    f"Tiff file is missing required tag {TiffTags(exception.args[0])}."
                ) from exception
            return offsets, lengths
        except UnidentifiedImageError:
            raise EmptyTiffFrameTagsException(
                "File is not a valid tiff file."
            ) from None
        finally:
            Pillow.MAX_IMAGE_PIXELS = max_image_pixels_restore
