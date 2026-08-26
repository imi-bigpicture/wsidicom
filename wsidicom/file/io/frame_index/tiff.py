from enum import Enum
from struct import error as StructError

from PIL.TiffImagePlugin import ImageFileDirectory_v2

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
        return list(zip(self._offsets, self._lengths, strict=True))

    def _get_tags(self):
        """Return the tags used for the TIFF table."""
        directory = self._read_image_file_directory()
        try:
            offsets: list[int] = directory[TiffTags.TILEOFFSETS.value]
            lengths: list[int] = directory[TiffTags.TILEBYTECOUNTS.value]
        except KeyError as exception:
            raise EmptyTiffFrameTagsException(
                f"Tiff file is missing required tag {TiffTags(exception.args[0])}."
            ) from exception
        if len(offsets) != len(lengths):
            raise EmptyTiffFrameTagsException(
                f"Tiff file has {len(offsets)} {TiffTags.TILEOFFSETS.name} but "
                f"{len(lengths)} {TiffTags.TILEBYTECOUNTS.name}."
            )
        return offsets, lengths

    def _read_image_file_directory(self) -> ImageFileDirectory_v2:
        """Read the first image file directory (IFD) of the tiff file.

        The directory is parsed straight from the stream rather than by opening the
        file as an image. Only the tags are needed, and opening an image is refused
        for one larger than `PIL.Image.MAX_IMAGE_PIXELS` -- a limit that guards
        against decompression bombs and that whole slide images routinely exceed.

        Returns
        -------
        ImageFileDirectory_v2
            The first image file directory of the tiff file.
        """
        HEADER_LENGTH = 8
        BIG_TIFF_VERSION = 43
        stream = self._file.stream
        stream.seek(0)
        header = stream.read(HEADER_LENGTH)
        try:
            # A big tiff has a header of twice the length. Detected as PIL does, so
            # that the same files are accepted.
            if header[2] == BIG_TIFF_VERSION:
                header += stream.read(HEADER_LENGTH)
            directory = ImageFileDirectory_v2(header)
            if directory.next == 0:
                raise EmptyTiffFrameTagsException(
                    "Tiff file does not contain an image file directory."
                )
            stream.seek(directory.next)
            directory.load(stream)
        except (IndexError, SyntaxError, StructError, OSError) as exception:
            raise EmptyTiffFrameTagsException(
                "File is not a valid tiff file."
            ) from exception
        return directory
