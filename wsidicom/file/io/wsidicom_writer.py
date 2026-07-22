#    Copyright 2021, 2022, 2023 SECTRA AB
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


from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
)

from pydicom.encaps import itemize_frame
from pydicom.tag import ItemTag, SequenceDelimiterTag
from pydicom.uid import UID, VLWholeSlideMicroscopyImageStorage
from pydicom.valuerep import DSfloat
from upath import UPath

from wsidicom.codec import Encoder
from wsidicom.errors import WsiDicomBotOverflow
from wsidicom.file.io.frame_index import (
    BotWriter,
    EotWriter,
    OffsetTableType,
    OffsetTableWriter,
)
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.file.wsidicom_stream_opener import WsiDicomStreamOpener
from wsidicom.instance.dataset import WsiDataset
from wsidicom.tags import LossyImageCompressionRatioTag, PixelDataTag


class PixelDataWriter(metaclass=ABCMeta):
    """Abstract interface for writing pixel data in a specific format.

    Implementations handle the format-specific details of writing
    encapsulated (compressed) or native (uncompressed) pixel data.
    """

    @abstractmethod
    def write_pixel_data_start(
        self, dataset: WsiDataset
    ) -> tuple[int, OffsetTableWriter | None]:
        """Write tags starting the pixel data section.

        Returns
        -------
        Tuple[int, Optional[OffsetTableWriter]]
            Start position of pixel data and optional offset table writer.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_tile(self, tile: bytes) -> int:
        """Write one tile frame to the file.

        Returns
        -------
        int
            Position of the frame in the file.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_pixel_data_end(
        self,
        offset_writer: OffsetTableWriter | None,
        pixels_start: int,
        dataset_start: int,
        dataset_end: int,
        dataset: WsiDataset,
        frame_positions: Sequence[int],
        transcoder: Encoder | None,
    ) -> None:
        """Finalize the pixel data section."""
        raise NotImplementedError()


class EncapsulatedPixelDataWriter(PixelDataWriter):
    """Writes encapsulated (compressed) pixel data with item headers."""

    def __init__(
        self,
        file: WsiDicomIO,
        offset_table: OffsetTableType,
        transfer_syntax: UID,
        file_options: dict[str, Any] | None = None,
    ) -> None:
        self._file = file
        self._offset_table = offset_table
        self._transfer_syntax = transfer_syntax
        self._file_options = file_options

    def write_pixel_data_start(
        self, dataset: WsiDataset
    ) -> tuple[int, OffsetTableWriter | None]:
        return self._write_pixel_data_start(dataset.frame_count)

    def write_tile(self, tile: bytes) -> int:
        position = self._file.tell()
        for frame in itemize_frame(tile, 1):
            self._file.write(frame)
        return position

    def write_pixel_data_end(
        self,
        offset_writer: OffsetTableWriter | None,
        pixels_start: int,
        dataset_start: int,
        dataset_end: int,
        dataset: WsiDataset,
        frame_positions: Sequence[int],
        transcoder: Encoder | None,
    ) -> None:
        last_frame_end = self._file.tell()
        if transcoder is not None and transcoder.lossy:
            compressed_size = self._calculate_size(frame_positions, last_frame_end)
            self._set_compression_ratio(dataset_start, dataset, compressed_size)
        self._finalize_pixel_data(
            offset_writer,
            pixels_start,
            last_frame_end,
            dataset_end,
            frame_positions,
        )

    def _write_pixel_data_start(
        self, frame_count: int
    ) -> tuple[int, OffsetTableWriter | None]:
        table_writer = None
        if self._offset_table == OffsetTableType.EXTENDED:
            table_writer = EotWriter(self._file)
            table_writer.reserve(frame_count)

        self._file.write_tag_of_vr_and_length(PixelDataTag, "OB")

        if self._offset_table == OffsetTableType.BASIC:
            table_writer = BotWriter(self._file)
            table_writer.reserve(frame_count)
        elif self._offset_table in (
            OffsetTableType.EMPTY,
            OffsetTableType.EXTENDED,
        ):
            self._file.write_tag(ItemTag)
            self._file.write_UL(0)

        return self._file.tell(), table_writer

    def _finalize_pixel_data(
        self,
        offset_writer: OffsetTableWriter | None,
        pixels_start: int,
        last_frame_end: int,
        dataset_end: int,
        frame_positions: Sequence[int],
    ) -> None:
        """Write sequence delimiter and offset table."""
        self._file.write_tag(SequenceDelimiterTag)
        self._file.write_UL(0)
        if offset_writer is not None:
            try:
                offset_writer.write(pixels_start, frame_positions, last_frame_end)
            except WsiDicomBotOverflow:
                if isinstance(offset_writer, BotWriter):
                    self._rewrite_as_eot(
                        dataset_end,
                        last_frame_end,
                        frame_positions,
                    )
                else:
                    raise

    def _rewrite_as_eot(
        self,
        dataset_end: int,
        pixels_end: int,
        frame_positions: Sequence[int],
    ) -> None:
        """Rewrite pixel data section with EOT on BOT overflow."""
        file_path = self._file.filepath
        temp_file_path = file_path.with_suffix(".tmp")
        opener = WsiDicomStreamOpener(self._file_options)
        temp_stream = opener.open_for_writing(
            temp_file_path, "w+b", self._transfer_syntax
        )
        temp_writer = EncapsulatedPixelDataWriter(
            temp_stream,
            OffsetTableType.EXTENDED,
            self._transfer_syntax,
            self._file_options,
        )

        # Copy dataset
        self._file.seek(0)
        temp_stream.write(self._file.read(dataset_end))

        # Write new pixel data start with EOT
        new_pixels_start, table_writer = temp_writer._write_pixel_data_start(
            len(frame_positions)
        )

        # Copy pixel data frames
        first_frame_position = frame_positions[0]
        self._file.seek(first_frame_position)
        while self._file.tell() < pixels_end:
            chunk_size = min(4096, pixels_end - self._file.tell())
            temp_stream.write(self._file.read(chunk_size))

        # Adjust frame positions
        frame_position_change = new_pixels_start - first_frame_position
        new_frame_positions = [pos + frame_position_change for pos in frame_positions]
        last_frame_end = temp_stream.tell()

        # Finalize temp file
        temp_writer._finalize_pixel_data(
            table_writer,
            new_pixels_start,
            last_frame_end,
            dataset_end,
            new_frame_positions,
        )
        temp_stream.close()

        # Replace original with temp
        self._file.close()
        temp_file_path.replace(file_path)
        self._file = next(opener.open([file_path]))

    @staticmethod
    def _calculate_size(
        frame_positions: Sequence[int], last_frame_position: int
    ) -> int:
        """Calculate compressed pixel data size excluding item headers."""
        first_frame_start = frame_positions[0]
        return last_frame_position - first_frame_start - len(frame_positions) * 8

    def _set_compression_ratio(
        self, dataset_start: int, dataset: WsiDataset, compressed_size: int
    ) -> None:
        uncompressed_size = (
            dataset.frame_count
            * dataset.tile_size.area
            * dataset.samples_per_pixel
            * (dataset.bits // 8)
        )
        ratio = DSfloat(round(uncompressed_size / compressed_size, 2))
        ratios = dataset.get_multi_value(LossyImageCompressionRatioTag)
        ratios[-1] = ratio
        self._file.update_dataset(
            dataset_start,
            {LossyImageCompressionRatioTag: ratios},
        )


class NativePixelDataWriter(PixelDataWriter):
    """Writes native (uncompressed) pixel data."""

    def __init__(self, file: WsiDicomIO) -> None:
        self._file = file

    def write_pixel_data_start(
        self, dataset: WsiDataset
    ) -> tuple[int, OffsetTableWriter | None]:
        length = (
            dataset.tile_size.area
            * dataset.samples_per_pixel
            * (dataset.bits // 8)
            * dataset.frame_count
        )
        self._file.write_tag_of_vr_and_length(PixelDataTag, "OW", length)
        return self._file.tell(), None

    def write_tile(self, tile: bytes) -> int:
        position = self._file.tell()
        self._file.write(tile)
        return position

    def write_pixel_data_end(
        self,
        offset_writer: OffsetTableWriter | None,
        pixels_start: int,
        dataset_start: int,
        dataset_end: int,
        dataset: WsiDataset,
        frame_positions: Sequence[int],
        transcoder: Encoder | None,
    ) -> None:
        pass  # Native format has no closing tags


class WsiDicomWriter:
    """Writer for DICOM WSI files.

    Delegates pixel data format handling to a composed PixelDataWriter.
    """

    def __init__(
        self,
        file: WsiDicomIO,
        transfer_syntax: UID,
        pixel_data_writer: PixelDataWriter,
    ) -> None:
        self._file = file
        self._transfer_syntax = transfer_syntax
        self._pixel_data_writer = pixel_data_writer
        self._frame_positions: list[int] = []
        self._dataset_start: int | None = None
        self._dataset_end: int | None = None
        self._pixels_start: int | None = None
        self._offset_writer: OffsetTableWriter | None = None

    @property
    def filepath(self) -> UPath:
        """File path of the output file."""
        return self._file.filepath

    @property
    def frame_positions(self) -> list[int]:
        """Frame positions collected from write_tiles() calls."""
        return self._frame_positions

    def write_header(self, dataset: WsiDataset) -> None:
        """Write DICOM preamble, file meta info, and dataset."""
        uid = UID(dataset.SOPInstanceUID)
        self._file.write_preamble()
        self._file.write_file_meta_info(
            uid,
            VLWholeSlideMicroscopyImageStorage,
            self._transfer_syntax,
        )
        self._dataset_start = self._file.tell()
        self._file.write_dataset(dataset, datetime.now())
        self._dataset_end = self._file.tell()

    def start_pixel_data(self, dataset: WsiDataset) -> None:
        """Write pixel data start tags."""
        self._pixels_start, self._offset_writer = (
            self._pixel_data_writer.write_pixel_data_start(dataset)
        )

    def write_tiles(self, tiles: Iterable[bytes]) -> int:
        """Write consecutive tiles and record their frame positions."""
        count = 0
        for tile in tiles:
            position = self._pixel_data_writer.write_tile(tile)
            self._frame_positions.append(position)
            count += 1
        return count

    def finalize(
        self,
        dataset: WsiDataset,
        transcoder: Encoder | None = None,
    ) -> None:
        """Finalize pixel data and close the writer."""
        if self._dataset_start is None or self._dataset_end is None:
            raise RuntimeError("write_header() must be called before finalize().")
        if self._pixels_start is None:
            raise RuntimeError("start_pixel_data() must be called before finalize().")
        self._pixel_data_writer.write_pixel_data_end(
            self._offset_writer,
            self._pixels_start,
            self._dataset_start,
            self._dataset_end,
            dataset,
            self._frame_positions,
            transcoder,
        )
        self.close()

    def close(self, force: bool | None = False) -> None:
        self._file.close(force)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def open(
        cls,
        file: str | Path | UPath,
        transfer_syntax: UID,
        offset_table: OffsetTableType,
        file_options: dict[str, Any] | None = None,
    ) -> "WsiDicomWriter":
        """Open file and create a WsiDicomWriter with the right pixel data writer."""
        stream = cls._open_stream(file, transfer_syntax, file_options)
        if transfer_syntax.is_encapsulated:
            pixel_writer: PixelDataWriter = EncapsulatedPixelDataWriter(
                stream, offset_table, transfer_syntax, file_options
            )
        else:
            pixel_writer = NativePixelDataWriter(stream)
        return cls(stream, transfer_syntax, pixel_writer)

    @classmethod
    def open_instance(
        cls,
        file: str | Path | UPath,
        transfer_syntax: UID,
        offset_table: OffsetTableType,
        file_options: dict[str, Any] | None,
        dataset: WsiDataset,
    ) -> "WsiDicomWriter":
        """Open a writer and write the dataset header and pixel-data preamble.

        `dataset.SOPInstanceUID` and `InstanceNumber` must already be set.
        """
        writer = cls.open(file, transfer_syntax, offset_table, file_options)
        try:
            writer.write_header(dataset)
            writer.start_pixel_data(dataset)
        except BaseException:
            writer.close()
            raise
        return writer

    @staticmethod
    def _open_stream(
        file: str | Path | UPath,
        transfer_syntax: UID,
        file_options: dict[str, Any] | None = None,
    ) -> WsiDicomIO:
        """Open file for writing."""
        return WsiDicomStreamOpener(file_options).open_for_writing(
            file, "w+b", transfer_syntax
        )
