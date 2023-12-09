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


import os
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from pydicom.encaps import itemize_frame
from pydicom.tag import ItemTag, SequenceDelimiterTag
from pydicom.uid import UID
from pydicom.valuerep import DSfloat

from wsidicom.codec import Encoder
from wsidicom.errors import WsiDicomBotOverflow
from wsidicom.file.io.frame_index import (
    BotWriter,
    EotWriter,
    OffsetTableType,
    OffsetTableWriter,
)
from wsidicom.tags import LossyImageCompressionRatioTag, PixelDataTag
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.geometry import Point, Region, Size
from wsidicom.instance import ImageData
from wsidicom.instance.dataset import WsiDataset
from wsidicom.thread import ConditionalThreadPoolExecutor
from wsidicom.uid import WSI_SOP_CLASS_UID


class WsiDicomWriter:
    """Writer for DICOM WSI files."""

    def __init__(
        self, file: WsiDicomIO, transfer_syntax: UID, offset_table: OffsetTableType
    ) -> None:
        """
        Create a writer for DICOM WSI data.

        Parameters
        ----------
        file: WsiDicomIO
            File to open for writing.
        transfer_syntax: UID
            Transfer syntax to use.
        offset_table: OffsetTableType
            Offset table to use.
        """
        if not self.supports_transfer_syntax(transfer_syntax):
            raise ValueError(
                f"Transfer syntax not supported for writer of type {type(self)}."
            )
        if not self.supports_offset_table(offset_table):
            raise ValueError(
                f"Offset table not supported for writer of type {type(self)}."
            )
        self._file = file
        self._file.is_implicit_VR = transfer_syntax.is_implicit_VR
        self._file.is_little_endian = transfer_syntax.is_little_endian
        self._transfer_syntax = transfer_syntax
        self._offset_table = offset_table

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def open(
        cls, file: Path, transfer_syntax: UID, offset_table: OffsetTableType
    ) -> "WsiDicomWriter":
        """Open file in path as WsiDicomWriter.

        Parameters
        ----------
        file: Path
            Path to file.
        transfer_syntax: UID
            Transfer syntax to use.
        offset_table: OffsetTableType
            Offset table to use.

        Returns
        ----------
        WsiDicomWriter
            WsiDicomWriter for file.
        """
        stream = WsiDicomIO.open(file, "w+b")
        if transfer_syntax.is_encapsulated:
            writer = WsiDicomEncapsulatedWriter
        else:
            writer = WsiDicomNativeWriter
        return writer(stream, transfer_syntax, offset_table)

    def write(
        self,
        uid: UID,
        dataset: WsiDataset,
        data: Dict[Tuple[str, float], ImageData],
        workers: int,
        chunk_size: int,
        instance_number: int,
        scale: int = 1,
        transcoder: Optional[Encoder] = None,
    ) -> None:
        """
        Write data to file.

        Parameters
        ----------
        uid: UID
            Instance UID for file.
        transfer_syntax: UID.
            Transfer syntax for file
        dataset: WsiDataset
            Dataset to write (excluding pixel data).
        data: Dict[Tuple[str, float], ImageData]
            Pixel data to write.
        workers: int
            Number of workers to use for writing pixel data.
        chunk_size: int
            Number of frames to give each worker.
        offset_table: OffsetTableType
            Offset table to use.
        instance_number: int
            Instance number for file.
        scale: int = 1
            Scale factor.
        transcoder: Optional[Encoder] = None,
            Encoder to use if transcoding pixel data. Default is None to not transcode.

        """
        if transcoder is not None:
            if transcoder.transfer_syntax != self._transfer_syntax:
                raise ValueError("Transcoder transfer syntax must match writer.")
        self._file.write_preamble()
        self._file.write_file_meta_info(uid, WSI_SOP_CLASS_UID, self._transfer_syntax)
        dataset.SOPInstanceUID = uid
        dataset.InstanceNumber = instance_number
        dataset_start = self._file.tell()
        self._file.write_dataset(dataset, datetime.now())
        dataset_end = self._file.tell()
        self._write_pixel_data(
            data,
            dataset,
            (dataset_start, dataset_end),
            workers,
            chunk_size,
            scale,
            transcoder,
        )

    @abstractmethod
    def _write_pixel_data(
        self,
        data: Dict[Tuple[str, float], ImageData],
        dataset: WsiDataset,
        dataset_range: Tuple[int, int],
        workers: int,
        chunk_size: int,
        scale: int,
        transcoder: Optional[Encoder],
    ) -> List[int]:
        """Write pixel data to file.

        Parameters
        ----------
        data: Dict[Tuple[str, float], ImageData]
            Pixel data to write.
        dataset: WsiDataset
            Dataset with parameters for image to write.
        dataset_start: int
            Position of dataset in file.
        workers: int
            Number of workers to use for writing pixel data.
        chunk_size: int
            Number of frames to give each worker.
        scale: int
            Scale factor. Set to 1 for no scaling.
        transcoder: Optional[Encoder]
            Encoder to use if transcoding image.
        """
        raise NotImplementedError()

    @abstractmethod
    def _write_tile(self, tile: bytes) -> int:
        """Write tile to file and return position of frame in file."""
        raise NotImplementedError()

    @abstractmethod
    def supports_transfer_syntax(self, transfer_syntax: UID) -> bool:
        """Return True if writer supports transfer syntax."""
        raise NotImplementedError()

    @abstractmethod
    def supports_offset_table(self, offset_table: OffsetTableType) -> bool:
        """Return True if writer supports offset table."""
        raise NotImplementedError()

    def close(self, force: Optional[bool] = False) -> None:
        self._file.close(force)

    def _write_tiles(
        self,
        image_data: ImageData,
        z: float,
        path: str,
        workers: int,
        chunk_size: int,
        scale: int = 1,
        transcoder: Optional[Encoder] = None,
    ) -> List[int]:
        """Write pixel data to file.

        Parameters
        ----------
        image_data: ImageData
            Image data to read pixel tiles from.
        encapsulate: bool
            If pixel data should be encapsulated.
        z: float
            Focal plane to write.
        path: str
            Optical path to write.
        workers: int
            Maximum number of thread workers to use.
        chunk_size: int
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.
        scale: int
            Scale factor (1 = No scaling).

        Returns
        ----------
        List[int]
            List of frame position (position of ItemTag), relative to start of
            file.
        """
        chunked_tile_points = self._chunk_tile_points(image_data, chunk_size, scale)

        def get_tiles(tile_points: Iterable[Point]) -> List[bytes]:
            """Function to get tiles as bytes."""
            return list(
                image_data.get_encoded_tiles(tile_points, z, path, scale, transcoder)
            )

        with ConditionalThreadPoolExecutor(max_workers=workers) as pool:
            return [
                self._write_tile(tile)
                for thread_result in pool.map(get_tiles, chunked_tile_points)
                for tile in thread_result
            ]

    def _chunk_tile_points(
        self, image_data: ImageData, chunk_size: int, scale: int = 1
    ) -> Iterator[Iterator[Point]]:
        """Divide tile positions in image_data into chunks.

        Parameters
        ----------
        image_data: ImageData
            Image data with tiles to chunk.
        chunk_size: int
            Requested chunk size
        scale: int = 1
            Scaling factor (1 = no scaling).

        Returns
        ----------
        Iterator[Iterator[Point]]
            Chunked tile positions
        """
        minimum_chunk_size = getattr(image_data, "suggested_minimum_chunk_size", 1)
        # If chunk_size is less than minimum_chunk_size, use minimum_chunk_size
        # Otherwise, set chunk_size to highest even multiple of
        # minimum_chunk_size
        chunk_size = max(
            minimum_chunk_size, chunk_size // minimum_chunk_size * minimum_chunk_size
        )
        new_tiled_size = image_data.tiled_size.ceil_div(scale)
        # Divide the image tiles up into chunk_size chunks (up to tiled size)
        chunked_tile_points = (
            Region(
                Point(x, y), Size(min(chunk_size, new_tiled_size.width - x), 1)
            ).iterate_all()
            for y in range(new_tiled_size.height)
            for x in range(0, new_tiled_size.width, chunk_size)
        )
        return chunked_tile_points


class WsiDicomEncapsulatedWriter(WsiDicomWriter):
    @classmethod
    def open(
        cls, file: Path, transfer_syntax: UID, offset_table: OffsetTableType
    ) -> "WsiDicomEncapsulatedWriter":
        """Open file in path as WsiDicomEncapsulatedWriter.

        Parameters
        ----------
        file: Path
            Path to file.
        transfer_syntax: UID
            Transfer syntax to use.
        offset_table: OffsetTableType
            Offset table to use.

        Returns
        ----------
        WsiDicomEncapsulatedWriter
            WsiDicomEncapsulatedWriter for file.
        """
        stream = WsiDicomIO.open(file, "w+b")
        return cls(stream, transfer_syntax, offset_table)

    def copy(
        self,
        copy_from: WsiDicomIO,
        dataset_end: int,
        pixels_end: int,
        frame_positions: List[int],
    ) -> List[int]:
        """Copy dataset and pixel data from other file to this.

        Parameters
        ----------
        copy_from: DicomFileLike
            File to copy from. Must have encapsulated pixel data.
        dataset_end: int
            Position of EOT or PixelData tag in copy_from.
        pixels_end: int
            End of PixelData in copy_from.
        frame_positions: List[int]
            List of frame positions in copy_from, relative to start of file.

        Returns
        ----------
        List[int]
            List of frame position relative to start of new file.
        """
        # Copy dataset until EOT or PixelData tag
        copy_from.seek(0)
        self._file.write(copy_from.read(dataset_end))

        # Write new pixel data start
        (
            new_pixels_start,
            table_writer,
        ) = self._write_pixel_data_start(len(frame_positions))

        # Copy pixel data
        first_frame_position = frame_positions[0]
        copy_from.seek(first_frame_position)
        while copy_from.tell() < pixels_end:
            chunk_size = min(4096, pixels_end - copy_from.tell())
            self._file.write(copy_from.read(chunk_size))

        # Adjust frame positions
        frame_position_change = new_pixels_start - first_frame_position
        new_frame_positions = [
            position + frame_position_change for position in frame_positions
        ]
        last_frame_end = self._file.tell()

        # Write pixel data end and EOT or BOT if used.
        frame_positions = self._write_pixel_data_end(
            table_writer,
            new_pixels_start,
            last_frame_end,
            dataset_end,
            new_frame_positions,
        )
        return frame_positions

    def _write_pixel_data(
        self,
        data: Dict[Tuple[str, float], ImageData],
        dataset: WsiDataset,
        dataset_range: Tuple[int, int],
        workers: int,
        chunk_size: int,
        scale: int,
        transcoder: Optional[Encoder],
    ) -> List[int]:
        (
            pixels_start,
            table_writer,
        ) = self._write_pixel_data_start(dataset.frame_count)
        frame_positions = [
            position
            for (path, z), image_data in sorted(data.items())
            for position in self._write_tiles(
                image_data, z, path, workers, chunk_size, scale, transcoder
            )
        ]
        last_frame_end = self._file.tell()
        if transcoder is not None and transcoder.lossy:
            compressed_size = self._calculate_size(frame_positions, last_frame_end)
            self._set_compression_ratio(dataset_range[0], dataset, compressed_size)

        frame_positions = self._write_pixel_data_end(
            table_writer,
            pixels_start,
            last_frame_end,
            dataset_range[1],
            frame_positions,
        )

        return frame_positions

    def _write_tile(
        self,
        tile: bytes,
    ) -> int:
        position = self._file.tell()
        for frame in itemize_frame(tile, 1):
            self._file.write(frame)
        return position

    def supports_transfer_syntax(self, transfer_syntax: UID) -> bool:
        return transfer_syntax.is_encapsulated

    def supports_offset_table(self, offset_table: OffsetTableType) -> bool:
        return offset_table is not OffsetTableType.NONE

    def _write_pixel_data_start(
        self,
        number_of_frames: int,
    ) -> Tuple[int, Optional[OffsetTableWriter]]:
        """Write tags starting pixel data and reserves space for BOT or EOT.

        Parameters
        ----------
        number_of_frames: int
            Number of frames to reserve space for in BOT or EOT.

        Returns
        ----------
        Tuple[int, Optional[OffsetTableWriter]]:
            Start of pixel data (after BOT) and optional offset table writer.
        """
        table_writer = None
        if self._offset_table == OffsetTableType.EXTENDED:
            table_writer = EotWriter(self._file)
            table_writer.reserve(number_of_frames)

        # Write pixel data tag
        self._file.write_tag_of_vr_and_length(PixelDataTag, "OB")

        if self._offset_table == OffsetTableType.BASIC:
            table_writer = BotWriter(self._file)
            table_writer.reserve(number_of_frames)
        elif (
            self._offset_table == OffsetTableType.EMPTY
            or self._offset_table == OffsetTableType.EXTENDED
        ):
            self._file.write_tag(ItemTag)
            self._file.write_leUL(0)

        pixel_data_start = self._file.tell()

        return pixel_data_start, table_writer

    def _write_pixel_data_end(
        self,
        offset_writer: Optional[OffsetTableWriter],
        pixels_start: int,
        last_frame_end: int,
        dataset_end: int,
        frame_positions: List[int],
    ):
        """Write tags ending pixel data and BOT or EOT.

        Parameters
        ----------
        offset_writer: Optional[OffsetTableWriter]
            Offset table writer to use.
        pixels_start: int
            Position of start of pixel data.
        dataset_end: int
            Position of EOT or PixelData tag.
        frame_positions: List[int]
            List of frame positions in file, relative to start of file.
        """
        self._write_pixel_data_end_tag()
        if offset_writer is not None:
            try:
                offset_writer.write(pixels_start, frame_positions, last_frame_end)
            except WsiDicomBotOverflow as exception:
                if self._file.filepath is not None and isinstance(
                    offset_writer, BotWriter
                ):
                    frame_positions = self._rewrite_as_table(
                        OffsetTableType.EXTENDED,
                        dataset_end,
                        last_frame_end,
                        frame_positions,
                    )
                else:
                    raise exception
        return frame_positions

    def _write_pixel_data_end_tag(self) -> None:
        """Writes tags ending pixel data."""
        self._file.write_tag(SequenceDelimiterTag)
        self._file.write_leUL(0)

    def _rewrite_as_table(
        self,
        offset_table: OffsetTableType,
        dataset_end: int,
        pixels_end: int,
        frame_positions: List[int],
    ) -> List[int]:
        """Rewrite file as encapsulated with EOT. Closes current file and replaces
        it with the new a new file.

        Parameters
        ----------
        offset_table: OffsetTableType
            Offset table to use in new file.
        dataset_end: int
            Position of EOT or PixelData tag in current file.
        pixels_end: int
            End of PixelData in current file.
        frame_positions: List[int]
            List of frame positions in current file, relative to start of file.

        Returns
        ----------
        List[int]
            List of frame position relative to start of new file.
        """
        if self._file.filepath is None:
            raise ValueError("Cannot rewrite file without filepath.")
        temp_file_path = self._file.filepath.with_suffix(".tmp")
        with WsiDicomEncapsulatedWriter.open(
            temp_file_path, self._transfer_syntax, offset_table
        ) as writer:
            frame_positions = writer.copy(
                self._file,
                dataset_end,
                pixels_end,
                frame_positions,
            )
        self._file.close()
        os.replace(temp_file_path, self._file.filepath)
        self._file = WsiDicomIO.open(self._file.filepath, "r+b")
        return frame_positions

    @staticmethod
    def _calculate_size(
        frame_positions: Sequence[int], last_frame_position: int
    ) -> int:
        first_frame_start = frame_positions[0]
        last_frame_end = last_frame_position
        return last_frame_end - first_frame_start - len(frame_positions) * 16

    def _set_compression_ratio(
        self, dataset_start: int, dataset: WsiDataset, compressed_size: int
    ):
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


class WsiDicomNativeWriter(WsiDicomWriter):
    @classmethod
    def open(
        cls, file: Path, transfer_syntax: UID, offset_table: OffsetTableType
    ) -> "WsiDicomNativeWriter":
        """Open file in path as WsiDicomNativeWriter.

        Parameters
        ----------
        file: Path
            Path to file.
        transfer_syntax: UID
            Transfer syntax to use.
        offset_table: OffsetTableType
            Offset table to use.

        Returns
        ----------
        WsiDicomNativeWriter
            WsiDicomNativeWriter for file.
        """
        stream = WsiDicomIO.open(file, "w+b")
        return cls(stream, transfer_syntax, offset_table)

    def _write_pixel_data(
        self,
        data: Dict[Tuple[str, float], ImageData],
        dataset: WsiDataset,
        dataset_range: Tuple[int, int],
        workers: int,
        chunk_size: int,
        scale: int,
        transcoder: Optional[Encoder],
    ) -> List[int]:
        length = (
            dataset.tile_size.area
            * dataset.samples_per_pixel
            * (dataset.bits // 8)
            * dataset.frame_count
        )
        self._file.write_tag_of_vr_and_length(PixelDataTag, "OB", length)
        return [
            position
            for (path, z), image_data in sorted(data.items())
            for position in self._write_tiles(
                image_data, z, path, workers, chunk_size, scale, transcoder
            )
        ]

    def _write_tile(self, tile: bytes) -> int:
        position = self._file.tell()
        self._file.write(tile)
        return position

    def supports_transfer_syntax(self, transfer_syntax: UID) -> bool:
        return not transfer_syntax.is_encapsulated

    def supports_offset_table(self, offset_table: OffsetTableType) -> bool:
        return offset_table is OffsetTableType.NONE
