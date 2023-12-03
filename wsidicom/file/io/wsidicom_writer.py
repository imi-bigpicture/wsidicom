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
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

from pydicom.dataset import Dataset, FileMetaDataset, validate_file_meta
from pydicom.encaps import itemize_frame
from pydicom.filewriter import write_dataset, write_file_meta_info
from pydicom.tag import ItemTag, SequenceDelimiterTag
from pydicom.uid import UID

from wsidicom.codec import Encoder
from wsidicom.errors import WsiDicomBotOverflow
from wsidicom.file.io.frame_index import (
    BotWriter,
    EotWriter,
    OffsetTableType,
    OffsetTableWriter,
)
from wsidicom.file.io.tags import (
    PixelDataTag,
)
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.geometry import Point, Region, Size
from wsidicom.instance import ImageData
from wsidicom.instance.dataset import WsiDataset
from wsidicom.thread import ConditionalThreadPoolExecutor
from wsidicom.uid import WSI_SOP_CLASS_UID


class WsiDicomWriter:
    """Writer for DICOM WSI files."""

    def __init__(self, file: WsiDicomIO) -> None:
        """
        Create a writer for DICOM WSI data.

        Parameters
        ----------
        file: WsiDicomIO
            File to open for writing.
        """
        self._file = file

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def open(cls, file: Path) -> "WsiDicomWriter":
        """Open file in path as WsiDicomWriter.

        Parameters
        ----------
        file: Path
            Path to file.

        Returns
        ----------
        WsiDicomWriter
            WsiDicomWriter for file.
        """
        stream = WsiDicomIO(
            open(file, "w+b"),
            filepath=file,
            owned=True,
        )
        return cls(stream)

    def write(
        self,
        uid: UID,
        transfer_syntax: UID,
        dataset: WsiDataset,
        data: Dict[Tuple[str, float], ImageData],
        workers: int,
        chunk_size: int,
        offset_table: OffsetTableType,
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
            transfer_syntax = transcoder.transfer_syntax
        if not transfer_syntax.is_encapsulated:
            offset_table = OffsetTableType.NONE
        elif offset_table is OffsetTableType.NONE:
            offset_table = OffsetTableType.EMPTY
        self._write_preamble()
        self._write_file_meta(uid, transfer_syntax)
        dataset.SOPInstanceUID = uid
        dataset.InstanceNumber = instance_number
        self._write_base(dataset)
        if offset_table is OffsetTableType.NONE:
            self._write_unencapsulated_pixel_data(
                dataset, data, workers, chunk_size, scale, transcoder
            )
        else:
            self._write_encapsulated_pixel_data(
                data,
                dataset.NumberOfFrames,
                workers,
                chunk_size,
                offset_table,
                transfer_syntax,
                scale,
                transcoder,
            )

    def copy_with_table(
        self,
        copy_from: WsiDicomIO,
        transfer_syntax: UID,
        offset_table: OffsetTableType,
        dataset_end: int,
        pixels_end: int,
        frame_positions: List[int],
    ) -> List[int]:
        """Copy dataset and pixel data from other file to this.

        Parameters
        ----------
        copy_from: DicomFileLike
            File to copy from. Must have encapsulated pixel data.
        transfer_syntax: UID
            Transfer syntax in file.
        offset_table: OffsetTableType
            Offset table to use in new file.
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
        if not transfer_syntax.is_encapsulated:
            raise ValueError("Transfer syntax must be encapsulated")
        if offset_table is OffsetTableType.NONE:
            raise ValueError("Offset table must be used")
        # Copy dataset until EOT or PixelData tag
        copy_from.seek(0)
        self._file.write(copy_from.read(dataset_end))
        # Write new pixel data start
        (
            new_dataset_end,
            new_pixels_start,
            table_writer,
        ) = self._write_encapsulated_pixel_data_start(
            offset_table, len(frame_positions)
        )
        # Copy pixel data
        first_frame_position = frame_positions[0]
        copy_from.seek(first_frame_position)
        self._file.write(copy_from.read(pixels_end - first_frame_position))

        # Adjust frame positions
        frame_position_change = new_pixels_start - first_frame_position
        new_frame_positions = [
            position + frame_position_change for position in frame_positions
        ]

        # Write pixel data end and EOT or BOT if used.
        return self._write_encapsulated_pixel_data_end(
            table_writer,
            transfer_syntax,
            new_pixels_start,
            new_dataset_end,
            new_frame_positions,
        )

    def close(self, force: Optional[bool] = False) -> None:
        self._file.close(force)

    def _write_encapsulated_pixel_data(
        self,
        data: Dict[Tuple[str, float], ImageData],
        number_of_frames: int,
        workers: int,
        chunk_size: int,
        offset_table: OffsetTableType,
        transfer_syntax: UID,
        scale: int,
        transcoder: Optional[Encoder],
    ) -> List[int]:
        """Write encapsulated pixel data to file.

        Parameters
        ----------
        data: Dict[Tuple[str, float], ImageData]
            Pixel data to write.
        number_of_frames: int
            Number of frames to write.
        workers: int
            Number of workers to use for writing pixel data.
        chunk_size: int
            Number of frames to give each worker.
        offset_table: OffsetTableType
            Offset table to use.
        transfer_syntax: UID
            Transfer syntax to use.
        scale: int = 1
            Scale factor.

        Returns
        ----------
        List[int]
            List of frame position relative to start of file.
        """
        (
            dataset_end,
            pixels_start,
            table_writer,
        ) = self._write_encapsulated_pixel_data_start(offset_table, number_of_frames)
        frame_positions = [
            position
            for (path, z), image_data in sorted(data.items())
            for position in self._write_pixel_data(
                image_data, True, z, path, workers, chunk_size, scale, transcoder
            )
        ]
        frame_positions = self._write_encapsulated_pixel_data_end(
            table_writer,
            transfer_syntax,
            pixels_start,
            dataset_end,
            frame_positions,
        )
        return frame_positions

    def _write_encapsulated_pixel_data_end(
        self,
        offset_writer: Optional[OffsetTableWriter],
        transfer_syntax: UID,
        pixels_start: int,
        dataset_end: int,
        frame_positions: List[int],
    ):
        last_frame_end = self._write_pixel_data_end_tag()
        if offset_writer is not None:
            try:
                offset_writer.write(pixels_start, frame_positions, last_frame_end)
            except WsiDicomBotOverflow as exception:
                if self._file.owned:
                    frame_positions = self._rewrite_as_table(
                        OffsetTableType.EXTENDED,
                        transfer_syntax,
                        dataset_end,
                        last_frame_end,
                        frame_positions,
                    )
                else:
                    raise exception
        return frame_positions

    def _write_unencapsulated_pixel_data(
        self,
        dataset: WsiDataset,
        data: Dict[Tuple[str, float], ImageData],
        workers: int,
        chunk_size: int,
        scale: int,
        transcoder: Optional[Encoder],
    ) -> None:
        """Write unencapsulated pixel data to file.

        Parameters
        ----------
        dataset: WsiDataset
            Dataset with parameters for image to write.
        data: Dict[Tuple[str, float], ImageData]
            Pixel data to write.
        workers: int
            Number of workers to use for writing pixel data.
        chunk_size: int
            Number of frames to give each worker.
        scale: int
            Scale factor.
        """
        length = (
            dataset.tile_size.area
            * dataset.samples_per_pixel
            * (dataset.bits // 8)
            * dataset.frame_count
        )
        self._file.write_tag_of_vr_and_length(PixelDataTag, "OB", length)
        for (path, z), image_data in sorted(data.items()):
            self._write_pixel_data(
                image_data, False, z, path, workers, chunk_size, scale, transcoder
            )

    def _write_preamble(self) -> None:
        """Write file preamble to file."""
        preamble = b"\x00" * 128
        self._file.write(preamble)
        self._file.write(b"DICM")

    def _write_file_meta(self, uid: UID, transfer_syntax: UID) -> None:
        """Write file meta dataset to file.

        Parameters
        ----------
        uid: UID
            SOP instance uid to include in file.
        transfer_syntax: UID
            Transfer syntax used in file.
        """
        meta_ds = FileMetaDataset()
        meta_ds.TransferSyntaxUID = transfer_syntax
        meta_ds.MediaStorageSOPInstanceUID = uid
        meta_ds.MediaStorageSOPClassUID = WSI_SOP_CLASS_UID
        validate_file_meta(meta_ds)
        write_file_meta_info(self._file, meta_ds)

    def _write_base(self, dataset: Dataset) -> None:
        """Write base dataset to file.

        Parameters
        ----------
        dataset: Dataset

        """
        now = datetime.now()
        dataset.ContentDate = datetime.date(now).strftime("%Y%m%d")
        dataset.ContentTime = datetime.time(now).strftime("%H%M%S.%f")
        write_dataset(self._file, dataset)

    def _write_encapsulated_pixel_data_start(
        self,
        offset_table: OffsetTableType,
        number_of_frames: int,
    ) -> Tuple[int, int, Optional[OffsetTableWriter]]:
        """Write tags starting pixel data and reserves space for BOT or EOT.

        Parameters
        ----------
        offset_table: OffsetTableType
            Offset table to use.
        number_of_frames: int
            Number of frames to reserve space for in BOT or EOT.

        Returns
        ----------
        Tuple[Optional[int], int]
            End of dataset (EOT or PixelData tag), start of table (BOT or EOT) and
            start of pixel data (after BOT).
        """
        dataset_end = self._file.tell()
        table_writer = None
        if offset_table == OffsetTableType.EXTENDED:
            table_writer = EotWriter(self._file)
            table_writer.reserve(number_of_frames)

        # Write pixel data tag
        self._file.write_tag_of_vr_and_length(PixelDataTag, "OB")

        if offset_table == OffsetTableType.BASIC:
            table_writer = BotWriter(self._file)
            table_writer.reserve(number_of_frames)
        elif (
            offset_table == OffsetTableType.EMPTY
            or offset_table == OffsetTableType.EXTENDED
        ):
            self._file.write_tag(ItemTag)
            self._file.write_leUL(0)

        pixel_data_start = self._file.tell()

        return dataset_end, pixel_data_start, table_writer

    def _write_pixel_data(
        self,
        image_data: ImageData,
        encapsulate: bool,
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
                self._write_tile(tile, encapsulate)
                for thread_result in pool.map(get_tiles, chunked_tile_points)
                for tile in thread_result
            ]

    def _write_tile(self, tile: bytes, encapslute: bool) -> int:
        if encapslute:
            frames = itemize_frame(tile, 1)
        else:
            frames = [tile]
        position = self._file.tell()
        for frame in frames:
            self._file.write(frame)
        return position

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

    def _write_pixel_data_end_tag(self) -> int:
        """Writes tags ending pixel data."""
        last_frame_end = self._file.tell()
        self._file.write_tag(SequenceDelimiterTag)
        self._file.write_leUL(0)
        return last_frame_end

    def _rewrite_as_table(
        self,
        offset_table: OffsetTableType,
        transfer_syntax: UID,
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
        transfer_syntax: UID
            Transfer syntax to use in new file.
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
        if self._file.filepath is not None:
            temp_file_path = self._file.filepath.with_suffix(".tmp")
            new_file = WsiDicomIO.open(temp_file_path, "w+b")
        else:
            temp_file_path = None
            new_file = WsiDicomIO(BytesIO())
        writer = WsiDicomWriter(new_file)
        frame_positions = writer.copy_with_table(
            self._file,
            transfer_syntax,
            offset_table,
            dataset_end,
            pixels_end,
            frame_positions,
        )
        self._file.close()
        if temp_file_path is not None and self._file.filepath is not None:
            os.replace(temp_file_path, self._file.filepath)
        self._file = new_file
        return frame_positions
