#    Copyright 2021, 2022 SECTRA AB
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

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from struct import pack
from typing import (Any, Dict, Generator, Iterable, List, Optional, Sequence,
                    Tuple)

from pydicom.dataset import Dataset, FileMetaDataset, validate_file_meta
from pydicom.encaps import itemize_frame
from pydicom.filewriter import write_dataset, write_file_meta_info
from pydicom.tag import ItemTag, SequenceDelimiterTag, Tag
from pydicom.uid import UID

from wsidicom.file import WsiDicomFileBase
from wsidicom.geometry import Point, Region, Size
from wsidicom.image_data import ImageData
from wsidicom.uid import WSI_SOP_CLASS_UID


class WsiDicomFileWriter(WsiDicomFileBase):
    def __init__(self, filepath: Path) -> None:
        """Return a dicom filepointer.

        Parameters
        ----------
        filepath: Path
            Path to filepointer.

        """
        super().__init__(filepath, mode='w+b')
        self._fp.is_little_endian = True
        self._fp.is_implicit_VR = False

    def write(
        self,
        uid: UID,
        transfer_syntax: UID,
        dataset: Dataset,
        data: Dict[Tuple[str, float], ImageData],
        workers: int,
        chunk_size: int,
        offset_table: Optional[str],
        scale: int = 1
    ) -> None:
        """Writes data to file.

        Parameters
        ----------
        uid: UID
            Instance UID for file.
        transfer_syntax: UID.
            Transfer syntax for file
        dataset: Dataset
            Dataset to write (exluding pixel data).
        data: Dict[Tuple[str, float], ImageData]
            Pixel data to write.
        workers: int
            Number of workers to use for writing pixel data.
        chunk_size: int
            Number of frames to give each worker.
        offset_table: Optional[str] = 'bot'
            Offset table to use, 'bot' basic offset table, 'eot' extended
            offset table, None - no offset table.
        scale: int = 1
            Scale factor.

        """
        self._write_preamble()
        self._write_file_meta(uid, transfer_syntax)
        dataset.SOPInstanceUID = uid
        self._write_base(dataset)
        table_start, pixels_start = self._write_pixel_data_start(
            dataset.NumberOfFrames,
            offset_table
        )
        frame_positions: List[int] = []
        for (path, z), image_data in sorted(data.items()):
            frame_positions += self._write_pixel_data(
                image_data,
                z,
                path,
                workers,
                chunk_size,
                scale
            )
        pixels_end = self._fp.tell()
        self._write_pixel_data_end()

        if offset_table is not None:
            if table_start is None:
                raise ValueError('Table start should not be None')
            elif offset_table == 'eot':
                self._write_eot(
                    table_start,
                    pixels_start,
                    frame_positions,
                    pixels_end
                )
            elif offset_table == 'bot':
                self._write_bot(table_start, pixels_start, frame_positions)
        self.close()

    def _write_preamble(self) -> None:
        """Writes file preamble to file."""
        preamble = b'\x00' * 128
        self._fp.write(preamble)
        self._fp.write(b'DICM')

    def _write_file_meta(self, uid: UID, transfer_syntax: UID) -> None:
        """Writes file meta dataset to file.

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
        write_file_meta_info(self._fp, meta_ds)

    def _write_base(self, dataset: Dataset) -> None:
        """Writes base dataset to file.

        Parameters
        ----------
        dataset: Dataset

        """
        now = datetime.now()
        dataset.ContentDate = datetime.date(now).strftime('%Y%m%d')
        dataset.ContentTime = datetime.time(now).strftime('%H%M%S.%f')
        write_dataset(self._fp, dataset)

    def _write_tag(
        self,
        tag: str,
        value_representation: str,
        length: Optional[int] = None
    ):
        """Write tag, tag VR and length.

        Parameters
        ----------
        tag: str
            Name of tag to write.
        value_representation: str.
            Value representation (VR) of tag to write.
        length: Optional[int] = None
            Length of data after tag. 'Unspecified' (0xFFFFFFFF) if None.

        """
        self._fp.write_tag(Tag(tag))
        self._fp.write(bytes(value_representation, "iso8859"))
        self._fp.write_leUS(0)
        if length is not None:
            self._fp.write_leUL(length)
        else:
            self._fp.write_leUL(0xFFFFFFFF)

    def _reserve_eot(
        self,
        number_of_frames: int
    ) -> int:
        """Reserve space in file for extended offset table.

        Parameters
        ----------
        number_of_frames: int
            Number of frames to reserve space for.

        """
        table_start = self._fp.tell()
        BYTES_PER_ITEM = 8
        eot_length = BYTES_PER_ITEM * number_of_frames
        self._write_tag('ExtendedOffsetTable', 'OV', eot_length)
        for index in range(number_of_frames):
            self._write_unsigned_long_long(0)
        self._write_tag('ExtendedOffsetTableLengths', 'OV', eot_length)
        for index in range(number_of_frames):
            self._write_unsigned_long_long(0)
        return table_start

    def _reserve_bot(
        self,
        number_of_frames: int
    ) -> int:
        """Reserve space in file for basic offset table.

        Parameters
        ----------
        number_of_frames: int
            Number of frames to reserve space for.

        """
        table_start = self._fp.tell()
        BYTES_PER_ITEM = 4
        tag_lengths = BYTES_PER_ITEM * number_of_frames
        self._fp.write_tag(ItemTag)
        self._fp.write_leUL(tag_lengths)
        for index in range(number_of_frames):
            self._fp.write_leUL(0)
        return table_start

    def _write_pixel_data_start(
        self,
        number_of_frames: int,
        offset_table: Optional[str]
    ) -> Tuple[Optional[int], int]:
        """Writes tags starting pixel data and reserves space for BOT or EOT.

        Parameters
        ----------
        number_of_frames: int
            Number of frames to reserve space for in BOT or EOT.
        offset_table: Optional[str] = 'bot'
            Offset table to use, 'bot' basic offset table, 'eot' extended
            offset table, None - no offset table.

        Returns
        ----------
        Tuple[Optional[int], int]
            Start of table (BOT or EOT) and start of pixel data (after BOT).
        """
        table_start: Optional[int] = None
        if offset_table == 'eot':
            table_start = self._reserve_eot(number_of_frames)

        # Write pixel data tag
        self._write_tag('PixelData', 'OB')

        if offset_table == 'bot':
            table_start = self._reserve_bot(number_of_frames)
        else:
            self._fp.write_tag(ItemTag)  # Empty BOT
            self._fp.write_leUL(0)

        pixel_data_start = self._fp.tell()

        return table_start, pixel_data_start

    def _write_bot(
        self,
        bot_start: int,
        pixel_data_start: int,
        frame_positions: Sequence[int]
    ) -> None:
        """Writes BOT to file.

        Parameters
        ----------
        bot_start: int
            File position of BOT start
        bot_end: int
            File position of BOT end
        frame_positions: Sequence[int]
            List of file positions for frames, relative to file start

        """
        BYTES_PER_ITEM = 4
        # Check that last BOT entry is not over 2^32 - 1
        last_entry = frame_positions[-1] - pixel_data_start
        if last_entry > 2**32 - 1:
            raise NotImplementedError(
                "Image data exceeds 2^32 - 1 bytes "
                "An extended offset table should be used"
            )

        self._fp.seek(bot_start)  # Go to first BOT entry
        self._check_tag_and_length(
            ItemTag,
            BYTES_PER_ITEM*len(frame_positions),
            False
        )

        for frame_position in frame_positions:  # Write BOT
            self._fp.write_leUL(frame_position-pixel_data_start)

    def _write_unsigned_long_long(
        self,
        value: int
    ):
        """Write unsigned long long integer (64 bits) as little endian.

        Parameters
        ----------
        value: int
            Value to write.

        """
        self._fp.write(pack('<Q', value))

    def _write_eot(
        self,
        eot_start: int,
        pixel_data_start: int,
        frame_positions: Sequence[int],
        last_frame_end: int
    ) -> None:
        """Writes EOT to file.

        Parameters
        ----------
        bot_start: int
            File position of EOT start
        pixel_data_start: int
            File position of EOT end
        frame_positions: Sequence[int]
            List of file positions for frames, relative to file start
        last_frame_end: int
            Position of last frame end.

        """
        BYTES_PER_ITEM = 8
        # Check that last BOT entry is not over 2^64 - 1
        last_entry = frame_positions[-1] - pixel_data_start
        if last_entry > 2**64 - 1:
            raise ValueError(
                "Image data exceeds 2^64 - 1 bytes, likely something is wrong"
            )
        self._fp.seek(eot_start)  # Go to EOT table
        self._check_tag_and_length(
            Tag('ExtendedOffsetTable'),
            BYTES_PER_ITEM*len(frame_positions)
        )
        for frame_position in frame_positions:  # Write EOT
            relative_position = frame_position-pixel_data_start
            self._write_unsigned_long_long(relative_position)

        # EOT LENGTHS
        self._check_tag_and_length(
            Tag('ExtendedOffsetTableLengths'),
            BYTES_PER_ITEM*len(frame_positions)
        )
        frame_start = frame_positions[0]
        for frame_end in frame_positions[1:]:  # Write EOT lengths
            frame_length = frame_end - frame_start
            self._write_unsigned_long_long(frame_length)
            frame_start = frame_end

        # Last frame length, end does not include tag and length
        TAG_BYTES = 4
        LENGHT_BYTES = 4
        last_frame_start = frame_start + TAG_BYTES + LENGHT_BYTES
        last_frame_length = last_frame_end - last_frame_start
        self._write_unsigned_long_long(last_frame_length)

    def _write_pixel_data(
        self,
        image_data: ImageData,
        z: float,
        path: str,
        workers: int,
        chunk_size: int,
        scale: int = 1,
        image_format: str = 'jpeg',
        image_options: Dict[str, Any] = {'quality': 95}
    ) -> List[int]:
        """Writes pixel data to file.

        Parameters
        ----------
        image_data: ImageData
            Image data to read pixel tiles from.
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
        image_format: str = 'jpeg'
            Image format if scaling.
        image_options: Dict[str, Any] = {'quality': 95}
            Image options if scaling.

        Returns
        ----------
        List[int]
            List of frame position (position of ItemTag), relative to start of
            file.
        """
        chunked_tile_points = self._chunk_tile_points(
            image_data,
            chunk_size,
            scale
        )

        if scale == 1:
            def get_tiles_thread(tile_points: Iterable[Point]) -> List[bytes]:
                """Thread function to get tiles as bytes."""
                return image_data.get_encoded_tiles(tile_points, z, path)
            get_tiles = get_tiles_thread
        else:
            def get_scaled_tiles_thread(
                scaled_tile_points: Iterable[Point]
            ) -> List[bytes]:
                """Thread function to get scaled tiles as bytes."""
                return image_data.get_scaled_encoded_tiles(
                    scaled_tile_points,
                    z,
                    path,
                    scale,
                    image_format,
                    image_options
                )
            get_tiles = get_scaled_tiles_thread

        def write_frame(frame: bytes) -> int:
            """Itemize and write frame to file. Return frame position."""
            position = self._fp.tell()
            self._fp.write(frame)
            return position

        if workers == 1:
            return [
                write_frame(frame)
                for chunk in chunked_tile_points
                for tile in get_tiles(chunk)
                for frame in itemize_frame(tile, 1)
            ]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return [
                write_frame(frame)
                for thread_result in pool.map(
                    get_tiles,
                    chunked_tile_points
                )
                for tile in thread_result
                for frame in itemize_frame(tile, 1)
            ]

    def _chunk_tile_points(
        self,
        image_data: ImageData,
        chunk_size: int,
        scale: int = 1
    ) -> Generator[Generator[Point, None, None], None, None]:
        """Divides tile positions in image_data into chunks.

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
        Generator[Generator[Point, None, None], None, None]
            Chunked tile positions
        """
        minimum_chunk_size = getattr(
            image_data,
            'suggested_minimum_chunk_size',
            1
        )
        # If chunk_size is less than minimum_chunk_size, use minimum_chunk_size
        # Otherwise, set chunk_size to highest even multiple of
        # minimum_chunk_size
        chunk_size = max(
            minimum_chunk_size,
            chunk_size//minimum_chunk_size * minimum_chunk_size
        )
        new_tiled_size = image_data.tiled_size.ceil_div(scale)
        # Divide the image tiles up into chunk_size chunks (up to tiled size)
        chunked_tile_points = (
            Region(
                Point(x, y),
                Size(min(chunk_size, new_tiled_size.width - x), 1)
            ).iterate_all()
            for y in range(new_tiled_size.height)
            for x in range(0, new_tiled_size.width, chunk_size)
        )
        return chunked_tile_points

    def _write_pixel_data_end(self) -> None:
        """Writes tags ending pixel data."""
        self._fp.write_tag(SequenceDelimiterTag)
        self._fp.write_leUL(0)
