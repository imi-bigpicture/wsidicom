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

from bisect import bisect_right
from collections.abc import Sequence

import numpy as np
from pydicom.uid import UID
from upath import UPath

from wsidicom.cache import DecodedFrameCache
from wsidicom.codec import Codec
from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.file.io import WsiDicomReader
from wsidicom.instance import WsiDicomImageData
from wsidicom.instance.tile_byte_ranges import FrameReader, TileByteRanges


class WsiDicomFileImageData(WsiDicomImageData):
    """
    Represents image data read from dicom file(s).

    Image data can be sparsely or fully tiled and/or concatenated.
    """

    def __init__(
        self,
        readers: WsiDicomReader | Sequence[WsiDicomReader],
        decoded_frame_cache: DecodedFrameCache,
    ) -> None:
        """
        Create WsiDicomFileImageData from frame data from readers.

        Parameters
        ----------
        readers: WsiDicomReader | Sequence[WsiDicomReader]
            Single or list of WsiDicomReader containing frame data.
        decoded_frame_cache: DecodedFrameCache
            Cache of decoded frames. Encoded frames are not cached, as a frame of a
            local file costs about as little to read again as to get from a cache,
            and a file opened through fsspec can be cached by fsspec.
        """
        if not isinstance(readers, Sequence):
            readers = [readers]

        self._readers = [
            file for file in sorted(readers, key=lambda file: file.frame_offset)
        ]
        self._reader_frame_starts = [reader.frame_offset for reader in self._readers]
        self._reader_frame_ends = [
            reader.frame_offset + reader.frame_count for reader in self._readers
        ]
        self._transfer_syntax = readers[0].transfer_syntax
        dataset = readers[0].dataset
        codec = Codec.create(
            self.transfer_syntax,
            dataset.samples_per_pixel,
            dataset.bits,
            dataset.tile_size,
            dataset.photometric_interpretation,
        )
        super().__init__(
            [file.dataset for file in self._readers], codec, decoded_frame_cache
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._readers})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of files {self._readers}"

    @property
    def files(self) -> list[UPath]:
        return [reader.filepath for reader in self._readers]

    @property
    def transfer_syntax(self) -> UID:
        """The uid of the transfer syntax of the image."""
        return self._transfer_syntax

    def _get_reader(self, frame_index: int) -> WsiDicomReader:
        """
        Return file containing frame index.

        Raises WsiDicomNotFoundError if frame is not found.

        Looked up for every frame read, so the frame ranges of the files are
        bisected rather than the lookup cached, which would cost more than it saves.

        Parameters
        ----------
        frame_index: int
             Frame index to get

        Returns
        -------
        WsiDicomFile
            File containing the frame
        """
        index = bisect_right(self._reader_frame_starts, frame_index) - 1
        if index < 0 or frame_index >= self._reader_frame_ends[index]:
            raise WsiDicomNotFoundError(f"Frame index {frame_index}", "instance")
        return self._readers[index]

    def get_tile_byte_ranges(self, z: float, path: str) -> TileByteRanges | None:
        """Return where the frames of the tiles of a plane are in the files, or None
        if the plane is not in the image data."""
        if z not in self.focal_planes or path not in self.optical_paths:
            return None
        frame_count = self._reader_frame_ends[-1]
        frame_positions = np.zeros(frame_count, np.int64)
        frame_lengths = np.full(
            frame_count, TileByteRanges.READ_THROUGH_IMAGE_DATA, np.int64
        )
        frame_files = np.zeros(frame_count, np.int64)
        for file_index, reader in enumerate(self._readers):
            start = self._reader_frame_starts[file_index]
            index = reader.frame_index
            end = min(self._reader_frame_ends[file_index], start + len(index))
            frame_positions[start:end] = index.positions[: end - start]
            frame_lengths[start:end] = index.lengths[: end - start]
            frame_files[start:end] = file_index

        frame_indices = self.tiles.get_plane_frame_indices(z, path)
        stored = (frame_indices >= 0) & (frame_indices < frame_count)
        stored_frames = frame_indices[stored]
        positions = np.zeros(len(frame_indices), np.int64)
        lengths = np.full(
            len(frame_indices), TileByteRanges.READ_THROUGH_IMAGE_DATA, np.int64
        )
        positions[stored] = frame_positions[stored_frames]
        lengths[stored] = frame_lengths[stored_frames]
        file_indices = None
        if len(self._readers) > 1:
            file_indices = np.zeros(len(frame_indices), np.int64)
            file_indices[stored] = frame_files[stored_frames]
        return TileByteRanges.from_arrays(
            self.image_size, self.tile_size, positions, lengths, file_indices
        )

    def get_frame_readers(self) -> Sequence[FrameReader] | None:
        """Return the readers of the files, or None if a file is not a local one
        (and so costs more to read a frame at a time from)."""
        readers = [reader.frame_reader for reader in self._readers]
        if any(reader is None for reader in readers):
            return None
        return [reader for reader in readers if reader is not None]

    def _get_tile_frame(self, frame_index: int) -> bytes:
        """Return tile frame for frame index.

        Parameters
        ----------
        frame_index: int
             Frame index to get

        Returns
        -------
        bytes
            The frame in bytes
        """
        reader = self._get_reader(frame_index)
        tile_frame = reader.read_frame(frame_index)
        return tile_frame
