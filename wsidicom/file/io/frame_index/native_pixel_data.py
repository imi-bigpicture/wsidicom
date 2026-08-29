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

"""Class for reading frame positions from non-encapsulated data."""

import math

import numpy as np

from wsidicom.file.io.frame_index.frame_index import FrameIndex
from wsidicom.file.io.frame_index.offset_table_type import OffsetTableType
from wsidicom.file.io.frame_index.parser import FrameIndexParser
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.geometry import Size


class NativePixelDataFrameIndexParser(FrameIndexParser):
    def __init__(
        self,
        file: WsiDicomIO,
        pixel_data_start: int,
        frame_count: int,
        tile_size: Size,
        samples_per_pixel: int,
        bits: int,
    ):
        self._tile_size = tile_size
        self._samples_per_pixel = samples_per_pixel
        self._bits = bits
        super().__init__(file, pixel_data_start, frame_count)

    @property
    def offset_table_type(self) -> OffsetTableType:
        return OffsetTableType.NONE

    @property
    def expected_length(self) -> int:
        return (
            math.ceil(
                self._tile_size.area
                * self._samples_per_pixel
                * (self._bits // 8)
                * self._frame_count
                / 2
            )
            * 2
        )

    def _get_pixels_start(self) -> int:
        self._validate_pixel_data_start(self.expected_length)
        return self._file.tell()

    def _get_index(self) -> FrameIndex:
        """Create frame positions for uncapsulated data.

        Every frame is the same size, so the positions are a series and the lengths are
        all the same.

        Returns
        -------
        FrameIndex
            Position and length of every frame.
        """
        frame_size = self._tile_size.area * self._samples_per_pixel * (self._bits // 8)
        frames = np.arange(self._frame_count, dtype=np.int64)
        return FrameIndex(
            self._pixels_start + frames * frame_size,
            np.full(self._frame_count, frame_size, dtype=np.int64),
        )
