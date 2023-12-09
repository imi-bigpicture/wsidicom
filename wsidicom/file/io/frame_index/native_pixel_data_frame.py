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

from typing import List, Tuple


from wsidicom.file.io.frame_index.frame_index import FrameIndex
from wsidicom.file.io.frame_index.offset_table_type import OffsetTableType
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.geometry import Size


class NativePixelData(FrameIndex):
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

    def _get_pixels_start(self) -> int:
        self._validate_pixel_data_start()
        return self._file.tell()

    def _get_index(self) -> List[Tuple[int, int]]:
        """Create frame positions for uncapsulated data.

        Parameters
        ----------
        pixel_data_start: int
            Offset to first frame in pixel data.

        Returns
        ----------
        List[Tuple[int, int]]
            A list with frame positions and frame lengths.
        """
        frame_size = self._tile_size.area * self._samples_per_pixel * (self._bits // 8)
        return [
            (self._pixels_start + index * frame_size, frame_size)
            for index in range(self._frame_count)
        ]

    def _validate_pixel_data_start(self):
        """Check that pixel data tag is present and that the tag length is equal to
        expected count. Raises WsiDicomFileError otherwise.

        """
        expected_length = (
            self._tile_size.area
            * self._samples_per_pixel
            * (self._bits // 8)
            * self._frame_count
        )
        super()._validate_pixel_data_start(expected_length)
