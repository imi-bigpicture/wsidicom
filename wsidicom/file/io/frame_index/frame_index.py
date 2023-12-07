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

"""Index for frame positions and length in image data."""

from abc import abstractmethod
from functools import cached_property
from typing import List, Optional, Tuple

from wsidicom.errors import WsiDicomFileError
from wsidicom.tags import PixelDataTag
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.file.io.frame_index.offset_table_type import OffsetTableType


class FrameIndex:
    def __init__(self, file: WsiDicomIO, pixel_data_start: int, frame_count: int):
        self._file = file
        self._frame_count = frame_count
        self._pixel_data_start = pixel_data_start
        self._file.seek(self._pixel_data_start)
        self._pixels_start = self._get_pixels_start()

    @cached_property
    def index(self) -> List[Tuple[int, int]]:
        """Return a list of frame positions and lengths."""
        self._file.seek(self._pixel_data_start)
        index = self._get_index()
        self._validate_frame_index(index)
        return index

    @property
    @abstractmethod
    def offset_table_type(self) -> OffsetTableType:
        raise NotImplementedError()

    @abstractmethod
    def _get_index(self) -> List[Tuple[int, int]]:
        """Return a list of frame positions and lengths."""
        raise NotImplementedError()

    @abstractmethod
    def _get_pixels_start(self) -> int:
        """Parse pixel data start and return position of first frame."""
        raise NotImplementedError()

    def _validate_frame_index(self, frame_index: List[Tuple[int, int]]):
        """Validate frame index.

        Parameters
        ----------
        frame_index: List[Tuple[int, int]]
            Frame index.
        """
        if len(frame_index) < self._frame_count:
            raise WsiDicomFileError(
                str(self._file),
                (
                    f"ImageData contained less frames {len(frame_index)} than "
                    f"NumberOfFrames {self._frame_count}."
                ),
            )
        if len(frame_index) > self._frame_count:
            raise WsiDicomFileError(
                str(self._file),
                (
                    f"ImageData contained more fragments {len(frame_index)} than "
                    f"NumberOfFrames {self._frame_count} and fragmented frames are not "
                    "supported."
                ),
            )

    def _validate_pixel_data_start(self, expected_length: Optional[int]):
        """Check that pixel data tag is present and that the tag length is equal to
        expected count. Raises WsiDicomFileError otherwise.

        """
        tag = self._file.read_tag()
        if tag != PixelDataTag:
            WsiDicomFileError(str(self._file), "Expected PixelData tag")
        self._file.read_tag_vr()
        length = self._file.read_tag_length(True)
        if expected_length is not None and length != expected_length:
            raise WsiDicomFileError(
                str(self._file),
                f"Expected {expected_length} length when reading Pixel data, got {length}.",
            )
        elif expected_length is None and length != 0xFFFFFFFF:
            raise WsiDicomFileError(
                str(self._file),
                f"Expected undefined length when reading Pixel data, got {length}.",
            )
