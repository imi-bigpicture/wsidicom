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

from pathlib import Path
from typing import List, OrderedDict, Sequence, Union

from pydicom.uid import UID

from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.instance import WsiDicomImageData
from wsidicom.file.wsidicom_file import WsiDicomFile


class WsiDicomFileImageData(WsiDicomImageData):
    """Represents image data read from dicom file(s). Image data can
    be sparsly or fully tiled and/or concatenated."""

    def __init__(self, files: Union[WsiDicomFile, Sequence[WsiDicomFile]]) -> None:
        """Create WsiDicomFileImageData from frame data in files.

        Parameters
        ----------
        files: Union[WsiDicomFile, Sequence[WsiDicomFile]]
            Single or list of WsiDicomFiles containing frame data.
        """
        if not isinstance(files, Sequence):
            files = [files]

        # Key is frame offset
        self._files = OrderedDict(
            (file.frame_offset, file)
            for file in sorted(files, key=lambda file: file.frame_offset)
        )
        self._transfer_syntax = files[0].transfer_syntax
        super().__init__([file.dataset for file in self._files.values()])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._files.values()})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of files {self._files.values()}"

    @property
    def files(self) -> List[Path]:
        return [file.filepath for file in self._files.values()]

    @property
    def transfer_syntax(self) -> UID:
        """The uid of the transfer syntax of the image."""
        return self._transfer_syntax

    def _get_file(self, frame_index: int) -> WsiDicomFile:
        """Return file contaning frame index. Raises WsiDicomNotFoundError if
        frame is not found.

        Parameters
        ----------
        frame_index: int
             Frame index to get

        Returns
        ----------
        WsiDicomFile
            File containing the frame
        """
        for frame_offset, file in self._files.items():
            if (
                frame_index < frame_offset + file.frame_count
                and frame_index >= frame_offset
            ):
                return file

        raise WsiDicomNotFoundError(f"Frame index {frame_index}", "instance")

    def _get_tile_frame(self, frame_index: int) -> bytes:
        """Return tile frame for frame index.

        Parameters
        ----------
        frame_index: int
             Frame index to get

        Returns
        ----------
        bytes
            The frame in bytes
        """
        file = self._get_file(frame_index)
        tile_frame = file.read_frame(frame_index)
        return tile_frame

    def close(self) -> None:
        for file in self._files.values():
            file.close()
