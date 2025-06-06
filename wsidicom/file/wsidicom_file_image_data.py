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

from functools import lru_cache
from typing import List, Sequence, Union

from pydicom.uid import UID
from upath import UPath

from wsidicom.cache import DecodedFrameCache, EncodedFrameCache
from wsidicom.codec import Codec
from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.file.io import WsiDicomReader
from wsidicom.instance import WsiDicomImageData


class WsiDicomFileImageData(WsiDicomImageData):
    """
    Represents image data read from dicom file(s).

    Image data can be sparsly or fully tiled and/or concatenated.
    """

    def __init__(
        self,
        readers: Union[WsiDicomReader, Sequence[WsiDicomReader]],
        decoded_frame_cache: DecodedFrameCache,
        encoded_frame_cache: EncodedFrameCache,
    ) -> None:
        """
        Create WsiDicomFileImageData from frame data from readers.

        Parameters
        ----------
        readers: Union[WsiDicomReader, Sequence[WsiDicomReader]]
            Single or list of WsiDicomReader containing frame data.
        """
        if not isinstance(readers, Sequence):
            readers = [readers]

        self._readers = [
            file for file in sorted(readers, key=lambda file: file.frame_offset)
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
            [file.dataset for file in self._readers],
            codec,
            decoded_frame_cache,
            encoded_frame_cache,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._readers})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of files {self._readers}"

    @property
    def files(self) -> List[UPath]:
        return [
            reader.filepath for reader in self._readers if reader.filepath is not None
        ]

    @property
    def transfer_syntax(self) -> UID:
        """The uid of the transfer syntax of the image."""
        return self._transfer_syntax

    @lru_cache
    def _get_reader(self, frame_index: int) -> WsiDicomReader:
        """
        Return file containing frame index.

        Raises WsiDicomNotFoundError if frame is not found.

        Parameters
        ----------
        frame_index: int
             Frame index to get

        Returns
        -------
        WsiDicomFile
            File containing the frame
        """
        try:
            return next(
                file
                for file in self._readers
                if frame_index >= file.frame_offset
                and frame_index < file.frame_offset + file.frame_count
            )
        except StopIteration:
            raise WsiDicomNotFoundError(f"Frame index {frame_index}", "instance")

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
