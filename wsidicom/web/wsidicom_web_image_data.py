#    Copyright 2023 SECTRA AB
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

from typing import Iterable, Iterator

from PIL.Image import Image
from pydicom.uid import UID

from wsidicom.codec import Codec
from wsidicom.geometry import Point
from wsidicom.instance import WsiDataset, WsiDicomImageData
from wsidicom.web.wsidicom_web_client import WsiDicomWebClient


class WsiDicomWebImageData(WsiDicomImageData):
    """
    ImageData for WSI DICOM instances read from DICOMWeb.

    Overrides get_decoded_tiles() and get_encoded_tiles() to fetch multiple frames.
    """

    def __init__(
        self,
        client: WsiDicomWebClient,
        dataset: WsiDataset,
        transfer_syntax: UID,
    ):
        """
        Create WsiDicomWebImageData from  dataset and provided client.

        Parameters
        ----------
        client: WsiDicomWebClient
            DICOMWeb client for reading image data.
        dataset: WsiDataset
            Dataset for the image data.
        transfer_syntax: UID
            Transfer syntax to request for image data, for example
            UID("1.2.840.10008.1.2.4.50") for JPEGBaseline8Bit.
        """
        self._client = client
        self._study_uid = dataset.uids.slide.study_instance
        self._series_uid = dataset.uids.slide.series_instance
        self._instance_uid = dataset.uids.instance
        self._transfer_syntax = transfer_syntax
        codec = Codec.create(
            self.transfer_syntax,
            dataset.samples_per_pixel,
            dataset.bits,
            dataset.tile_size,
            dataset.photometric_interpretation,
        )
        super().__init__([dataset], codec)

    @property
    def transfer_syntax(self) -> UID:
        """The uid of the transfer syntax of the image."""
        return self._transfer_syntax

    def _get_decoded_tiles(
        self, tiles: Iterable[Point], z: float, path: str
    ) -> Iterator[Image]:
        """
        Return Pillow images for tiles.

        Parameters
        ----------
        tiles: Iterable[Point]
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        ----------
        Iterator[Image]
            Tiles as Images.
        """
        frame_indices = [self._get_frame_index(tile, z, path) for tile in tiles]

        frames = self._get_tile_frames(
            frame_index for frame_index in frame_indices if frame_index != -1
        )
        for frame_index in frame_indices:
            if frame_index == -1:
                yield self.blank_tile
            frame = next(frames)
            yield self.decoder.decode(frame)

    def _get_encoded_tiles(
        self, tiles: Iterable[Point], z: float, path: str
    ) -> Iterator[bytes]:
        """
        Return bytes for tiles.

        Parameters
        ----------
        tiles: Iterable[Point]
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        ----------
        Iterator[Image]
            Tiles as Images.
        """
        frame_indices = [self._get_frame_index(tile, z, path) for tile in tiles]

        frames = self._get_tile_frames(
            frame_index for frame_index in frame_indices if frame_index != -1
        )
        for frame_index in frame_indices:
            if frame_index == -1:
                yield self.blank_encoded_tile
            return next(frames)

    def _get_tile_frame(self, frame_index: int) -> bytes:
        # First frame for DICOM web is 1.
        return next(
            self._client.get_frames(
                self._study_uid,
                self._series_uid,
                self._instance_uid,
                [frame_index + 1],
                self._transfer_syntax,
            )
        )

    def _get_tile_frames(self, frame_indices: Iterable[int]) -> Iterator[bytes]:
        # First frame for DICOM web is 1.
        return self._client.get_frames(
            self._study_uid,
            self._series_uid,
            self._instance_uid,
            [frame_index + 1 for frame_index in frame_indices],
            self._transfer_syntax,
        )
