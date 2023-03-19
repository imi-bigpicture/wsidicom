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

from pathlib import Path
from typing import List
from pydicom.uid import UID

from wsidicom.dataset import WsiDataset
from wsidicom.image_data import WsiDicomImageData
from wsidicom.wsidicom_web.wsidicom_web_client import WsiDicomWebClient


class WsiDicomWebImageData(WsiDicomImageData):
    """ImageData for WSI DICOM instances read from DICOM Web."""

    def __init__(
        self,
        client: WsiDicomWebClient,
        dataset: WsiDataset,
    ):
        """Create WsiDicomWebImageData from provided dataset and read image data using
        provided client.

        Parameters
        ----------
        client: WsiDicomWebClient
            DICOM Web client for reading image data.
        dataset: WsiDataset
            Dataset for the image data.
        """
        self._client = client
        self._study_uid = dataset.uids.slide.study_instance
        self._series_uid = dataset.uids.slide.series_instance
        self._instance_uid = dataset.uids.instance
        super().__init__([dataset])

    @property
    def files(self) -> List[Path]:
        return []

    @property
    def transfer_syntax(self) -> UID:
        """The uid of the transfer syntax of the image."""
        return self._dataset.transfer_syntax  # TODO

    def close(self) -> None:
        """Should close any open files."""
        pass

    def _get_tile_frame(self, frame_index: int) -> bytes:
        # First frame for DICOM web is 1.
        return self._client.get_frame(
            self._study_uid, self._series_uid, self._instance_uid, frame_index + 1
        )
