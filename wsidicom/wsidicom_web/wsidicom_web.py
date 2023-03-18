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

from functools import cached_property

from pydicom.uid import UID

from wsidicom.dataset import WsiDataset
from wsidicom.wsidicom_web.wsidicom_web_client import WsiDicomWebClient


class WsiDicomWeb:
    def __init__(
        self,
        client: WsiDicomWebClient,
        study_uid: UID,
        series_uid: UID,
        instance_uid: UID,
    ):
        self._study_uid = study_uid
        self._series_uid = series_uid
        self._instance_uid = instance_uid
        self._client = client

    @cached_property
    def dataset(self) -> WsiDataset:
        dataset = self._client.get_instance(
            self._study_uid, self._series_uid, self._instance_uid
        )
        self._dataset = WsiDataset(dataset)
        return self._dataset

    @property
    def transfer_syntax(self) -> UID:
        return UID(self.dataset["TransferSyntaxUID"].value)

    def get_tile(self, tile: int) -> bytes:
        # First frame for DICOM web is 1.
        return self._client.get_frame(
            self._study_uid, self._series_uid, self._instance_uid, tile + 1
        )
