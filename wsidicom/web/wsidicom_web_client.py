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

from enum import Enum
from pathlib import Path
from typing import List
from dicomweb_client import DICOMfileClient

from dicomweb_client.api import DICOMwebClient
from dicomweb_client.session_utils import create_session_from_auth
from pydicom import Dataset
from pydicom.uid import UID, JPEGBaseline8Bit
from requests.auth import AuthBase

from wsidicom.uid import ANN_SOP_CLASS_UID, WSI_SOP_CLASS_UID


class DicomTags(Enum):
    SOP_CLASS_UID = "00080016"
    SOP_INSTANCE_UID = "00080018"


class WsiDicomWebClient:
    def __init__(
        self, hostname: str, qido_prefix: str, wado_prefix: str, auth: AuthBase
    ):
        self._client = DICOMwebClient(
            hostname,
            qido_url_prefix=qido_prefix,
            wado_url_prefix=wado_prefix,
            session=create_session_from_auth(auth),
        )

    def get_wsi_instances(self, study_uid: UID, series_uid: UID) -> List[UID]:
        return [
            UID(instance[DicomTags.SOP_INSTANCE_UID.value]["Value"][0])
            for instance in self._client.search_for_instances(study_uid, series_uid)
            if UID(instance[DicomTags.SOP_CLASS_UID.value]["Value"][0])
            == WSI_SOP_CLASS_UID
        ]

    def get_ann_instances(self, study_uid: UID, series_uid: UID) -> List[UID]:
        return [
            UID(instance[DicomTags.SOP_INSTANCE_UID.value]["Value"][0])
            for instance in self._client.search_for_instances(study_uid, series_uid)
            if UID(instance[DicomTags.SOP_CLASS_UID.value]["Value"][0])
            == ANN_SOP_CLASS_UID
        ]

    def get_instance(
        self, study_uid: UID, series_uid: UID, instance_uid: UID
    ) -> Dataset:
        instance = self._client.retrieve_instance_metadata(
            study_uid, series_uid, instance_uid
        )
        return Dataset.from_json(instance)

    def get_frame(
        self, study_uid: UID, series_uid: UID, instance_uid: UID, frame_index: int
    ) -> bytes:
        frames = self._client.retrieve_instance_frames(
            study_uid,
            series_uid,
            instance_uid,
            frame_numbers=[frame_index],
            # media_types=(
            #     (
            #         "image/jpeg",
            #         JPEGBaseline8Bit,
            #     ),
            # ),
        )
        return frames[0]


class WsiDicomFileClient(WsiDicomWebClient):
    def __init__(self, path: Path):
        self._client = DICOMfileClient(
            f"file://{path.absolute().as_posix()}", in_memory=True
        )
