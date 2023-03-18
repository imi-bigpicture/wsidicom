from enum import Enum
from functools import cached_property
from typing import List

from dicomweb_client.api import DICOMwebClient
from dicomweb_client.session_utils import create_session_from_auth
from pydicom import Dataset
from pydicom.uid import UID
from requests.auth import AuthBase

from wsidicom.dataset import WsiDataset
from wsidicom.uid import WSI_SOP_CLASS_UID


class DicomTags(Enum):
    SOP_CLASS_UID = "00080016"
    SOP_INSTANCE_UID = "00080018"


class DicomWebClient:
    def __init__(
        self, hostname: str, qido_prefix: str, wado_prefix: str, auth: AuthBase
    ):
        self._client = DICOMwebClient(
            hostname,
            qido_url_prefix=qido_prefix,
            wado_url_prefix=wado_prefix,
            session=create_session_from_auth(auth),
        )

    def get_instances(self, study_uid: UID, series_uid: UID) -> List[UID]:
        return [
            UID(instance[DicomTags.SOP_INSTANCE_UID.value]["Value"][0])
            for instance in self._client.search_for_instances(study_uid, series_uid)
            if UID(instance[DicomTags.SOP_CLASS_UID.value]["Value"][0])
            == WSI_SOP_CLASS_UID
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
            media_types=("image/jpeg",),
        )
        return frames[0]


class WsiDicomWeb:
    def __init__(
        self, client: DicomWebClient, study_uid: UID, series_uid: UID, instance_uid: UID
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
