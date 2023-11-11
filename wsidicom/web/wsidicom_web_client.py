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

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from dicomweb_client.api import DICOMwebClient, DICOMfileClient
from dicomweb_client.session_utils import create_session_from_auth
from pydicom import Dataset
from pydicom.uid import (
    JPEG2000,
    UID,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGExtended12Bit,
)
from requests import Session
from requests.auth import AuthBase

from wsidicom.uid import ANN_SOP_CLASS_UID, WSI_SOP_CLASS_UID

SOP_CLASS_UID = "00080016"
SOP_INSTANCE_UID = "00080018"


class WsiDicomWebClient:
    def __init__(self, client: Union[DICOMwebClient, DICOMfileClient]):
        """Create a WsiDicomWebClient.

        Parameters
        ----------
        client: Union[DICOMwebClient, DICOMfileClient]
            The client to use
        """
        self._client = client

    @classmethod
    def create_client(
        cls,
        hostname: str,
        qido_prefix: Optional[str] = None,
        wado_prefix: Optional[str] = None,
        auth: Optional[Union[AuthBase, Session]] = None,
    ):
        """Create a WsiDicomWebClient.

        Parameters
        ----------
        hostname: str
            The URL of the DICOMweb server
        qido_prefix: Optional[str]
            If needed by the server, provide the prefix for QIDO services
        wado_prefix: Optional[str]
            If needed by the server, provide the prefix for WADO services
        auth: Optional[Union[AuthBase, Session]]
            If needed by the server, provide authentication credentials.
            This may be provided by either passing an object that
            inherits from requests.auth.AuthBase, or by passing a
            requests.Session object.
        """
        if isinstance(auth, AuthBase):
            session = create_session_from_auth(auth)
        else:
            session = auth

        client = DICOMwebClient(
            hostname,
            qido_url_prefix=qido_prefix,
            wado_url_prefix=wado_prefix,
            session=session,
        )

        return cls(client)

    def get_wsi_instances(self, study_uid: UID, series_uid: UID) -> Iterator[UID]:
        return self._get_instances_of_class(study_uid, series_uid, WSI_SOP_CLASS_UID)

    def get_ann_instances(self, study_uid: UID, series_uid: UID) -> Iterator[UID]:
        return self._get_instances_of_class(study_uid, series_uid, ANN_SOP_CLASS_UID)

    def get_instance(
        self, study_uid: UID, series_uid: UID, instance_uid: UID
    ) -> Dataset:
        self._client.retrieve_instance
        instance = self._client.retrieve_instance_metadata(
            study_uid, series_uid, instance_uid
        )
        return Dataset.from_json(instance)

    def get_frames(
        self,
        study_uid: UID,
        series_uid: UID,
        instance_uid: UID,
        frame_indices: List[int],
        transfer_syntax: UID,
    ) -> Iterator[bytes]:
        return self._client.iter_instance_frames(
            study_uid,
            series_uid,
            instance_uid,
            frame_numbers=frame_indices,
            media_types=(self._transfer_syntax_to_media_type(transfer_syntax),),
        )

    def _get_instances_of_class(
        self, study_uid: UID, series_uid: UID, sop_class_uid: UID
    ) -> Iterator[UID]:
        return (
            self._get_sop_instance_uid_from_response(instance)
            for instance in self._client.search_for_instances(study_uid, series_uid)
            if self._get_sop_class_uid_from_response(instance) == sop_class_uid
        )

    @staticmethod
    def _get_sop_instance_uid_from_response(response: Dict[str, Dict[Any, Any]]) -> UID:
        return UID(response[SOP_INSTANCE_UID]["Value"][0])

    @staticmethod
    def _get_sop_class_uid_from_response(response: Dict[str, Dict[Any, Any]]) -> UID:
        return UID(response[SOP_CLASS_UID]["Value"][0])

    @staticmethod
    def _transfer_syntax_to_media_type(transfer_syntax: UID) -> Tuple[str, str]:
        if transfer_syntax == JPEGBaseline8Bit or transfer_syntax == JPEGExtended12Bit:
            return (
                "image/jpeg",
                transfer_syntax,
            )
        elif transfer_syntax == JPEG2000 or transfer_syntax == JPEG2000Lossless:
            return (
                "image/jp2",
                transfer_syntax,
            )
        raise NotImplementedError()
