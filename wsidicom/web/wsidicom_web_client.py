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

from http import HTTPStatus
import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

from dicomweb_client.api import DICOMfileClient, DICOMwebClient
from dicomweb_client.session_utils import create_session_from_auth
from pydicom import Dataset
from pydicom.uid import (
    UID,
)
from requests import HTTPError, Session
from requests.auth import AuthBase

from wsidicom.codec import determine_media_type
from wsidicom.uid import ANN_SOP_CLASS_UID, WSI_SOP_CLASS_UID

SOP_CLASS_UID = "00080016"
SOP_INSTANCE_UID = "00080018"
SERIES_INSTANCE_UID = "0020000E"
AVAILABLE_SOP_TRANSFER_SYNTAX_UID = "00083002"


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

    def get_wsi_instances(
        self, study_uid: UID, series_uids: Iterable[UID]
    ) -> Iterator[Tuple[UID, UID, Optional[Set[UID]]]]:
        """
        Get instance uids for WSI instances in a study.

        Parameters
        ----------
        study_uid: UID
            Study UID of the study.
        series_uids: Iterable[UID]
            Series UIDs in the study.

        Returns
        ----------
        Iterator[Tuple[UID, UID, Optional[Set[UID]]]]
            Iterator of series and instance uid and optionally available transfer syntax
             uids for WSI instances in the study and series.
        """
        return self._get_intances(study_uid, series_uids, WSI_SOP_CLASS_UID)

    def get_annotation_instances(
        self, study_uid: UID, series_uids: Iterable[UID]
    ) -> Iterator[Tuple[UID, UID, Optional[Set[UID]]]]:
        """
        Get instance uids of Annotation instances in a study.

        Parameters
        ----------
        study_uid: UID
            Study UID of the study.
        series_uids: Iterable[UID]
            Series UIDs in the study.

        Returns
        ----------
        Iterator[Tuple[UID, UID, Optional[Set[UID]]]]
            Iterator of series and instance uid and optionally available transfer syntax
            uids for Annotation instances in the study and series.
        """
        return self._get_intances(study_uid, series_uids, ANN_SOP_CLASS_UID)

    def get_instance(
        self, study_uid: UID, series_uid: UID, instance_uid: UID
    ) -> Dataset:
        """
        Get instance metadata.

        Parameters
        ----------
        study_uid: UID
            Study UID of the instance.
        series_uid: UID
            Series UID of the instance.
        instance_uid: UID
            Instance UID of the instance.

        Returns
        ----------
        Dataset
            Instance metadata.
        """
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
        """
        Get frames from an instance.

        Parameters
        ----------
        study_uid: UID
            Study UID of the instance.
        series_uid: UID
            Series UID of the instance.
        instance_uid: UID
            Instance UID of the instance.
        frame_indices: List[int]
            List of frame indices to get. Note frames are indexed starting from 1.
        transfer_syntax: UID
            Transfer syntax of the to get frames.

        Returns
        ----------
        Iterator[bytes]
            Iterator of frames.
        """
        return self._client.iter_instance_frames(
            study_uid,
            series_uid,
            instance_uid,
            frame_numbers=frame_indices,
            media_types=(self._transfer_syntax_to_media_type(transfer_syntax),),
        )

    def is_transfer_syntax_supported(
        self, study_uid: UID, series_uid: UID, instance_uid: UID, transfer_syntax: UID
    ) -> bool:
        """Check if transfer syntax is supported for retrieving frames.

        Parameters
        ----------
        study_uid: UID
            Study UID of the instance.
        series_uid: UID
            Series UID of the instance.
        instance_uid: UID
            Instance UID of the instance.
        transfer_syntax: UID
            Transfer syntax to check.

        Returns
        ----------
        bool
            True if transfer syntax is supported, False otherwise.
        """
        try:
            self.get_frames(study_uid, series_uid, instance_uid, [1], transfer_syntax)
        except HTTPError as exception:
            if exception.response.status_code == HTTPStatus.NOT_ACCEPTABLE:
                logging.debug(
                    f"Transfer syntax {transfer_syntax} not supported "
                    f"for {instance_uid}."
                )
                return False
            raise exception
        logging.debug(
            f"Transfer syntax {transfer_syntax} supported for {instance_uid}."
        )
        return True

    def _get_intances(
        self, study_uid: UID, series_uids: Iterable[UID], sop_class_uid
    ) -> Iterator[Tuple[UID, UID, Optional[Set[UID]]]]:
        """Get series, instance, and optionally available transfer syntax uids for
        instances of SOP class in study.

        Parameters
        ----------
        study_uid: UID
            Study UID of the study.
        series_uids: Iterable[UID]
            Series UIDs in the study.
        sop_class_uid: UID
            SOP Class UID of the instances.

        Returns
        ----------
        Iterator[Tuple[UID, UID, Optional[Set[UID]]]]
            Iterator of series and instance uid and optionally available transfer syntax
            uids for instances in the study and series.
        """
        if isinstance(self._client, DICOMfileClient):
            # DICOMfileClient does not support searching for instances by
            # series instance uid as search filter
            return (
                self._get_uids_from_response(instance, series_uid)
                for series_uid in series_uids
                for instance in self._client.search_for_instances(
                    study_uid,
                    series_uid,
                    search_filters={SOP_CLASS_UID: sop_class_uid},
                )
            )
        return (
            self._get_uids_from_response(instance)
            for instance in self._client.search_for_instances(
                study_uid,
                fields=["AvailableTransferSyntaxUID"],
                search_filters={
                    SOP_CLASS_UID: sop_class_uid,
                    SERIES_INSTANCE_UID: series_uids,
                },
            )
        )

    @staticmethod
    def _get_uids_from_response(
        response: Dict[str, Dict[Any, Any]], series_uid: Optional[UID] = None
    ) -> Tuple[UID, UID, Optional[Set[UID]]]:
        """Get series, instance, and optionally transfer syntax uids from response.

        Parameters
        ----------
        response: Dict[str, Dict[Any, Any]]
            Response from server for an instance.

        Returns
        ----------
        Tuple[UID, UID, Optional[Set[UID]]]
            Series and instance uid and optionally available transfer syntax
            uids for instances in response.
        """
        available_transfer_syntaxes = response.get(
            AVAILABLE_SOP_TRANSFER_SYNTAX_UID, None
        )
        return (
            series_uid
            if series_uid is not None
            else UID(response[SERIES_INSTANCE_UID]["Value"][0]),
            UID(response[SOP_INSTANCE_UID]["Value"][0]),
            set(available_transfer_syntaxes["Value"])
            if available_transfer_syntaxes
            else None,
        )

    @staticmethod
    def _get_sop_class_uid_from_response(response: Dict[str, Dict[Any, Any]]) -> UID:
        """Get SOP Class UID from response.

        Parameters
        ----------
        response: Dict[str, Dict[Any, Any]]
            Response from server.

        Returns
        ----------
        UID
            SOP Class UID from response.
        """
        return UID(response[SOP_CLASS_UID]["Value"][0])

    @staticmethod
    def _transfer_syntax_to_media_type(transfer_syntax: UID) -> Tuple[str, str]:
        """Convert transfer syntax to media type.

        Parameters
        ----------
        transfer_syntax: UID
            Transfer syntax to convert.

        Returns
        ----------
        Tuple[str, str]
            Media type and transfer syntax.
        """
        try:
            return determine_media_type(transfer_syntax), transfer_syntax
        except NotImplementedError as exception:
            raise ValueError(
                f"Could not determine media type for transfer syntax {transfer_syntax}"
            ) from exception
