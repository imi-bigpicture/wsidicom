from http import HTTPStatus
from typing import Any, Dict, Optional

import pytest
from decoy import Decoy
from dicomweb_client import DICOMwebClient
from pydicom.uid import (
    UID,
    JPEGBaseline8Bit,
    VLWholeSlideMicroscopyImageStorage,
    generate_uid,
)
from requests import HTTPError, Response

from wsidicom.web.wsidicom_web_client import (
    AVAILABLE_SOP_TRANSFER_SYNTAX_UID,
    MODALITY,
    SERIES_INSTANCE_UID,
    SOP_CLASS_UID,
    SOP_INSTANCE_UID,
    WsiDicomWebClient,
)

STUDY_INSTANCE_UID = "0020000D"


@pytest.fixture()
def study_instance_uid():
    yield generate_uid()


@pytest.fixture()
def series_instance_uid():
    yield generate_uid()


@pytest.fixture()
def sop_instance_uid():
    yield generate_uid()


@pytest.fixture()
def available_sop_transfer_syntax_uid():
    yield JPEGBaseline8Bit


@pytest.fixture()
def instance_metadata(
    study_instance_uid: UID,
    series_instance_uid: UID,
    sop_instance_uid: UID,
    available_sop_transfer_syntax_uid: Optional[UID],
):
    wsi_instance = {
        STUDY_INSTANCE_UID: {"vr": "UI", "Value": [str(study_instance_uid)]},
        SERIES_INSTANCE_UID: {
            "vr": "UI",
            "Value": [str(series_instance_uid)],
        },
        SOP_INSTANCE_UID: {"vr": "UI", "Value": [str(sop_instance_uid)]},
        SOP_CLASS_UID: {
            "vr": "UI",
            "Value": [str(VLWholeSlideMicroscopyImageStorage)],
        },
    }
    if available_sop_transfer_syntax_uid is not None:
        wsi_instance.update(
            {
                AVAILABLE_SOP_TRANSFER_SYNTAX_UID: {
                    "vr": "UI",
                    "Value": [str(available_sop_transfer_syntax_uid)],
                },
            }
        )
    yield wsi_instance


# @pytest.fixture()
# def dicom_web_client(
#     study_instance_uid: UID,
#     series_instance_uid: UID,
#     sop_instance_uid: UID,
#     throws_on_include_field: Optional[Dict[str, str]],
#     throws_on_search_filter: Optional[Dict[str, str]],
#     include_other_sop_classes: bool,
#     instance_metadata: Dataset,
#     decoy: Decoy,
# ):
#     fixture_study_instance_uid = study_instance_uid
#     fixture_series_instance_uid = series_instance_uid
#     client = decoy.mock(cls=DICOMwebClient)

#     def raise_error(status_code: HTTPStatus, message: str):
#         response = Response()
#         response.status_code = status_code
#         response._content = message.encode()
#         raise HTTPError("", response=response)

#     def search_for_instances(
#         study_instance_uid: Optional[str] = None,
#         series_instance_uid: Optional[str] = None,
#         fuzzymatching: Optional[bool] = None,
#         limit: Optional[int] = None,
#         offset: Optional[int] = None,
#         fields: Optional[Sequence[str]] = None,
#         search_filters: Optional[Dict[str, Any]] = None,
#         get_remaining: bool = False,
#     ) -> List[Dict[str, dict]]:
#         if throws_on_include_field is not None and fields is not None:
#             for field_key, error in throws_on_include_field.items():
#                 if field_key in fields:
#                     raise_error(HTTPStatus.BAD_REQUEST, error)
#         if throws_on_search_filter is not None and search_filters is not None:
#             for search_filter_key, error in throws_on_search_filter.items():
#                 if search_filter_key in search_filters:
#                     raise_error(HTTPStatus.BAD_REQUEST, error)
#         wsi_instance = {
#             STUDY_INSTANCE_UID: {"vr": "UI", "Value": [fixture_study_instance_uid]},
#             SERIES_INSTANCE_UID: {
#                 "vr": "UI",
#                 "Value": [fixture_series_instance_uid],
#             },
#             SOP_INSTANCE_UID: {"vr": "UI", "Value": [sop_instance_uid]},
#             SOP_CLASS_UID: {"vr": "UI", "Value": [VLWholeSlideMicroscopyImageStorage]},
#         }
#         if not include_other_sop_classes:
#             return [wsi_instance]
#         other_instance = {
#             STUDY_INSTANCE_UID: {"vr": "UI", "Value": [fixture_study_instance_uid]},
#             SERIES_INSTANCE_UID: {
#                 "vr": "UI",
#                 "Value": [fixture_series_instance_uid],
#             },
#             SOP_INSTANCE_UID: {"vr": "UI", "Value": [generate_uid()]},
#             SOP_CLASS_UID: {"vr": "UI", "Value": [generate_uid()]},
#         }
#         return [wsi_instance, other_instance]

#     def retrieve_instance_metadata(
#         study_instance_uid: str,
#         series_instance_uid: str,
#         sop_instance_uid: str,
#     ) -> Dict[str, dict]:
#         return instance_metadata.to_json_dict()

#     client.search_for_instances = mocker.MagicMock(
#         client.search_for_instances, search_for_instances
#     )
#     client.retrieve_instance_metadata = mocker.MagicMock(
#         client.retrieve_instance_metadata, retrieve_instance_metadata
#     )
#     yield client


class TestWsiDicomWebClient:
    @pytest.mark.parametrize("available_sop_transfer_syntax_uid", [None])
    def test_get_wsi_instances_available_transfer_syntax_uid_field_not_supported_fallback(
        self,
        decoy: Decoy,
        study_instance_uid: UID,
        series_instance_uid: UID,
        sop_instance_uid: UID,
        instance_metadata: Dict[str, Any],
    ):
        # Arrange
        def create_response(code: HTTPStatus, message: str) -> Response:
            response = Response()
            response.status_code = code
            response._content = message.encode()
            return response

        dicom_web_client = decoy.mock(cls=DICOMwebClient)
        decoy.when(
            dicom_web_client.search_for_instances(
                study_instance_uid=study_instance_uid,
                fields=["AvailableTransferSyntaxUID"],
                search_filters={
                    SOP_CLASS_UID: VLWholeSlideMicroscopyImageStorage,
                    SERIES_INSTANCE_UID: [series_instance_uid],
                    MODALITY: "SM",
                },
            )
        ).then_raise(
            HTTPError(
                response=create_response(
                    HTTPStatus.BAD_REQUEST,
                    "unknown/unsupported QIDO attribute: AvailableTransferSyntaxUID",
                )
            )
        )
        decoy.when(
            dicom_web_client.search_for_instances(
                study_instance_uid=study_instance_uid,
                fields=[],
                search_filters={
                    SOP_CLASS_UID: VLWholeSlideMicroscopyImageStorage,
                    SERIES_INSTANCE_UID: [series_instance_uid],
                    MODALITY: "SM",
                },
            )
        ).then_return([instance_metadata])

        client = WsiDicomWebClient(dicom_web_client)

        # Act
        instances = client.get_wsi_instances(study_instance_uid, [series_instance_uid])

        # Assert
        instances = list(instances)
        assert len(instances) == 1
        assert instances[0][0] == series_instance_uid
        assert instances[0][1] == sop_instance_uid
        assert instances[0][2] is None

    def test_get_wsi_instances_sop_class_uid_search_filter_not_supported_fallback(
        self,
        decoy: Decoy,
        study_instance_uid: UID,
        series_instance_uid: UID,
        sop_instance_uid: UID,
        available_sop_transfer_syntax_uid: UID,
        instance_metadata: Dict[str, Any],
    ):
        # Arrange

        def create_response(code: HTTPStatus, message: str) -> Response:
            response = Response()
            response.status_code = code
            response._content = message.encode()
            return response

        dicom_web_client = decoy.mock(cls=DICOMwebClient)
        decoy.when(
            dicom_web_client.search_for_instances(
                study_instance_uid=study_instance_uid,
                fields=["AvailableTransferSyntaxUID"],
                search_filters={
                    SOP_CLASS_UID: VLWholeSlideMicroscopyImageStorage,
                    SERIES_INSTANCE_UID: [series_instance_uid],
                    MODALITY: "SM",
                },
            )
        ).then_raise(
            HTTPError(
                response=create_response(
                    HTTPStatus.BAD_REQUEST, "SOPClassUID is not a supported instance"
                )
            )
        )
        decoy.when(
            dicom_web_client.search_for_instances(
                study_instance_uid=study_instance_uid,
                fields=["AvailableTransferSyntaxUID"],
                search_filters={
                    SERIES_INSTANCE_UID: [series_instance_uid],
                    MODALITY: "SM",
                },
            )
        ).then_return([instance_metadata])

        client = WsiDicomWebClient(dicom_web_client)

        # Act
        instances = client.get_wsi_instances(study_instance_uid, [series_instance_uid])

        # Assert
        instances = list(instances)
        assert len(instances) == 1
        assert instances[0][0] == series_instance_uid
        assert instances[0][1] == sop_instance_uid
        assert instances[0][2] == set([available_sop_transfer_syntax_uid])

    def test_get_wsi_instances_other_search_filter_not_supported_throw(
        self, decoy: Decoy, study_instance_uid: UID, series_instance_uid: UID
    ):
        # Arrange

        def create_response(code: HTTPStatus, message: str) -> Response:
            response = Response()
            response.status_code = code
            response._content = message.encode()
            return response

        dicom_web_client = decoy.mock(cls=DICOMwebClient)
        decoy.when(
            dicom_web_client.search_for_instances(
                study_instance_uid=study_instance_uid,
                fields=["AvailableTransferSyntaxUID"],
                search_filters={
                    SOP_CLASS_UID: VLWholeSlideMicroscopyImageStorage,
                    SERIES_INSTANCE_UID: [series_instance_uid],
                    MODALITY: "SM",
                },
            )
        ).then_raise(
            HTTPError(
                response=create_response(
                    HTTPStatus.BAD_REQUEST, "Some other error message"
                )
            )
        )

        client = WsiDicomWebClient(dicom_web_client)

        # Act and Assert
        with pytest.raises(HTTPError):
            list(client.get_wsi_instances(study_instance_uid, [series_instance_uid]))

    def test_get_instance(
        self,
        decoy: Decoy,
        study_instance_uid: UID,
        series_instance_uid: UID,
        sop_instance_uid: UID,
        available_sop_transfer_syntax_uid: UID,
        instance_metadata: Dict[str, Any],
    ):

        dicom_web_client = decoy.mock(cls=DICOMwebClient)
        decoy.when(
            dicom_web_client.search_for_instances(
                study_instance_uid=study_instance_uid,
                fields=["AvailableTransferSyntaxUID"],
                search_filters={
                    SOP_CLASS_UID: VLWholeSlideMicroscopyImageStorage,
                    SERIES_INSTANCE_UID: [series_instance_uid],
                    MODALITY: "SM",
                },
            )
        ).then_return([instance_metadata])

        client = WsiDicomWebClient(dicom_web_client)

        # Act
        instances = client.get_wsi_instances(study_instance_uid, [series_instance_uid])

        # Assert
        instances = list(instances)
        assert len(instances) == 1
        assert instances[0][0] == series_instance_uid
        assert instances[0][1] == sop_instance_uid
        assert instances[0][2] == set([available_sop_transfer_syntax_uid])
