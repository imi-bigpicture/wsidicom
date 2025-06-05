from http import HTTPStatus
from typing import Any, Dict, List, Optional, Sequence

import pytest
from dicomweb_client import DICOMwebClient
from pydicom import Dataset
from pydicom.uid import UID, VLWholeSlideMicroscopyImageStorage, generate_uid
from pytest_mock import MockerFixture
from requests import HTTPError, Response

from tests.data_gen import create_main_dataset
from wsidicom.web.wsidicom_web_client import (
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
def throws_on_include_field():
    yield None


@pytest.fixture()
def throws_on_search_filter():
    yield None


@pytest.fixture()
def include_other_sop_classes():
    yield False


@pytest.fixture()
def instance_metadata():
    yield create_main_dataset()


@pytest.fixture()
def dicom_web_client(
    study_instance_uid: UID,
    series_instance_uid: UID,
    sop_instance_uid: UID,
    throws_on_include_field: Optional[Dict[str, str]],
    throws_on_search_filter: Optional[Dict[str, str]],
    include_other_sop_classes: bool,
    instance_metadata: Dataset,
    mocker: MockerFixture,
):
    fixture_study_instance_uid = study_instance_uid
    fixture_series_instance_uid = series_instance_uid
    client = DICOMwebClient("http://localhost")

    def raise_error(status_code: HTTPStatus, message: str):
        response = Response()
        response.status_code = status_code
        response._content = message.encode()
        raise HTTPError("", response=response)

    def search_for_instances(
        study_instance_uid: Optional[str] = None,
        series_instance_uid: Optional[str] = None,
        fuzzymatching: Optional[bool] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        fields: Optional[Sequence[str]] = None,
        search_filters: Optional[Dict[str, Any]] = None,
        get_remaining: bool = False,
    ) -> List[Dict[str, dict]]:
        if throws_on_include_field is not None and fields is not None:
            for field_key, error in throws_on_include_field.items():
                if field_key in fields:
                    raise_error(HTTPStatus.BAD_REQUEST, error)
        if throws_on_search_filter is not None and search_filters is not None:
            for search_filter_key, error in throws_on_search_filter.items():
                if search_filter_key in search_filters:
                    raise_error(HTTPStatus.BAD_REQUEST, error)
        wsi_instance = {
            STUDY_INSTANCE_UID: {"vr": "UI", "Value": [fixture_study_instance_uid]},
            SERIES_INSTANCE_UID: {
                "vr": "UI",
                "Value": [fixture_series_instance_uid],
            },
            SOP_INSTANCE_UID: {"vr": "UI", "Value": [sop_instance_uid]},
            SOP_CLASS_UID: {"vr": "UI", "Value": [VLWholeSlideMicroscopyImageStorage]},
        }
        if not include_other_sop_classes:
            return [wsi_instance]
        other_instance = {
            STUDY_INSTANCE_UID: {"vr": "UI", "Value": [fixture_study_instance_uid]},
            SERIES_INSTANCE_UID: {
                "vr": "UI",
                "Value": [fixture_series_instance_uid],
            },
            SOP_INSTANCE_UID: {"vr": "UI", "Value": [generate_uid()]},
            SOP_CLASS_UID: {"vr": "UI", "Value": [generate_uid()]},
        }
        return [wsi_instance, other_instance]

    def retrieve_instance_metadata(
        study_instance_uid: str,
        series_instance_uid: str,
        sop_instance_uid: str,
    ) -> Dict[str, dict]:
        return instance_metadata.to_json_dict()

    client.search_for_instances = mocker.MagicMock(
        client.search_for_instances, search_for_instances
    )
    client.retrieve_instance_metadata = mocker.MagicMock(
        client.retrieve_instance_metadata, retrieve_instance_metadata
    )
    yield client


class TestWsiDicomWebClient:
    @pytest.mark.parametrize(
        "throws_on_include_field",
        [
            {
                "AvailableTransferSyntaxUID": "unknown/unsupported QIDO attribute: AvailableTransferSyntaxUID"
            },
            None,
        ],
    )
    @pytest.mark.parametrize(
        "throws_on_search_filter",
        [{SOP_CLASS_UID: "SOPClassUID is not a supported instance"}, None],
    )
    def test_get_wsi_instances(
        self,
        dicom_web_client: DICOMwebClient,
        study_instance_uid: UID,
        series_instance_uid: UID,
    ):
        # Arrange
        client = WsiDicomWebClient(dicom_web_client)

        # Act
        instances = client.get_wsi_instances(study_instance_uid, [series_instance_uid])

        # Assert
        assert len(list(instances)) == 1
        assert dicom_web_client.search_for_instances.called

    @pytest.mark.parametrize(
        "throws_on_search_filter",
        [{SERIES_INSTANCE_UID: "Some other error message"}],
    )
    @pytest.mark.parametrize("include_other_sop_classes", [False, True])
    def test_get_wsi_instances_raise_on_search_filter_is_not_handled(
        self,
        dicom_web_client: DICOMwebClient,
        study_instance_uid: UID,
        series_instance_uid: UID,
    ):
        # Arrange
        client = WsiDicomWebClient(dicom_web_client)

        # Act and Assert
        with pytest.raises(HTTPError):
            list(client.get_wsi_instances(study_instance_uid, [series_instance_uid]))

        # Assert
        assert dicom_web_client.search_for_instances.called

    def test_get_instance(
        self,
        dicom_web_client: DICOMwebClient,
        study_instance_uid: UID,
        series_instance_uid: UID,
        sop_instance_uid: UID,
        instance_metadata: Dataset,
    ):
        # Arrange
        client = WsiDicomWebClient(dicom_web_client)

        # Act
        instance = client.get_instance(
            study_instance_uid, series_instance_uid, sop_instance_uid
        )

        # Assert
        assert instance == instance_metadata
        assert dicom_web_client.retrieve_instance_metadata.called
        assert dicom_web_client.retrieve_instance_metadata.call_count == 1
