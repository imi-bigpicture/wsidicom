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

import json
import os
from enum import Enum
from io import BufferedReader
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pytest
from dicomweb_client import DICOMfileClient
from pydicom.uid import JPEGBaseline8Bit

from tests.data_gen import create_layer_file
from wsidicom import WsiDicom
from wsidicom.web.wsidicom_web_client import WsiDicomWebClient

SLIDE_FOLDER = Path(os.environ.get("WSIDICOM_TESTDIR", "tests/testdata/slides"))
REGION_DEFINITIONS_FILE = "tests/testdata/region_definitions.json"


class WsiInputType(Enum):
    FILE = "file"
    STREAM = "stream"
    WEB = "web"


class WsiTestDefinitions:
    """Interface for reading test parameters from definition file."""

    with open(REGION_DEFINITIONS_FILE) as json_file:
        test_definitions: Dict[str, Dict[str, Any]] = json.load(json_file)
    if len(test_definitions) == 0:
        pytest.skip("No test definition found, skipping.")

    @classmethod
    def folders(cls) -> Iterable[Path]:
        return (
            SLIDE_FOLDER.joinpath(wsi_name, path)
            for wsi_name, path in cls._get_parameter("path")
        )

    @classmethod
    def wsi_names(cls) -> Iterable[str]:
        return cls.test_definitions.keys()

    @classmethod
    def read_region(cls) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return cls._get_dict("read_region")

    @classmethod
    def read_region_mm(cls) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return cls._get_dict("read_region_mm")

    @classmethod
    def read_region_mpp(cls) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return cls._get_dict("read_region_mpp")

    @classmethod
    def read_tile(cls) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return cls._get_dict("read_tile")

    @classmethod
    def read_encoded_tile(cls) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return cls._get_dict("read_encoded_tile")

    @classmethod
    def read_thumbnail(cls) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return cls._get_dict("read_thumbnail")

    @classmethod
    def levels(cls) -> Iterable[Tuple[str, int]]:
        return cls._get_parameter("levels")

    @classmethod
    def has_label(cls) -> Iterable[Tuple[str, bool]]:
        return cls._get_parameter("label")

    @classmethod
    def has_overview(cls) -> Iterable[Tuple[str, bool]]:
        return cls._get_parameter("overview")

    @classmethod
    def _get_dict(cls, region_name: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return [
            (wsi_name, region)
            for wsi_name, wsi_definition in cls.test_definitions.items()
            for region in wsi_definition[region_name]
        ]

    @classmethod
    def _get_parameter(cls, parameter_name: str) -> Iterable[Tuple[str, Any]]:
        return [
            (
                wsi_name,
                wsi_definition[parameter_name],
            )
            for wsi_name, wsi_definition in cls.test_definitions.items()
        ]


@pytest.fixture()
def wsi(tmp_path: Path):
    test_file_path = tmp_path.joinpath("test_im.dcm")
    create_layer_file(test_file_path)
    with WsiDicom.open(tmp_path) as wsi:
        yield wsi


@pytest.fixture(scope="module")
def wsi_factory():
    """Fixture providing a callable that takes a wsi name and input type and returns a
    WsiDicom object. Caches opened objects and closes when tests using fixture are done.
    """
    streams: List[BufferedReader] = []
    wsis: Dict[Tuple[WsiInputType, Path], WsiDicom] = {}

    def open_wsi(
        wsi_name: str, input_type: WsiInputType = WsiInputType.FILE
    ) -> WsiDicom:
        test_definition = WsiTestDefinitions.test_definitions[wsi_name]
        folder = Path(SLIDE_FOLDER).joinpath(wsi_name, test_definition["path"])
        if (input_type, folder) in wsis:
            return wsis[(input_type, folder)]
        if not folder.exists():
            pytest.skip(f"Folder {folder} does not exist.")
        if input_type == WsiInputType.FILE:
            wsi = WsiDicom.open(folder)
        elif input_type == WsiInputType.WEB:
            client = WsiDicomWebClient(
                DICOMfileClient(f"file://{folder.absolute().as_posix()}")
            )
            wsi = WsiDicom.open_web(
                client,
                test_definition["study_instance_uid"],
                test_definition["series_instance_uid"],
                JPEGBaseline8Bit,
            )
        elif input_type == WsiInputType.STREAM:
            streams = [open(file, "rb") for file in folder.iterdir() if file.is_file()]
            wsi = WsiDicom.open(streams)
        else:
            raise NotImplementedError()
        wsis[(input_type, folder)] = wsi
        return wsi

    yield open_wsi
    for wsi in wsis.values():
        wsi.close()
    for stream in streams:
        stream.close()
