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
import random
import sys
from enum import Enum
from io import BufferedReader
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest
from dicomweb_client import DICOMfileClient
from pydicom.misc import is_dicom
from pydicom.uid import (
    JPEG2000,
    UID,
    ExplicitVRLittleEndian,
    JPEGBaseline8Bit,
)
from upath import UPath

from tests.data_gen import create_layer_file
from tests.file.io.test_wsidicom_writer import WsiDicomTestImageData
from wsidicom import WsiDicom
from wsidicom.config import settings
from wsidicom.geometry import Size
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
        return (SLIDE_FOLDER.joinpath(path) for _, path in cls._get_parameter("path"))

    @classmethod
    def folders_and_counts(cls) -> Iterable[Tuple[Path, int, bool, bool]]:
        return (
            (
                SLIDE_FOLDER.joinpath(wsi_definition["path"]),
                wsi_definition["levels"],
                wsi_definition["label"],
                wsi_definition["overview"],
            )
            for wsi_definition in cls.test_definitions.values()
        )

    @classmethod
    def folders_and_instance_counts(cls) -> Iterable[Tuple[Path, int]]:
        return (
            (
                SLIDE_FOLDER.joinpath(wsi_definition["path"]),
                wsi_definition["instances"],
            )
            for wsi_definition in cls.test_definitions.values()
        )

    @classmethod
    def wsi_names(cls, tiling: Optional[str] = None) -> Iterable[str]:
        return (
            key
            for key, value in cls.test_definitions.items()
            if tiling is None or value["tiled"] == tiling
        )

    @classmethod
    def read_region(cls) -> Iterable[Tuple[str, UID, Dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_region", "level_transfer_syntax"
        )

    @classmethod
    def read_region_mm(cls) -> Iterable[Tuple[str, UID, Dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_region_mm", "level_transfer_syntax"
        )

    @classmethod
    def read_region_mpp(cls) -> Iterable[Tuple[str, UID, Dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_region_mpp", "level_transfer_syntax"
        )

    @classmethod
    def read_tile(cls) -> Iterable[Tuple[str, UID, Dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_tile", "level_transfer_syntax"
        )

    @classmethod
    def read_encoded_tile(cls) -> Iterable[Tuple[str, UID, Dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_encoded_tile", "level_transfer_syntax"
        )

    @classmethod
    def read_thumbnail(cls) -> Iterable[Tuple[str, UID, Dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_thumbnail", "level_transfer_syntax"
        )

    @classmethod
    def levels(cls) -> Iterable[Tuple[str, int]]:
        return cls._get_parameter("levels")

    @classmethod
    def label_hash(cls) -> Iterable[Tuple[str, UID, Optional[str]]]:
        return cls._get_parameter_and_transfer_syntax("label", "label_transfer_syntax")

    @classmethod
    def overview_hash(cls) -> Iterable[Tuple[str, UID, Optional[str]]]:
        return cls._get_parameter_and_transfer_syntax(
            "overview", "overview_transfer_syntax"
        )

    @classmethod
    def _get_test_values_and_transfer_syntax(
        cls, test_value_name: str, transfer_syntax_name: str
    ) -> Iterable[Tuple[str, UID, Dict[str, Any]]]:
        return [
            (wsi_name, UID(wsi_definition[transfer_syntax_name]), region)
            for wsi_name, wsi_definition in cls.test_definitions.items()
            for region in wsi_definition[test_value_name]
        ]

    @classmethod
    def _get_parameter_and_transfer_syntax(
        cls, parameter_name: str, transfer_syntax_name: str
    ) -> Iterable[Tuple[str, UID, Any]]:
        return [
            (
                wsi_name,
                UID(wsi_definition[transfer_syntax_name]),
                wsi_definition[parameter_name],
            )
            for wsi_name, wsi_definition in cls.test_definitions.items()
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
def wsi_file(tmp_path: Path):
    test_file_path = tmp_path.joinpath("test_im.dcm")
    create_layer_file(test_file_path)
    yield test_file_path


@pytest.fixture()
def wsi(wsi_file: Path):
    with WsiDicom.open(wsi_file) as wsi:
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
        folder = UPath(SLIDE_FOLDER).joinpath(test_definition["path"])
        if (input_type, folder) in wsis:
            return wsis[(input_type, folder)]
        if not folder.exists():
            pytest.skip(f"Folder {folder} does not exist.")
        if input_type == WsiInputType.FILE:
            wsi = WsiDicom.open(folder)
        elif input_type == WsiInputType.WEB:
            settings.open_web_theads = 1
            client = WsiDicomWebClient(
                DICOMfileClient(f"file://{folder.absolute().as_posix()}")
            )
            wsi = WsiDicom.open_web(
                client,
                test_definition["study_instance_uid"],
                test_definition["series_instance_uid"],
                [JPEGBaseline8Bit, JPEG2000, ExplicitVRLittleEndian],
            )
        elif input_type == WsiInputType.STREAM:
            streams = [
                open(file, "rb")
                for file in folder.iterdir()
                if file.is_file() and is_dicom(file)
            ]
            wsi = WsiDicom.open_streams(streams)
        else:
            raise NotImplementedError()
        wsis[(input_type, folder)] = wsi
        return wsi

    yield open_wsi
    for wsi in wsis.values():
        wsi.close()
    for stream in streams:
        stream.close()


@pytest.fixture()
def tiled_size():
    yield Size(2, 2)


@pytest.fixture()
def frame_count(tiled_size: Size):
    yield tiled_size.area


@pytest.fixture()
def rng():
    SEED = 0
    yield random.Random(SEED)


@pytest.fixture()
def bits():
    yield 8


@pytest.fixture
def samples_per_pixel():
    yield 3


@pytest.fixture
def tile_size():
    yield Size(10, 10)


@pytest.fixture()
def transfer_syntax():
    yield JPEGBaseline8Bit


@pytest.fixture()
def frames(
    rng: random.Random,
    transfer_syntax: UID,
    frame_count: int,
    bits: int,
    samples_per_pixel: int,
    tile_size: Size,
):
    if not transfer_syntax.is_encapsulated:
        min_frame_length = bits * tile_size.area * samples_per_pixel // 8
        max_frame_length = min_frame_length
    else:
        min_frame_length = 2
        max_frame_length = 100
    lengths = [
        rng.randint(min_frame_length, max_frame_length) for i in range(frame_count)
    ]
    yield [
        rng.getrandbits(length * 8).to_bytes(length, sys.byteorder)
        for length in lengths
    ]


@pytest.fixture()
def image_data(frames: List[bytes], tiled_size: Size):
    yield WsiDicomTestImageData(frames, tiled_size)
