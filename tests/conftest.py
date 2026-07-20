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
import platform
import random
import sys
from concurrent.futures import Executor, ThreadPoolExecutor
from enum import Enum
from io import BufferedReader
from pathlib import Path
from typing import Any

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
from wsidicom.config import Settings
from wsidicom.geometry import Size
from wsidicom.metadata.uid_generator import CallableUidGenerator
from wsidicom.thread import ReadExecutor
from wsidicom.web.wsidicom_web_client import WsiDicomWebClient

SLIDE_FOLDER = Path(os.environ.get("WSIDICOM_TESTDIR", "tests/testdata/slides"))
REGION_DEFINITIONS_FILE = "tests/testdata/region_definitions.json"


def skip_if_hash_unstable(transfer_syntax: UID) -> None:
    """Skip a pixel-hash test when the transfer syntax does not decode
    byte-identically on the current platform, so its md5 comparison would
    spuriously fail. Lossy JPEG2000 decodes differently on macOS.

    Call at the top of a test that asserts an md5 checksum; tests that do not
    hash simply never call it and so are never skipped.
    """
    if transfer_syntax == JPEG2000 and platform.system() == "Darwin":
        pytest.skip("Lossy JPEG2000 does not produce identical output on macOS.")


def skip_if_shared_pool_unsupported(
    input_type: "WsiInputType", use_shared: bool
) -> None:
    """Skip a shared-executor read for web input, whose SQLite-backed
    DICOMfileClient is not thread-safe across a persistent pool.

    This is a limitation of the file-based test client only; a real DICOMweb
    HTTP client reads safely across a shared pool.
    """
    if use_shared and input_type == WsiInputType.WEB:
        pytest.skip(
            "The SQLite-backed DICOMfileClient used for web tests is not "
            "thread-safe across a persistent pool; a real DICOMweb HTTP client "
            "would be."
        )


class WsiInputType(Enum):
    FILE = "file"
    STREAM = "stream"
    WEB = "web"


class WsiTestDefinitions:
    """Interface for reading test parameters from definition file."""

    with open(REGION_DEFINITIONS_FILE) as json_file:
        test_definitions: dict[str, dict[str, Any]] = json.load(json_file)
    if len(test_definitions) == 0:
        pytest.skip("No test definition found, skipping.")

    @classmethod
    def folders(cls) -> list[Path]:
        return [SLIDE_FOLDER.joinpath(path) for _, path in cls._get_parameter("path")]

    @classmethod
    def folders_and_counts(cls) -> list[tuple[Path, int, bool, bool]]:
        return [
            (
                SLIDE_FOLDER.joinpath(wsi_definition["path"]),
                wsi_definition["levels"],
                wsi_definition["label"],
                wsi_definition["overview"],
            )
            for wsi_definition in cls.test_definitions.values()
        ]

    @classmethod
    def folders_and_instance_counts(cls) -> list[tuple[Path, int]]:
        return [
            (
                SLIDE_FOLDER.joinpath(wsi_definition["path"]),
                wsi_definition["instances"],
            )
            for wsi_definition in cls.test_definitions.values()
        ]

    @classmethod
    def wsi_names(cls, tiling: str | None = None) -> list[str]:
        return [
            key
            for key, value in cls.test_definitions.items()
            if tiling is None or value["tiled"] == tiling
        ]

    @classmethod
    def read_region(cls) -> list[tuple[str, UID, dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_region", "level_transfer_syntax"
        )

    @classmethod
    def read_region_mm(cls) -> list[tuple[str, UID, dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_region_mm", "level_transfer_syntax"
        )

    @classmethod
    def read_region_mpp(cls) -> list[tuple[str, UID, dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_region_mpp", "level_transfer_syntax"
        )

    @classmethod
    def read_tile(cls) -> list[tuple[str, UID, dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_tile", "level_transfer_syntax"
        )

    @classmethod
    def read_encoded_tile(cls) -> list[tuple[str, UID, dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_encoded_tile", "level_transfer_syntax"
        )

    @classmethod
    def read_thumbnail(cls) -> list[tuple[str, UID, dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_thumbnail", "level_transfer_syntax"
        )

    @classmethod
    def levels(cls) -> list[tuple[str, int]]:
        return cls._get_parameter("levels")

    @classmethod
    def label_hash(cls) -> list[tuple[str, UID, str | None]]:
        return cls._get_parameter_and_transfer_syntax("label", "label_transfer_syntax")

    @classmethod
    def overview_hash(cls) -> list[tuple[str, UID, str | None]]:
        return cls._get_parameter_and_transfer_syntax(
            "overview", "overview_transfer_syntax"
        )

    @classmethod
    def _get_test_values_and_transfer_syntax(
        cls, test_value_name: str, transfer_syntax_name: str
    ) -> list[tuple[str, UID, dict[str, Any]]]:
        return [
            (wsi_name, UID(wsi_definition[transfer_syntax_name]), region)
            for wsi_name, wsi_definition in cls.test_definitions.items()
            for region in wsi_definition[test_value_name]
        ]

    @classmethod
    def _get_parameter_and_transfer_syntax(
        cls, parameter_name: str, transfer_syntax_name: str
    ) -> list[tuple[str, UID, Any]]:
        return [
            (
                wsi_name,
                UID(wsi_definition[transfer_syntax_name]),
                wsi_definition[parameter_name],
            )
            for wsi_name, wsi_definition in cls.test_definitions.items()
        ]

    @classmethod
    def _get_parameter(cls, parameter_name: str) -> list[tuple[str, Any]]:
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
def shared_threadpool_executor():
    """A shared thread pool for exercising the supplied-executor read path.

    Module-scoped and requested by `wsi_factory` so it is torn down after the
    wsis that read through it are closed.
    """
    with ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="TestSharedRead"
    ) as executor:
        yield executor


@pytest.fixture(scope="module")
def wsi_factory(shared_threadpool_executor: Executor):
    """Fixture providing a callable that takes a wsi name and input type and returns a
    WsiDicom object. Caches opened objects and closes when tests using fixture are done.

    The callable also takes an optional `read_executor` so tests can exercise the
    supplied-executor read path; wsis opened with and without one are cached
    separately.
    """
    streams: list[BufferedReader] = []
    wsis: dict[tuple[WsiInputType, Path, bool], WsiDicom] = {}

    def open_wsi(
        wsi_name: str,
        input_type: WsiInputType = WsiInputType.FILE,
        use_shared_executor: bool = False,
    ) -> WsiDicom:
        test_definition = WsiTestDefinitions.test_definitions[wsi_name]
        folder = UPath(SLIDE_FOLDER).joinpath(test_definition["path"])
        key = (input_type, folder, use_shared_executor)
        if key in wsis:
            return wsis[key]
        read_executor = shared_threadpool_executor if use_shared_executor else None
        if not folder.exists():
            pytest.skip(f"Folder {folder} does not exist.")
        if input_type == WsiInputType.FILE:
            wsi = WsiDicom.open(folder, read_executor=read_executor)
        elif input_type == WsiInputType.WEB:
            client = WsiDicomWebClient(
                DICOMfileClient(f"file://{folder.absolute().as_posix()}")
            )
            wsi = WsiDicom.open_web(
                client,
                test_definition["study_instance_uid"],
                test_definition["series_instance_uid"],
                [JPEGBaseline8Bit, JPEG2000, ExplicitVRLittleEndian],
                read_executor=read_executor,
                settings=Settings(open_web_threads=1),
            )
        elif input_type == WsiInputType.STREAM:
            new_streams = [
                open(file, "rb")
                for file in folder.iterdir()
                if file.is_file() and is_dicom(file)
            ]
            streams.extend(new_streams)
            wsi = WsiDicom.open_streams(new_streams, read_executor=read_executor)
        else:
            raise NotImplementedError()
        wsis[key] = wsi
        return wsi

    yield open_wsi
    for wsi in wsis.values():
        wsi.close()
    for stream in streams:
        stream.close()


@pytest.fixture()
def read_executor():
    """An inline (single-threaded) read executor for read methods under test."""
    yield ReadExecutor(None, None)


@pytest.fixture(scope="module")
def tiled_size():
    yield Size(2, 2)


@pytest.fixture()
def frame_count(tiled_size: Size):
    yield tiled_size.area


@pytest.fixture(scope="module")
def rng():
    SEED = 0
    yield random.Random(SEED)


@pytest.fixture(scope="module")
def bits():
    yield 8


@pytest.fixture(scope="module")
def samples_per_pixel():
    yield 3


@pytest.fixture(scope="module")
def tile_size():
    yield Size(10, 10)


@pytest.fixture(scope="module")
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
def image_data(frames: list[bytes], tiled_size: Size):
    yield WsiDicomTestImageData(frames, tiled_size)


@pytest.fixture(scope="module")
def uid_generator():
    return CallableUidGenerator()
