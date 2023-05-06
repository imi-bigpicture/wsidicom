#    Copyright 2021, 2022, 2023 SECTRA AB
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
import unittest
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Tuple

from wsidicom import WsiDicom
from wsidicom.web.wsidicom_web_client import WsiDicomFileClient

SLIDE_FOLDER = Path(os.environ.get("WSIDICOM_TESTDIR", "tests/testdata/slides"))
REGION_DEFINITIONS_FILE = "tests/testdata/region_definitions.json"


class WsiInputType(Enum):
    FILE = "file"
    STREAM = "stream"
    WEB = "web"


class WsiTestDefinitions:
    with open(REGION_DEFINITIONS_FILE) as json_file:
        test_definitions: Dict[str, Dict[str, Any]] = json.load(json_file)
    if len(test_definitions) == 0:
        raise unittest.SkipTest("No test definition found, skipping.")

    @classmethod
    def read_region(cls) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return cls._get_region("read_region")

    @classmethod
    def read_region_mm(cls) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return cls._get_region("read_region_mm")

    @classmethod
    def read_region_mpp(cls) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return cls._get_region("read_region_mpp")

    @classmethod
    def read_tile(cls) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return cls._get_region("read_tile")

    @classmethod
    def read_encoded_tile(cls) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return cls._get_region("read_encoded_tile")

    @classmethod
    def read_thumbnail(cls) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return cls._get_region("read_thumbnail")

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
    def _get_region(cls, region_name: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
        return [
            (folder, region)
            for folder, folder_definition in cls.test_definitions.items()
            for region in folder_definition[region_name]
        ]

    @classmethod
    def _get_parameter(cls, parameter_name: str) -> Iterable[Tuple[str, Any]]:
        return [
            (folder, folder_definition[parameter_name])
            for folder, folder_definition in cls.test_definitions.items()
        ]


class WsiTestFiles:
    def __init__(self, input_type: WsiInputType):
        self._input_type = input_type
        self._wsis: Dict[str, WsiDicom] = {}
        self._opened_streams: List[BinaryIO] = []
        self._wsi_folders: Dict[str, Path] = self._get_folders()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self):
        for wsi in self._wsis.values():
            wsi.close()
        for stream in self._opened_streams:
            stream.close()

    @property
    def wsi_folders(self) -> Dict[str, Path]:
        return self._wsi_folders

    def get_wsi(self, wsi_name: str) -> WsiDicom:
        if wsi_name in self._wsis:
            return self._wsis[wsi_name]
        if not wsi_name in self._wsi_folders:
            raise unittest.SkipTest("WSI files not found, skipping.")
        folder = self._wsi_folders[wsi_name]
        try:
            while next(folder.iterdir()).is_dir():
                folder = next(folder.iterdir())
        except StopIteration:
            raise unittest.SkipTest("WSI files not found, skipping.")
        if self._input_type == WsiInputType.FILE:
            wsi = WsiDicom.open(folder)
        elif self._input_type == WsiInputType.WEB:
            client = WsiDicomFileClient(folder)
            test_definition = WsiTestDefinitions.test_definitions[wsi_name]
            wsi = WsiDicom.open_web(
                client,
                test_definition["study_instance_uid"],
                test_definition["series_instance_uid"],
            )
        elif self._input_type == WsiInputType.STREAM:
            streams = [open(file, "rb") for file in folder.iterdir() if file.is_file()]
            self._opened_streams.extend(streams)
            wsi = WsiDicom.open(streams)
        else:
            raise ValueError(f"Unknown test_type {self._input_type}.")
        self._wsis[(wsi_name)] = wsi
        return wsi

    @classmethod
    def _get_folders(cls) -> Dict[str, Path]:
        folders = {}
        if SLIDE_FOLDER.exists():
            folders = {
                cls._get_wsi_name(item): item
                for item in SLIDE_FOLDER.iterdir()
                if item.is_dir
            }
        if len(folders) == 0:
            raise unittest.SkipTest(
                f"No test slide files found for {SLIDE_FOLDER}, skipping."
            )
        return folders

    @staticmethod
    def _get_wsi_name(slide_path: Path) -> str:
        parts = slide_path.parts
        return parts[-1]
