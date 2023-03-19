#    Copyright 2021 SECTRA AB
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
import unittest
from pathlib import Path
from typing import Any, Dict

import pytest
from dicomweb_client import DICOMfileClient

from tests.test_dcmfiles import (
    WsiDicomFilesTests,
    SLIDE_FOLDER,
    REGION_DEFINITIONS_FILE,
)
from wsidicom import WsiDicom
from wsidicom.web.wsidicom_web_client import WsiDicomWebClient


class WsiDicomFileClient(WsiDicomWebClient):
    def __init__(self, path: Path):
        self._client = DICOMfileClient(
            f"file://{path.absolute().as_posix()}", in_memory=True
        )


@pytest.mark.integration
class WsiDicomWebTests(WsiDicomFilesTests):
    @classmethod
    def setUpClass(cls):
        folders = cls._get_folders(SLIDE_FOLDER)
        cls.test_folders: Dict[Path, WsiDicom] = {}

        with open(REGION_DEFINITIONS_FILE) as json_file:
            cls.test_definitions: Dict[str, Dict[str, Any]] = json.load(json_file)
        if len(cls.test_definitions) == 0:
            raise unittest.SkipTest("no test definition found, skipping.")
        for folder in folders:
            relative_path = cls._get_relative_path(folder)
            test_file_definition = cls.test_definitions[str(relative_path)]
            cls.test_folders[relative_path] = cls.open(folder, test_file_definition)

        if len(cls.test_folders) == 0:
            raise unittest.SkipTest(
                f"no test slide files found for {SLIDE_FOLDER}, " "skipping"
            )

    @staticmethod
    def open(folder: Path, test_file_definition: Dict[str, Any]) -> WsiDicom:
        while next(folder.iterdir()).is_dir():
            folder = next(folder.iterdir())
        client = WsiDicomFileClient(folder)
        return WsiDicom.open_web(
            client,
            test_file_definition["StudyInstanceUID"],
            test_file_definition["SeriesInstanceUID"],
        )
