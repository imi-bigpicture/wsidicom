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
import os
import unittest
from hashlib import md5
from pathlib import Path
from typing import Any, Dict, List
from PIL import Image

import pytest
from wsidicom import WsiDicom

SLIDE_FOLDER = Path(os.environ.get("WSIDICOM_TESTDIR", "tests/testdata/slides"))
REGION_DEFINITIONS_FILE = "tests/testdata/region_definitions.json"


@pytest.mark.integration
class WsiDicomFilesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folders = cls._get_folders(SLIDE_FOLDER)
        cls.test_folders: Dict[Path, WsiDicom] = {}
        for folder in folders:
            relative_path = cls._get_relative_path(folder)
            cls.test_folders[relative_path] = cls.open(folder)

        if len(cls.test_folders) == 0:
            raise unittest.SkipTest(
                f"no test slide files found for {SLIDE_FOLDER}, " "skipping"
            )
        with open(REGION_DEFINITIONS_FILE) as json_file:
            cls.test_definitions: Dict[str, Dict[str, Any]] = json.load(json_file)
        if len(cls.test_definitions) == 0:
            raise unittest.SkipTest("no test definition found, skipping.")

    @classmethod
    def tearDownClass(cls):
        for folder, wsi_dicom in cls.test_folders.items():
            wsi_dicom.close()

    @staticmethod
    def open(folder: Path) -> WsiDicom:
        while next(folder.iterdir()).is_dir():
            folder = next(folder.iterdir())
        return WsiDicom.open(folder)

    @staticmethod
    def _get_folders(SLIDE_FOLDER: Path) -> List[Path]:
        if not SLIDE_FOLDER.exists():
            print("slide folder does not exist")
            return []
        return [item for item in SLIDE_FOLDER.iterdir() if item.is_dir]

    @staticmethod
    def _get_relative_path(slide_path: Path) -> Path:
        parts = slide_path.parts
        return Path(parts[-1])

    def test_read_region(self):
        for folder, test_definitions in self.test_definitions.items():
            if not Path(folder) in self.test_folders:
                continue
            wsi = self.test_folders[Path(folder)]
            for region in test_definitions["read_region"]:
                im = wsi.read_region(
                    (region["location"]["x"], region["location"]["y"]),
                    region["level"],
                    (region["size"]["width"], region["size"]["height"]),
                )
                print(region)
                self.assertEqual(  # type: ignore
                    md5(im.tobytes()).hexdigest(), region["md5"], msg=region
                )

    def test_read_region_mm(self):
        for folder, test_definitions in self.test_definitions.items():
            if not Path(folder) in self.test_folders:
                continue
            wsi = self.test_folders[Path(folder)]
            for region in test_definitions["read_region_mm"]:
                im = wsi.read_region_mm(
                    (region["location"]["x"], region["location"]["y"]),
                    region["level"],
                    (region["size"]["width"], region["size"]["height"]),
                )
                print(region)
                self.assertEqual(  # type: ignore
                    md5(im.tobytes()).hexdigest(), region["md5"], msg=region
                )

    def test_read_region_mpp(self):
        for folder, test_definitions in self.test_definitions.items():
            if not Path(folder) in self.test_folders:
                continue
            wsi = self.test_folders[Path(folder)]
            for region in test_definitions["read_region_mpp"]:
                im = wsi.read_region_mpp(
                    (region["location"]["x"], region["location"]["y"]),
                    region["mpp"],
                    (region["size"]["width"], region["size"]["height"]),
                )
                print(region)
                self.assertEqual(  # type: ignore
                    md5(im.tobytes()).hexdigest(), region["md5"], msg=region
                )

    def test_read_tile(self):
        for folder, test_definitions in self.test_definitions.items():
            if not Path(folder) in self.test_folders:
                continue
            wsi = self.test_folders[Path(folder)]
            for region in test_definitions["read_tile"]:
                im = wsi.read_tile(
                    region["level"], (region["location"]["x"], region["location"]["y"])
                )
                print(region)
                self.assertEqual(  # type: ignore
                    md5(im.tobytes()).hexdigest(), region["md5"], msg=region
                )

    def test_read_encoded_tile(self):
        for folder, test_definitions in self.test_definitions.items():
            if not Path(folder) in self.test_folders:
                continue
            wsi = self.test_folders[Path(folder)]
            for region in test_definitions["read_encoded_tile"]:
                im = wsi.read_encoded_tile(
                    region["level"], (region["location"]["x"], region["location"]["y"])
                )
                print(region)
                self.assertEqual(  # type: ignore
                    md5(im).hexdigest(), region["md5"], msg=region
                )

    def test_read_thumbnail(self):
        for folder, test_definitions in self.test_definitions.items():
            if not Path(folder) in self.test_folders:
                continue
            wsi = self.test_folders[Path(folder)]
            for region in test_definitions["read_thumbnail"]:
                im = wsi.read_thumbnail(
                    (region["size"]["width"], region["size"]["height"])
                )
                print(region)
                self.assertEqual(  # type: ignore
                    md5(im.tobytes()).hexdigest(), region["md5"], msg=region
                )

    def test_replace_label(self):
        path = next(folders for folders in self._get_folders(SLIDE_FOLDER))
        while next(path.iterdir()).is_dir():
            path = next(path.iterdir())
        image = Image.new("RGB", (256, 256), (128, 128, 128))
        with WsiDicom.open(path, label=image) as wsi:
            self.assertEqual(image, wsi.read_label())
