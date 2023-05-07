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

import unittest
from hashlib import md5
from typing import Any, Dict

import pytest
from parameterized import parameterized, parameterized_class
from PIL import Image

from tests.wsi_test_files import WsiInputType, WsiTestDefinitions, WsiTestFiles
from wsidicom import WsiDicom


@pytest.mark.integration
@parameterized_class(
    [
        {"input_type": WsiInputType.FILE},
        {"input_type": WsiInputType.STREAM},
        {"input_type": WsiInputType.WEB},
    ]
)
class WsiDicomIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_type: WsiInputType
        cls.wsi_test_files = WsiTestFiles(cls.input_type)

    @classmethod
    def tearDownClass(cls):
        cls.wsi_test_files.close()

    @parameterized.expand(WsiTestDefinitions.read_region)
    def test_read_region(self, folder: str, region: Dict[str, Any]):
        # Arrange
        wsi = self.wsi_test_files.get_wsi(folder)

        # Act
        im = wsi.read_region(
            (region["location"]["x"], region["location"]["y"]),
            region["level"],
            (region["size"]["width"], region["size"]["height"]),
        )

        # Assert
        self.assertEqual(md5(im.tobytes()).hexdigest(), region["md5"], msg=region)

    @parameterized.expand(WsiTestDefinitions.read_region_mm)
    def test_read_region_mm(self, folder: str, region: Dict[str, Any]):
        # Arrange
        wsi = self.wsi_test_files.get_wsi(folder)

        # Act
        im = wsi.read_region_mm(
            (region["location"]["x"], region["location"]["y"]),
            region["level"],
            (region["size"]["width"], region["size"]["height"]),
        )

        # Assert
        self.assertEqual(md5(im.tobytes()).hexdigest(), region["md5"], msg=region)

    @parameterized.expand(WsiTestDefinitions.read_region_mpp)
    def test_read_region_mpp(self, folder: str, region: Dict[str, Any]):
        # Arrange
        wsi = self.wsi_test_files.get_wsi(folder)

        # Act
        im = wsi.read_region_mpp(
            (region["location"]["x"], region["location"]["y"]),
            region["mpp"],
            (region["size"]["width"], region["size"]["height"]),
        )

        # Assert
        self.assertEqual(md5(im.tobytes()).hexdigest(), region["md5"], msg=region)

    @parameterized.expand(WsiTestDefinitions.read_tile)
    def test_read_tile(self, folder: str, region: Dict[str, Any]):
        # Arrange
        wsi = self.wsi_test_files.get_wsi(folder)

        # Act
        im = wsi.read_tile(
            region["level"],
            (region["location"]["x"], region["location"]["y"]),
        )

        # Assert
        self.assertEqual(md5(im.tobytes()).hexdigest(), region["md5"], msg=region)

    @parameterized.expand(WsiTestDefinitions.read_encoded_tile)
    def test_read_encoded_tile(self, folder: str, region: Dict[str, Any]):
        # Arrange
        wsi = self.wsi_test_files.get_wsi(folder)

        # Act
        im = wsi.read_encoded_tile(
            region["level"],
            (region["location"]["x"], region["location"]["y"]),
        )

        # Assert
        self.assertEqual(md5(im).hexdigest(), region["md5"], msg=region)

    @parameterized.expand(WsiTestDefinitions.read_thumbnail)
    def test_read_thumbnail(self, folder: str, region: Dict[str, Any]):
        # Arrange
        wsi = self.wsi_test_files.get_wsi(folder)

        # Act
        im = wsi.read_thumbnail((region["size"]["width"], region["size"]["height"]))

        # Assert
        self.assertEqual(md5(im.tobytes()).hexdigest(), region["md5"], msg=region)

    def test_replace_label(self):
        # Arrange
        path = next(folders for folders in self.wsi_test_files.folders.values())
        while next(path.iterdir()).is_dir():
            path = next(path.iterdir())
        image = Image.new("RGB", (256, 256), (128, 128, 128))

        # Act
        with WsiDicom.open(path, label=image) as wsi:
            label = wsi.read_label()

        # Assert
        self.assertEqual(image, label)

    @parameterized.expand(WsiTestDefinitions.levels)
    def test_number_of_levels(self, folder: str, expected_level_count: int):
        # Arrange
        wsi = self.wsi_test_files.get_wsi(folder)

        # Act
        levels_count = len(wsi.levels)

        # Assert
        self.assertEqual(levels_count, expected_level_count)

    @parameterized.expand(WsiTestDefinitions.has_label)
    def test_has_label(self, folder: str, expected_has_label: bool):
        # Arrange
        wsi = self.wsi_test_files.get_wsi(folder)

        # Act
        has_label = wsi.labels is not None

        # Assert
        self.assertEqual(has_label, expected_has_label)

    @parameterized.expand(WsiTestDefinitions.has_overview)
    def test_has_overview(self, folder: str, expected_has_overview: bool):
        # Arrange
        wsi = self.wsi_test_files.get_wsi(folder)

        # Act
        has_overview = wsi.overviews is not None

        # Assert
        self.assertEqual(has_overview, expected_has_overview)
