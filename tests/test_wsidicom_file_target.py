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

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pytest
from parameterized import parameterized
from pydicom.uid import generate_uid
from tests.wsi_test_files import WsiTestFiles
from wsidicom import WsiDicom
from wsidicom.file.wsidicom_file_target import WsiDicomFileTarget
from wsidicom.series.levels import Levels


@pytest.mark.integration
class WsiDicomFileTargetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wsi_test_files = WsiTestFiles()

    @classmethod
    def tearDownClass(cls):
        cls.wsi_test_files.close()

    def setUp(self):
        self.tempdir = TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    @parameterized.expand((WsiTestFiles.folders.keys))
    def test_save_levels(self, wsi_name: str):
        # Arrange
        wsi = self.wsi_test_files.get_wsi(wsi_name)
        expected_levels_count = len(wsi.levels)

        # Act
        with WsiDicomFileTarget(
            Path(self.tempdir.name), generate_uid, 1, 16, "bot"
        ) as target:
            target.save_levels(wsi.levels)

        # Assert
        with WsiDicom.open(self.tempdir.name) as saved_wsi:
            self.assertEqual(expected_levels_count, len(saved_wsi.levels))

    @parameterized.expand((WsiTestFiles.folders.keys))
    def test_save_levels_add_missing(self, wsi_name: str):
        # Arrange
        wsi = self.wsi_test_files.get_wsi(wsi_name)
        levels_larger_than_tile_size = [
            level for level in wsi.levels if level.size.any_greater_than(wsi.tile_size)
        ]
        expected_levels_count = len(levels_larger_than_tile_size) + 1
        levels_missing_smallest_levels = Levels(levels_larger_than_tile_size)

        # Act
        with WsiDicomFileTarget(
            Path(self.tempdir.name), generate_uid, 1, 16, "bot", True
        ) as target:
            target.save_levels(levels_missing_smallest_levels)

        # Assert
        with WsiDicom.open(self.tempdir.name) as saved_wsi:
            self.assertEqual(expected_levels_count, len(saved_wsi.levels))
            self.assertTrue(
                saved_wsi.levels[-1].size.all_less_than_or_equal(saved_wsi.tile_size)
            )
            self.assertTrue(
                saved_wsi.levels[-2].size.any_greater_than(saved_wsi.tile_size)
            )
