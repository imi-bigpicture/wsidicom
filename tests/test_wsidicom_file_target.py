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
from typing import Callable, cast

import pytest
from PIL import ImageChops, ImageFilter, ImageStat
from pydicom.uid import generate_uid

from tests.conftest import WsiTestDefinitions
from wsidicom import WsiDicom
from wsidicom.file import OffsetTableType, WsiDicomFileTarget
from wsidicom.group.level import Level
from wsidicom.series.levels import Levels


@pytest.mark.integration
class TestWsiDicomFileTargetIntegration:
    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names())
    def test_save_levels(
        self, wsi_name: str, wsi_factory: Callable[[str], WsiDicom], tmp_path: Path
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        expected_levels_count = len(wsi.levels)

        # Act
        with WsiDicomFileTarget(
            tmp_path, generate_uid, 1, 16, OffsetTableType.BASIC, None, False
        ) as target:
            target.save_levels(wsi.levels)

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            assert expected_levels_count == len(saved_wsi.levels)

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names())
    def test_save_levels_add_missing(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        levels_larger_than_tile_size = [
            level for level in wsi.levels if level.size.any_greater_than(wsi.tile_size)
        ]
        expected_levels_count = len(levels_larger_than_tile_size) + 1
        levels_missing_smallest_levels = Levels(levels_larger_than_tile_size)

        # Act
        with WsiDicomFileTarget(
            tmp_path, generate_uid, 1, 16, OffsetTableType.BASIC, None, True
        ) as target:
            target.save_levels(levels_missing_smallest_levels)

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            assert expected_levels_count == len(saved_wsi.levels)
            assert saved_wsi.levels[-1].size.all_less_than_or_equal(saved_wsi.tile_size)
            assert saved_wsi.levels[-2].size.any_greater_than(saved_wsi.tile_size)

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names())
    def test_create_child(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        target_level = cast(Level, wsi.levels[-2])
        source_level = cast(Level, wsi.levels[-3])

        # Act
        with WsiDicomFileTarget(
            tmp_path,
            generate_uid,
            1,
            100,
            OffsetTableType.BASIC,
            None,
            False,
        ) as target:
            target._save_and_open_level(source_level, wsi.pixel_spacing, 2)

        # Assert
        with WsiDicom.open(tmp_path) as created_wsi:
            created_size = created_wsi.levels[0].size.to_tuple()
            target_size = target_level.size.to_tuple()
            assert created_size == target_size

            created = created_wsi.read_region((0, 0), 0, created_size)
            original = wsi.read_region((0, 0), target_level.level, target_size)
            blur = ImageFilter.GaussianBlur(2)
            diff = ImageChops.difference(created.filter(blur), original.filter(blur))
            for band_rms in ImageStat.Stat(diff).rms:
                assert band_rms < 2
