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
from typing import Callable, List

import pytest
from PIL import ImageChops, ImageFilter, ImageStat
from pydicom.uid import generate_uid

from tests.conftest import WsiTestDefinitions
from wsidicom import WsiDicom
from wsidicom.codec import Jpeg2kSettings, JpegSettings, Settings
from wsidicom.codec.encoder import Encoder
from wsidicom.file import OffsetTableType, WsiDicomFileTarget
from wsidicom.series import Pyramid, Pyramids


@pytest.mark.integration
class TestWsiDicomFileTargetIntegration:
    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_save_levels(
        self, wsi_name: str, wsi_factory: Callable[[str], WsiDicom], tmp_path: Path
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        expected_levels_count = len(wsi.pyramids[0])

        # Act
        with WsiDicomFileTarget(
            tmp_path, generate_uid, 1, 16, OffsetTableType.BASIC, None, None, False
        ) as target:
            target.save_pyramids(wsi.pyramids)

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            assert expected_levels_count == len(saved_wsi.pyramids[0])

    @pytest.mark.parametrize("include_levels", [[-1], [-1, -2]])
    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_save_limited_levels(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        include_levels: List[int],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)

        # Act
        with WsiDicomFileTarget(
            tmp_path,
            generate_uid,
            1,
            16,
            OffsetTableType.BASIC,
            None,
            include_levels,
            False,
        ) as target:
            target.save_pyramids(wsi.pyramids)

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            assert len(saved_wsi.pyramids[0]) == len(include_levels)

    @pytest.mark.parametrize("settings", [Jpeg2kSettings(), JpegSettings()])
    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_transcode(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        settings: Settings,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        transcoder = Encoder.create_for_settings(settings)

        # Act
        with WsiDicomFileTarget(
            tmp_path,
            generate_uid,
            1,
            16,
            OffsetTableType.BASIC,
            None,
            [-1],
            False,
            transcoder,
        ) as target:
            target.save_pyramids(wsi.pyramids)

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            original_level = wsi.pyramids[0][-1]
            level = saved_wsi.pyramids[0].base_level
            assert (
                level.default_instance.image_data.transfer_syntax
                == transcoder.transfer_syntax
            )
            assert level.default_instance.dataset.LossyImageCompression == "01"
            original_methods = (
                original_level.default_instance.dataset.LossyImageCompressionMethod
            )
            original_ratios = (
                original_level.default_instance.dataset.LossyImageCompressionRatio
            )
            methods = level.default_instance.dataset.LossyImageCompressionMethod
            ratios = level.default_instance.dataset.LossyImageCompressionRatio

            if transcoder.lossy_method is not None:
                assert level.default_instance.dataset
                assert transcoder.lossy_method.value == methods[-1]
                assert len(methods) == len(original_methods) + 1
                assert len(ratios) == len(original_ratios) + 1
                assert methods[:-1] == original_methods
                assert ratios[:-1] == original_ratios
                assert isinstance(ratios[-1], float)
            else:
                assert (
                    level.default_instance.dataset.lossy_compressed
                    == original_level.default_instance.dataset.lossy_compressed
                )
                assert methods == original_methods
                assert ratios == original_ratios

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_save_levels_add_missing(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        levels_larger_than_tile_size = [
            level
            for level in wsi.pyramids[0]
            if level.size.any_greater_than(wsi.pyramids[0].tile_size)
        ]
        expected_levels_count = len(levels_larger_than_tile_size) + 1
        pyramid_missing_smallest_levels = Pyramid(levels_larger_than_tile_size, [])
        pyramids = Pyramids([pyramid_missing_smallest_levels])

        # Act
        with WsiDicomFileTarget(
            tmp_path, generate_uid, 1, 16, OffsetTableType.BASIC, None, None, True
        ) as target:
            target.save_pyramids(pyramids)

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            assert expected_levels_count == len(saved_wsi.pyramids[0])
            assert saved_wsi.pyramids[0][-1].size.all_less_than_or_equal(
                saved_wsi.pyramids[0].tile_size
            )
            assert saved_wsi.pyramids[0][-2].size.any_greater_than(
                saved_wsi.pyramids[0].tile_size
            )

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_create_child(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        target_level = wsi.pyramids[0][-2]
        source_level = wsi.pyramids[0][-3]

        # Act
        with WsiDicomFileTarget(
            tmp_path,
            generate_uid,
            1,
            100,
            OffsetTableType.BASIC,
            None,
            None,
            False,
        ) as target:
            target._save_and_open_level(source_level, wsi.pyramids[0].pixel_spacing, 2)

        # Assert
        with WsiDicom.open(tmp_path) as created_wsi:
            created_size = created_wsi.pyramids[0][0].size.to_tuple()
            target_size = target_level.size.to_tuple()
            assert created_size == target_size

            created = created_wsi.read_region((0, 0), 0, created_size)
            original = wsi.read_region((0, 0), target_level.level, target_size)
            blur = ImageFilter.GaussianBlur(2)
            diff = ImageChops.difference(created.filter(blur), original.filter(blur))
            for band_rms in ImageStat.Stat(diff).rms:
                assert band_rms < 2
