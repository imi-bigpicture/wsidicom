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

from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import ImageChops, ImageFilter, ImageStat

from tests.conftest import WsiTestDefinitions
from wsidicom import WsiDicom
from wsidicom.codec import Jpeg2kSettings, JpegSettings, Settings
from wsidicom.codec.encoder import Encoder
from wsidicom.file import OffsetTableType, WsiDicomFileTarget
from wsidicom.metadata.uid_generator import CallableUidGenerator
from wsidicom.series import Pyramid, Pyramids
from wsidicom.tags import LossyImageCompressionMethodTag, LossyImageCompressionRatioTag


@pytest.mark.integration
class TestWsiDicomFileTargetIntegration:
    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_save_levels(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        uid_generator: CallableUidGenerator,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        expected_levels_count = len(wsi.pyramids[0])

        # Act
        with WsiDicomFileTarget(
            tmp_path,
            uid_generator,
            1,
            16,
            OffsetTableType.BASIC,
            None,
            None,
            False,
        ) as target:
            target.save(wsi.pyramids, None, None, True)

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
        include_levels: list[int],
        uid_generator: CallableUidGenerator,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)

        # Act
        with WsiDicomFileTarget(
            tmp_path,
            uid_generator,
            1,
            None,
            OffsetTableType.BASIC,
            None,
            include_levels,
            False,
        ) as target:
            target.save(wsi.pyramids, None, None, True)

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            assert len(saved_wsi.pyramids[0]) == len(include_levels)

    @pytest.mark.parametrize("settings", [Jpeg2kSettings(), JpegSettings()])
    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    @pytest.mark.parametrize("force_transcoding", [False, True])
    def test_transcode(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        settings: Settings,
        force_transcoding: bool,
        uid_generator: CallableUidGenerator,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        original_transfer_syntax = wsi.levels[
            0
        ].default_instance.image_data.transfer_syntax
        transcoder = Encoder.create_for_settings(settings)
        expect_transcoding = force_transcoding or (
            original_transfer_syntax != transcoder.transfer_syntax
        )

        # Act
        with WsiDicomFileTarget(
            tmp_path,
            uid_generator,
            1,
            None,
            OffsetTableType.BASIC,
            None,
            [-1],
            False,
            transcoder,
            force_transcoding,
        ) as target:
            target.save(wsi.pyramids, None, None, True)

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            original_level = wsi.pyramids[0][-1]
            level = saved_wsi.pyramids[0].base_level
            assert (
                level.default_instance.image_data.transfer_syntax
                == transcoder.transfer_syntax
            )
            assert level.default_instance.dataset.LossyImageCompression == "01"
            original_methods = original_level.default_instance.dataset.get_multi_value(
                LossyImageCompressionMethodTag
            )
            original_ratios = original_level.default_instance.dataset.get_multi_value(
                LossyImageCompressionRatioTag
            )
            methods = level.default_instance.dataset.get_multi_value(
                LossyImageCompressionMethodTag
            )
            ratios = level.default_instance.dataset.get_multi_value(
                LossyImageCompressionRatioTag
            )

            if expect_transcoding and transcoder.lossy_method is not None:
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
        uid_generator: CallableUidGenerator,
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
            tmp_path,
            uid_generator,
            1,
            16,
            OffsetTableType.BASIC,
            None,
            None,
            True,
        ) as target:
            target.save(pyramids, None, None, True)

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
        uid_generator: CallableUidGenerator,
    ):
        # Arrange — create a pyramid missing the smallest levels so that
        # add_missing_levels generates downsampled children.
        wsi = wsi_factory(wsi_name)
        pyramid = wsi.pyramids[0]
        tile_size = pyramid.tile_size
        levels_larger_than_tile = [
            level for level in pyramid if level.size.any_greater_than(tile_size)
        ]
        if len(levels_larger_than_tile) < 2:
            pytest.skip("Not enough multi-tile levels for test")

        # Keep only multi-tile levels — child levels will be generated
        pyramid_missing_smallest = Pyramid(levels_larger_than_tile, [])
        pyramids = Pyramids([pyramid_missing_smallest])

        # Act
        with WsiDicomFileTarget(
            tmp_path,
            uid_generator,
            1,
            None,
            OffsetTableType.BASIC,
            None,
            None,
            True,
        ) as target:
            target.save(pyramids, None, None, False)

        # Assert — verify a generated child level is visually close
        with WsiDicom.open(tmp_path) as created_wsi:
            assert len(created_wsi.pyramids[0]) > len(levels_larger_than_tile)
            # The first generated level should be the single-tile level
            last_source = levels_larger_than_tile[-1]
            child_level_idx = last_source.level + 1
            created_level = created_wsi.pyramids[0].get(child_level_idx)
            created_size = created_level.size.to_tuple()

            created = created_wsi.read_region((0, 0), child_level_idx, created_size)
            original = wsi.read_region((0, 0), child_level_idx, created_size)
            blur = ImageFilter.GaussianBlur(2)
            diff = ImageChops.difference(created.filter(blur), original.filter(blur))
            for band_rms in ImageStat.Stat(diff).rms:
                assert band_rms < 2
