#    Copyright 2026 SECTRA AB
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

"""Integration tests for the files described by `WsiDicom.save` and `WsiDicom.files`."""

import dataclasses
from collections.abc import Callable
from pathlib import Path

import pytest
from pydicom import dcmread
from pydicom.multival import MultiValue

from tests.conftest import WsiTestDefinitions
from wsidicom import (
    WsiAssociatedFile,
    WsiDicom,
    WsiLevelFile,
    WsiThumbnailFile,
)
from wsidicom.codec import JpegSettings
from wsidicom.file import WsiDicomFileTarget
from wsidicom.metadata import ImageType
from wsidicom.metadata.uid_generator import CallableUidGenerator
from wsidicom.series import Pyramids


@pytest.mark.integration
class TestWsiFilesIntegration:
    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names())
    def test_save_describes_every_written_file(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)

        # Act
        created_files = wsi.save(tmp_path, include_levels=[-1])

        # Assert
        assert {Path(str(path)) for path in created_files} == set(tmp_path.iterdir())

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names())
    def test_created_files_match_written_datasets(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)

        # Act
        created_files = wsi.save(tmp_path, include_levels=[-1])

        # Assert
        for path, created_file in created_files.items():
            dataset = dcmread(str(path), stop_before_pixels=True)
            assert created_file.uids.instance == dataset.SOPInstanceUID
            assert created_file.uids.slide.series_instance == dataset.SeriesInstanceUID
            assert created_file.image_type.value == dataset.ImageType[2]
            assert created_file.image_size.width == dataset.TotalPixelMatrixColumns
            assert created_file.image_size.height == dataset.TotalPixelMatrixRows
            assert created_file.tile_size.width == dataset.Columns
            assert created_file.tile_size.height == dataset.Rows
            assert created_file.transfer_syntax == dataset.file_meta.TransferSyntaxUID
            assert (
                created_file.photometric_interpretation
                == dataset.PhotometricInterpretation
            )
            if created_file.pixel_spacing is not None:
                pixel_measures = dataset.SharedFunctionalGroupsSequence[
                    0
                ].PixelMeasuresSequence[0]
                assert type(created_file.pixel_spacing.width) is float
                assert type(created_file.pixel_spacing.height) is float
                assert created_file.pixel_spacing.width == float(
                    pixel_measures.PixelSpacing[1]
                )
                assert created_file.pixel_spacing.height == float(
                    pixel_measures.PixelSpacing[0]
                )

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names())
    def test_level_files_give_pyramid_and_level_index(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        expected_level_index = wsi.pyramids[0].levels[-1].level

        # Act
        created_files = wsi.save(tmp_path, include_pyramids=[0], include_levels=[-1])

        # Assert
        level_files = [
            created_file
            for created_file in created_files.values()
            if isinstance(created_file, WsiLevelFile)
        ]
        assert len(level_files) > 0
        for level_file in level_files:
            assert level_file.pyramid_index == 0
            assert level_file.level_index == expected_level_index

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names())
    def test_thumbnail_and_associated_files_by_image_type(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        thumbnails = wsi.pyramids[0].thumbnails
        expected_thumbnail_count = (
            0 if thumbnails is None else len(list(thumbnails.groups))
        )

        # Act
        created_files = wsi.save(tmp_path, include_pyramids=[0], include_levels=[-1])

        # Assert
        thumbnail_files = [
            created_file
            for created_file in created_files.values()
            if isinstance(created_file, WsiThumbnailFile)
        ]
        associated_files = [
            created_file
            for created_file in created_files.values()
            if isinstance(created_file, WsiAssociatedFile)
        ]
        assert len(thumbnail_files) == expected_thumbnail_count
        assert all(
            thumbnail_file.pyramid_index == 0 for thumbnail_file in thumbnail_files
        )
        assert all(
            associated_file.image_type in (ImageType.LABEL, ImageType.OVERVIEW)
            for associated_file in associated_files
        )

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_transcoded_file_gives_compression_ratio_as_written(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)

        # Act
        created_files = wsi.save(
            tmp_path,
            include_levels=[-1],
            include_labels=False,
            include_overviews=False,
            include_thumbnails=False,
            transcoding=JpegSettings(),
            force_transcoding=True,
        )

        # Assert
        for path, created_file in created_files.items():
            dataset = dcmread(str(path), stop_before_pixels=True)
            ratios = dataset.LossyImageCompressionRatio
            written_ratios = (
                list(ratios) if isinstance(ratios, MultiValue) else [ratios]
            )
            assert [
                compression.ratio for compression in created_file.lossy_compressions
            ] == [float(ratio) for ratio in written_ratios]

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_negative_pyramid_index_is_given_as_position(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        uid_generator: CallableUidGenerator,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        pyramid = wsi.pyramids[0]
        pyramids = Pyramids([pyramid, pyramid])

        # Act
        with WsiDicomFileTarget(
            tmp_path,
            uid_generator,
            workers=1,
            include_pyramids=[-1],
            include_levels=[-1],
        ) as target:
            target.save(pyramids, None, None, False)
            created_files = target.created_files

        # Assert
        assert len(created_files) > 0
        for created_file in created_files.values():
            assert isinstance(created_file, WsiLevelFile)
            assert created_file.pyramid_index == 1

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names())
    def test_opened_files_match_read_datasets(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)

        # Act
        opened_files = wsi.files

        # Assert
        assert opened_files is not None
        assert len(opened_files) > 0
        for path, opened_file in opened_files.items():
            dataset = dcmread(str(path), stop_before_pixels=True)
            assert opened_file.uids.instance == dataset.SOPInstanceUID
            assert opened_file.image_type.value == dataset.ImageType[2]
            assert opened_file.image_size.width == dataset.TotalPixelMatrixColumns
            assert opened_file.image_size.height == dataset.TotalPixelMatrixRows
            assert opened_file.transfer_syntax == dataset.file_meta.TransferSyntaxUID

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names())
    def test_opened_files_match_level_indices(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        expected_level_indices = {
            (pyramid_index, level.level)
            for pyramid_index, pyramid in enumerate(wsi.pyramids)
            for level in pyramid.levels
        }

        # Act
        opened_files = wsi.files

        # Assert
        assert opened_files is not None
        level_indices = {
            (opened_file.pyramid_index, opened_file.level_index)
            for opened_file in opened_files.values()
            if isinstance(opened_file, WsiLevelFile)
        }
        assert level_indices == expected_level_indices

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names())
    def test_opened_files_match_saved_files(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name)
        created_files = wsi.save(tmp_path, include_levels=[-1])
        # Only the smallest level is saved, so it is the base level when opened.
        expected_files = {
            str(path): (
                dataclasses.replace(created_file, level_index=0)
                if isinstance(created_file, WsiLevelFile)
                else created_file
            )
            for path, created_file in created_files.items()
        }

        # Act
        with WsiDicom.open(tmp_path) as opened:
            opened_files = opened.files

        # Assert
        assert opened_files is not None
        assert {
            str(path): opened_file for path, opened_file in opened_files.items()
        } == expected_files
