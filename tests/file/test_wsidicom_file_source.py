#    Copyright 2024 SECTRA AB
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

import logging
from copy import deepcopy
from pathlib import Path

import pytest
from pydicom.dataset import Dataset
from pydicom.fileset import FileSet
from pydicom.uid import generate_uid

from tests.conftest import WsiTestDefinitions
from tests.data_gen import create_layer_file, create_main_dataset
from wsidicom.file.wsidicom_file_source import WsiDicomFileSource


@pytest.fixture
def level_dataset() -> Dataset:
    return create_main_dataset()


@pytest.fixture
def level_file(tmp_path: Path, level_dataset: Dataset) -> Path:
    path = tmp_path.joinpath("level.dcm")
    create_layer_file(path, deepcopy(level_dataset))
    return path


def create_thumbnail_file(
    path: Path, level_dataset: Dataset, with_pixel_spacing: bool
) -> Path:
    dataset = deepcopy(level_dataset)
    dataset.SOPInstanceUID = generate_uid()
    dataset.ImageType = ["DERIVED", "PRIMARY", "THUMBNAIL", "RESAMPLED"]
    if not with_pixel_spacing:
        pixel_measures = dataset.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence
        del pixel_measures[0].PixelSpacing
    create_layer_file(path, dataset)
    return path


@pytest.mark.unittest
class TestWsiDicomFileSource:
    @pytest.mark.parametrize(
        "wsi_folder_and_counts", WsiTestDefinitions.folders_and_counts()
    )
    def test_open(self, wsi_folder_and_counts: tuple[Path, int, bool, bool]) -> None:
        # Arrange
        wsi_folder, level_count, label, overview = wsi_folder_and_counts

        # Act
        with WsiDicomFileSource.open(wsi_folder) as source:
            instances = list(source.level_instances)
            label_instances = list(source.label_instances)
            overview_instances = list(source.overview_instances)

        # Assert
        assert len(instances) == level_count
        assert len(label_instances) == (1 if label else 0)
        assert len(overview_instances) == (1 if overview else 0)

    @pytest.mark.parametrize("path_is_folder", [True, False])
    def test_open_dicomdir(
        self, tmp_path: Path, wsi_file: Path, path_is_folder: bool
    ) -> None:
        # Arrange
        dicom_dir_path = tmp_path.joinpath("dicomdir")
        dicom_dir = FileSet()
        dicom_dir.add(wsi_file)
        dicom_dir.write(dicom_dir_path)
        if not path_is_folder:
            dicom_dir_path = dicom_dir_path.joinpath("DICOMDIR")

        # Act
        with WsiDicomFileSource.open_dicomdir(dicom_dir_path) as source:
            instances = list(source.level_instances)

        # Assert
        assert len(instances) == 1

    def test_open_uses_thumbnail_with_pixel_spacing(
        self, tmp_path: Path, level_dataset: Dataset, level_file: Path
    ) -> None:
        # Arrange
        thumbnail_file = create_thumbnail_file(
            tmp_path.joinpath("thumbnail.dcm"), level_dataset, True
        )

        # Act
        with WsiDicomFileSource.open([level_file, thumbnail_file]) as source:
            files = source.files
            thumbnail_instances = list(source.thumbnail_instances)

        # Assert
        assert files == [level_file, thumbnail_file]
        assert len(thumbnail_instances) == 1

    def test_open_ignores_thumbnail_without_pixel_spacing(
        self,
        tmp_path: Path,
        level_dataset: Dataset,
        level_file: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        thumbnail_file = create_thumbnail_file(
            tmp_path.joinpath("thumbnail.dcm"), level_dataset, False
        )

        # Act
        with (
            caplog.at_level(logging.WARNING),
            WsiDicomFileSource.open([level_file, thumbnail_file]) as source,
        ):
            files = source.files
            thumbnail_instances = list(source.thumbnail_instances)

        # Assert
        assert files == [level_file]
        assert thumbnail_instances == []
        assert "Ignoring thumbnail" in caplog.text

    def test_open_leaves_out_files_of_other_slide(
        self, tmp_path: Path, level_dataset: Dataset, level_file: Path
    ) -> None:
        # Arrange
        other_dataset = deepcopy(level_dataset)
        other_dataset.SOPInstanceUID = generate_uid()
        other_dataset.StudyInstanceUID = generate_uid()
        other_file = tmp_path.joinpath("other.dcm")
        create_layer_file(other_file, other_dataset)

        # Act
        with WsiDicomFileSource.open([level_file, other_file]) as source:
            files = source.files
            instances = list(source.level_instances)

        # Assert
        assert files == [level_file]
        assert len(instances) == 1
