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

from pathlib import Path
from typing import Tuple

import pytest
from pydicom.fileset import FileSet

from tests.conftest import WsiTestDefinitions
from wsidicom.file.wsidicom_file_source import WsiDicomFileSource


@pytest.mark.unittest
class TestWsiDicomFileSource:
    @pytest.mark.parametrize(
        "wsi_folder_and_counts", WsiTestDefinitions.folders_and_counts()
    )
    def test_open(self, wsi_folder_and_counts: Tuple[Path, int, bool, bool]):
        # Arrange
        wsi_folder, level_count, label, overview = wsi_folder_and_counts

        # Act
        with WsiDicomFileSource(wsi_folder) as source:
            instances = list(source.level_instances)
            label_instances = list(source.label_instances)
            overview_instances = list(source.overview_instances)

        # Assert
        assert len(instances) == level_count
        assert len(label_instances) == (1 if label else 0)
        assert len(overview_instances) == (1 if overview else 0)

    @pytest.mark.parametrize("path_is_folder", [True, False])
    def test_open_dicomdir(self, tmp_path: Path, wsi_file: Path, path_is_folder: bool):
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
