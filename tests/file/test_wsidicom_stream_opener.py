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

from tests.conftest import WsiTestDefinitions
from wsidicom.file.wsidicom_stream_opener import WsiDicomStreamOpener
from wsidicom.uid import WSI_SOP_CLASS_UID


class TestWsiDicomStreamOpener:
    @pytest.mark.parametrize(
        "wsi_folder_and_counts", WsiTestDefinitions.folders_and_instance_counts()
    )
    def test_open_folder(self, wsi_folder_and_counts: Tuple[Path, int]):
        # Arrange
        wsi_folder, instances_count = wsi_folder_and_counts

        # Act
        streams = list(WsiDicomStreamOpener().open(wsi_folder, WSI_SOP_CLASS_UID))
        for stream in streams:
            stream.close()

        # Assert
        assert len(streams) == instances_count

    @pytest.mark.parametrize(
        "wsi_folder_and_counts", WsiTestDefinitions.folders_and_instance_counts()
    )
    def test_open_files(self, wsi_folder_and_counts: Tuple[Path, int]):
        # Arrange
        wsi_folder, instances_count = wsi_folder_and_counts
        files = list(wsi_folder.iterdir())

        # Act
        streams = list(WsiDicomStreamOpener().open(files, WSI_SOP_CLASS_UID))
        for stream in streams:
            stream.close()

        # Assert
        assert len(streams) == instances_count

    @pytest.mark.parametrize(
        "wsi_folder_and_counts", WsiTestDefinitions.folders_and_instance_counts()
    )
    def test_open_streams(self, wsi_folder_and_counts: Tuple[Path, int]):
        # Arrange
        wsi_folder, instances_count = wsi_folder_and_counts
        files = [open(file, "rb") for file in wsi_folder.iterdir() if file.is_file()]

        # Act
        streams = list(WsiDicomStreamOpener().open(files, WSI_SOP_CLASS_UID))
        for stream in streams:
            stream.close()

        # Assert
        assert len(streams) == instances_count
