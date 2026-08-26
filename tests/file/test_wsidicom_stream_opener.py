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

import fsspec
import pytest
from pydicom.uid import ExplicitVRLittleEndian, VLWholeSlideMicroscopyImageStorage

from tests.conftest import WsiTestDefinitions
from wsidicom.file import WsiDicomStreamOpener


@pytest.mark.unittest
class TestWsiDicomStreamOpener:
    @pytest.mark.parametrize(
        "wsi_folder_and_counts", WsiTestDefinitions.folders_and_instance_counts()
    )
    def test_open_folder(self, wsi_folder_and_counts: tuple[Path, int]):
        # Arrange
        wsi_folder, instances_count = wsi_folder_and_counts

        # Act
        streams = list(
            WsiDicomStreamOpener().open(wsi_folder, VLWholeSlideMicroscopyImageStorage)
        )
        for stream in streams:
            stream.close()

        # Assert
        assert len(streams) == instances_count

    @pytest.mark.parametrize(
        "wsi_folder_and_counts", WsiTestDefinitions.folders_and_instance_counts()
    )
    def test_open_files(self, wsi_folder_and_counts: tuple[Path, int]):
        # Arrange
        wsi_folder, instances_count = wsi_folder_and_counts
        files = list(wsi_folder.iterdir())

        # Act
        streams = list(
            WsiDicomStreamOpener().open(files, VLWholeSlideMicroscopyImageStorage)
        )
        for stream in streams:
            stream.close()

        # Assert
        assert len(streams) == instances_count

    @pytest.mark.parametrize(
        ["pattern", "expected_count"],
        [("memory://test-glob/*.dcm", 1), ("memory://test-glob/**/*.dcm", 2)],
    )
    def test_open_glob_pattern(self, pattern: str, expected_count: int, wsi_file: Path):
        # Arrange
        filesystem = fsspec.filesystem("memory")
        for path in ("/test-glob/one.dcm", "/test-glob/sub/two.dcm"):
            filesystem.pipe(path, wsi_file.read_bytes())

        # Act
        streams = list(WsiDicomStreamOpener().open(pattern))
        for stream in streams:
            stream.close()

        # Assert
        assert len(streams) == expected_count

    def test_open_for_writing_fsspec_path_should_keep_protocol(self):
        # Arrange
        path = "memory://test-opener/instance.dcm"

        # Act
        with WsiDicomStreamOpener().open_for_writing(
            path, "w+b", ExplicitVRLittleEndian
        ) as stream:
            filepath = stream.filepath

        # Assert
        assert str(filepath) == path
