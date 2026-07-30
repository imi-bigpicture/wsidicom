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

from pathlib import Path

import fsspec
import pytest
from upath import UPath

from wsidicom import WsiDicom
from wsidicom.paths import as_local_path, as_upath


@pytest.mark.unittest
class TestAsPath:
    @pytest.mark.parametrize(
        "path, expected",
        [
            # Relative string paths
            ("relative-output", Path.cwd().joinpath("relative-output")),
            ("./relative-output", Path.cwd().joinpath("relative-output")),
            # Relative string path with a parent segment is normalized away
            ("sub/../relative-output", Path.cwd().joinpath("relative-output")),
            # Relative Path objects
            (Path("relative-output"), Path.cwd().joinpath("relative-output")),
            # "::" in a Path is a filename, not a protocol chain
            (Path("local::name"), Path.cwd().joinpath("local::name")),
            # Relative UPath objects
            (UPath("relative-output"), UPath.cwd().joinpath("relative-output")),
            (UPath("./relative-output"), UPath.cwd().joinpath("relative-output")),
        ],
    )
    def test_as_path_resolves_relative_local_path(
        self, path: str | Path | UPath, expected: Path | UPath
    ) -> None:
        # Act
        normalized = as_upath(path)

        # Assert
        assert normalized == expected
        assert normalized.is_absolute()

    @pytest.mark.parametrize(
        "path",
        [
            # Absolute string path
            str(Path.cwd().joinpath("out")),
            # Local URI strings (fsspec absolutizes these at construction)
            "file:///tmp/out",
            "local:///tmp/out",
            "local://path/to/file",
            # Absolute Path objects
            Path.cwd().joinpath("out"),
            Path(Path.cwd().anchor).joinpath("out"),
            # Local URI UPath objects
            UPath("file:///C:/out"),
            UPath("local://path/to/file"),
            # Absolute UPath objects
            UPath.cwd().joinpath("out"),
            UPath(Path.cwd().anchor).joinpath("out"),
        ],
    )
    def test_as_path_returns_absolute_local_path_unchanged(
        self, path: str | Path | UPath
    ) -> None:
        # Arrange
        expected = UPath(path) if isinstance(path, str) else path

        # Act
        normalized = as_upath(path)

        # Assert
        assert normalized == expected
        assert normalized.is_absolute()

    @pytest.mark.parametrize(
        "path",
        [
            "relative-output",
            UPath("relative-output"),
            Path("relative-output"),
            "s3://bucket/out",
        ],
    )
    def test_as_path_returns_upath(self, path: str | Path | UPath) -> None:
        # Act
        normalized = as_upath(path)

        # Assert
        assert isinstance(normalized, UPath)

    @pytest.mark.parametrize(
        "path",
        [
            # Remote string paths
            "s3://bucket/out",
            "simplecache::s3://bucket/out",
            "zip::s3://bucket/archive.zip",
            # Remote UPath objects
            UPath("s3://bucket/out"),
            UPath("simplecache::s3://bucket/out"),
            UPath("zip::s3://bucket/archive.zip"),
        ],
    )
    def test_as_path_remote_path(self, path: str | Path | UPath) -> None:
        # Arrange
        expected = UPath(path) if isinstance(path, str) else path

        # Act
        normalized = as_upath(path)

        # Assert
        assert isinstance(normalized, UPath)
        assert normalized == expected


class TestWsiDicomFsspecRoundtrip:
    def test_save_and_open_over_fsspec(self, wsi: WsiDicom) -> None:
        # Arrange
        output_path = "memory://test-roundtrip/wsi"

        # Act
        created_files = wsi.save(output_path)

        # Assert
        with WsiDicom.open(output_path) as opened:
            read_size = opened.pyramids[0].base_level.size.to_tuple()
            assert opened.read_region((0, 0), 0, read_size).size == read_size
            opened_files = opened.files
            assert opened_files is not None
            assert {str(file) for file in opened_files} == {
                str(file) for file in created_files
            }
        assert all(str(file).startswith(output_path) for file in created_files)


@pytest.mark.unittest
class TestAsLocalPath:
    def test_path_on_other_filesystem_has_no_local_path(self) -> None:
        # Arrange
        filesystem = fsspec.filesystem("memory")
        filesystem.pipe("/test-local/slide.bin", b"slide")

        # Act
        local_path = as_local_path("memory://test-local/slide.bin")

        # Assert
        assert local_path is None

    def test_cached_path_has_no_local_path(self) -> None:
        # Act
        local_path = as_local_path("simplecache::memory://test-local/slide.bin")

        # Assert
        assert local_path is None
