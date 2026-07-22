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

import pytest
from upath import UPath

from wsidicom import WsiDicom


@pytest.mark.unittest
class TestWsiDicomNormalizeOutputPath:
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
    def test_normalize_resolves_relative_local_path(
        self, path: str | Path | UPath, expected: Path | UPath
    ) -> None:
        # Act
        normalized = WsiDicom._normalize_path(path)

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
    def test_normalize_returns_absolute_local_path_unchanged(
        self, path: str | Path | UPath
    ) -> None:
        # Arrange
        expected = UPath(path) if isinstance(path, str) else path

        # Act
        normalized = WsiDicom._normalize_path(path)

        # Assert
        assert normalized == expected
        assert normalized.is_absolute()

    @pytest.mark.parametrize(
        "path, expected_normalized_to_upath",
        [
            ("relative-output", True),  # str normalizes to UPath
            (UPath("relative-output"), True),  # UPath stays UPath
            (Path("relative-output"), False),  # plain Path stays plain Path
        ],
    )
    def test_normalize_preserves_path_type(
        self, path: str | Path | UPath, expected_normalized_to_upath: bool
    ) -> None:
        # Act
        normalized = WsiDicom._normalize_path(path)

        # Assert
        assert isinstance(normalized, UPath) == expected_normalized_to_upath

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
    def test_normalize_remote_path(self, path: str | Path | UPath) -> None:
        # Arrange
        expected = UPath(path) if isinstance(path, str) else path

        # Act
        normalized = WsiDicom._normalize_path(path)

        # Assert
        assert isinstance(normalized, UPath)
        assert normalized == expected
