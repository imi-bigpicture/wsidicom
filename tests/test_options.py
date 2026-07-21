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

"""Unit tests for the concatenation option value parsing and validation."""

import pytest

from wsidicom import ConcatenationByBytes, ConcatenationByFrames


class TestConcatenationByFrames:
    @pytest.mark.parametrize("count", [-10, -1, 0])
    def test_rejects_non_positive_count(self, count: int) -> None:
        # Arrange

        # Act & Assert
        with pytest.raises(ValueError):
            ConcatenationByFrames(count)

    @pytest.mark.parametrize("count", [2, 10, 100])
    def test_accepts_positive_count(self, count: int) -> None:
        # Arrange

        # Act
        concatenation = ConcatenationByFrames(count)

        # Assert
        assert concatenation.count == count


class TestConcatenationByBytes:
    @pytest.mark.parametrize(
        ("size", "expected"),
        [
            (1000, 1000),
            ("1000", 1000),
            ("500K", 500 * 1024),
            ("100m", 100 * 1024**2),
            ("2G", 2 * 1024**3),
            ("1000B", 1000),
            ("500KB", 500 * 1024),
            ("100mb", 100 * 1024**2),
            ("2GB", 2 * 1024**3),
        ],
    )
    def test_size_accepts_int_and_suffixed_string(
        self, size: int | str, expected: int
    ) -> None:
        # Arrange & Act
        concatenation = ConcatenationByBytes(size)

        # Assert
        assert concatenation.size == expected

    @pytest.mark.parametrize("size", [0, -1, "0", "0K", "-1K"])
    def test_rejects_non_positive_size(self, size: int | str) -> None:
        # Arrange

        # Act & Assert
        with pytest.raises(ValueError):
            ConcatenationByBytes(size)

    @pytest.mark.parametrize("size", ["100X", "100XB", "100xb"])
    def test_rejects_unknown_suffix(self, size: str) -> None:
        # Arrange

        # Act & Assert
        with pytest.raises(ValueError):
            ConcatenationByBytes(size)
