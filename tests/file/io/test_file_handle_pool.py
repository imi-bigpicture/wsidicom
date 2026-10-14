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

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from wsidicom.file.io.file_handle_pool import FileHandlePool

CONTENT = bytes(range(256)) * 64
MAX_HANDLES = 2
"""Fewer than the threads reading at once in the tests, so that reads wait for a
handle."""


@pytest.fixture()
def pool(tmp_path: Path) -> Iterator[FileHandlePool]:
    path = tmp_path.joinpath("file.bin")
    path.write_bytes(CONTENT)
    pool = FileHandlePool(str(path), MAX_HANDLES)
    yield pool
    pool.close()


@pytest.mark.unittest
class TestFileHandlePool:
    @pytest.mark.parametrize(["position", "length"], [(0, 10), (1000, 4096)])
    def test_read_returns_bytes_at_position(
        self, pool: FileHandlePool, position: int, length: int
    ):
        # Arrange

        # Act
        data = pool.read(position, length)

        # Assert
        assert data == CONTENT[position : position + length]

    def test_read_past_end_returns_bytes_up_to_end(self, pool: FileHandlePool):
        # Arrange
        position = len(CONTENT) - 10

        # Act
        data = pool.read(position, 100)

        # Assert
        assert data == CONTENT[position:]

    def test_reads_from_threads_at_once_return_bytes_at_their_positions(
        self, pool: FileHandlePool
    ):
        # Arrange
        positions = list(range(0, len(CONTENT) - 100, 37))

        # Act
        with ThreadPoolExecutor(8) as executor:
            data = list(
                executor.map(lambda position: pool.read(position, 100), positions)
            )

        # Assert
        assert data == [CONTENT[position : position + 100] for position in positions]

    def test_reads_from_more_threads_than_max_handles_open_max_handles(
        self, pool: FileHandlePool
    ):
        """Reads beyond the most handles wait for one instead of opening another,
        and still read their own bytes."""
        # Arrange
        positions = [position for _ in range(20) for position in range(0, 16000, 97)]

        # Act
        with ThreadPoolExecutor(16) as executor:
            data = list(
                executor.map(lambda position: pool.read(position, 100), positions)
            )

        # Assert
        assert data == [CONTENT[position : position + 100] for position in positions]
        assert 1 <= pool.handle_count <= MAX_HANDLES

    def test_reads_in_turn_open_one_handle(self, pool: FileHandlePool):
        # Arrange

        # Act
        for position in range(0, 1000, 100):
            pool.read(position, 10)

        # Assert
        assert pool.handle_count == 1

    def test_create_with_no_handles_raises(self, tmp_path: Path):
        # Arrange
        path = tmp_path.joinpath("file.bin")
        path.write_bytes(CONTENT)

        # Act & Assert
        with pytest.raises(ValueError):
            FileHandlePool(str(path), 0)

    def test_read_after_close_raises(self, pool: FileHandlePool):
        # Arrange
        pool.read(0, 10)
        pool.close()

        # Act & Assert
        with pytest.raises(ValueError):
            pool.read(0, 10)
