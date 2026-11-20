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

"""Tests for the tile readers and the submitters of encoded tiles."""

from dataclasses import dataclass, field
from typing import TypeVar

import numpy as np
import pytest
from decoy import Decoy

from wsidicom.codec import Encoder, Recoder
from wsidicom.geometry import Point
from wsidicom.instance import ImageData
from wsidicom.thread import CancellationToken, FifoCancelableQueue
from wsidicom.writing.models import EncodeTask, EncodingTaskResult, RecodeTask
from wsidicom.writing.tile_readers import (
    EncodedTileReader,
    PassthroughEncodedTileSubmitter,
    RecodeEncodedTileSubmitter,
    TranscodeTileReader,
)

QueuedType = TypeVar("QueuedType")

ENCODED_TILES = [b"tile_0_0", b"tile_1_0", b"tile_0_1"]


@dataclass
class RecordingAccumulator:
    """Accumulator that records the decoded tiles it is given."""

    chain_depth: int = 1
    """Number of generated levels in the cascade starting from this one."""
    added: list[tuple[int, int, int, int, np.ndarray]] = field(default_factory=list)
    """The tiles added, as (x, y, z, path, tile)."""

    def add_tile(self, x: int, y: int, z: int, path: int, tile: np.ndarray) -> None:
        self.added.append((x, y, z, path, tile))


@pytest.fixture
def decoy() -> Decoy:
    """Create a Decoy instance for mocking."""
    return Decoy()


@pytest.fixture
def token() -> CancellationToken:
    """Create a cancellation token for queue operations."""
    return CancellationToken()


@pytest.fixture
def positions() -> list[Point]:
    """Two tiles in the first row and one in the second."""
    return [Point(0, 0), Point(1, 0), Point(0, 1)]


@pytest.fixture
def decoded_tiles() -> list[np.ndarray]:
    """Return decoded tiles for the positions, each filled with its index."""
    return [np.full((4, 4, 3), value, dtype=np.uint8) for value in range(3)]


def drain(
    queue: FifoCancelableQueue[QueuedType], token: CancellationToken
) -> list[QueuedType]:
    """Return the items put on a queue, in order."""
    items: list[QueuedType] = []
    while queue.qsize() > 0:
        items.append(queue.get(token))
    return items


@pytest.mark.unittest
class TestEncodedTileReader:
    def test_passthrough_puts_one_result_per_row_segment_on_output_queue(
        self, decoy: Decoy, token: CancellationToken, positions: list[Point]
    ) -> None:
        # Arrange
        image_data = decoy.mock(cls=ImageData)
        decoy.when(image_data.get_encoded_tiles(positions, 0.0, "0")).then_return(
            iter(ENCODED_TILES)
        )
        output_queue: FifoCancelableQueue[EncodingTaskResult] = FifoCancelableQueue()
        reader = EncodedTileReader(
            2, PassthroughEncodedTileSubmitter(token), None, token
        )

        # Act
        reader.read_and_submit(image_data, positions, 0.0, "0", 0, 0, output_queue)

        # Assert
        results = drain(output_queue, token)
        assert [result.tiles for result in results] == [
            [b"tile_0_0", b"tile_1_0"],
            [b"tile_0_1"],
        ]
        assert [
            (
                result.coordinates.level,
                result.coordinates.x_index,
                result.coordinates.y_index,
            )
            for result in results
        ] == [(2, 0, 0), (2, 0, 1)]

    def test_recode_submits_one_recode_task_per_row_segment(
        self, decoy: Decoy, token: CancellationToken, positions: list[Point]
    ) -> None:
        # Arrange
        image_data = decoy.mock(cls=ImageData)
        decoy.when(image_data.get_encoded_tiles(positions, 0.0, "0")).then_return(
            iter(ENCODED_TILES)
        )
        pool_queue: FifoCancelableQueue[RecodeTask] = FifoCancelableQueue()
        output_queue: FifoCancelableQueue[EncodingTaskResult] = FifoCancelableQueue()
        recoder = decoy.mock(cls=Recoder)
        reader = EncodedTileReader(
            2, RecodeEncodedTileSubmitter(pool_queue, recoder, token), None, token
        )

        # Act
        reader.read_and_submit(image_data, positions, 0.0, "0", 0, 0, output_queue)

        # Assert
        tasks = drain(pool_queue, token)
        assert [task.tiles for task in tasks] == [
            [b"tile_0_0", b"tile_1_0"],
            [b"tile_0_1"],
        ]
        assert [
            (task.coordinates.level, task.coordinates.x_index, task.coordinates.y_index)
            for task in tasks
        ] == [(2, 0, 0), (2, 0, 1)]
        assert all(task.output_queue is output_queue for task in tasks)
        assert all(task.recoder is recoder for task in tasks)
        assert output_queue.qsize() == 0

    def test_feeds_decoded_tiles_to_accumulator(
        self,
        decoy: Decoy,
        token: CancellationToken,
        positions: list[Point],
        decoded_tiles: list[np.ndarray],
    ) -> None:
        # Arrange
        image_data = decoy.mock(cls=ImageData)
        decoy.when(
            image_data.get_encoded_and_decoded_tiles(positions, 0.0, "0")
        ).then_return(iter(zip(ENCODED_TILES, decoded_tiles, strict=True)))
        output_queue: FifoCancelableQueue[EncodingTaskResult] = FifoCancelableQueue()
        accumulator = RecordingAccumulator()
        reader = EncodedTileReader(
            0, PassthroughEncodedTileSubmitter(token), accumulator, token
        )

        # Act
        reader.read_and_submit(image_data, positions, 0.0, "0", 0, 0, output_queue)

        # Assert
        assert [result.tiles for result in drain(output_queue, token)] == [
            [b"tile_0_0", b"tile_1_0"],
            [b"tile_0_1"],
        ]
        assert [(x, y) for x, y, _, _, _ in accumulator.added] == [
            (0, 0),
            (1, 0),
            (0, 1),
        ]
        for (_, _, _, _, tile), expected in zip(
            accumulator.added, decoded_tiles, strict=True
        ):
            assert np.array_equal(tile, expected)

    @pytest.mark.parametrize(
        ["accumulator", "expected"],
        [(None, 0), (RecordingAccumulator(chain_depth=3), 3)],
    )
    def test_reports_accumulator_chain_depth(
        self,
        token: CancellationToken,
        accumulator: RecordingAccumulator | None,
        expected: int,
    ) -> None:
        # Arrange

        # Act
        reader = EncodedTileReader(
            0, PassthroughEncodedTileSubmitter(token), accumulator, token
        )

        # Assert
        assert reader.accumulator_chain_depth == expected


@pytest.mark.unittest
class TestTranscodeTileReader:
    def test_submits_one_encode_task_per_row_segment(
        self,
        decoy: Decoy,
        token: CancellationToken,
        positions: list[Point],
        decoded_tiles: list[np.ndarray],
    ) -> None:
        # Arrange
        image_data = decoy.mock(cls=ImageData)
        decoy.when(
            image_data.get_decoded_tiles(positions, 0.0, "0", cache=False)
        ).then_return(iter(decoded_tiles))
        pool_queue: FifoCancelableQueue[EncodeTask] = FifoCancelableQueue()
        output_queue: FifoCancelableQueue[EncodingTaskResult] = FifoCancelableQueue()
        encoder = decoy.mock(cls=Encoder)
        reader = TranscodeTileReader(2, pool_queue, encoder, None, token)

        # Act
        reader.read_and_submit(image_data, positions, 0.0, "0", 0, 0, output_queue)

        # Assert
        tasks = drain(pool_queue, token)
        assert [len(task.tiles) for task in tasks] == [2, 1]
        assert [
            (task.coordinates.level, task.coordinates.x_index, task.coordinates.y_index)
            for task in tasks
        ] == [(2, 0, 0), (2, 0, 1)]
        assert all(task.encoder is encoder for task in tasks)
        assert all(task.output_queue is output_queue for task in tasks)
        assert output_queue.qsize() == 0

    def test_feeds_decoded_tiles_to_accumulator(
        self,
        decoy: Decoy,
        token: CancellationToken,
        positions: list[Point],
        decoded_tiles: list[np.ndarray],
    ) -> None:
        # Arrange
        image_data = decoy.mock(cls=ImageData)
        decoy.when(
            image_data.get_decoded_tiles(positions, 0.0, "0", cache=False)
        ).then_return(iter(decoded_tiles))
        pool_queue: FifoCancelableQueue[EncodeTask] = FifoCancelableQueue()
        output_queue: FifoCancelableQueue[EncodingTaskResult] = FifoCancelableQueue()
        accumulator = RecordingAccumulator()
        reader = TranscodeTileReader(
            0, pool_queue, decoy.mock(cls=Encoder), accumulator, token
        )

        # Act
        reader.read_and_submit(image_data, positions, 0.0, "0", 0, 0, output_queue)

        # Assert
        assert [(x, y) for x, y, _, _, _ in accumulator.added] == [
            (0, 0),
            (1, 0),
            (0, 1),
        ]
        for (_, _, _, _, tile), expected in zip(
            accumulator.added, decoded_tiles, strict=True
        ):
            assert np.array_equal(tile, expected)
