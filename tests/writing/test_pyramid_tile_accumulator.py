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

"""Tests for PyramidTileAccumulator."""

import numpy as np
import pytest

from wsidicom.geometry import Size
from wsidicom.thread import CancellationToken, PriorityCancelableQueue
from wsidicom.writing.models import DownsampleEncodeTask
from wsidicom.writing.pyramid_tile_accumulator import (
    PyramidTileAccumulator,
    WritingPyramidTileAccumulator,
)


def make_tile() -> np.ndarray:
    return np.zeros((256, 256, 3), dtype=np.uint8)


@pytest.fixture
def token() -> CancellationToken:
    """Create a cancellation token for accumulator construction."""
    return CancellationToken()


@pytest.mark.unittest
class TestChainDepth:
    """Tests for chain_depth traversal."""

    def test_single_accumulator_has_depth_1(self, token: CancellationToken):
        # Arrange
        accumulator = WritingPyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(4, 4),
            encoder_pool_queue=PriorityCancelableQueue(),
        )

        # Act
        depth = accumulator.chain_depth

        # Assert
        assert depth == 1

    def test_two_level_chain(self, token: CancellationToken):
        # Arrange
        encoder_pool_queue: PriorityCancelableQueue = PriorityCancelableQueue()
        upper = WritingPyramidTileAccumulator(
            token=token,
            level_index=2,
            input_tiled_size=Size(2, 2),
            encoder_pool_queue=encoder_pool_queue,
        )
        lower = WritingPyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(4, 4),
            encoder_pool_queue=encoder_pool_queue,
            next_accumulator=upper,
        )

        # Act
        lower_depth = lower.chain_depth
        upper_depth = upper.chain_depth

        # Assert
        assert lower_depth == 2
        assert upper_depth == 1

    def test_three_level_chain(self, token: CancellationToken):
        # Arrange
        encoder_pool_queue: PriorityCancelableQueue = PriorityCancelableQueue()
        top = WritingPyramidTileAccumulator(
            token=token,
            level_index=3,
            input_tiled_size=Size(1, 1),
            encoder_pool_queue=encoder_pool_queue,
        )
        middle = WritingPyramidTileAccumulator(
            token=token,
            level_index=2,
            input_tiled_size=Size(2, 2),
            encoder_pool_queue=encoder_pool_queue,
            next_accumulator=top,
        )
        bottom = WritingPyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(4, 4),
            encoder_pool_queue=encoder_pool_queue,
            next_accumulator=middle,
        )

        # Act
        bottom_depth = bottom.chain_depth
        middle_depth = middle.chain_depth
        top_depth = top.chain_depth

        # Assert
        assert bottom_depth == 3
        assert middle_depth == 2
        assert top_depth == 1


@pytest.mark.unittest
class TestBlockPositions:
    """Tests for _block_positions edge-block handling."""

    def test_full_2x2_block_interior(self, token: CancellationToken):
        # Arrange
        accumulator = WritingPyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(10, 10),
            encoder_pool_queue=PriorityCancelableQueue(),
        )

        # Act — output (2, 3) maps to input columns 4-5, rows 6-7
        positions = accumulator._block_positions(output_x=2, output_y=3)

        # Assert
        assert positions == [[(4, 6), (5, 6)], [(4, 7), (5, 7)]]

    def test_right_edge_block(self, token: CancellationToken):
        # Arrange — input width=5 means input column 5 is out of bounds
        accumulator = WritingPyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(5, 10),
            encoder_pool_queue=PriorityCancelableQueue(),
        )

        # Act
        positions = accumulator._block_positions(output_x=2, output_y=0)

        # Assert — only one column per row
        assert positions == [[(4, 0)], [(4, 1)]]

    def test_bottom_edge_block(self, token: CancellationToken):
        # Arrange — input height=5 means input row 5 is out of bounds
        accumulator = WritingPyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(10, 5),
            encoder_pool_queue=PriorityCancelableQueue(),
        )

        # Act
        positions = accumulator._block_positions(output_x=0, output_y=2)

        # Assert — only one row
        assert positions == [[(0, 4), (1, 4)]]

    def test_corner_1x1_block(self, token: CancellationToken):
        # Arrange
        accumulator = WritingPyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(5, 5),
            encoder_pool_queue=PriorityCancelableQueue(),
        )

        # Act
        positions = accumulator._block_positions(output_x=2, output_y=2)

        # Assert — single tile at the corner
        assert positions == [[(4, 4)]]


@pytest.mark.unittest
class TestSingleAccumulatorDrain:
    """Tests for accumulator consumer loop and block-completion behaviour."""

    def test_full_2x2_block_submits_task(self, token: CancellationToken):
        # Arrange
        encoder_pool_queue: PriorityCancelableQueue = PriorityCancelableQueue()
        accumulator = WritingPyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(2, 2),
            encoder_pool_queue=encoder_pool_queue,
            is_chain_start=True,
        )
        accumulator.start()
        try:
            tiles = [make_tile() for _ in range(4)]

            # Act — feed all 4 tiles for output block (0, 0)
            accumulator.add_tile(0, 0, 0, 0, tiles[0])
            accumulator.add_tile(1, 0, 0, 0, tiles[1])
            accumulator.add_tile(0, 1, 0, 0, tiles[2])
            accumulator.add_tile(1, 1, 0, 0, tiles[3])

            # Assert — task is submitted
            task = encoder_pool_queue.get(CancellationToken())
            assert isinstance(task, DownsampleEncodeTask)
            assert task.coordinates.level == 1
            assert task.coordinates.x_index == 0
            assert task.coordinates.y_index == 0
            # Arrays compare elementwise, so assert pass-through identity.
            assert task.tiles[0][0] is tiles[0]
            assert task.tiles[0][1] is tiles[1]
            assert task.tiles[1][0] is tiles[2]
            assert task.tiles[1][1] is tiles[3]
            assert task.output_queue is accumulator.output_queue

            # Simulate encoder worker completion so wait_for_zero unblocks shutdown
            task.cascade_tracker.decrement()
        finally:
            accumulator.shutdown()

    def test_partial_block_does_not_submit(self, token: CancellationToken):
        # Arrange
        encoder_pool_queue: PriorityCancelableQueue = PriorityCancelableQueue()
        accumulator = WritingPyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(2, 2),
            encoder_pool_queue=encoder_pool_queue,
            is_chain_start=True,
        )
        accumulator.start()

        # Act — feed only 3 of the 4 tiles for block (0, 0)
        accumulator.add_tile(0, 0, 0, 0, make_tile())
        accumulator.add_tile(1, 0, 0, 0, make_tile())
        accumulator.add_tile(0, 1, 0, 0, make_tile())
        # Shutdown drains the input queue, so we know consumer has processed everything
        accumulator.shutdown()

        # Assert — no task submitted because block never completed
        assert encoder_pool_queue.qsize() == 0

    def test_unwritten_level_submits_task_without_output_queue(
        self, token: CancellationToken
    ):
        """An intermediate level cascades but produces no output to encode.

        Levels that only bridge a gap in a sparse pyramid are downsampled to feed
        the level above, but no instance is written for them.
        """
        # Arrange - the base accumulator has no output queue
        encoder_pool_queue: PriorityCancelableQueue = PriorityCancelableQueue()
        accumulator = PyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(2, 2),
            encoder_pool_queue=encoder_pool_queue,
            is_chain_start=True,
        )
        accumulator.start()
        try:
            # Act — a full 2x2 block
            accumulator.add_tile(0, 0, 0, 0, make_tile())
            accumulator.add_tile(1, 0, 0, 0, make_tile())
            accumulator.add_tile(0, 1, 0, 0, make_tile())
            accumulator.add_tile(1, 1, 0, 0, make_tile())

            # Assert — task carries no output queue, so it is not encoded
            task = encoder_pool_queue.get(CancellationToken())
            assert isinstance(task, DownsampleEncodeTask)
            assert task.output_queue is None

            task.cascade_tracker.decrement()
        finally:
            accumulator.shutdown()

    def test_tile_outside_input_grid_raises(self, token: CancellationToken):
        """A tile outside the input grid fails loudly, naming the coordinate.

        Regression: `_block_positions` returned no positions for such a tile, the
        completeness check passed vacuously (`all()` of nothing is True), and an
        empty block was submitted to the encoder pool, surfacing there as an
        unrelated stitching error.
        """
        # Arrange
        encoder_pool_queue: PriorityCancelableQueue = PriorityCancelableQueue()
        accumulator = WritingPyramidTileAccumulator(
            token=token,
            level_index=2,
            input_tiled_size=Size(2, 2),
            encoder_pool_queue=encoder_pool_queue,
            is_chain_start=True,
        )
        accumulator.start()

        # Act — column 5 is outside the 2x2 input grid
        accumulator.add_tile(5, 0, 0, 0, make_tile())

        # Assert — the consumer failure is re-raised on shutdown, and no empty
        # block reached the encoder pool
        with pytest.raises(ValueError, match="outside its input grid"):
            accumulator.shutdown()
        assert encoder_pool_queue.qsize() == 0


@pytest.mark.unittest
class TestCascadeShutdown:
    """Tests for shutdown sentinel propagation through the cascade chain."""

    def test_shutdown_propagates_through_chain(self, token: CancellationToken):
        # Arrange — two-level chain: lower (chain start) → upper
        encoder_pool_queue: PriorityCancelableQueue = PriorityCancelableQueue()
        upper = WritingPyramidTileAccumulator(
            token=token,
            level_index=2,
            input_tiled_size=Size(2, 2),
            encoder_pool_queue=encoder_pool_queue,
        )
        lower = WritingPyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(4, 4),
            encoder_pool_queue=encoder_pool_queue,
            next_accumulator=upper,
            is_chain_start=True,
        )
        upper.start()
        lower.start()

        # Act — shutting down lower (chain_start) propagates sentinel to upper
        lower.shutdown()

        # Assert — upper's consumer thread also exits after receiving the
        # propagated sentinel
        upper._consumer_thread.join(timeout=5)
        assert not upper._consumer_thread.is_alive()

    def test_non_chain_start_does_not_send_sentinel_on_shutdown(
        self, token: CancellationToken
    ):
        # Arrange — upper accumulator with is_chain_start=False
        encoder_pool_queue: PriorityCancelableQueue = PriorityCancelableQueue()
        upper = WritingPyramidTileAccumulator(
            token=token,
            level_index=2,
            input_tiled_size=Size(2, 2),
            encoder_pool_queue=encoder_pool_queue,
            is_chain_start=False,
        )
        upper.start()

        # Act — shutdown() does NOT send the sentinel itself; thread stays alive
        # until something else feeds it a sentinel (e.g., the lower chain start).
        # Simulate that by sending one ourselves via the input queue.
        from wsidicom.thread import ShutdownSentinel

        upper._input_queue.put(ShutdownSentinel(), CancellationToken())
        upper.shutdown()

        # Assert — thread exited cleanly
        assert not upper._consumer_thread.is_alive()


@pytest.mark.unittest
class TestCleanup:
    """Tests for cleanup() error-path behaviour."""

    def test_cleanup_unblocks_consumer_thread(self, token: CancellationToken):
        # Arrange — non-chain-start accumulator (cleanup must still unblock)
        accumulator = WritingPyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(2, 2),
            encoder_pool_queue=PriorityCancelableQueue(),
            is_chain_start=False,
        )
        accumulator.start()

        # Act
        accumulator.cleanup()

        # Assert — consumer thread exits
        assert not accumulator._consumer_thread.is_alive()


@pytest.mark.unittest
class TestConsumerFailureWatchdog:
    """Consumer thread failures must surface on shutdown and unblock the cascade."""

    def test_consumer_failure_raised_on_shutdown(self, token: CancellationToken):
        # Arrange — encoder_pool_queue that raises on put to simulate a worker error
        class FailingQueue:
            def put(self, item, token):
                raise RuntimeError("encoder pool put failed")

        accumulator = WritingPyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(2, 2),
            encoder_pool_queue=FailingQueue(),
            is_chain_start=True,
        )
        accumulator.start()
        # Feed enough tiles to complete a 2x2 block, triggering the failing put
        for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            accumulator.add_tile(x, y, z=0, path=0, tile=make_tile())

        # Act / Assert — shutdown surfaces the worker failure
        with pytest.raises(RuntimeError, match="encoder pool put failed"):
            accumulator.shutdown()

    def test_consumer_failure_propagates_shutdown_to_next(
        self, token: CancellationToken
    ):
        # Arrange — two-level chain sharing one cancellation token; lower consumer
        # fails, and upper must still exit (via the shared cancellation token).
        class FailingQueue:
            def put(self, item, token):
                raise RuntimeError("encoder pool put failed")

        upper = WritingPyramidTileAccumulator(
            token=token,
            level_index=2,
            input_tiled_size=Size(2, 2),
            encoder_pool_queue=PriorityCancelableQueue(),
        )
        lower = WritingPyramidTileAccumulator(
            token=token,
            level_index=1,
            input_tiled_size=Size(4, 4),
            encoder_pool_queue=FailingQueue(),
            next_accumulator=upper,
            is_chain_start=True,
        )
        upper.start()
        lower.start()
        # Feed a complete 2x2 block to lower to trigger failure
        for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            lower.add_tile(x, y, z=0, path=0, tile=make_tile())

        # Act — lower fails and cancels the shared token; upper must not hang
        with pytest.raises(RuntimeError):
            lower.shutdown()

        # Assert — lower recorded the failure and upper's consumer unwound via
        # the shared cancellation token
        assert isinstance(token.exception, RuntimeError)
        upper._consumer_thread.join(timeout=5)
        assert not upper._consumer_thread.is_alive()
