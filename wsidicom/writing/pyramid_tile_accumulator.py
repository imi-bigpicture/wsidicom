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

"""Accumulator for cascaded tiles in pyramid generation."""

import contextlib
from threading import Thread
from typing import Optional

from PIL import Image

from wsidicom.geometry import Size
from wsidicom.thread import (
    CancellationToken,
    Cancelled,
    CompletionTracker,
    FifoCancelableQueue,
    PriorityCancelableQueue,
    ReadOnlyQueue,
    ShutdownSentinel,
    WriteOnlyQueue,
)
from wsidicom.writing.models import (
    CascadedTile,
    DownsampleEncodeTask,
    EncodingTaskResult,
    PyramidTilePosition,
)


class PyramidTileAccumulator:
    """Buffers tiles from the higher-resolution pyramid level, builds 2x2 blocks,
    submits downsample tasks, and cascades the downsampled result up the pyramid.

    Each accumulator handles one generated pyramid level. It receives decoded
    tiles from the level immediately below (one level higher in resolution)
    via `add_tile`. Once a 2x2 input block is complete, it submits a
    `DownsampleEncodeTask` to the encoder pool. The resulting decoded
    downsampled tile is forwarded to the next accumulator (one level lower
    in resolution) in the cascade chain.

    Memory bound
    ------------
    The internal staging buffer (`_buffer`) holds decoded tiles whose 2x2
    output block is not yet complete. It has no hard size limit, but is
    kept small in practice by upstream invariants:

    - Tiles arrive in Z-order (recursive 2x2 sub-blocks) from
      `SourcePyramidLevelWriter._iter_block_recursive`, so blocks complete
      within a few tiles of being started.
    - The source iterates serially over `(optical_path, focal_plane)`
      pairs, so only one slice accumulates concurrently.
    - The encoder pool input queue and this accumulator's input queue are
      bounded, so upstream stalls when downstream backpressures.

    Under those invariants, the buffer's peak size is on the order of a
    few tiles per active block (typically <10 across all in-flight
    blocks). If upstream tile arrival ever stops being Z-ordered (e.g. a
    custom tile reader that emits in raster order), the buffer could grow
    to roughly one input row of tiles per active slice. The current
    upstream wiring guarantees Z-order; consider adding a memory+disk
    tiered buffer (analogous to `ByteBudgetTileCache`) if that invariant
    is ever weakened.
    """

    def __init__(
        self,
        level_index: int,
        input_tiled_size: Size,
        encoder_pool_queue: WriteOnlyQueue[DownsampleEncodeTask],
        next_accumulator: Optional["PyramidTileAccumulator"] = None,
        is_chain_start: bool = False,
        queue_maxsize: int = 100,
        *,
        token: CancellationToken,
    ):
        """Create an accumulator for one generated pyramid level.

        Parameters
        ----------
        level_index: int
            Pyramid level index for the tiles this accumulator produces.
        input_tiled_size: Size
            Tiled size (columns, rows) of the level below — used to detect
            edge blocks that are smaller than 2x2.
        encoder_pool_queue: WriteOnlyQueue[DownsampleEncodeTask]
            Queue for submitting downsample+encode tasks.
        next_accumulator: Optional[PyramidTileAccumulator]
            The accumulator one level higher in the pyramid (smaller image).
            Cascaded downsampled tiles are forwarded to its input queue.
        is_chain_start: bool
            True if this accumulator is the bottom of a cascade chain (the
            first generated level above a source level). The chain start
            is responsible for forwarding the shutdown sentinel.
        queue_maxsize: int
            Maximum size of the internal input queue. Provides backpressure.
        token: CancellationToken
            Shared cancellation token for fail-fast teardown. The consumer
            cancels it if it fails, and all blocking queue operations observe
            it so the chain unwinds instead of deadlocking.
        """
        self._level_index = level_index
        self._input_tiled_size = input_tiled_size
        self._encoder_pool_queue = encoder_pool_queue
        self._next = next_accumulator
        self._is_chain_start = is_chain_start
        self._token = token
        self._cascade_tracker = CompletionTracker()
        self._cascade_queue: WriteOnlyQueue[CascadedTile] | None = (
            next_accumulator.input_queue if next_accumulator is not None else None
        )
        self._buffer: dict[tuple[int, int, int, int], Image.Image] = {}
        self._input_queue: FifoCancelableQueue[CascadedTile | ShutdownSentinel] = (
            FifoCancelableQueue(maxsize=queue_maxsize)
        )
        self._output_queue: PriorityCancelableQueue[
            EncodingTaskResult | ShutdownSentinel
        ] = PriorityCancelableQueue(maxsize=queue_maxsize)
        self._failure: BaseException | None = None
        self._consumer_thread = Thread(
            target=self._consumer_loop,
            name=f"Accumulator-L{level_index}",
            daemon=True,
        )

    @property
    def output_queue(
        self,
    ) -> ReadOnlyQueue[EncodingTaskResult | ShutdownSentinel]:
        """Queue where encoder pool results land for this level.

        Exposed as read-only because the accumulator owns the queue and
        emits the shutdown sentinel itself; downstream consumers should
        only read from it.
        """
        return self._output_queue

    @property
    def input_queue(self) -> WriteOnlyQueue[CascadedTile | ShutdownSentinel]:
        """Queue producers write decoded tiles (or the shutdown sentinel) to.

        Exposed as write-only because the accumulator owns the queue and
        reads from it itself; upstream producers (cascading tile readers,
        and the next-lower accumulator's encoder pool tasks) should only
        write to it.
        """
        return self._input_queue

    @property
    def next_accumulator(self) -> Optional["PyramidTileAccumulator"]:
        """The accumulator one pyramid level higher than this one, or None
        at the top of the chain.
        """
        return self._next

    @property
    def chain_depth(self) -> int:
        """Count of this accumulator plus every accumulator higher up the
        pyramid in this chain.

        The chain extends via `next_accumulator` to the accumulator one
        level higher — coarser, lower-resolution, smaller image — and
        ends when `next_accumulator` is None (no more generated levels
        above). The source-level tile reader uses this depth to
        pre-arrange tile reads in nested 2x2 Z-order to the matching
        depth, so each accumulator in the chain receives its input tiles
        grouped into complete 2x2 blocks.
        """
        depth = 1
        current = self.next_accumulator
        while current is not None:
            depth += 1
            current = current.next_accumulator
        return depth

    def add_tile(self, x: int, y: int, z: int, path: int, tile: Image.Image) -> None:
        """Submit a decoded tile for accumulation."""
        self._input_queue.put(
            CascadedTile(
                x_index=x, y_index=y, z_index=z, optical_path_index=path, tile=tile
            ),
            self._token,
        )

    def start(self) -> None:
        """Start the consumer thread."""
        self._consumer_thread.start()

    def shutdown(self) -> None:
        """Signal graceful shutdown and wait for the consumer thread to finish.

        Emits the shutdown sentinel if this is the chain start; non-chain-start
        accumulators receive it from the level below. The join is untimed but
        cannot hang: any failure cancels the token, which unblocks the
        consumer's queue waits. Re-raises an exception that crashed this
        accumulator's consumer.
        """
        if self._is_chain_start:
            with contextlib.suppress(Cancelled):
                self._input_queue.put(ShutdownSentinel(), self._token)
        self._consumer_thread.join()
        if self._failure is not None:
            raise self._failure

    def cleanup(self) -> None:
        """Best-effort teardown on error.

        Cancels the token (if still live) so the consumer leaves its blocking
        get, then joins with a timeout so teardown cannot hang. Downstream is
        unblocked by the token, not by a sentinel from here, so no defensive
        output-queue emission is needed.
        """
        self._token.cancel(RuntimeError("accumulator cleanup"))
        if self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=5.0)

    def _consumer_loop(self) -> None:
        try:
            item = self._input_queue.get(self._token)
            while not isinstance(item, ShutdownSentinel):
                self._process_item(item)
                item = self._input_queue.get(self._token)
            self._cascade_tracker.wait_for_zero()
        except Cancelled:
            return
        except BaseException as exc:
            self._failure = exc
            self._token.cancel(exc)
            return
        self._emit_completion_sentinels()

    def _emit_completion_sentinels(self) -> None:
        """On normal completion, tell downstream that no more results will come.

        `wait_for_zero()` has returned, so all in-flight encoder pool tasks have
        put their results on the output queue and cascaded their tiles to the
        next accumulator; the sentinels therefore sort/arrive after them.
        Skipped if the token cancels concurrently — downstream then unwinds via
        the token instead.
        """
        try:
            self._output_queue.put(ShutdownSentinel(), self._token)
            if self.next_accumulator is not None:
                self.next_accumulator.input_queue.put(ShutdownSentinel(), self._token)
        except Cancelled:
            pass

    def _process_item(self, item: CascadedTile) -> None:
        x, y, z, path, tile = (
            item.x_index,
            item.y_index,
            item.z_index,
            item.optical_path_index,
            item.tile,
        )
        output_x, output_y = x // 2, y // 2

        self._buffer[(x, y, z, path)] = tile
        position_rows = self._block_positions(output_x, output_y)
        if not all(
            (px, py, z, path) in self._buffer for row in position_rows for px, py in row
        ):
            return

        block_tiles = [
            [self._buffer.pop((px, py, z, path)) for px, py in row]
            for row in position_rows
        ]

        coords = PyramidTilePosition(
            level=self._level_index,
            x_index=output_x,
            y_index=output_y,
            z_index=z,
            optical_path_index=path,
        )
        # Increment before handing off so wait_for_zero never observes a false
        # zero while a task is in flight. On the cancel path the tracker may be
        # left unbalanced (a task enqueued after cancel is never processed by the
        # stopped dispatcher, or a full-queue put raises Cancelled) — harmless,
        # because the consumer returns on Cancelled without calling wait_for_zero.
        self._cascade_tracker.increment()
        self._encoder_pool_queue.put(
            DownsampleEncodeTask(
                coordinates=coords,
                tiles=block_tiles,
                output_queue=self._output_queue,
                cascade_tracker=self._cascade_tracker,
                cascade_queue=self._cascade_queue,
            ),
            self._token,
        )

    def _block_positions(
        self, output_x: int, output_y: int
    ) -> list[list[tuple[int, int]]]:
        rows: list[list[tuple[int, int]]] = []
        for py in [output_y * 2, output_y * 2 + 1]:
            if py >= self._input_tiled_size.height:
                break
            row = [
                (px, py)
                for px in [output_x * 2, output_x * 2 + 1]
                if px < self._input_tiled_size.width
            ]
            if row:
                rows.append(row)
        return rows
