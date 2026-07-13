#    Copyright 2021, 2022, 2023 SECTRA AB
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

"""Priority-based encoder pool for parallel tile encoding."""

import contextlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from types import TracebackType
from typing import Any

from PIL import Image

from wsidicom.codec import Encoder
from wsidicom.downsampler import Downsampler
from wsidicom.geometry import Size
from wsidicom.stitcher import Stitcher
from wsidicom.thread import (
    CancelableQueue,
    CancellationToken,
    Cancelled,
    PriorityCancelableQueue,
    ShutdownSentinel,
    WriteOnlyQueue,
)
from wsidicom.writing.models import (
    CascadedTile,
    DownsampleEncodeTask,
    EncodeTask,
    EncodingTaskResult,
)


class EncoderPool:
    """Pool of encoder workers using ThreadPoolExecutor."""

    def __init__(
        self,
        encoder: Encoder,
        num_workers: int,
        downsampler: Downsampler,
        stitcher: Stitcher,
        tile_size: Size,
        blank_tile: Image.Image,
        input_queue_maxsize: int = 100,
        *,
        token: CancellationToken,
    ):
        """Create an encoder pool.

        Parameters
        ----------
        encoder: Encoder
            Encoder to use for encoding tiles.
        num_workers: int
            Number of encoder worker threads.
        downsampler: Downsampler
            Downsampler for downsample+encode tasks.
        stitcher: Stitcher
            Stitcher used to combine input tile blocks before downsampling.
        tile_size: Size
            Output tile size for downsample+encode tasks.
        blank_tile: Image.Image
            Background tile of the output tile size, used to pad a downsampled
            edge block back to a full tile. Treated as read-only.
        input_queue_maxsize: int
            Maximum size of input queue. Provides backpressure when full.
        token: CancellationToken
            Shared cancellation token for fail-fast teardown. A worker that
            fails cancels it, and all blocking queue operations observe it so
            the pipeline unwinds instead of deadlocking.
        """
        self._encoder = encoder
        self._downsampler = downsampler
        self._stitcher = stitcher
        self._tile_size = tile_size
        self._blank_tile = blank_tile
        self._token = token
        self._input_queue: CancelableQueue[
            EncodeTask | DownsampleEncodeTask | ShutdownSentinel
        ] = PriorityCancelableQueue(maxsize=input_queue_maxsize)
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="EncoderWorker"
        )
        self._dispatcher_thread = Thread(
            target=self._dispatcher_loop, name="EncoderDispatcher", daemon=False
        )

    def start(self) -> None:
        """Start the dispatcher thread."""
        if self._dispatcher_thread.is_alive():
            raise RuntimeError("Encoder pool is already running")
        self._dispatcher_thread.start()

    @property
    def queue(self) -> WriteOnlyQueue[EncodeTask | DownsampleEncodeTask]:
        """The queue for submitting encoding tasks."""
        return self._input_queue

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the encoder pool.

        Parameters
        ----------
        wait: bool
            If True, wait for dispatcher and executor to finish.
        """
        # Cancel-aware: on the cancel path the dispatcher has already exited and
        # nothing drains the input queue, so a plain put could block forever.
        with contextlib.suppress(Cancelled):
            self._input_queue.put(ShutdownSentinel(), self._token)
        if wait and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join()

        self._executor.shutdown(wait=wait)

    def _dispatcher_loop(self) -> None:
        """Dispatcher thread: pulls from priority queue, submits to executor.

        Exits on a `ShutdownSentinel` (graceful) or once the token is cancelled
        (fail-fast).
        """
        try:
            task = self._input_queue.get(self._token)
            while not isinstance(task, ShutdownSentinel):
                self._executor.submit(self._dispatch_task, task)
                task = self._input_queue.get(self._token)
        except Cancelled:
            return

    def _dispatch_task(self, task: EncodeTask | DownsampleEncodeTask) -> None:
        """Dispatch a task to the appropriate handler."""
        if isinstance(task, DownsampleEncodeTask):
            self._downsample_and_encode(task)
        else:
            self._encode(task)

    def _safe_put(self, queue: WriteOnlyQueue[Any], item: Any) -> None:
        """Put on a downstream queue, observing the cancellation token.

        Swallows `Cancelled` (raised when the token cancels while waiting on a
        full queue) so a worker tearing down does not raise; the originating
        failure is already held by the token.
        """
        with contextlib.suppress(Cancelled):
            queue.put(item, self._token)

    def _encode(self, task: EncodeTask) -> None:
        """Encode tiles and submit the result to the output queue.

        A failure cancels the token (the single failure channel) and the worker
        returns; the pipeline then unwinds without a result.
        """
        try:
            encoded_tiles = [self._encoder.encode(tile) for tile in task.tiles]
        except Exception as exception:
            self._token.cancel(exception)
            return
        self._safe_put(
            task.output_queue,
            EncodingTaskResult(coordinates=task.coordinates, tiles=encoded_tiles),
        )

    def _downsample_and_encode(self, task: DownsampleEncodeTask) -> None:
        """Stitch, downsample, encode, and submit to output and cascade queues.

        A failure cancels the token and the worker returns. The cascade push
        always precedes the tracker decrement so a level's shutdown (which
        waits for the tracker to reach zero) cannot forward its sentinel ahead
        of a cascaded tile.
        """
        try:
            try:
                downsampled = self._downsample_block(task.tiles)
                encoded = self._encoder.encode(downsampled)
            except Exception as exception:
                self._token.cancel(exception)
                return
            self._safe_put(
                task.output_queue,
                EncodingTaskResult(coordinates=task.coordinates, tiles=[encoded]),
            )
            if task.cascade_queue is not None:
                self._safe_put(
                    task.cascade_queue,
                    CascadedTile(
                        x_index=task.coordinates.x_index,
                        y_index=task.coordinates.y_index,
                        z_index=task.coordinates.z_index,
                        optical_path_index=task.coordinates.optical_path_index,
                        tile=downsampled,
                    ),
                )
        finally:
            task.cascade_tracker.decrement()

    def _downsample_block(
        self,
        tiles: Sequence[Sequence[Image.Image]],
    ) -> Image.Image:
        """Downsample a block of tiles into one output tile.

        Chooses between reduce-each-then-stitch (fast path when the downsampler
        supports it) and stitch-then-halve (standard path). Both halve the block
        by exactly 2x; the result is then padded to the full output tile size,
        which matters at the right and bottom edges of a level where the input
        block is smaller than 2x2.
        """
        if self._downsampler.commutes_with_stitch:
            reduced = [[tile.reduce(2) for tile in row] for row in tiles]
            block = self._stitcher.stitch_grid(reduced)
        else:
            composite = self._stitcher.stitch_grid(tiles)
            # Halve the composite, rather than resizing it to the output tile
            # size: a partial edge block is smaller than 2x2, and resizing it to
            # a full tile would stretch it instead of downsampling it.
            half = Size(composite.width // 2, composite.height // 2)
            block = self._downsampler.downsample(composite, half)
        return self._pad_to_tile(block)

    def _pad_to_tile(self, block: Image.Image) -> Image.Image:
        """Pad a downsampled block to the full output tile size.

        At the right and bottom edges of a level the input block can be smaller
        than 2x2 (the level below has an odd number of tiles), so the halved
        block covers only part of the output tile. The remainder lies outside
        the image and is filled with the background color, matching how source
        edge tiles are padded. Without this the written frame is smaller than
        the Rows/Columns declared in the dataset.

        Parameters
        ----------
        block: Image.Image
            The halved block, at most the output tile size.

        Returns
        -------
        Image.Image
            The block padded to the full output tile size.
        """
        if (block.width, block.height) == (
            self._tile_size.width,
            self._tile_size.height,
        ):
            return block
        # blank_tile is a cached, shared image - copy before pasting into it.
        canvas = self._blank_tile.copy()
        canvas.paste(block, (0, 0))
        return canvas

    def __enter__(self) -> "EncoderPool":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Context manager exit."""
        self.shutdown(wait=True)
        return False
