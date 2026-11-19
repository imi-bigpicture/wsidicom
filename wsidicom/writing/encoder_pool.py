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
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from types import TracebackType
from typing import Any

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
    RecodeTask,
)


class EncoderPool:
    """Pool of encoder workers using ThreadPoolExecutor."""

    def __init__(
        self,
        num_workers: int,
        input_queue_maxsize: int = 100,
        *,
        token: CancellationToken,
    ) -> None:
        """Create an encoder pool.

        Parameters
        ----------
        num_workers: int
            Number of encoder worker threads.
        input_queue_maxsize: int
            Maximum size of input queue. Provides backpressure when full.
        token: CancellationToken
            Shared cancellation token for fail-fast teardown. A worker that
            fails cancels it, and all blocking queue operations observe it so
            the pipeline unwinds instead of deadlocking.
        """
        self._token = token
        self._input_queue: CancelableQueue[
            EncodeTask | RecodeTask | DownsampleEncodeTask | ShutdownSentinel
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
    def queue(self) -> WriteOnlyQueue[EncodeTask | RecodeTask | DownsampleEncodeTask]:
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

    def _dispatch_task(
        self, task: EncodeTask | RecodeTask | DownsampleEncodeTask
    ) -> None:
        """Dispatch a task to the appropriate handler."""
        if isinstance(task, DownsampleEncodeTask):
            self._downsample_and_encode(task)
        elif isinstance(task, RecodeTask):
            self._recode(task)
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

        Any failure, including one raised while handing the result off, cancels
        the token (the single failure channel) and the pipeline unwinds without a
        result. A put that raised outside it would be swallowed by the worker's
        future and silently drop the tiles.
        """
        try:
            self._safe_put(
                task.output_queue,
                EncodingTaskResult(
                    coordinates=task.coordinates,
                    tiles=[task.encoder.encode(tile) for tile in task.tiles],
                ),
            )
        except Exception as exception:
            self._token.cancel(exception)

    def _recode(self, task: RecodeTask) -> None:
        """Recode encoded tiles with the task's recoder and submit them.

        Failures cancel the token, as for `_encode`.
        """
        try:
            self._safe_put(
                task.output_queue,
                EncodingTaskResult(
                    coordinates=task.coordinates,
                    tiles=[task.recoder.recode(tile) for tile in task.tiles],
                ),
            )
        except Exception as exception:
            self._token.cancel(exception)

    def _downsample_and_encode(self, task: DownsampleEncodeTask) -> None:
        """Stitch and downsample a block, encode it, and cascade it.

        A level with no output queue produces no output: it is downsampled and
        cascaded to feed the level above, but never encoded.

        Any failure, including one raised while handing a result off, cancels the
        token: that is the single failure channel, and a put that raised outside
        it would be swallowed by the worker's future and silently drop a tile. The
        cascade push always precedes the tracker decrement so a level's shutdown
        (which waits for the tracker to reach zero) cannot forward its sentinel
        ahead of a cascaded tile.
        """
        try:
            downsampled = task.block_downsampler.downsample(task.tiles)
            if task.output_queue is not None:
                encoded = [task.encoder.encode(downsampled)]
                self._safe_put(
                    task.output_queue,
                    EncodingTaskResult(
                        coordinates=task.coordinates,
                        tiles=encoded,
                    ),
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
        except Exception as exception:
            self._token.cancel(exception)
        finally:
            task.cascade_tracker.decrement()

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
