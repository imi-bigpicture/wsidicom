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

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from heapq import heappop, heappush
from itertools import count
from threading import Condition, Lock
from typing import Any, Generic, Protocol, TypeVar

ReturnType = TypeVar("ReturnType")
QueueItemType = TypeVar("QueueItemType", contravariant=True)
QueueItemTypeOut = TypeVar("QueueItemTypeOut", covariant=True)
ItemType = TypeVar("ItemType")


class WriteOnlyQueue(Protocol[QueueItemType]):
    """Queue capability restricted to producer-side operations."""

    def put(self, item: QueueItemType, token: "CancellationToken") -> None:
        """Put `item` on the queue, blocking until there is space or `token`
        is cancelled."""
        ...


class ReadOnlyQueue(Protocol[QueueItemTypeOut]):
    """Queue capability restricted to consumer-side operations."""

    def get(self, token: "CancellationToken") -> QueueItemTypeOut:
        """Get an item from the queue, blocking until one is available or
        `token` is cancelled."""
        ...


class ShutdownSentinel:
    """Sentinel that compares greater than any other object, thus processed last
    in priority queue."""

    def __lt__(self, other: object) -> bool:
        return False

    def __gt__(self, other: object) -> bool:
        return not isinstance(other, ShutdownSentinel)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ShutdownSentinel)

    def __le__(self, other: object) -> bool:
        return isinstance(other, ShutdownSentinel)

    def __ge__(self, other: object) -> bool:
        return True


class CompletionTracker:
    """Thread-safe counter with `wait_for_zero` blocking semantics.

    Increments and decrements may interleave freely across threads; a
    thread waiting in `wait_for_zero` resumes once the count reaches zero.
    """

    def __init__(self) -> None:
        self._pending = 0
        self._condition = Condition()

    def increment(self) -> None:
        """Increase the pending count by one."""
        with self._condition:
            self._pending += 1

    def decrement(self) -> None:
        """Decrease the pending count by one.

        Wakes any threads blocked in `wait_for_zero` if the count reaches zero.
        """
        with self._condition:
            self._pending -= 1
            if self._pending == 0:
                self._condition.notify_all()

    def wait_for_zero(self) -> None:
        """Block until the pending count reaches zero.

        Returns immediately if the count is already zero.
        """
        with self._condition:
            while self._pending > 0:
                self._condition.wait()


class Cancelled(Exception):
    """Raised by `CancelableQueue.get`/`put` once its token has been cancelled.

    Carries no payload of its own; the originating failure is held by the
    `CancellationToken`, so workers catching `Cancelled` should exit quietly
    without recording it as a new cause.
    """


class CancellationToken:
    """Set-once cancellation flag shared across a multi-stage pipeline.

    The first thread to fail calls `cancel`, recording its exception and waking
    every `CancelableQueue` blocked on this token (via the queue's condition).
    Later `cancel` calls are ignored, so the first failure is preserved as the
    root cause and can be re-raised once the pipeline has unwound.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._exception: BaseException | None = None
        self._conditions: set[Condition] = set()

    def cancel(self, exception: BaseException) -> None:
        """Record `exception` as the cause and wake every watching queue.

        Only the first call has an effect; later calls are ignored so the first
        failure wins.
        """
        with self._lock:
            if self._exception is not None:
                return
            self._exception = exception
            conditions = list(self._conditions)
        # Snapshot under the lock, then notify outside it. `watch` and `cancel`
        # each only ever hold one lock at a time, so no lock-ordering cycle is
        # possible with a queue waiter. Setting the exception before notifying
        # guarantees a woken waiter re-reads a cancelled token (no missed wake).
        for condition in conditions:
            with condition:
                condition.notify_all()

    def is_cancelled(self) -> bool:
        """Return whether the token has been cancelled."""
        return self._exception is not None

    @property
    def exception(self) -> BaseException | None:
        """The first recorded cause, or None if not cancelled."""
        return self._exception

    def raise_if_cancelled(self) -> None:
        """Re-raise the recorded cause if the token has been cancelled."""
        exception = self._exception
        if exception is not None:
            raise exception

    def watch(self, condition: Condition) -> None:
        """Register a queue's condition to be woken when this token cancels."""
        with self._lock:
            self._conditions.add(condition)


class CancelableQueue(Generic[ItemType], ABC):
    """Bounded queue whose `get`/`put` are woken by a `CancellationToken`
    without polling.

    A blocked `get`/`put` wakes either on the normal queue event (an item
    arriving / space freeing) or on `token.cancel(...)`, which `notify_all`s the
    queue's condition; the woken operation then raises `Cancelled`.

    Subclasses choose ordering by encoding each item into a sortable heap entry
    (`PriorityCancelableQueue`, `FifoCancelableQueue`).
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._heap: list[Any] = []
        self._counter = count()
        self._maxsize = maxsize
        self._condition = Condition()

    def qsize(self) -> int:
        """Return the number of queued items (a snapshot under concurrency)."""
        with self._condition:
            return len(self._heap)

    @abstractmethod
    def _entry(self, item: ItemType) -> Any:
        """Wrap `item` in a heap entry whose sort order defines dequeue order."""
        ...

    @abstractmethod
    def _item(self, entry: Any) -> ItemType:
        """Extract the original item from a heap entry."""
        ...

    def put(self, item: ItemType, token: CancellationToken) -> None:
        """Put `item`, blocking until there is space or `token` is cancelled.

        Raises
        ------
        Cancelled
            If `token` is cancelled before the item could be enqueued.
        """
        token.watch(self._condition)
        with self._condition:
            while self._maxsize and len(self._heap) >= self._maxsize:
                if token.is_cancelled():
                    raise Cancelled()
                self._condition.wait()
            heappush(self._heap, self._entry(item))
            self._condition.notify_all()

    def get(self, token: CancellationToken) -> ItemType:
        """Get an item, blocking until one is available or `token` is cancelled.

        Raises
        ------
        Cancelled
            If `token` is cancelled before an item could be retrieved.
        """
        token.watch(self._condition)
        with self._condition:
            while True:
                if token.is_cancelled():
                    raise Cancelled()
                if self._heap:
                    break
                self._condition.wait()
            entry = heappop(self._heap)
            self._condition.notify_all()
            return self._item(entry)


class PriorityCancelableQueue(CancelableQueue[ItemType]):
    """Cancelable queue that dequeues items in priority order.

    Items must be mutually comparable; a sequence counter breaks ties stably so
    equal-priority items keep their arrival order.
    """

    def _entry(self, item: ItemType) -> Any:
        return (item, next(self._counter))

    def _item(self, entry: Any) -> ItemType:
        return entry[0]


class FifoCancelableQueue(CancelableQueue[ItemType]):
    """Cancelable queue that dequeues items in arrival order.

    Orders solely by a sequence counter, so items need not be comparable.
    """

    def _entry(self, item: ItemType) -> Any:
        return (next(self._counter), item)

    def _item(self, entry: Any) -> ItemType:
        return entry[1]


class ConditionalThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that uses a single thread if max workers is 1."""

    def __init__(
        self, max_workers: int | None = None, force_iteration: bool = False, **kwargs
    ) -> None:
        self._force_iteration = force_iteration
        super().__init__(max_workers, **kwargs)

    def submit(
        self, fn: Callable[..., ReturnType], /, *args: Any, **kwargs: Any
    ) -> Future[ReturnType]:
        if self._max_workers == 1:
            future: Future[ReturnType] = Future()
            try:
                result = fn(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            return future
        return super().submit(fn, *args, **kwargs)

    def map(
        self,
        fn: Callable[..., ReturnType],
        *iterables: Iterable[Any],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterator[ReturnType]:
        if self._max_workers == 1:
            if self._force_iteration:
                # Make sure items are iterated through
                return iter(list(map(fn, *iterables)))
            return map(fn, *iterables)
        return super().map(fn, *iterables, timeout=timeout, chunksize=chunksize)
