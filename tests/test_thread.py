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

"""Tests for wsidicom.thread."""

import contextvars
import time
from concurrent.futures import Executor, Future
from threading import Thread, get_ident

import pytest
from decoy import Decoy, matchers

from wsidicom.thread import (
    CancellationToken,
    Cancelled,
    CompletionTracker,
    FifoCancelableQueue,
    PriorityCancelableQueue,
    ReadExecutor,
    ShutdownSentinel,
)


@pytest.mark.unittest
class TestCompletionTracker:
    def test_increment_and_decrement(self):
        """Test basic increment/decrement cycle."""
        tracker = CompletionTracker()
        tracker.increment()
        tracker.increment()
        tracker.decrement()
        tracker.decrement()
        tracker.wait_for_zero()  # Should not block

    def test_wait_for_zero_when_already_zero(self):
        """Test that wait_for_zero returns immediately when count is zero."""
        tracker = CompletionTracker()
        tracker.wait_for_zero()  # Should not block


@pytest.mark.unittest
class TestCancellationToken:
    def test_not_cancelled_initially(self):
        """A fresh token is not cancelled and holds no exception."""
        # Arrange
        token = CancellationToken()

        # Act & Assert
        assert token.is_cancelled() is False
        assert token.exception is None

    def test_cancel_records_exception_and_sets(self):
        """Cancelling records the cause and sets the flag."""
        # Arrange
        token = CancellationToken()
        error = ValueError("boom")

        # Act
        token.cancel(error)

        # Assert
        assert token.is_cancelled() is True
        assert token.exception is error

    def test_first_cancel_wins(self):
        """The first recorded cause is preserved across later cancels."""
        # Arrange
        token = CancellationToken()
        first = ValueError("first")
        second = RuntimeError("second")

        # Act
        token.cancel(first)
        token.cancel(second)

        # Assert
        assert token.exception is first

    def test_raise_if_cancelled_reraises_cause(self):
        """raise_if_cancelled re-raises the recorded exception."""
        # Arrange
        token = CancellationToken()
        token.cancel(ValueError("boom"))

        # Act & Assert
        with pytest.raises(ValueError, match="boom"):
            token.raise_if_cancelled()

    def test_raise_if_cancelled_noop_when_not_cancelled(self):
        """raise_if_cancelled does nothing when the token is live."""
        # Arrange
        token = CancellationToken()

        # Act & Assert
        token.raise_if_cancelled()  # should not raise


@pytest.mark.unittest
class TestCancelableQueue:
    def test_put_get_roundtrip(self):
        """An item put on the queue is returned by get."""
        # Arrange
        queue: PriorityCancelableQueue = PriorityCancelableQueue()
        token = CancellationToken()

        # Act
        queue.put("item", token)

        # Assert
        assert queue.get(token) == "item"

    def test_priority_ordering(self):
        """A prioritized queue returns items in priority order."""
        # Arrange
        queue: PriorityCancelableQueue = PriorityCancelableQueue()
        token = CancellationToken()

        # Act
        for value in (3, 1, 2):
            queue.put(value, token)

        # Assert
        assert [queue.get(token) for _ in range(3)] == [1, 2, 3]

    def test_fifo_ordering_allows_incomparable_items(self):
        """A FIFO queue preserves arrival order and needs no item ordering."""
        # Arrange
        queue: FifoCancelableQueue = FifoCancelableQueue()
        token = CancellationToken()
        items = [object(), object(), object()]

        # Act
        for item in items:
            queue.put(item, token)

        # Assert
        assert [queue.get(token) for _ in range(3)] == items

    def test_get_raises_if_already_cancelled(self):
        """An already-cancelled token takes precedence over an available item."""
        # Arrange
        queue: PriorityCancelableQueue = PriorityCancelableQueue()
        token = CancellationToken()
        queue.put("item", token)
        token.cancel(ValueError("x"))

        # Act & Assert
        with pytest.raises(Cancelled):
            queue.get(token)

    def test_put_raises_if_already_cancelled(self):
        """An already-cancelled token prevents enqueueing into a full queue."""
        # Arrange
        queue: PriorityCancelableQueue = PriorityCancelableQueue(maxsize=1)
        token = CancellationToken()
        queue.put("fill", token)
        token.cancel(ValueError("x"))

        # Act & Assert
        with pytest.raises(Cancelled):
            queue.put("blocked", token)

    def test_get_unblocks_on_cancel_when_empty(self):
        """A get blocked on an empty queue raises Cancelled once cancelled."""
        # Arrange
        queue: PriorityCancelableQueue = PriorityCancelableQueue()
        token = CancellationToken()
        outcome: dict = {}

        def worker() -> None:
            try:
                queue.get(token)
            except Cancelled:
                outcome["cancelled"] = True

        thread = Thread(target=worker)

        # Act
        thread.start()
        time.sleep(0.05)  # let the worker reach the empty-queue block
        token.cancel(ValueError("stop"))
        thread.join(timeout=2.0)

        # Assert
        assert thread.is_alive() is False
        assert outcome.get("cancelled") is True

    def test_put_unblocks_on_cancel_when_full(self):
        """A put blocked on a full queue raises Cancelled once cancelled."""
        # Arrange
        queue: PriorityCancelableQueue = PriorityCancelableQueue(maxsize=1)
        token = CancellationToken()
        queue.put("fill", token)
        outcome: dict = {}

        def worker() -> None:
            try:
                queue.put("blocked", token)
            except Cancelled:
                outcome["cancelled"] = True

        thread = Thread(target=worker)

        # Act
        thread.start()
        time.sleep(0.05)  # let the worker reach the full-queue block
        token.cancel(ValueError("stop"))
        thread.join(timeout=2.0)

        # Assert
        assert thread.is_alive() is False
        assert outcome.get("cancelled") is True

    def test_cancel_wakes_all_blocked_getters(self):
        """One cancel releases every thread blocked on an empty-queue get."""
        # Arrange
        queue: PriorityCancelableQueue = PriorityCancelableQueue()
        token = CancellationToken()
        cancelled: list = []

        def worker() -> None:
            try:
                queue.get(token)
            except Cancelled:
                cancelled.append(True)  # list.append is atomic under the GIL

        threads = [Thread(target=worker) for _ in range(5)]

        # Act
        for thread in threads:
            thread.start()
        time.sleep(0.05)  # let all workers block on the empty queue
        token.cancel(ValueError("stop"))
        for thread in threads:
            thread.join(timeout=2.0)

        # Assert — every waiter woke and raised, none left blocked
        assert all(not thread.is_alive() for thread in threads)
        assert len(cancelled) == 5

    def test_cancel_wakes_all_blocked_putters(self):
        """One cancel releases every thread blocked on a full-queue put."""
        # Arrange
        queue: PriorityCancelableQueue = PriorityCancelableQueue(maxsize=1)
        token = CancellationToken()
        queue.put("fill", token)
        cancelled: list = []

        def worker() -> None:
            try:
                queue.put("blocked", token)
            except Cancelled:
                cancelled.append(True)

        threads = [Thread(target=worker) for _ in range(5)]

        # Act
        for thread in threads:
            thread.start()
        time.sleep(0.05)  # let all workers block on the full queue
        token.cancel(ValueError("stop"))
        for thread in threads:
            thread.join(timeout=2.0)

        # Assert
        assert all(not thread.is_alive() for thread in threads)
        assert len(cancelled) == 5

    def test_priority_orders_shutdown_sentinel_last(self):
        """A ShutdownSentinel dequeues after real items, regardless of arrival."""
        # Arrange
        queue: PriorityCancelableQueue = PriorityCancelableQueue()
        token = CancellationToken()

        # Act — enqueue the sentinel first and the items out of order
        queue.put(ShutdownSentinel(), token)
        for value in (3, 1, 2):
            queue.put(value, token)
        dequeued = [queue.get(token) for _ in range(4)]

        # Assert — items come out in priority order, sentinel strictly last
        assert dequeued[:3] == [1, 2, 3]
        assert isinstance(dequeued[3], ShutdownSentinel)


@pytest.mark.unittest
class TestReadExecutor:
    def test_workers_explicit_value(self):
        """An explicit thread count is used as the worker budget."""
        # Arrange
        expected_workers = 4
        read_executor = ReadExecutor(expected_workers, None)

        # Act
        workers = read_executor.workers

        # Assert
        assert workers == expected_workers

    def test_workers_none_without_shared_is_one(self):
        """No count and no shared executor reads single-threaded."""
        # Arrange
        expected_workers = 1
        read_executor = ReadExecutor(None, None)

        # Act
        workers = read_executor.workers

        # Assert
        assert workers == expected_workers

    def test_workers_none_with_shared_is_cpu_count(
        self, decoy: Decoy, monkeypatch: pytest.MonkeyPatch
    ):
        """No count with a shared executor fans out across cpu_count."""
        # Arrange
        expected_workers = 7
        monkeypatch.setattr("wsidicom.thread.os.cpu_count", lambda: expected_workers)
        shared = decoy.mock(cls=Executor)
        read_executor = ReadExecutor(None, shared)

        # Act
        workers = read_executor.workers

        # Assert
        assert workers == expected_workers

    def test_single_worker_runs_inline_on_calling_thread(self):
        """workers == 1 runs the callable on the calling thread, not a worker."""
        # Arrange
        this_thread_id = get_ident()
        read_executor = ReadExecutor(1, None)

        # Act — the callable reports the thread it ran on
        future = read_executor.submit(get_ident)

        # Assert — that thread is this (calling) thread, i.e. it ran inline
        assert future.result() == this_thread_id

    def test_inline_exception_is_captured_in_future(self):
        """An inline callable's exception is delivered through the future."""

        # Arrange
        def boom():
            raise ValueError("boom")

        read_executor = ReadExecutor(1, None)

        # Act
        future = read_executor.submit(boom)

        # Assert
        with pytest.raises(ValueError, match="boom"):
            future.result()

    def test_delegates_to_shared_executor(self, decoy: Decoy):
        """With workers > 1 and a shared executor, submit forwards the call and
        returns the shared executor's future."""
        # Arrange
        shared = decoy.mock(cls=Executor)
        read_executor = ReadExecutor(4, shared)
        expected_future: Future = Future()

        def work(value: int) -> int:
            return value

        argument = 7
        # submit wraps the call in the caller's context (matchers.Anything() is
        # that copy_context().run wrapper) for context-local settings propagation.
        decoy.when(
            shared.submit(matchers.Anything(), work, argument)
        ).then_return(expected_future)

        # Act
        result = read_executor.submit(work, argument)

        # Assert — the shared executor's future is returned unchanged
        assert result is expected_future

    def test_propagates_context_to_workers(self):
        """A context-local value set by the caller is visible in pool workers.

        Guards the copy_context wrap in submit: without it, worker threads start
        with an empty context and context-local settings silently fall back to
        the process default.
        """
        # Arrange
        variable: contextvars.ContextVar[str] = contextvars.ContextVar(
            "test", default="default"
        )
        variable.set("scoped")

        # Act — read the variable inside real pool worker threads
        with ReadExecutor(4) as read_executor:
            results = list(read_executor.map(lambda _: variable.get(), range(8)))

        # Assert
        assert set(results) == {"scoped"}

    def test_does_not_shut_down_shared_executor(self, decoy: Decoy):
        """Closing a ReadExecutor never shuts down a borrowed shared executor."""
        # Arrange
        shared = decoy.mock(cls=Executor)

        # Act
        with ReadExecutor(4, shared) as read_executor:
            read_executor.submit(lambda: None)

        # Assert
        decoy.verify(shared.shutdown(), times=0)

    def test_owns_and_shuts_down_per_read_pool(self):
        """With workers > 1 and no shared executor, the owned pool is created and
        shut down on close."""
        # Arrange
        read_executor = ReadExecutor(2, None)
        expected = object()

        # Act — use the executor (creating its pool), then close it
        with read_executor:
            result = read_executor.submit(lambda: expected).result()
            own_pool = read_executor._own

        # Assert — the pool ran the work, then rejects new work after close
        assert result is expected
        assert own_pool is not None
        with pytest.raises(RuntimeError):
            own_pool.submit(lambda: None)
