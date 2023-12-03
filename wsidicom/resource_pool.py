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

"""Generic pool for accessing a resource."""

from abc import abstractmethod
from contextlib import contextmanager
import os
from queue import SimpleQueue, Empty
import threading
from typing import (
    Generator,
    Generic,
    Optional,
    TypeVar,
)

ResourceType = TypeVar("ResourceType")


class ResourcePool(Generic[ResourceType]):
    def __init__(
        self,
        max_pool_size: Optional[int] = None,
    ):
        if max_pool_size is None:
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                max_pool_size = cpu_count
            else:
                max_pool_size = 1
        self._max_pool_size = max_pool_size
        self._current_count = 0
        self._queue: SimpleQueue[ResourceType] = SimpleQueue()
        self._lock = threading.Lock()

    @contextmanager
    def get_resource(self) -> Generator[ResourceType, None, None]:
        """Return a resource. Should be used as a context manager. Will block
        if no resource is available."""
        resource = self._get_resource()
        try:
            yield resource
        finally:
            self._return_resource(resource)

    def close(self) -> None:
        """Close all resources."""
        with self._lock:
            while not self._is_empty:
                resource = self._queue.get()
                self._close_resource(resource)
                self._decrement()

    @abstractmethod
    def _create_new_resource(self) -> ResourceType:
        """Return a new resource."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def _close_resource(resource: ResourceType) -> None:
        """Close a resource."""
        raise NotImplementedError()

    @property
    def _is_full(self) -> bool:
        return self._current_count == self._max_pool_size

    @property
    def _is_empty(self) -> bool:
        return self._current_count == 0

    def _increment(self):
        if self._current_count < self._max_pool_size:
            self._current_count += 1
            return
        raise ValueError()

    def _decrement(self):
        if self._current_count > 0:
            self._current_count -= 1
            return
        raise ValueError()

    def _get_resource(self) -> ResourceType:
        """Return a resource with no wait if one is available. Else if current
        resource count is less than maximum resource count return a new resource.
        Otherwise wait for an available resourcen."""
        try:
            return self._queue.get_nowait()
        except Empty:
            pass
        with self._lock:
            if not self._is_full:
                self._increment()
                return self._create_new_resource()
        return self._queue.get(block=True, timeout=None)

    def _return_resource(self, resource: ResourceType) -> None:
        """Release a used resource."""
        self._queue.put(resource)
