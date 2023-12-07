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

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, TypeVar

ReturnType = TypeVar("ReturnType")


class ConditionalThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that uses a single thread if max workers is 1."""

    def __init__(
        self, max_workers: Optional[int] = None, force_iteration: bool = False, **kwargs
    ) -> None:
        self._force_iteration = force_iteration
        super().__init__(max_workers, **kwargs)

    def map(
        self,
        fn: Callable[..., ReturnType],
        *iterables: Iterable[Any],
        timeout: Optional[float] = None,
        chunksize: int = 1
    ) -> Iterator[ReturnType]:
        if self._max_workers == 1:
            if self._force_iteration:
                # Make sure items are iterated through
                return iter(list(map(fn, *iterables)))
            return map(fn, *iterables)
        return super().map(fn, *iterables, timeout=timeout, chunksize=chunksize)
