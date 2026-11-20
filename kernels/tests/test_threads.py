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

"""Tests of the GIL-released paths under threads."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import ResizeFilter, downscale, resize

THREADS = 8
ROUNDS = 30


@pytest.mark.unittest
class TestThreads:
    def test_shared_input_downscales_equal(
        self, random_image: Callable[[tuple[int, ...], Any], np.ndarray]
    ) -> None:
        """Concurrent downscales of one shared image equal a single-threaded one."""
        # Arrange
        image = random_image((512, 512), np.uint8)
        single = downscale(image, 4, ResizeFilter.BOX)

        # Act
        with ThreadPoolExecutor(THREADS) as pool:
            per_thread = list(
                pool.map(
                    lambda _: [
                        downscale(image, 4, ResizeFilter.BOX) for _ in range(ROUNDS)
                    ],
                    range(THREADS),
                )
            )

        # Assert
        for downscaled in per_thread:
            assert all((result == single).all() for result in downscaled)

    def test_concurrent_resizes_of_different_sizes_equal_single_threaded(
        self, random_image: Callable[[tuple[int, ...], Any], np.ndarray]
    ) -> None:
        """Each thread's resizes equal single-threaded ones, whatever the others run.

        Every thread cycles through different output sizes, so the threads'
        plan caches hold different plans at the same time.
        """
        # Arrange
        image = random_image((257, 263, 3), np.uint8)
        sizes = [(128, 131), (85, 87), (200, 150), (64, 66)]
        single = {size: resize(image, *size, ResizeFilter.BILINEAR) for size in sizes}

        def resize_rounds(offset: int) -> list[tuple[tuple[int, int], np.ndarray]]:
            results = []
            for round_index in range(ROUNDS):
                size = sizes[(offset + round_index) % len(sizes)]
                results.append((size, resize(image, *size, ResizeFilter.BILINEAR)))
            return results

        # Act
        with ThreadPoolExecutor(THREADS) as pool:
            per_thread = list(pool.map(resize_rounds, range(THREADS)))

        # Assert
        for results in per_thread:
            assert all((result == single[size]).all() for size, result in results)
