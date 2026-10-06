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

import contextlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import Interpolation, ResizeFilter, compose, downscale

THREADS = 8
ROUNDS = 30
POSITION = np.array([[3.3, 2.6]])


@pytest.fixture
def tile(random_image: Callable[[tuple[int, ...], Any], np.ndarray]) -> np.ndarray:
    """Make an odd-sized random tile that resamples across vector body and tail."""
    return random_image((257, 263), np.uint8)


@pytest.mark.unittest
class TestThreads:
    def test_separate_canvases_equal_single_threaded(self, tile: np.ndarray) -> None:
        """Concurrent composes onto separate canvases equal a single-threaded one."""
        # Arrange
        reference = np.zeros((300, 300), np.uint8)
        compose(reference, [tile], POSITION, Interpolation.BICUBIC)

        def compose_rounds(_: int) -> list[np.ndarray]:
            canvases = []
            for _ in range(ROUNDS):
                canvas = np.zeros((300, 300), np.uint8)
                compose(canvas, [tile], POSITION, Interpolation.BICUBIC)
                canvases.append(canvas)
            return canvases

        # Act
        with ThreadPoolExecutor(THREADS) as pool:
            per_thread = list(pool.map(compose_rounds, range(THREADS)))

        # Assert
        for canvases in per_thread:
            assert all((canvas == reference).all() for canvas in canvases)

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

    def test_shared_canvas_raises_value_error_at_worst(self, tile: np.ndarray) -> None:
        """Concurrent composes onto one canvas each succeed or raise ValueError.

        A clean ValueError (the canvas is borrowed): never another error.
        """
        # Arrange
        canvas = np.zeros((300, 300), np.uint8)

        def count_successful_rounds(_: int) -> int:
            succeeded = 0
            for _ in range(ROUNDS):
                with contextlib.suppress(ValueError):
                    compose(canvas, [tile], POSITION, Interpolation.BILINEAR)
                    succeeded += 1
            return succeeded

        # Act
        with ThreadPoolExecutor(THREADS) as pool:
            succeeded = sum(pool.map(count_successful_rounds, range(THREADS)))

        # Assert
        assert succeeded > 0
