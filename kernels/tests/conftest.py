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

"""Fixtures shared by the wsidicom_kernels tests."""

import contextlib
import math
import zlib
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import ResizeFilter
from wsidicom_kernels._kernels import set_simd


@pytest.fixture(autouse=True)
def simd_restored() -> Iterator[None]:
    """Every test starts and ends with the SIMD kernels enabled."""
    set_simd(True)
    yield
    set_simd(True)


@pytest.fixture
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """Make a generator seeded from the test's id.

    Reproducible, and different per test and parameter set.
    """
    return np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))


@pytest.fixture
def random_image(
    rng: np.random.Generator,
) -> Callable[[tuple[int, ...], Any], np.ndarray]:
    """Return a factory of random images.

    Uniform over an integer dtype's range, or [0, 1) for float32.
    """

    def create(shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        if np.dtype(dtype) == np.float32:
            return rng.random(shape, np.float32)
        return rng.integers(0, np.iinfo(dtype).max, shape, endpoint=True).astype(dtype)

    return create


@pytest.fixture
def scalar_kernels() -> Callable[[], contextlib.AbstractContextManager[None]]:
    """Return a context manager running the forced-scalar reference kernels.

    The kernels are scalar within its block, and SIMD again after it.
    """

    @contextlib.contextmanager
    def scalar() -> Iterator[None]:
        set_simd(False)
        try:
            yield
        finally:
            set_simd(True)

    return scalar


@pytest.fixture
def simd_and_scalar(
    scalar_kernels: Callable[[], contextlib.AbstractContextManager[None]],
) -> Callable[[Callable[[], Any]], tuple[Any, Any]]:
    """Run a call with the SIMD kernels and with the scalar reference; return both."""

    def run(call: Callable[[], Any]) -> tuple[Any, Any]:
        simd = call()
        with scalar_kernels():
            scalar = call()
        return simd, scalar

    return run


@pytest.fixture
def exact_resize_float() -> Callable[[np.ndarray, int, int, ResizeFilter], np.ndarray]:
    """Return the exact convolution every resize is held to.

    The float64 separable convolution with fast_image_resize's window rules
    (kernel scaled when minifying, window clipped at the edge, renormalized).
    """

    def kernel(x: np.ndarray, filter: ResizeFilter) -> np.ndarray:
        a, ax = -0.5, np.abs(x)
        if filter == "box":
            return ((x > -0.5) & (x <= 0.5)).astype(float)
        if filter == "bilinear":
            return np.maximum(0.0, 1.0 - ax)
        if filter == "catmullrom":
            return np.where(
                ax < 1,
                ((a + 2) * ax - (a + 3)) * ax * ax + 1,
                np.where(ax < 2, (((ax - 5) * ax + 8) * ax - 4) * a, 0.0),
            )
        return np.where(ax < 3, np.sinc(x) * np.sinc(x / 3), 0.0)

    support = {"box": 0.5, "bilinear": 1.0, "catmullrom": 2.0, "lanczos3": 3.0}

    def matrix(n_in: int, n_out: int, filter: ResizeFilter) -> np.ndarray:
        scale = n_in / n_out
        ks = max(scale, 1.0)
        m = np.zeros((n_out, n_in))
        for o in range(n_out):
            center = (o + 0.5) * scale
            lo = max(0, math.floor(center - support[filter] * ks))
            hi = min(n_in, math.ceil(center + support[filter] * ks))
            weights = kernel((np.arange(lo, hi) - (center - 0.5)) / ks, filter)
            m[o, lo:hi] = weights / weights.sum()
        return m

    def resize(image: np.ndarray, oh: int, ow: int, filter: ResizeFilter) -> np.ndarray:
        planes = (
            [image]
            if image.ndim == 2
            else [image[..., c] for c in range(image.shape[2])]
        )
        rows = matrix(image.shape[0], oh, filter)
        cols = matrix(image.shape[1], ow, filter).T
        out = [rows @ p.astype(float) @ cols for p in planes]
        return out[0] if image.ndim == 2 else np.stack(out, axis=-1)

    return resize


@pytest.fixture
def exact_resize_u8(
    exact_resize_float: Callable[[np.ndarray, int, int, ResizeFilter], np.ndarray],
) -> Callable[[np.ndarray, int, int, ResizeFilter], np.ndarray]:
    """`exact_resize_float` rounded half up and clamped to the uint8 range."""

    def resize(image: np.ndarray, oh: int, ow: int, filter: ResizeFilter) -> np.ndarray:
        exact = exact_resize_float(image, oh, ow, filter)
        return np.clip(np.floor(exact + 0.5), 0, 255)

    return resize
