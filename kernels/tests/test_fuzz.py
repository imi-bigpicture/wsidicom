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

"""Property tests over generated shapes, positions, factors and filters.

Every input (and the compose canvas) ends flush with the end of an mmap'd
region, so a kernel that reads or writes one element past its buffer hits
unmapped memory and crashes, instead of quietly touching heap slack that no
value-based test sees.

Each case checks that nothing crashes and that the SIMD kernels match the
forced-scalar reference.

Hypothesis runs many examples inside one pytest test, so a function-scoped
fixture is not reset between them and Hypothesis refuses one by default. The
tests here only take the conftest ``simd_and_scalar`` factory, which holds no
state between calls (it restores the SIMD kernels after each), so sharing it
across examples is safe and that health check is suppressed. Generated images
come from the plain ``page_end`` helper, drawn per example from a seed.
"""

import mmap
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from wsidicom_kernels import (
    Interpolation,
    ResizeFilter,
    compose,
    downscale,
    resize,
    rgb_to_rgbx,
)

SimdAndScalar = Callable[[Callable[[], Any]], tuple[Any, Any]]
Size = tuple[int, int]

# The real page size (16 KiB on Apple silicon): a smaller guess would leave
# readable slack past the array, hiding overreads.
PAGE = mmap.PAGESIZE
DTYPES = [np.uint8, np.uint16, np.float32]

fuzz = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
dtypes = st.sampled_from(DTYPES)
channel_counts = st.sampled_from([1, 3, 4])
seeds = st.integers(0, 2**32 - 1)


def page_end(shape: tuple[int, ...], dtype: Any, seed: int) -> np.ndarray:
    """Make a random array whose last byte is the last byte of an mmap."""
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    size = max(1, -(-nbytes // PAGE)) * PAGE
    array = np.frombuffer(mmap.mmap(-1, size), np.uint8, nbytes, size - nbytes)
    array = array.view(dtype).reshape(shape)
    rng = np.random.default_rng(seed)
    if dtype.kind == "u":
        array[...] = rng.integers(0, np.iinfo(dtype).max, shape, endpoint=True)
    else:
        array[...] = rng.random(shape)
    return array


def image_shape(size: Size, channels: int) -> tuple[int, ...]:
    """Return the array shape of a size with a channel count (no axis for one)."""
    return size if channels == 1 else (*size, channels)


def assert_parity(
    simd: np.ndarray, scalar: np.ndarray, integer_tolerance: int = 0
) -> None:
    """float32 results agree to rounding, integer ones within the tolerance."""
    if simd.dtype == np.float32:
        assert np.allclose(simd, scalar, atol=1e-4, rtol=1e-5)
    else:
        difference = np.abs(simd.astype(np.int64) - scalar.astype(np.int64))
        assert difference.max(initial=0) <= integer_tolerance


@st.composite
def positions(draw: st.DrawFn) -> float:
    """Draw a position component: integer, fractional, half, far or negative."""
    whole = draw(st.integers(-6, 12))
    fraction = draw(
        st.one_of(st.just(0.0), st.just(0.5), st.floats(0, 1, exclude_max=True))
    )
    return float(whole) + fraction


@st.composite
def resizes(draw: st.DrawFn) -> tuple[Size, Size]:
    """Draw an input size (h, w) and an output size (oh, ow).

    An exact integer reduction (the phased plans), an exact 2x or 4x
    enlargement, or any size up to twice the input.
    """
    h, w = draw(st.integers(1, 300)), draw(st.integers(1, 300))
    kind = draw(st.sampled_from(["reduce", "enlarge", "any"]))
    if kind == "reduce":
        k = draw(st.integers(2, 4))
        h, w = max(k, h - h % k), max(k, w - w % k)
        return (h, w), (h // k, w // k)
    if kind == "enlarge":
        k = draw(st.sampled_from([2, 4]))
        return (h, w), (h * k, w * k)
    return (h, w), (draw(st.integers(1, 2 * h)), draw(st.integers(1, 2 * w)))


@pytest.mark.unittest
class TestCompose:
    """compose over generated tiles, canvases, positions and filters."""

    @fuzz
    @given(
        dtype=dtypes,
        layout=st.sampled_from([(1, 1), (3, 3), (4, 4), (3, 4)]),
        tile_size=st.tuples(st.integers(1, 160), st.integers(1, 160)),
        canvas_size=st.tuples(st.integers(1, 168), st.integers(1, 168)),
        position=st.tuples(positions(), positions()),
        interpolation=st.sampled_from(list(Interpolation)),
        seed=seeds,
    )
    def test_simd_matches_scalar(
        self,
        simd_and_scalar: SimdAndScalar,
        dtype: Any,
        layout: tuple[int, int],
        tile_size: Size,
        canvas_size: Size,
        position: tuple[float, float],
        interpolation: Interpolation,
        seed: int,
    ) -> None:
        """Composing stays in bounds and matches the scalar reference.

        The SIMD kernels match the scalar reference within one level.
        """
        # Arrange
        tile_channels, canvas_channels = layout
        if canvas_channels != tile_channels:  # RGB onto RGBX: uint8, resampled
            dtype = np.uint8
            if interpolation == Interpolation.NEAREST:
                interpolation = Interpolation.BILINEAR
        tile = page_end(image_shape(tile_size, tile_channels), dtype, seed)

        def composed() -> np.ndarray:
            canvas_shape = image_shape(canvas_size, canvas_channels)
            canvas = page_end(canvas_shape, dtype, seed + 1)
            compose(canvas, [tile], np.array([position], np.float64), interpolation)
            return canvas

        # Act
        simd, scalar = simd_and_scalar(composed)

        # Assert
        assert_parity(simd, scalar, integer_tolerance=1)


@pytest.mark.unittest
class TestBoxDownscale:
    """Box downscale over generated sizes, channels and factors."""

    @fuzz
    @given(
        dtype=dtypes,
        channels=channel_counts,
        factor=st.sampled_from([1, 2, 4, 8, 16]),
        data=st.data(),
        seed=seeds,
    )
    def test_simd_matches_scalar(
        self,
        simd_and_scalar: SimdAndScalar,
        dtype: Any,
        channels: int,
        factor: int,
        data: st.DataObject,
        seed: int,
    ) -> None:
        """Downscaling stays in bounds and matches the scalar reference.

        The SIMD kernels match the scalar reference within one level per halving.
        """
        # Arrange
        height = data.draw(st.integers(factor, 300))
        width = data.draw(st.integers(factor, 300))
        image = page_end(image_shape((height, width), channels), dtype, seed)

        # Act
        simd, scalar = simd_and_scalar(
            lambda: downscale(image, factor, ResizeFilter.BOX)
        )
        if dtype == np.uint8 and channels == 3:
            downscale(
                image, factor, ResizeFilter.BOX, pad=255
            )  # the padded path's bounds

        # Assert — fused and chained passes may round differently: one step per
        # level.
        levels = factor.bit_length() - 1
        assert_parity(simd, scalar, integer_tolerance=levels + 1)


@pytest.mark.unittest
class TestRgbToRgbx:
    """rgb_to_rgbx over generated sizes."""

    @fuzz
    @given(height=st.integers(1, 40), width=st.integers(1, 40), seed=seeds)
    def test_simd_matches_scalar_and_packs(
        self, simd_and_scalar: SimdAndScalar, height: int, width: int, seed: int
    ) -> None:
        """Packing stays in bounds and the SIMD kernels match the scalar reference.

        They match exactly, and the color channels are kept with the pad value
        after.
        """
        # Arrange
        image = page_end((height, width, 3), np.uint8, seed)

        # Act
        simd, scalar = simd_and_scalar(lambda: rgb_to_rgbx(image, 7))

        # Assert
        assert (simd == scalar).all()
        assert (simd[:, :, :3] == image).all()
        assert (simd[:, :, 3] == 7).all()


@pytest.mark.unittest
class TestResize:
    """resize over generated sizes, channels and filters."""

    @fuzz
    @given(
        dtype=dtypes,
        channels=channel_counts,
        sizes=resizes(),
        filter=st.sampled_from(
            [
                ResizeFilter.BOX,
                ResizeFilter.BILINEAR,
                ResizeFilter.CATMULLROM,
                ResizeFilter.LANCZOS3,
            ]
        ),
        seed=seeds,
    )
    def test_simd_matches_scalar(
        self,
        simd_and_scalar: SimdAndScalar,
        dtype: Any,
        channels: int,
        sizes: tuple[Size, Size],
        filter: ResizeFilter,
        seed: int,
    ) -> None:
        """Resizing stays in bounds and the SIMD kernels match the scalar reference."""
        # Arrange
        (height, width), (output_height, output_width) = sizes
        image = page_end(image_shape((height, width), channels), dtype, seed)

        # Act
        simd, scalar = simd_and_scalar(
            lambda: resize(image, output_height, output_width, filter)
        )

        # Assert — identical on every tier, except the power-of-2 box kernels'
        # fused versus chained rounding.
        is_power_of_2_box = (
            filter == ResizeFilter.BOX
            and height % output_height == 0
            and width % output_width == 0
            and height // output_height == width // output_width
        )
        assert_parity(simd, scalar, integer_tolerance=1 if is_power_of_2_box else 0)
