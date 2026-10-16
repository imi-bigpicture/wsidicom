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

"""Tests of box downscale and rgb_to_rgbx against Pillow.

Box reduce(2) is bit for bit Pillow's (uint8 gray and RGB; uint16 through mode
"I"), reduce(4) within 1, both with Pillow's partial edge boxes where the
factor doesn't divide the size, and rgb_to_rgbx is Pillow's own RGBX packing.
"""

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import ResizeFilter, downscale, rgb_to_rgbx

Image = pytest.importorskip("PIL.Image")

PillowReduce = Callable[[np.ndarray, str, int], np.ndarray]
RandomImage = Callable[[tuple[int, ...], Any], np.ndarray]

SIZES = [(64, 64), (63, 65), (130, 34), (33, 129), (16, 16), (17, 19), (256, 254)]


@pytest.fixture(params=["simd", "scalar"])
def kernel_tier(
    request: pytest.FixtureRequest,
    scalar_kernels: Callable[[], contextlib.AbstractContextManager[None]],
) -> Iterator[str]:
    """Run the test with the SIMD kernels and with the scalar reference."""
    if request.param == "scalar":
        with scalar_kernels():
            yield request.param
    else:
        yield request.param


@pytest.fixture
def pillow_reduce() -> PillowReduce:
    """Reduce with Pillow, partial edge boxes averaged over what they cover."""

    def reduce(image: np.ndarray, mode: str, factor: int) -> np.ndarray:
        return np.asarray(Image.fromarray(image, mode).reduce(factor))

    return reduce


@pytest.fixture
def pillow_reduce_u16(
    pillow_reduce: PillowReduce,
) -> Callable[[np.ndarray], np.ndarray]:
    """Reduce uint16 by 2 per channel with Pillow, through mode "I" (int32).

    "I" holds uint16 exactly and rounds half up as for uint8 (Pillow cannot
    reduce "I;16").
    """

    def reduce(image: np.ndarray) -> np.ndarray:
        planes = image[..., None] if image.ndim == 2 else image
        reduced = [
            pillow_reduce(
                np.ascontiguousarray(planes[..., channel]).astype(np.int32), "I", 2
            )
            for channel in range(planes.shape[2])
        ]
        return reduced[0] if image.ndim == 2 else np.stack(reduced, axis=2)

    return reduce


@pytest.mark.unittest
@pytest.mark.usefixtures("kernel_tier")
@pytest.mark.parametrize(["height", "width"], SIZES)
class TestBoxEqualsPillowReduce:
    """Box downscale against Pillow's reduce, on both kernel tiers."""

    @pytest.mark.parametrize(["samples", "mode"], [(1, "L"), (3, "RGB")])
    def test_uint8_reduce_by_2_is_bit_exact(
        self,
        random_image: RandomImage,
        pillow_reduce: PillowReduce,
        height: int,
        width: int,
        samples: int,
        mode: str,
    ) -> None:
        """A uint8 box downscale by 2 equals Pillow's reduce(2)."""
        # Arrange
        shape = (height, width) if samples == 1 else (height, width, samples)
        image = random_image(shape, np.uint8)

        # Act
        result = downscale(image, 2, ResizeFilter.BOX)

        # Assert
        assert (result == pillow_reduce(image, mode, 2)).all()

    def test_uint8_reduce_by_4_is_within_one(
        self,
        random_image: RandomImage,
        pillow_reduce: PillowReduce,
        height: int,
        width: int,
    ) -> None:
        """A uint8 box downscale by 4 is within one of Pillow's reduce(4)."""
        # Arrange
        image = random_image((height, width), np.uint8)

        # Act
        result = downscale(image, 4, ResizeFilter.BOX)

        # Assert — the fused 4x rounds once, Pillow's reduce(4) too but from
        # other sums.
        difference = result.astype(int) - pillow_reduce(image, "L", 4)
        assert np.abs(difference).max() <= 1

    @pytest.mark.parametrize("samples", [1, 3])
    def test_uint16_reduce_by_2_is_bit_exact(
        self,
        random_image: RandomImage,
        pillow_reduce_u16: Callable[[np.ndarray], np.ndarray],
        height: int,
        width: int,
        samples: int,
    ) -> None:
        """A uint16 box downscale by 2 equals Pillow's reduce(2) in mode "I"."""
        # Arrange
        shape = (height, width) if samples == 1 else (height, width, samples)
        image = random_image(shape, np.uint16)

        # Act
        result = downscale(image, 2, ResizeFilter.BOX)

        # Assert
        assert (result.astype(np.int64) == pillow_reduce_u16(image)).all()


@pytest.mark.unittest
class TestRgbToRgbxEqualsPillow:
    """rgb_to_rgbx against Pillow's RGBX packing."""

    @pytest.mark.parametrize(["height", "width"], SIZES)
    def test_equals_pillow_packing(
        self, random_image: RandomImage, height: int, width: int
    ) -> None:
        """The color channels equal Pillow's "RGBX" raw packing."""
        # Arrange
        rgb = random_image((height, width, 3), np.uint8)
        pillow = np.frombuffer(
            Image.fromarray(rgb, "RGB").tobytes("raw", "RGBX"), np.uint8
        ).reshape(height, width, 4)

        # Act
        result = rgb_to_rgbx(rgb, 0)

        # Assert
        assert (result[:, :, :3] == pillow[:, :, :3]).all()
