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

"""Tests for rgb_to_rgbx: expanding uint8 RGB to RGBX with a pad sample."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from wsidicom_kernels import rgb_to_rgbx

RandomImage = Callable[[tuple[int, ...], Any], np.ndarray]


@pytest.fixture
def odd_rgb_image(random_image: RandomImage) -> np.ndarray:
    """Make a uint8 RGB image of odd size, so the SIMD body and scalar tail run."""
    return random_image((17, 19, 3), np.uint8)


@pytest.mark.unittest
class TestRgbToRgbx:
    def test_copies_rgb_and_pads_with_zero(self, odd_rgb_image: np.ndarray) -> None:
        """The RGB samples are copied, and the fourth sample defaults to 0."""
        # Act
        rgbx = rgb_to_rgbx(odd_rgb_image)

        # Assert
        assert rgbx.shape == (17, 19, 4)
        assert rgbx.dtype == np.uint8
        assert (rgbx[..., :3] == odd_rgb_image).all()
        assert (rgbx[..., 3] == 0).all()

    def test_pads_with_given_value(self, odd_rgb_image: np.ndarray) -> None:
        """The fourth sample is the given pad value."""
        # Act
        rgbx = rgb_to_rgbx(odd_rgb_image, pad=255)

        # Assert
        assert (rgbx[..., 3] == 255).all()
