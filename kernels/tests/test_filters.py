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

"""Tests for the Interpolation and ResizeFilter enums.

Every member is valid for its function.
"""

import numpy as np
import pytest
from wsidicom_kernels import Interpolation, ResizeFilter, compose, downscale


@pytest.fixture
def image(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, (32, 32), np.uint8)


@pytest.mark.unittest
class TestFilterEnums:
    @pytest.mark.parametrize("interpolation", list(Interpolation))
    def test_every_interpolation_member_composes(
        self, interpolation: Interpolation
    ) -> None:
        """Every declared Interpolation member is a valid compose interpolation."""
        # Arrange
        canvas = np.zeros((8, 8), np.uint8)

        # Act & Assert
        compose(
            canvas, [np.ones((4, 4), np.uint8)], np.array([[1.5, 1.5]]), interpolation
        )

    @pytest.mark.parametrize("resize_filter", list(ResizeFilter))
    def test_every_resample_member_downscales(
        self, image: np.ndarray, resize_filter: ResizeFilter
    ) -> None:
        """Every declared ResizeFilter member is a valid downscale filter."""
        # Act & Assert
        downscale(image, 2, resize_filter)
