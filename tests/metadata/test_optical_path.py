#    Copyright 2023 SECTRA AB
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

import numpy as np
import pytest

from wsidicom.metadata.optical_path import (
    ConstantLutSegment,
    DiscreteLutSegment,
    LinearLutSegment,
    Lut,
)


class TestDicomLut:
    @pytest.mark.parametrize(
        ["lut", "expected_table_component"],
        [
            (
                Lut(
                    [ConstantLutSegment(0, 256)],
                    [ConstantLutSegment(0, 256)],
                    [LinearLutSegment(0, 255, 256)],
                    np.uint16,
                ),
                [
                    np.full(256, 0, dtype=np.uint16),
                    np.full(256, 0, dtype=np.uint16),
                    np.linspace(0, 255, 256, dtype=np.uint16),
                ],
            ),
            (
                Lut(
                    [ConstantLutSegment(0, 256)],
                    [ConstantLutSegment(0, 256)],
                    [
                        ConstantLutSegment(0, 100),
                        LinearLutSegment(0, 255, 100),
                        ConstantLutSegment(255, 56),
                    ],
                    np.uint16,
                ),
                [
                    np.full(256, 0, dtype=np.uint16),
                    np.full(256, 0, dtype=np.uint16),
                    np.concatenate(
                        [
                            np.full(100, 0, dtype=np.uint16),
                            np.linspace(0, 255, 100, dtype=np.uint16),
                            np.full(56, 255, dtype=np.uint16),
                        ]
                    ),
                ],
            ),
            (
                Lut(
                    [DiscreteLutSegment([index for index in range(256)])],
                    [DiscreteLutSegment([0 for _ in range(256)])],
                    [DiscreteLutSegment([255 for _ in range(256)])],
                    np.uint16,
                ),
                [
                    np.array([index for index in range(256)], dtype=np.uint16),
                    np.full(256, 0, dtype=np.uint16),
                    np.full(256, 255, dtype=np.uint16),
                ],
            ),
        ],
    )
    def test_parse_to_table(self, lut: Lut, expected_table_component: np.ndarray):
        # Arrange

        # Act
        table = lut.table

        # Assert
        for component, expected_component in zip(table, expected_table_component):
            assert len(component) == len(expected_component)
            assert np.array_equal(component, expected_component)
