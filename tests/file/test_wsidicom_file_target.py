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

from collections.abc import Sequence

import pytest

from wsidicom.file import WsiDicomFileTarget


@pytest.mark.unittest
class TestWsiDicomFileTarget:
    @pytest.mark.parametrize(
        "candidate_levels",
        [[0, 1, 2], [0, 2, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8]],
    )
    def test_select_included_levels_include_indices_is_none_returns_all_candidates(
        self,
        candidate_levels: Sequence[int],
    ):
        # Act
        selected = WsiDicomFileTarget._select_included_levels(
            candidate_levels, None
        )

        # Assert
        assert selected == set(candidate_levels)

    @pytest.mark.parametrize(
        ["candidate_levels", "include_indices", "expected"],
        [
            # All indices in range
            ([0, 1, 2], [0, 1, 2], {0, 1, 2}),
            # Single positive index
            ([0, 1, 2], [0], {0}),
            ([0, 1, 2], [1], {1}),
            ([0, 1, 2], [2], {2}),
            # Negative indices
            ([0, 1, 2], [-3], {0}),
            ([0, 1, 2], [-2], {1}),
            ([0, 1, 2], [-1], {2}),
            # Mix of positive and negative
            ([0, 1, 2], [0, -1], {0, 2}),
            # Out-of-range indices silently dropped
            ([0, 1, 2], [0, 10, -10], {0}),
            # Sparse present_levels: indices map to absolute level numbers
            ([0, 2, 4], [0], {0}),
            ([0, 2, 4], [1], {2}),
            ([0, 2, 4], [2], {4}),
            ([0, 2, 4], [-1], {4}),
            # Indices beyond sparse list are dropped (legacy behaviour)
            ([0, 2, 4], [0, 1, 2, 3, 4, 5], {0, 2, 4}),
        ],
    )
    def test_select_included_levels_returns_levels_at_indices(
        self,
        candidate_levels: Sequence[int],
        include_indices: Sequence[int],
        expected: set[int],
    ):
        # Act
        selected = WsiDicomFileTarget._select_included_levels(
            candidate_levels, include_indices
        )

        # Assert
        assert selected == expected

    def test_select_included_levels_empty_indices_returns_empty(self):
        # Act
        selected = WsiDicomFileTarget._select_included_levels([0, 1, 2], [])

        # Assert
        assert selected == set()

    @pytest.mark.parametrize(
        ["candidate_levels", "include_indices", "expected"],
        [
            # Sparse pyramid (e.g. APERIO_JP2000_RGB at indices 0, 2, 4)
            # extended with missing levels up to lowest_single_tile_level=8.
            # Reproduces #140: include_indices=range(6) should select 6 levels,
            # not 9. The candidate list is the extended virtual pyramid.
            (
                list(range(9)),
                list(range(6)),
                {0, 1, 2, 3, 4, 5},
            ),
            # First three levels of the extended list.
            (list(range(9)), [0, 1, 2], {0, 1, 2}),
            # Last three levels.
            (list(range(9)), [-1, -2, -3], {6, 7, 8}),
        ],
    )
    def test_select_included_levels_extended_pyramid_for_add_missing_levels(
        self,
        candidate_levels: Sequence[int],
        include_indices: Sequence[int],
        expected: set[int],
    ):
        # Act
        selected = WsiDicomFileTarget._select_included_levels(
            candidate_levels, include_indices
        )

        # Assert
        assert selected == expected
