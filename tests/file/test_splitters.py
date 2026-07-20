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

"""Unit tests for the concatenation part splitters (pure boundary logic)."""

import pytest

from wsidicom.file.file_writer import (
    ByteSizeSplitter,
    FrameCountSplitter,
    NoSplitter,
)

# Encapsulated frame overhead the byte splitter accounts per tile (item tag +
# length). Kept independent of the implementation constant on purpose.
_ITEM_HEADER_BYTES = 8

# Frame budget for the parametrized next_part_frame_count cases.
_MAX_FRAMES = 100


class TestNoSplitter:
    def test_never_starts_new_part(self) -> None:
        # Arrange
        splitter = NoSplitter()
        splitter.account(b"tile")

        # Act
        starts_new = splitter.should_start_new_part(b"tile")

        # Assert
        assert starts_new is False

    def test_next_part_takes_all_remaining_frames(self) -> None:
        # Arrange
        splitter = NoSplitter()
        remaining = 37

        # Act
        count = splitter.next_part_frame_count(remaining)

        # Assert — the whole level is one part, so it takes all remaining frames.
        assert count == remaining


class TestFrameCountSplitter:
    def test_starts_new_part_only_once_max_frames_reached(self) -> None:
        # Arrange
        max_frames = 3
        splitter = FrameCountSplitter(max_frames)
        for _ in range(max_frames - 1):
            splitter.account(b"a")

        # Act
        below_max = splitter.should_start_new_part(b"a")
        splitter.account(b"a")  # reaches max_frames
        at_max = splitter.should_start_new_part(b"a")

        # Assert
        assert below_max is False
        assert at_max is True

    def test_reset_restarts_the_count(self) -> None:
        # Arrange — fill the part to its max.
        max_frames = 2
        splitter = FrameCountSplitter(max_frames)
        for _ in range(max_frames):
            splitter.account(b"a")

        # Act
        splitter.reset()

        # Assert
        assert splitter.should_start_new_part(b"a") is False

    @pytest.mark.parametrize(
        ("remaining", "expected"),
        [
            (_MAX_FRAMES * 2, _MAX_FRAMES),  # more than budget -> full part
            (_MAX_FRAMES // 2, _MAX_FRAMES // 2),  # fewer -> short last part
            (_MAX_FRAMES, _MAX_FRAMES),  # exact fit
        ],
    )
    def test_next_part_frame_count_is_bounded_by_remaining(
        self, remaining: int, expected: int
    ) -> None:
        # Arrange
        splitter = FrameCountSplitter(_MAX_FRAMES)

        # Act
        count = splitter.next_part_frame_count(remaining)

        # Assert
        assert count == expected


class TestByteSizeSplitter:
    def test_accumulates_tiles_with_header_until_budget_exceeded(self) -> None:
        # Arrange — budget holds exactly two tiles, each counted with its header.
        tile = b"x" * 40
        tile_cost = len(tile) + _ITEM_HEADER_BYTES
        splitter = ByteSizeSplitter(2 * tile_cost)
        splitter.account(tile)

        # Act
        after_first = splitter.should_start_new_part(tile)
        splitter.account(tile)
        after_second = splitter.should_start_new_part(tile)

        # Assert
        assert after_first is False  # room for the second tile
        assert after_second is True  # a third tile would overflow

    def test_budget_boundary_counts_the_item_header(self) -> None:
        # Arrange — a tile whose size plus header exactly fills the budget.
        budget = 100
        splitter = ByteSizeSplitter(budget)
        exact = b"x" * (budget - _ITEM_HEADER_BYTES)
        splitter.account(exact)

        # Act
        starts_new = splitter.should_start_new_part(b"x")

        # Assert — the part is exactly full, so any further tile overflows.
        assert starts_new is True

    def test_reset_restarts_the_byte_count(self) -> None:
        # Arrange — a budget holding exactly one tile, then fill it.
        tile = b"x" * 40
        splitter = ByteSizeSplitter(len(tile) + _ITEM_HEADER_BYTES)
        splitter.account(tile)

        # Act
        splitter.reset()

        # Assert
        assert splitter.should_start_new_part(tile) is False

    def test_frame_count_unknown_until_part_fills(self) -> None:
        # Arrange
        splitter = ByteSizeSplitter(100)
        remaining = 1000

        # Act
        count = splitter.next_part_frame_count(remaining)

        # Assert — a byte budget can't predict the count, so parts must buffer.
        assert count is None
