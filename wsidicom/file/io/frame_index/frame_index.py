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

"""Index of where the frames of an image are in the file."""

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray


class FrameIndex:
    """Where every frame of an image is in the file, as a position and a length.

    Both values come from a table that is read in one piece, so they are kept as an
    array each and a frame becomes a pair of numbers only when one is asked for.
    """

    def __init__(self, positions: NDArray[np.int64], lengths: NDArray[np.int64]):
        """Create a frame index.

        Parameters
        ----------
        positions: NDArray[np.int64]
            Position in the file of the first byte of each frame.
        lengths: NDArray[np.int64]
            Length in bytes of each frame.
        """
        if len(positions) != len(lengths):
            raise ValueError(
                f"Got {len(positions)} frame positions and {len(lengths)} frame "
                "lengths, expected as many of one as of the other."
            )
        self._positions = positions
        self._lengths = lengths

    def __getitem__(self, index: int) -> tuple[int, int]:
        """Return the position and length of the frame at the given index.

        Parameters
        ----------
        index: int
            Index of the frame in the file.

        Returns
        -------
        tuple[int, int]
            Position in the file of the first byte of the frame, and its length.
        """
        return int(self._positions[index]), int(self._lengths[index])

    def __len__(self) -> int:
        return len(self._positions)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return zip(self._positions.tolist(), self._lengths.tolist(), strict=True)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} frames)"
