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

"""Tile positions taken from the Per Frame Functional Groups Sequence."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PerFrameGroupPositions:
    """Tile positions for every frame of an instance, in frame order.

    Parameters
    ----------
    columns: NDArray[np.int64]
        Column position in the total image pixel matrix, one per frame.
    rows: NDArray[np.int64]
        Row position in the total image pixel matrix, one per frame.
    z_offsets: NDArray[np.float64] | None
        Z offset in the slide coordinate system, one per frame, or None if the frames
        do not carry one.
    optical_path_identifiers: NDArray[np.str_] | None
        Optical path identifier, one per frame, or None if the frames do not carry one.
    """

    columns: NDArray[np.int64] = field(repr=False)
    rows: NDArray[np.int64] = field(repr=False)
    z_offsets: NDArray[np.float64] | None = field(repr=False)
    optical_path_identifiers: NDArray[np.str_] | None = field(repr=False)
