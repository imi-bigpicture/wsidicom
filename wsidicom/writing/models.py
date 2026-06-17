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

"""Shared data models for the write pipeline."""

from dataclasses import dataclass, field

from PIL import Image

from wsidicom.thread import CompletionTracker, WriteOnlyQueue


@dataclass(frozen=True)
class CascadedTile:
    """Decoded tile tagged with its source-grid position for cascade forwarding.

    Parameters
    ----------
    x_index: int
        Column index of the tile in the source (input) grid.
    y_index: int
        Row index of the tile in the source (input) grid.
    z_index: int
        Focal plane index.
    optical_path_index: int
        Optical path index.
    tile: Image.Image
        The decoded tile image.
    """

    x_index: int
    y_index: int
    z_index: int
    optical_path_index: int
    tile: Image.Image


@dataclass(frozen=True)
class PyramidTilePosition:
    """Coordinates identifying a tile position in the pyramid.

    Parameters
    ----------
    level: int
        Pyramid level index (0 = base).
    x_index: int
        Column index of the tile at this level.
    y_index: int
        Row index of the tile at this level.
    z_index: int
        Focal plane index.
    optical_path_index: int
        Optical path index.
    """

    level: int
    x_index: int
    y_index: int
    z_index: int
    optical_path_index: int

    def __lt__(self, other: object) -> bool:
        """Compare from slowest to fastest changing index.

        Order: level -> optical path -> focal plane (z) -> row (y) -> column (x).
        """
        if not isinstance(other, PyramidTilePosition):
            return NotImplemented
        return (
            self.level,
            self.optical_path_index,
            self.z_index,
            self.y_index,
            self.x_index,
        ) < (
            other.level,
            other.optical_path_index,
            other.z_index,
            other.y_index,
            other.x_index,
        )


@dataclass(order=True, frozen=True)
class EncodingTaskResult:
    """A batch of successfully encoded tiles.

    A failed encoding cancels the shared `CancellationToken` rather than
    producing a result, so a result always carries encoded tiles.

    Parameters
    ----------
    coordinates: PyramidTilePosition
        Position of the first tile in the batch. Used as the priority key.
    tiles: List[bytes]
        Encoded tile bytes.
    """

    coordinates: PyramidTilePosition = field(compare=True)
    tiles: list[bytes] = field(compare=False, default_factory=list)


@dataclass(frozen=True)
class CoordinatePriority:
    """Base for priority-queue items ordered by their `coordinates`.

    Comparison succeeds across any subclass; returns `NotImplemented`
    for unrelated types so Python's reflection falls through to the
    other side's `__gt__`.

    Parameters
    ----------
    coordinates: PyramidTilePosition
        Position used as the priority key.
    """

    coordinates: PyramidTilePosition

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, CoordinatePriority):
            return NotImplemented
        return self.coordinates < other.coordinates


@dataclass(frozen=True)
class EncodeTask(CoordinatePriority):
    """Encode decoded tiles and submit the result to an output queue.

    Parameters
    ----------
    tiles: List[Image.Image]
        Decoded tiles to encode.
    output_queue: WriteOnlyQueue[EncodingTaskResult]
        Queue to put the encoded result on.
    """

    tiles: list[Image.Image]
    output_queue: WriteOnlyQueue[EncodingTaskResult]


@dataclass(frozen=True)
class DownsampleEncodeTask(CoordinatePriority):
    """Stitch, downsample, and encode a row-major 2D block of tiles.

    At pyramid edges the block can be smaller than 2x2.

    Parameters
    ----------
    tiles: List[List[Image.Image]]
        Decoded tiles forming the input block, as a row-major 2D matrix.
    output_queue: WriteOnlyQueue[EncodingTaskResult]
        Queue to put the encoded result on.
    cascade_tracker: CompletionTracker
        Tracker to decrement when the task completes.
    cascade_queue: Optional[WriteOnlyQueue[CascadedTile]]
        Queue to forward the decoded downsampled tile to the next level's
        accumulator, or None if this is the top of the cascade.
    """

    tiles: list[list[Image.Image]]
    output_queue: WriteOnlyQueue[EncodingTaskResult]
    cascade_tracker: CompletionTracker
    cascade_queue: WriteOnlyQueue[CascadedTile] | None = None
