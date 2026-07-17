#    Copyright 2021, 2022, 2023 SECTRA AB
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

"""Tile readers for pyramid generation.

Each reader reads tiles from source ImageData and dispatches them:
- PassthroughTileReader: encoded only, no accumulator
- CascadingPassthroughTileReader: encoded + decoded, feeds accumulator
- TranscodeTileReader: decoded only, submits to encoder pool
- CascadingTranscodeTileReader: decoded, submits to encoder pool + feeds accumulator
"""

from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from wsidicom.geometry import Point
from wsidicom.instance import ImageData
from wsidicom.thread import CancellationToken, WriteOnlyQueue
from wsidicom.writing.models import (
    EncodeTask,
    EncodingTaskResult,
    PyramidTilePosition,
)


class TileAccumulator(Protocol):
    """Sink that receives decoded tiles for cascaded downsampling."""

    @property
    def chain_depth(self) -> int:
        """Number of generated levels in the cascade starting from this one."""
        ...

    def add_tile(self, x: int, y: int, z: int, path: int, tile: np.ndarray) -> None:
        """Submit a decoded tile for accumulation."""
        ...


class TileReader(metaclass=ABCMeta):
    """Reads tiles from source and submits them for writing."""

    def __init__(self, level_index: int, token: CancellationToken):
        self._level_index = level_index
        self._token = token

    @property
    def level_index(self) -> int:
        return self._level_index

    @property
    def accumulator_chain_depth(self) -> int:
        """Depth of the accumulator chain. 0 if no accumulator."""
        return 0

    @abstractmethod
    def read_and_submit(
        self,
        image_data: ImageData,
        positions: Sequence[Point],
        z: float,
        path: str,
        z_index: int,
        path_index: int,
        output_queue: WriteOnlyQueue[EncodingTaskResult],
    ) -> None:
        """Read tiles from source, submit for writing, and feed accumulator."""
        raise NotImplementedError()

    def _make_coordinates(
        self, position: Point, z_index: int, path_index: int
    ) -> PyramidTilePosition:
        return PyramidTilePosition(
            level=self._level_index,
            x_index=position.x,
            y_index=position.y,
            z_index=z_index,
            optical_path_index=path_index,
        )

    def _submit_row(
        self,
        output_queue: WriteOnlyQueue[EncodingTaskResult],
        row_start: Point,
        tiles: list[bytes],
        z_index: int,
        path_index: int,
    ) -> None:
        """Submit a row of consecutive encoded tiles as one result."""
        output_queue.put(
            EncodingTaskResult(
                coordinates=self._make_coordinates(row_start, z_index, path_index),
                tiles=tiles,
            ),
            self._token,
        )


class PassthroughTileReader(TileReader):
    """Reads encoded tiles and submits to writer."""

    def read_and_submit(
        self,
        image_data: ImageData,
        positions: Sequence[Point],
        z: float,
        path: str,
        z_index: int,
        path_index: int,
        output_queue: WriteOnlyQueue[EncodingTaskResult],
    ) -> None:
        row_start: Point | None = None
        row_tiles: list[bytes] = []
        for position, encoded in zip(
            positions, image_data.get_encoded_tiles(positions, z, path), strict=True
        ):
            if row_start is not None and position.y != row_start.y:
                self._submit_row(
                    output_queue, row_start, row_tiles, z_index, path_index
                )
                row_tiles = []
            if not row_tiles:
                row_start = position
            row_tiles.append(encoded)
        if row_tiles and row_start is not None:
            self._submit_row(output_queue, row_start, row_tiles, z_index, path_index)


class CascadingPassthroughTileReader(TileReader):
    """Reads encoded+decoded tiles, submits encoded, feeds decoded to accumulator."""

    def __init__(
        self,
        level_index: int,
        accumulator: TileAccumulator,
        token: CancellationToken,
    ):
        super().__init__(level_index, token)
        self._accumulator = accumulator

    @property
    def accumulator_chain_depth(self) -> int:
        return self._accumulator.chain_depth

    def read_and_submit(
        self,
        image_data: ImageData,
        positions: Sequence[Point],
        z: float,
        path: str,
        z_index: int,
        path_index: int,
        output_queue: WriteOnlyQueue[EncodingTaskResult],
    ) -> None:
        row_start: Point | None = None
        row_tiles: list[bytes] = []
        for position, (encoded, decoded) in zip(
            positions,
            image_data.get_encoded_and_decoded_tiles(positions, z, path),
            strict=True,
        ):
            if row_start is not None and position.y != row_start.y:
                self._submit_row(
                    output_queue, row_start, row_tiles, z_index, path_index
                )
                row_tiles = []
            if not row_tiles:
                row_start = position
            row_tiles.append(encoded)
            self._accumulator.add_tile(
                position.x, position.y, z_index, path_index, decoded
            )
        if row_tiles and row_start is not None:
            self._submit_row(output_queue, row_start, row_tiles, z_index, path_index)


class TranscodeTileReader(TileReader):
    """Reads decoded tiles and submits to encoder pool."""

    def __init__(
        self,
        level_index: int,
        encoder_pool_queue: WriteOnlyQueue[EncodeTask],
        token: CancellationToken,
    ):
        super().__init__(level_index, token)
        self._encoder_pool_queue = encoder_pool_queue

    def read_and_submit(
        self,
        image_data: ImageData,
        positions: Sequence[Point],
        z: float,
        path: str,
        z_index: int,
        path_index: int,
        output_queue: WriteOnlyQueue[EncodingTaskResult],
    ) -> None:
        row_start: Point | None = None
        row_tiles: list[np.ndarray] = []
        for position, decoded in zip(
            positions,
            image_data.get_decoded_tiles(positions, z, path, cache=False),
            strict=True,
        ):
            if row_start is not None and position.y != row_start.y:
                self._encoder_pool_queue.put(
                    EncodeTask(
                        coordinates=self._make_coordinates(
                            row_start, z_index, path_index
                        ),
                        tiles=row_tiles,
                        output_queue=output_queue,
                    ),
                    self._token,
                )
                row_tiles = []
            if not row_tiles:
                row_start = position
            row_tiles.append(decoded)
        if row_tiles and row_start is not None:
            self._encoder_pool_queue.put(
                EncodeTask(
                    coordinates=self._make_coordinates(row_start, z_index, path_index),
                    tiles=row_tiles,
                    output_queue=output_queue,
                ),
                self._token,
            )


class CascadingTranscodeTileReader(TileReader):
    """Reads decoded tiles, submits to encoder pool, feeds accumulator."""

    def __init__(
        self,
        level_index: int,
        encoder_pool_queue: WriteOnlyQueue[EncodeTask],
        accumulator: TileAccumulator,
        token: CancellationToken,
    ):
        super().__init__(level_index, token)
        self._encoder_pool_queue = encoder_pool_queue
        self._accumulator = accumulator

    @property
    def accumulator_chain_depth(self) -> int:
        return self._accumulator.chain_depth

    def read_and_submit(
        self,
        image_data: ImageData,
        positions: Sequence[Point],
        z: float,
        path: str,
        z_index: int,
        path_index: int,
        output_queue: WriteOnlyQueue[EncodingTaskResult],
    ) -> None:
        row_start: Point | None = None
        row_tiles: list[np.ndarray] = []
        for position, decoded in zip(
            positions,
            image_data.get_decoded_tiles(positions, z, path, cache=False),
            strict=True,
        ):
            if row_start is not None and position.y != row_start.y:
                self._encoder_pool_queue.put(
                    EncodeTask(
                        coordinates=self._make_coordinates(
                            row_start, z_index, path_index
                        ),
                        tiles=row_tiles,
                        output_queue=output_queue,
                    ),
                    self._token,
                )
                row_tiles = []
            if not row_tiles:
                row_start = position
            row_tiles.append(decoded)
            self._accumulator.add_tile(
                position.x, position.y, z_index, path_index, decoded
            )
        if row_tiles and row_start is not None:
            self._encoder_pool_queue.put(
                EncodeTask(
                    coordinates=self._make_coordinates(row_start, z_index, path_index),
                    tiles=row_tiles,
                    output_queue=output_queue,
                ),
                self._token,
            )
