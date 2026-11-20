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

Each reader reads tiles from source ImageData, submits them for writing in
segments of consecutive tiles of a tile row, and feeds the decoded tiles to an
accumulator if it has one:
- EncodedTileReader: reads encoded tiles and hands each segment to an encoded
  tile submitter, which puts them on the output queue as they are
  (PassthroughEncodedTileSubmitter) or has them recoded first
  (RecodeEncodedTileSubmitter).
- TranscodeTileReader: reads decoded tiles and submits each segment to the
  encoder pool to be encoded.
"""

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol, TypeVar

import numpy as np

from wsidicom.codec import Encoder, Recoder
from wsidicom.geometry import Point
from wsidicom.instance import ImageData
from wsidicom.thread import CancellationToken, WriteOnlyQueue
from wsidicom.writing.models import (
    EncodeTask,
    EncodingTaskResult,
    PyramidTilePosition,
    RecodeTask,
)

TileType = TypeVar("TileType")


class TileAccumulator(Protocol):
    """Sink that receives decoded tiles for cascaded downsampling."""

    @property
    def chain_depth(self) -> int:
        """Number of generated levels in the cascade starting from this one."""
        ...

    def add_tile(self, x: int, y: int, z: int, path: int, tile: np.ndarray) -> None:
        """Submit a decoded tile for accumulation."""
        ...


class EncodedTileSubmitter(metaclass=ABCMeta):
    """Gets encoded tiles onto an output queue, changed or not on the way."""

    def __init__(self, token: CancellationToken) -> None:
        self._token = token

    @abstractmethod
    def submit(
        self,
        output_queue: WriteOnlyQueue[EncodingTaskResult],
        coordinates: PyramidTilePosition,
        tiles: list[bytes],
    ) -> None:
        """Submit encoded tiles to end up on the output queue.

        The tiles are consecutive in x within one tile row.

        Parameters
        ----------
        output_queue: WriteOnlyQueue[EncodingTaskResult]
            Queue the tiles are to end up on.
        coordinates: PyramidTilePosition
            Position of the first of the tiles.
        tiles: list[bytes]
            The encoded tiles, in increasing x.
        """
        raise NotImplementedError()


class PassthroughEncodedTileSubmitter(EncodedTileSubmitter):
    """Puts encoded tiles on the output queue as they are."""

    def submit(
        self,
        output_queue: WriteOnlyQueue[EncodingTaskResult],
        coordinates: PyramidTilePosition,
        tiles: list[bytes],
    ) -> None:
        output_queue.put(
            EncodingTaskResult(coordinates=coordinates, tiles=tiles), self._token
        )


class RecodeEncodedTileSubmitter(EncodedTileSubmitter):
    """Submits encoded tiles to the encoder pool to be recoded."""

    def __init__(
        self,
        encoder_pool_queue: WriteOnlyQueue[RecodeTask],
        recoder: Recoder,
        token: CancellationToken,
    ) -> None:
        super().__init__(token)
        self._encoder_pool_queue = encoder_pool_queue
        self._recoder = recoder

    def submit(
        self,
        output_queue: WriteOnlyQueue[EncodingTaskResult],
        coordinates: PyramidTilePosition,
        tiles: list[bytes],
    ) -> None:
        self._encoder_pool_queue.put(
            RecodeTask(
                coordinates=coordinates,
                tiles=tiles,
                recoder=self._recoder,
                output_queue=output_queue,
            ),
            self._token,
        )


class TileReader(metaclass=ABCMeta):
    """Reads tiles from source and submits them for writing."""

    def __init__(
        self,
        level_index: int,
        accumulator: TileAccumulator | None,
        token: CancellationToken,
    ) -> None:
        self._level_index = level_index
        self._accumulator = accumulator
        self._token = token

    @property
    def level_index(self) -> int:
        return self._level_index

    @property
    def accumulator_chain_depth(self) -> int:
        """Depth of the accumulator chain. 0 if no accumulator."""
        if self._accumulator is None:
            return 0
        return self._accumulator.chain_depth

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

    @staticmethod
    def _iterate_row_segments(
        positions: Sequence[Point], tiles: Iterable[TileType]
    ) -> Iterator[tuple[list[Point], list[TileType]]]:
        """Group tiles into segments of a tile row.

        A segment ends where the next position is in another tile row. The
        positions within a tile row are taken to be consecutive in x, as they
        are batched by the level writer.

        Parameters
        ----------
        positions: Sequence[Point]
            Positions of the tiles.
        tiles: Iterable[TileType]
            The tiles, in the order of the positions.

        Returns
        -------
        Iterator[tuple[list[Point], list[TileType]]]
            The positions and the tiles of each segment.
        """
        segment_positions: list[Point] = []
        segment_tiles: list[TileType] = []
        for position, tile in zip(positions, tiles, strict=True):
            if segment_positions and position.y != segment_positions[0].y:
                yield segment_positions, segment_tiles
                segment_positions = []
                segment_tiles = []
            segment_positions.append(position)
            segment_tiles.append(tile)
        if segment_positions:
            yield segment_positions, segment_tiles


class EncodedTileReader(TileReader):
    """Reads encoded tiles and hands them segment by segment to a submitter.

    With an accumulator the tiles are also read decoded and fed to it.
    """

    def __init__(
        self,
        level_index: int,
        encoded_tile_submitter: EncodedTileSubmitter,
        accumulator: TileAccumulator | None,
        token: CancellationToken,
    ) -> None:
        super().__init__(level_index, accumulator, token)
        self._encoded_tile_submitter = encoded_tile_submitter

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
        accumulator = self._accumulator
        if accumulator is None:
            for segment_positions, encoded_tiles in self._iterate_row_segments(
                positions, image_data.get_encoded_tiles(positions, z, path)
            ):
                self._encoded_tile_submitter.submit(
                    output_queue,
                    self._make_coordinates(segment_positions[0], z_index, path_index),
                    encoded_tiles,
                )
            return
        for segment_positions, tiles in self._iterate_row_segments(
            positions, image_data.get_encoded_and_decoded_tiles(positions, z, path)
        ):
            for position, (_, decoded) in zip(segment_positions, tiles, strict=True):
                accumulator.add_tile(
                    position.x, position.y, z_index, path_index, decoded
                )
            self._encoded_tile_submitter.submit(
                output_queue,
                self._make_coordinates(segment_positions[0], z_index, path_index),
                [encoded for encoded, _ in tiles],
            )


class TranscodeTileReader(TileReader):
    """Reads decoded tiles and submits them segment by segment to be encoded.

    With an accumulator the decoded tiles are also fed to it.
    """

    def __init__(
        self,
        level_index: int,
        encoder_pool_queue: WriteOnlyQueue[EncodeTask],
        encoder: Encoder,
        accumulator: TileAccumulator | None,
        token: CancellationToken,
    ) -> None:
        super().__init__(level_index, accumulator, token)
        self._encoder_pool_queue = encoder_pool_queue
        self._encoder = encoder

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
        for segment_positions, decoded_tiles in self._iterate_row_segments(
            positions, image_data.get_decoded_tiles(positions, z, path, cache=False)
        ):
            if self._accumulator is not None:
                for position, decoded in zip(
                    segment_positions, decoded_tiles, strict=True
                ):
                    self._accumulator.add_tile(
                        position.x, position.y, z_index, path_index, decoded
                    )
            self._encoder_pool_queue.put(
                EncodeTask(
                    coordinates=self._make_coordinates(
                        segment_positions[0], z_index, path_index
                    ),
                    tiles=decoded_tiles,
                    encoder=self._encoder,
                    output_queue=output_queue,
                ),
                self._token,
            )
