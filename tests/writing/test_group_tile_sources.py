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

"""Tests for the sources of the tiles written for group instances."""

import numpy as np
import pytest
from decoy import Decoy, matchers

from wsidicom.codec import Encoder, Recoder
from wsidicom.group import Instances
from wsidicom.writing.group_tile_sources import (
    CopyingGroupTileSource,
    RecodingGroupTileSource,
    TranscodingGroupTileSource,
)

FOCAL_PLANES = [0.0]
OPTICAL_PATHS = ["0"]


@pytest.fixture
def decoy() -> Decoy:
    """Create a Decoy instance for mocking."""
    return Decoy()


@pytest.mark.unittest
class TestCopyingGroupTileSource:
    def test_gives_encoded_tiles_unchanged(self, decoy: Decoy) -> None:
        # Arrange
        instances = decoy.mock(cls=Instances)
        decoy.when(
            instances.iter_encoded_tiles(FOCAL_PLANES, OPTICAL_PATHS)
        ).then_return(iter([b"first", b"second"]))
        source = CopyingGroupTileSource()

        # Act
        tiles = list(source.iter_tiles(instances, FOCAL_PLANES, OPTICAL_PATHS))

        # Assert
        assert tiles == [b"first", b"second"]


@pytest.mark.unittest
class TestRecodingGroupTileSource:
    def test_gives_encoded_tiles_recoded(self, decoy: Decoy) -> None:
        # Arrange
        instances = decoy.mock(cls=Instances)
        decoy.when(
            instances.iter_encoded_tiles(FOCAL_PLANES, OPTICAL_PATHS)
        ).then_return(iter([b"first", b"second"]))
        recoder = decoy.mock(cls=Recoder)
        decoy.when(recoder.recode(b"first")).then_return(b"first recoded")
        decoy.when(recoder.recode(b"second")).then_return(b"second recoded")
        source = RecodingGroupTileSource(recoder)

        # Act
        tiles = list(source.iter_tiles(instances, FOCAL_PLANES, OPTICAL_PATHS))

        # Assert
        assert tiles == [b"first recoded", b"second recoded"]


@pytest.mark.unittest
class TestTranscodingGroupTileSource:
    def test_gives_decoded_tiles_encoded(self, decoy: Decoy) -> None:
        # Arrange
        first = np.zeros((2, 2, 3), dtype=np.uint8)
        second = np.ones((2, 2, 3), dtype=np.uint8)
        instances = decoy.mock(cls=Instances)
        decoy.when(
            instances.iter_decoded_tiles(FOCAL_PLANES, OPTICAL_PATHS)
        ).then_return(iter([first, second]))
        encoder = decoy.mock(cls=Encoder)
        # Numpy arrays can not be matched by equality, so the encoder answers
        # with the value of the tile it is given.
        decoy.when(encoder.encode(matchers.IsA(np.ndarray))).then_do(
            lambda tile: f"encoded {tile[0, 0, 0]}".encode()
        )
        source = TranscodingGroupTileSource(encoder)

        # Act
        tiles = list(source.iter_tiles(instances, FOCAL_PLANES, OPTICAL_PATHS))

        # Assert
        assert tiles == [b"encoded 0", b"encoded 1"]
