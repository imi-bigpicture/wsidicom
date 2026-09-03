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

from collections.abc import Sequence

import numpy as np
import pytest
from pydicom import Dataset
from pydicom.sequence import Sequence as DicomSequence

from wsidicom.errors import WsiDicomError
from wsidicom.geometry import Point
from wsidicom.instance.dataset import WsiDataset
from wsidicom.instance.per_frame_group_positions import PerFrameGroupPositions
from wsidicom.instance.tile_index.sparse_tile_index import SparseTileIndex

TILE_SIZE = 10
IMAGE_SIZE = 30
"""Three by three tiles, so a frame can be placed in any of nine positions."""


def create_dataset(
    columns: Sequence[int],
    rows: Sequence[int],
    z_offsets: Sequence[float] | None = None,
    optical_path_identifiers: Sequence[str] | None = None,
    frame_offset: int = 0,
) -> WsiDataset:
    """Create a dataset whose frames sit at the given positions.

    The positions are given as the reader gives them, one sequence per element, which
    is what a dataset read without parsing the per frame functional groups carries.
    """
    dataset = WsiDataset(
        Dataset(),
        frame_positions=PerFrameGroupPositions(
            columns=np.asarray(columns, dtype=np.int64),
            rows=np.asarray(rows, dtype=np.int64),
            z_offsets=(
                None if z_offsets is None else np.asarray(z_offsets, dtype=np.float64)
            ),
            optical_path_identifiers=(
                None
                if optical_path_identifiers is None
                else np.asarray(optical_path_identifiers, dtype=np.str_)
            ),
        ),
    )
    dataset = dataset.replace(
        {
            "TotalPixelMatrixColumns": IMAGE_SIZE,
            "TotalPixelMatrixRows": IMAGE_SIZE,
            "Columns": TILE_SIZE,
            "Rows": TILE_SIZE,
            "NumberOfFrames": len(columns),
            "SOPInstanceUID": "1.2.3.4",
            "StudyInstanceUID": "1.2.3",
            "SeriesInstanceUID": "1.2.3.5",
        }
    )
    if frame_offset:
        # Only a concatenated instance has its frames offset from the first.
        dataset = dataset.replace(
            {
                "SOPInstanceUIDOfConcatenationSource": "1.2.3.6",
                "ConcatenationFrameOffsetNumber": frame_offset,
            }
        )
    return dataset


def create_parsed_dataset(
    columns: Sequence[int],
    rows: Sequence[int],
    z_offsets: Sequence[float] | None = None,
    optical_path_identifiers: Sequence[str] | None = None,
) -> WsiDataset:
    """Create the same dataset with the per frame functional groups in it.

    The positions then have to be parsed out of the sequence, which has to put the
    frames in the same planes as the sequences the reader gives.
    """
    dataset = create_dataset(columns, rows, z_offsets, optical_path_identifiers)
    dataset._frame_positions = None
    frames = DicomSequence()
    for index, (column, row) in enumerate(zip(columns, rows, strict=True)):
        position = Dataset()
        position.ColumnPositionInTotalImagePixelMatrix = column
        position.RowPositionInTotalImagePixelMatrix = row
        if z_offsets is not None:
            position.ZOffsetInSlideCoordinateSystem = z_offsets[index]
        frame = Dataset()
        frame.PlanePositionSlideSequence = DicomSequence([position])
        if optical_path_identifiers is not None:
            optical_path = Dataset()
            optical_path.OpticalPathIdentifier = optical_path_identifiers[index]
            frame.OpticalPathIdentificationSequence = DicomSequence([optical_path])
        frames.append(frame)
    dataset = dataset.replace({"PerFrameFunctionalGroupsSequence": frames})
    return dataset


@pytest.mark.unittest
class TestSparseTileIndex:
    def test_places_frames_at_their_tiles(self):
        # Arrange - positions are one based, so 1 is the first tile and 21 the third.
        dataset = create_dataset(columns=[1, 21, 11], rows=[1, 1, 21])

        # Act
        index = SparseTileIndex([dataset])

        # Assert
        assert index.get_frame_index(Point(0, 0), 0, "0") == 0
        assert index.get_frame_index(Point(2, 0), 0, "0") == 1
        assert index.get_frame_index(Point(1, 2), 0, "0") == 2

    def test_missing_tile_is_reported_as_missing(self):
        # Arrange
        dataset = create_dataset(columns=[1], rows=[1])

        # Act
        index = SparseTileIndex([dataset])

        # Assert
        assert index.get_frame_index(Point(2, 2), 0, "0") == -1

    def test_frames_without_a_z_offset_are_in_the_plane_at_zero(self):
        # Arrange
        dataset = create_dataset(columns=[1], rows=[1])

        # Act
        index = SparseTileIndex([dataset])

        # Assert
        assert index.focal_planes == [0]
        assert index.get_frame_index(Point(0, 0), 0, "0") == 0

    def test_splits_frames_into_focal_planes(self):
        # Arrange - the same tile at two focal planes, and a third tile at one of them.
        dataset = create_dataset(
            columns=[1, 1, 21], rows=[1, 1, 1], z_offsets=[0.0, 1.5, 1.5]
        )

        # Act
        index = SparseTileIndex([dataset])

        # Assert
        assert index.focal_planes == [0.0, 1.5]
        assert index.get_frame_index(Point(0, 0), 0.0, "0") == 0
        assert index.get_frame_index(Point(0, 0), 1.5, "0") == 1
        assert index.get_frame_index(Point(2, 0), 1.5, "0") == 2
        assert index.get_frame_index(Point(2, 0), 0.0, "0") == -1

    def test_splits_frames_into_optical_paths(self):
        # Arrange - the same tile down two optical paths.
        dataset = create_dataset(
            columns=[1, 1], rows=[1, 1], optical_path_identifiers=["0", "1"]
        )

        # Act
        index = SparseTileIndex([dataset])

        # Assert
        assert index.get_frame_index(Point(0, 0), 0, "0") == 0
        assert index.get_frame_index(Point(0, 0), 0, "1") == 1

    def test_splits_frames_into_focal_planes_and_optical_paths(self):
        # Arrange - every combination of two focal planes and two optical paths.
        dataset = create_dataset(
            columns=[1, 1, 1, 1],
            rows=[1, 1, 1, 1],
            z_offsets=[0.0, 0.0, 1.0, 1.0],
            optical_path_identifiers=["0", "1", "0", "1"],
        )

        # Act
        index = SparseTileIndex([dataset])

        # Assert
        assert index.focal_planes == [0.0, 1.0]
        assert index.get_frame_index(Point(0, 0), 0.0, "0") == 0
        assert index.get_frame_index(Point(0, 0), 0.0, "1") == 1
        assert index.get_frame_index(Point(0, 0), 1.0, "0") == 2
        assert index.get_frame_index(Point(0, 0), 1.0, "1") == 3

    def test_z_offsets_are_rounded_into_the_same_focal_plane(self):
        # Arrange - closer together than the decimals a focal plane is keyed by.
        dataset = create_dataset(columns=[1, 21], rows=[1, 1], z_offsets=[1.0, 1.00001])

        # Act
        index = SparseTileIndex([dataset])

        # Assert
        assert index.focal_planes == [1.0]
        assert index.get_frame_index(Point(2, 0), 1.0, "0") == 1

    def test_frames_of_a_concatenated_instance_keep_their_frame_numbers(self):
        # Arrange - the second dataset holds the frames after the first.
        first = create_dataset(columns=[1, 21], rows=[1, 1])
        second = create_dataset(columns=[1], rows=[21], frame_offset=2)

        # Act
        index = SparseTileIndex([first, second])

        # Assert
        assert index.get_frame_index(Point(0, 0), 0, "0") == 0
        assert index.get_frame_index(Point(2, 0), 0, "0") == 1
        assert index.get_frame_index(Point(0, 2), 0, "0") == 2

    @pytest.mark.parametrize(
        ["z_offsets", "optical_path_identifiers"],
        [
            (None, None),
            ([0.0, 1.5], None),
            (None, ["0", "1"]),
            ([0.0, 1.5], ["0", "1"]),
        ],
    )
    def test_parsed_sequence_gives_the_same_planes(
        self,
        z_offsets: Sequence[float] | None,
        optical_path_identifiers: Sequence[str] | None,
    ):
        # Arrange - the same frames, once as read positions and once as a sequence.
        columns, rows = [1, 21], [1, 11]
        read = create_dataset(columns, rows, z_offsets, optical_path_identifiers)
        parsed = create_parsed_dataset(
            columns, rows, z_offsets, optical_path_identifiers
        )

        # Act
        from_read = SparseTileIndex([read])
        from_parsed = SparseTileIndex([parsed])

        # Assert
        assert from_read.focal_planes == from_parsed.focal_planes
        assert set(from_read.planes) == set(from_parsed.planes)
        for key, plane in from_read.planes.items():
            assert (plane.plane == from_parsed.planes[key].plane).all()

    @pytest.mark.parametrize(
        ["columns", "rows", "why"],
        [
            ([1, IMAGE_SIZE + 1], [1, 1], "past the last column"),
            ([1, 1], [1, IMAGE_SIZE + 1], "past the last row"),
            ([1, 0], [1, 1], "before the first column"),
            ([1, 1], [1, 0], "before the first row"),
        ],
    )
    def test_refuses_a_frame_outside_the_tiling(
        self, columns: Sequence[int], rows: Sequence[int], why: str
    ):
        """A position above the tiling would be an IndexError from numpy, and one
        below it no error at all, as a negative index counts from the far end."""
        # Arrange - positions are one based, so zero is before the first tile.
        dataset = create_dataset(columns=columns, rows=rows)

        # Act & Assert
        with pytest.raises(WsiDicomError):
            _ = SparseTileIndex([dataset]).planes
