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

"""Reads what a dataset read in parts has left to read."""

from collections.abc import Iterable
from typing import Final

from pydicom import DataElement
from pydicom.dataset import Dataset

from wsidicom.file.io.deferred_element import DeferredElement
from wsidicom.file.io.per_frame_functional_groups_reader import (
    PerFrameFunctionalGroupsReader,
)
from wsidicom.file.io.wsidicom_io import WsiDicomReadIO
from wsidicom.instance import WsiDataset
from wsidicom.instance.dataset import DeferredDatasetReader
from wsidicom.instance.per_frame_group_positions import PerFrameGroupPositions
from wsidicom.tags import (
    ExtendedOffsetTableTag,
    NumberOfFramesTag,
    SpecificCharacterSetTag,
)


class FileDeferredDatasetReader(DeferredDatasetReader):
    """Reads the rest of a dataset that was read only as far as opening needs.

    Opening stops at the per frame functional groups sequence and passes over the
    values too large to be worth reading, neither of which the way to a tile goes
    through. What was left is read here when something asks for it: the tile
    positions when a frame is read, and the rest when the whole dataset is wanted.

    The sequence holds the tile positions and says where the pixel data starts, so
    it is read once whichever of them is asked for first, and not at all for an
    instance whose frames are never read. It is not kept, the positions being what
    it holds.
    """

    def __init__(
        self,
        io: WsiDicomReadIO,
        dataset: Dataset,
        deferred_elements: Iterable[DeferredElement],
        per_frame_position: int | None,
        pixel_data_position: int | None,
    ):
        """Create a reader for the rest of a dataset.

        Parameters
        ----------
        io: WsiDicomReadIO
            File the dataset was read from.
        dataset: Dataset
            The dataset, as far as it has been read.
        deferred_elements: Iterable[DeferredElement]
            Elements whose values were passed over while reading.
        per_frame_position: int | None
            Offset of the per frame functional groups sequence the read stopped at,
            or None where the instance has no sequence to stop at.
        pixel_data_position: int | None
            Offset the pixel data starts at, which the read reached where there was
            no sequence to stop at, or None while the sequence is still in the way.
        """
        self._io: Final = io
        self._dataset: Final = dataset
        self._deferred_elements: Final = list(deferred_elements)
        self._per_frame_position: Final = per_frame_position
        self._pixel_data_position = pixel_data_position
        self._end_of_per_frame_groups: int | None = None
        self._positions_read = False
        self._positions: PerFrameGroupPositions | None = None

    def seek_to_pixel_data(self) -> int:
        """Read past the per frame functional groups sequence to the pixel data.

        The attributes between the sequence and the pixel data are read into the
        dataset on the way. The sequence itself is not: what it holds is the tile
        positions, which :func:`read_frame_positions` gives.

        Returns
        -------
        int
            Offset the pixel data starts at.

        Raises
        ------
        ValueError
            If the offset is not known and there is no sequence to read past for it.
        """
        with self._io.exclusive():
            if self._pixel_data_position is not None:
                return self._pixel_data_position
            if self._per_frame_position is None:
                raise ValueError(
                    "Where the pixel data starts is not known and there is no per "
                    "frame functional groups sequence to read past for it."
                )
            if self._end_of_per_frame_groups is None:
                self._end_of_per_frame_groups = self._seek_past_per_frame_groups(
                    self._per_frame_position
                )
            return self._read_elements_until_pixel_data(self._end_of_per_frame_groups)

    def read_frame_positions(self) -> PerFrameGroupPositions | None:
        """Read the tile position of every frame.

        Reading them gets past the per frame functional groups sequence, so the
        attributes between it and the pixel data are read into the dataset on the
        way, and where the pixel data starts is found. The sequence itself is not
        put in the dataset.

        Returns
        -------
        PerFrameGroupPositions | None
            Position of every frame, or None where the instance has no sequence or
            its frames do not state where they sit.
        """
        if self._per_frame_position is None:
            return None
        with self._io.exclusive():
            if not self._positions_read:
                self._positions, self._end_of_per_frame_groups = (
                    self._read_per_frame_groups(self._per_frame_position)
                )
                self._positions_read = True
                self._read_elements_until_pixel_data(self._end_of_per_frame_groups)
            return self._positions

    def complete_dataset(self) -> None:
        """Read everything left unread into the dataset.

        Does nothing once there is nothing left, so a caller wanting a whole dataset
        can ask without first asking whether it is already whole.
        """
        with self._io.exclusive():
            self.seek_to_pixel_data()
            while self._deferred_elements:
                element = self._deferred_elements.pop()
                self._io.seek(element.offset)
                element.set(self._io.read(element.length, need_exact_length=True))

    def _read_per_frame_groups(
        self, per_frame_position: int
    ) -> tuple[PerFrameGroupPositions | None, int]:
        """Find the tile position of every frame, and where their groups end.

        Parameters
        ----------
        per_frame_position: int
            Offset the per frame functional groups sequence starts at.

        Returns
        -------
        tuple[PerFrameGroupPositions | None, int]
            Positions of every frame, or None where the instance has no sequence or
            its frames do not state where they sit, and the offset where the per frame
            functional groups sequence ends.
        """
        reader = self._create_reader(per_frame_position)
        positions = reader.read_positions()
        return positions, reader.end_of_sequence

    def _seek_past_per_frame_groups(self, per_frame_position: int) -> int:
        """Find where the per frame functional groups sequence ends.

        Reads no more of it than it takes to get past it.

        Parameters
        ----------
        per_frame_position: int
            Offset the per frame functional groups sequence starts at.
        """
        reader = self._create_reader(per_frame_position)
        return reader.read_end_of_sequence()

    def _create_reader(self, per_frame_position: int) -> PerFrameFunctionalGroupsReader:
        """Create a reader for the per frame functional groups sequence.

        Parameters
        ----------
        per_frame_position: int
            Offset the per frame functional groups sequence starts at.

        Returns
        -------
        PerFrameFunctionalGroupsReader
            Reader for the sequence at `per_frame_position`.
        """
        number_of_frames = WsiDataset.get_value(self._dataset, NumberOfFramesTag, 1)
        character_set: DataElement | None = self._dataset.get(
            SpecificCharacterSetTag, None
        )
        return PerFrameFunctionalGroupsReader(
            self._io,
            per_frame_position,
            int(number_of_frames),
            specific_character_set=(
                character_set.value if character_set is not None else None
            ),
        )

    def _read_elements_until_pixel_data(self, end_of_per_frame_groups: int) -> int:
        """Read what lies between the per frame groups and the pixel data.

        The attributes there are read into the dataset.

        Does nothing once the offset is known, which is from the start for an
        instance with no sequence to read past.

        Returns
        -------
        int
            Offset the pixel data starts at.

        Raises
        ------
        ValueError
            If the sequence has not been read past, so where to read from is unknown.
        """
        if self._pixel_data_position is not None:
            return self._pixel_data_position
        self._io.read_dataset_from(
            end_of_per_frame_groups, ExtendedOffsetTableTag, self._dataset
        )
        self._pixel_data_position = self._io.tell()
        return self._pixel_data_position
