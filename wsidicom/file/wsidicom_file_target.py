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

"""A target for writing WSI DICOM files to disk."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import (
    Any,
)

from upath import UPath

from wsidicom.codec import Encoder
from wsidicom.codec import Settings as EncoderSettings
from wsidicom.file.file_writer import (
    BaseFileWriter,
    GroupFileWriter,
    PyramidFileWriter,
)
from wsidicom.file.io import OffsetTableType
from wsidicom.group import Label, Level, Overview, Thumbnail
from wsidicom.metadata import WsiMetadata
from wsidicom.metadata.uid_generator import UidGenerator
from wsidicom.options import (
    ConcatenationByBytes,
    ConcatenationByFrames,
    InstanceSplit,
)
from wsidicom.series import Labels, Overviews, Pyramids
from wsidicom.target import Target


class WsiDicomFileTarget(Target):
    """Target for writing WSI DICOM instances to disk."""

    def __init__(
        self,
        output_path: str | Path | UPath,
        uid_generator: UidGenerator,
        workers: int,
        chunk_size: int | None = None,
        offset_table: OffsetTableType | None = None,
        include_pyramids: Sequence[int] | None = None,
        include_levels: Sequence[int] | None = None,
        add_missing_levels: bool = False,
        regenerate_pyramid: bool = False,
        transcoding: EncoderSettings | Encoder | None = None,
        force_transcoding: bool = False,
        file_options: dict[str, Any] | None = None,
        metadata: WsiMetadata | None = None,
        replace_metadata: bool = True,
        instance_split: InstanceSplit = InstanceSplit.NONE,
        concatenation: ConcatenationByFrames | ConcatenationByBytes | None = None,
    ):
        """
        Create a WsiDicomFileTarget.

        Parameters
        ----------
        output_path: str | Path | UPath
            Folder path to save files to.
        uid_generator: UidGenerator
            Generator for producing UIDs.
        workers: int
            Maximum number of thread workers to use.
        chunk_size: Optional[int] = None
            Per-batch tile width hint for source tile reading. When None,
            each source's `ImageData.suggested_minimum_chunk_size` is used.
        offset_table: Optional[OffsetTableType]
            Offset table to use. If None, determined automatically.
        include_pyramids: Optional[Sequence[int]] = None
            Optional list indices (in present pyramids) to include.
        include_levels: Sequence[int] | None = None
            Optional list indices (in all pyramids) to include, e.g. [0, 1]
            includes the two lowest levels. Negative indices can be used,
            e.g. [-1, -2] includes the two highest levels.
        add_missing_levels: bool
            If to add missing dyadic levels up to the single tile level.
        regenerate_pyramid: bool
            If to re-derive every non-base level by downsampling from the base
            level instead of reading the source's stored pyramid. Orthogonal to
            `add_missing_levels`.
        transcoding: EncoderSettings | Encoder | None = None,
            Optional settings or encoder for transcoding image data. If None, image data
            will be copied as is.
        force_transcoding: bool
            If to force transcoding even if transfer syntax already matches the encoding
            settings.
        file_options: dict[str, Any] | None = None
            Keyword arguments for saving files to output path.
        metadata: WsiMetadata | None = None
            Optional metadata to apply to the written files. See
            `replace_metadata` for how it is applied.
        replace_metadata: bool = True
            Only used when `metadata` is set. If True (default), the output
            datasets are rebuilt from `metadata` combined with the technical
            attributes of the source image data, dropping any attributes not
            modeled by the metadata schema (e.g. private tags). If False,
            `metadata` is overlaid on the source datasets instead.
        instance_split: InstanceSplit = InstanceSplit.NONE
            Controls how optical paths and focal planes are split across output
            instances. See `InstanceSplit`.
        concatenation: ConcatenationByFrames | ConcatenationByBytes | None = None
            If set, split each pyramid level into concatenated instances by frame
            count (`ConcatenationByFrames`) or byte size (`ConcatenationByBytes`).
        """
        self._output_path = UPath(output_path)
        self._offset_table = offset_table
        self._filepaths: list[UPath] = []
        self._file_options = file_options
        self._chunk_size = chunk_size
        self._metadata = metadata
        self._replace_metadata = replace_metadata
        self._instance_split = instance_split
        self._concatenation = concatenation
        super().__init__(
            uid_generator,
            workers,
            include_pyramids,
            include_levels,
            add_missing_levels,
            regenerate_pyramid,
            transcoding,
            force_transcoding,
        )

    @property
    def filepaths(self) -> list[UPath]:
        """Return filepaths for created files."""
        return self._filepaths

    def save(
        self,
        pyramids: Pyramids,
        labels: Labels | None,
        overviews: Overviews | None,
        include_thumbnails: bool,
    ) -> None:
        """Save pyramids, labels, and overviews to target.

        Parameters
        ----------
        pyramids: Pyramids
            Pyramids to save.
        labels: Optional[Labels]
            Labels to save, or None to skip.
        overviews: Optional[Overviews]
            Overviews to save, or None to skip.
        include_thumbnails: bool
            If to include thumbnails from pyramids.
        """
        for writer in self._collect_writers(
            pyramids, labels, overviews, include_thumbnails
        ):
            filepaths = writer.write()
            self._filepaths.extend(filepaths)
            self._instance_number += len(filepaths)

    def close(self) -> None:
        pass

    def _collect_writers(
        self,
        pyramids: Pyramids,
        labels: Labels | None,
        overviews: Overviews | None,
        include_thumbnails: bool,
    ) -> Iterator[BaseFileWriter]:
        """Collect all writers needed for the save operation."""
        if self._include_pyramids is not None:
            pyramids_to_save = [pyramids[index] for index in self._include_pyramids]
        else:
            pyramids_to_save = pyramids

        for pyramid in pyramids_to_save:
            yield PyramidFileWriter(
                pyramid=pyramid,
                output_path=self._output_path,
                uid_generator=self._uid_generator,
                max_threads=self._workers,
                offset_table=self._offset_table,
                transcoder=self._transcoder,
                force_transcoding=self._force_transcoding,
                include_levels=self._include_levels,
                add_missing_levels=self._add_missing_levels,
                regenerate_pyramid=self._regenerate_pyramid,
                file_options=self._file_options,
                instance_number_start=self._instance_number,
                chunk_size=self._chunk_size,
                metadata=self._metadata,
                replace_metadata=self._replace_metadata,
                instance_split=self._instance_split,
                concatenation=self._concatenation,
            )
            if include_thumbnails and pyramid.thumbnails is not None:
                for group in pyramid.thumbnails.groups:
                    yield self._make_group_writer(group)

        if overviews is not None:
            for overview in overviews:
                yield self._make_group_writer(overview)

        if labels is not None:
            for label in labels:
                yield self._make_group_writer(label)

    def _make_group_writer(
        self, group: Label | Level | Overview | Thumbnail
    ) -> GroupFileWriter:
        """Create a GroupFileWriter for a group."""
        return GroupFileWriter(
            group=group,
            output_path=self._output_path,
            uid_generator=self._uid_generator,
            transcoder=self._transcoder,
            force_transcoding=self._force_transcoding,
            offset_table=self._offset_table,
            file_options=self._file_options,
            instance_number_start=self._instance_number,
            metadata=self._metadata,
            replace_metadata=self._replace_metadata,
            instance_split=self._instance_split,
        )
