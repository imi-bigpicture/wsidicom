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

"""Writers for WSI DICOM pyramid and group files."""

import contextlib
import itertools
import logging
import os
import shutil
import tempfile
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from threading import Thread
from typing import Any, BinaryIO

from pydicom.uid import UID
from upath import UPath

from wsidicom.codec import Encoder
from wsidicom.downsampler import Downsampler
from wsidicom.file.io import OffsetTableType, WsiDicomWriter
from wsidicom.geometry import Size
from wsidicom.group import Instances, Label, Level, Overview, Thumbnail
from wsidicom.instance import ImageData, WsiDataset, WsiInstance
from wsidicom.metadata import ImageType, WsiMetadata
from wsidicom.metadata.schema.dicom.wsi import WsiMetadataDicomSchema
from wsidicom.metadata.uid_generator import UidGenerator
from wsidicom.options import (
    ConcatenationByBytes,
    ConcatenationByFrames,
    InstanceSplit,
)
from wsidicom.series import Pyramid
from wsidicom.stitcher import NumpyStitcher
from wsidicom.thread import CancellationToken, Cancelled, ReadExecutor
from wsidicom.writing.encoder_pool import EncoderPool
from wsidicom.writing.instance_writers import (
    GeneratedPyramidLevelWriter,
    GroupInstanceWriter,
    PyramidLevelWriter,
    SourcePyramidLevelWriter,
)
from wsidicom.writing.pyramid_tile_accumulator import (
    PyramidTileAccumulator,
    WritingPyramidTileAccumulator,
)
from wsidicom.writing.tile_cache import ByteBudgetTileCache, DictTileCache, TileCache
from wsidicom.writing.tile_readers import (
    CascadingPassthroughTileReader,
    CascadingTranscodeTileReader,
    PassthroughTileReader,
    TileReader,
    TranscodeTileReader,
)


class PartFactory:
    """Builds and opens the instance-file parts of one level.

    Each part gets a copy of the level dataset with its own per-part
    `NumberOfFrames`, `SOPInstanceUID`, and `InstanceNumber`; concatenated parts
    also get the concatenation attributes. Owns the shared `ConcatenationUID` and
    notional source SOP Instance UID, minted on the first concatenated part, so a
    caller decides only *when* to split, not how a part is built.
    """

    def __init__(
        self,
        base_dataset: WsiDataset,
        uid_generator: UidGenerator,
        output_path: UPath,
        file_options: dict[str, Any] | None,
        transfer_syntax: UID,
        offset_table: OffsetTableType,
        instance_counter: Iterator[int],
        part_total: int | None,
    ) -> None:
        """Set up the part factory for one level.

        Parameters
        ----------
        base_dataset: WsiDataset
            The level's dataset that every part is derived from.
        uid_generator: UidGenerator
            Generates each part's SOP Instance UID and the shared
            ConcatenationUID / source UID.
        output_path: UPath
            Directory the part files are written to.
        file_options: dict[str, Any] | None
            Options forwarded to file operations (e.g. fsspec credentials).
        transfer_syntax: UID
            Transfer syntax of the written instances.
        offset_table: OffsetTableType
            Offset table type to use for the pixel data.
        instance_counter: Iterator[int]
            Supplies each part's InstanceNumber.
        part_total: int | None
            Total number of parts (`InConcatenationTotalNumber`), or None when
            it is not known in advance (byte-size splitting).
        """
        self._base_dataset = base_dataset
        self._uid_generator = uid_generator
        self._output_path = output_path
        self._file_options = file_options
        self._transfer_syntax = transfer_syntax
        self._offset_table = offset_table
        self._instance_counter = instance_counter
        self._part_total = part_total
        self._part_number = 0

    @cached_property
    def _concatenation_uid(self) -> UID:
        """The ConcatenationUID shared by all parts, minted on first use."""
        return self._uid_generator.concatenation_uid(self._base_dataset)

    @cached_property
    def _source_uid(self) -> UID:
        """The notional source SOP Instance UID shared by all parts, minted on
        first use."""
        return self._uid_generator.concatenation_source_uid(self._base_dataset)

    def open(
        self, frame_offset: int, frame_count: int, *, concatenated: bool
    ) -> tuple[WsiDicomWriter, WsiDataset]:
        """Build and open one part of `frame_count` frames starting at `frame_offset`.

        When `concatenated`, stamps the concatenation attributes and shares the
        level's ConcatenationUID / source UID (`InConcatenationTotalNumber` is
        omitted when the level's `part_total` is None, e.g. streaming byte splits).

        A concatenated part gets its own deep copy so the shared `base_dataset`
        stays pristine for the remaining parts. A non-concatenated part is the
        sole part of its level, so `base_dataset` is reused in place and the
        (possibly expensive) deepcopy of its nested sequences is skipped.
        """
        if concatenated:
            dataset = WsiDataset(deepcopy(self._base_dataset))
        else:
            dataset = self._base_dataset
        dataset.NumberOfFrames = frame_count
        dataset.SOPInstanceUID = self._uid_generator.sop_uid(dataset)
        dataset.InstanceNumber = next(self._instance_counter)
        if concatenated:
            self._part_number += 1
            dataset.ConcatenationUID = self._concatenation_uid
            dataset.SOPInstanceUIDOfConcatenationSource = self._source_uid
            dataset.InConcatenationNumber = self._part_number
            dataset.ConcatenationFrameOffsetNumber = frame_offset
            if self._part_total is not None:
                dataset.InConcatenationTotalNumber = self._part_total
        writer = WsiDicomWriter.open_instance(
            self._output_path.joinpath(str(dataset.SOPInstanceUID) + ".dcm"),
            self._transfer_syntax,
            self._offset_table,
            self._file_options,
            dataset,
        )
        return writer, dataset


class PartSplitter(metaclass=ABCMeta):
    """Decides where a level's raster-ordered tiles are split into concatenation
    parts, and — when it can — how many frames the next part will hold."""

    @abstractmethod
    def should_start_new_part(self, next_tile: bytes) -> bool:
        """Whether `next_tile` begins a new part (the current one is full).

        Only called when the current part already holds at least one tile.
        """

    @abstractmethod
    def account(self, tile: bytes) -> None:
        """Account for `tile` now in the current part (update the running measure
        `should_start_new_part` reads)."""

    @abstractmethod
    def reset(self) -> None:
        """Reset accounting for a new part."""

    @abstractmethod
    def next_part_frame_count(self, remaining_frames: int) -> int | None:
        """The next part's frame count if fixed in advance (so it can be streamed
        straight to its file), else None (the part must be buffered until full)."""


class NoSplitter(PartSplitter):
    """Never splits: the level is written as a single, non-concatenated instance."""

    def should_start_new_part(self, next_tile: bytes) -> bool:
        return False

    def account(self, tile: bytes) -> None:
        pass

    def reset(self) -> None:
        pass

    def next_part_frame_count(self, remaining_frames: int) -> int | None:
        return remaining_frames


class FrameCountSplitter(PartSplitter):
    """Splits off a new part every `max_frames` frames."""

    def __init__(self, max_frames: int) -> None:
        """Split by frame count.

        Parameters
        ----------
        max_frames: int
            Maximum number of frames per part.
        """
        self._max_frames = max_frames
        self._frames = 0

    def should_start_new_part(self, next_tile: bytes) -> bool:
        return self._frames >= self._max_frames

    def account(self, tile: bytes) -> None:
        self._frames += 1

    def reset(self) -> None:
        self._frames = 0

    def next_part_frame_count(self, remaining_frames: int) -> int | None:
        return min(self._max_frames, remaining_frames)


class ByteSizeSplitter(PartSplitter):
    """Splits off a new part when the next tile would push the current part's
    encapsulated pixel data past `max_bytes` (each frame counted with its 8-byte
    item header)."""

    _ITEM_HEADER_BYTES = 8

    def __init__(self, max_bytes: int) -> None:
        """Split by encapsulated pixel-data size.

        Parameters
        ----------
        max_bytes: int
            Maximum size in bytes of a part's encapsulated pixel data (each frame
            counted with its item header).
        """
        self._max_bytes = max_bytes
        self._bytes = 0

    def should_start_new_part(self, next_tile: bytes) -> bool:
        return self._bytes + len(next_tile) + self._ITEM_HEADER_BYTES > self._max_bytes

    def account(self, tile: bytes) -> None:
        self._bytes += len(tile) + self._ITEM_HEADER_BYTES

    def reset(self) -> None:
        self._bytes = 0

    def next_part_frame_count(self, remaining_frames: int) -> int | None:
        return None


class PartSink(metaclass=ABCMeta):
    """Writes the frames of one concatenation part to its instance file."""

    @abstractmethod
    def write(self, tiles: Iterable[bytes]) -> None:
        """Add a batch of raster-order tiles to the part."""

    @abstractmethod
    def finalize(self, concatenated: bool) -> UPath:
        """Complete the part's instance and return its file path."""

    @abstractmethod
    def close(self) -> None:
        """Best-effort cleanup on error."""


class DirectPartSink(PartSink):
    """Streams a part's tiles straight to an already-open instance file, used when
    the frame count is fixed in advance (frame-count splitting)."""

    def __init__(
        self, writer: WsiDicomWriter, dataset: WsiDataset, transcoder: Encoder | None
    ) -> None:
        """Stream tiles to an already-open instance.

        Parameters
        ----------
        writer: WsiDicomWriter
            The already-opened instance file to stream the part's tiles to.
        dataset: WsiDataset
            The part's dataset, used to finalize the instance.
        transcoder: Encoder | None
            Encoder to transcode tiles with, or None to write them through.
        """
        self._writer = writer
        self._dataset = dataset
        self._transcoder = transcoder

    def write(self, tiles: Iterable[bytes]) -> None:
        self._writer.write_tiles(tiles)

    def finalize(self, concatenated: bool) -> UPath:
        filepath = self._writer.filepath
        assert filepath is not None
        self._writer.finalize(self._dataset, self._transcoder)
        return filepath

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._writer.close()


class BufferedPartSink(PartSink):
    """Buffers a part's raster-order tiles to one sequential scratch file and, on
    finalize, opens the instance with the now-known frame count and writes the
    tiles into it — used when the count is discovered only as tiles arrive
    (byte-size splitting).
    """

    def __init__(
        self,
        temp_dir: UPath,
        part_factory: PartFactory,
        frame_offset: int,
        transcoder: Encoder | None,
    ) -> None:
        """Buffer tiles until the part's frame count is known.

        Parameters
        ----------
        temp_dir: UPath
            Directory for the part's scratch file.
        part_factory: PartFactory
            Opens the instance once the buffered frame count is known.
        frame_offset: int
            Index of the part's first frame within the level.
        transcoder: Encoder | None
            Encoder to transcode tiles with, or None to write them through.
        """
        self._temp_dir = temp_dir
        self._part_factory = part_factory
        self._frame_offset = frame_offset
        self._transcoder = transcoder
        self._file: BinaryIO | None = None
        self._path: Path | None = None
        self._lengths: list[int] = []

    def write(self, tiles: Iterable[bytes]) -> None:
        if self._file is None:
            fd, path = tempfile.mkstemp(
                prefix="concat_", suffix=".bin", dir=str(self._temp_dir)
            )
            self._path = Path(path)
            self._file = os.fdopen(fd, "w+b")
        for tile in tiles:
            self._file.write(tile)
            self._lengths.append(len(tile))

    def finalize(self, concatenated: bool) -> UPath:
        writer, dataset = self._part_factory.open(
            self._frame_offset, len(self._lengths), concatenated=concatenated
        )
        filepath = writer.filepath
        assert filepath is not None
        try:
            writer.write_tiles(self._read())
            writer.finalize(dataset, self._transcoder)
        except BaseException:
            writer.close()
            raise
        finally:
            self.close()
        return filepath

    def _read(self) -> Iterator[bytes]:
        """Yield the buffered tiles back in write order from the scratch file."""
        if self._file is not None:
            self._file.seek(0)
            for length in self._lengths:
                yield self._file.read(length)

    def close(self) -> None:
        self._lengths = []
        if self._file is not None:
            with contextlib.suppress(Exception):
                self._file.close()
            self._file = None
        if self._path is not None:
            with contextlib.suppress(Exception):
                self._path.unlink()
            self._path = None


class InstanceFileWriter:
    """Writes a level to files provided by `part_factory`, splitting into parts where
    `splitter` says.
    """

    def __init__(
        self,
        part_factory: PartFactory,
        splitter: PartSplitter,
        total_frames: int,
        transcoder: Encoder | None,
        temp_dir: UPath,
    ) -> None:
        """Write a level's instance file(s).

        Parameters
        ----------
        part_factory: PartFactory
            Builds and opens each part's instance file.
        splitter: PartSplitter
            Decides where the level's tiles are split into parts.
        total_frames: int
            Total number of frames in the level, used to recognize a part that
            covers the whole level (which is written non-concatenated).
        transcoder: Encoder | None
            Encoder to transcode tiles with, or None to write them through.
        temp_dir: UPath
            Scratch directory for buffered (byte-split) parts.
        """
        self._part_factory = part_factory
        self._splitter = splitter
        self._total_frames = total_frames
        self._transcoder = transcoder
        self._temp_dir = temp_dir
        self._frame_offset = 0
        self._part_frames = 0
        self._did_split = False
        self._filepaths: list[UPath] = []
        self._sink: PartSink | None = None

    @property
    def filepaths(self) -> list[UPath]:
        """The instance file(s) finalized so far, in part order."""
        return list(self._filepaths)

    def write_tiles(self, tiles: Iterable[bytes]) -> int:
        """Batch same-part tiles to the current sink, splitting where the splitter
        says (a batch that crosses a part boundary is split)."""
        count = 0
        batch: list[bytes] = []
        for tile in tiles:
            if self._part_frames > 0 and self._splitter.should_start_new_part(tile):
                self._write_batch(batch)
                batch = []
                self._finalize_part(concatenated=True)
                self._splitter.reset()
            batch.append(tile)
            self._splitter.account(tile)
            self._part_frames += 1
            count += 1
        self._write_batch(batch)
        return count

    def finalize(self) -> None:
        """Finalize the last part (a plain instance if it is the only one)."""
        self._finalize_part(concatenated=self._did_split)

    def close(self) -> None:
        """Best-effort cleanup of the current unfinalized part on error."""
        if self._sink is not None:
            self._sink.close()
            self._sink = None

    def _write_batch(self, batch: list[bytes]) -> None:
        """Write a run of same-part tiles, opening the part's sink on first use."""
        if not batch:
            return
        if self._sink is None:
            self._sink = self._open_sink()
        self._sink.write(batch)

    def _open_sink(self) -> PartSink:
        """Open the next part's sink: a streaming `DirectPartSink` when the
        splitter fixes the frame count in advance, else a buffering
        `BufferedPartSink`."""
        remaining = self._total_frames - self._frame_offset
        expected = self._splitter.next_part_frame_count(remaining)
        if expected is not None:
            # Count fixed in advance: stream straight to the final file. A first
            # part that takes the whole level is the only part, so it is written
            # as a plain, non-concatenated instance.
            concatenated = not (self._frame_offset == 0 and expected == remaining)
            writer, dataset = self._part_factory.open(
                self._frame_offset, expected, concatenated=concatenated
            )
            return DirectPartSink(writer, dataset, self._transcoder)
        # Count unknown: buffer to a temp file, splice on finalize.
        return BufferedPartSink(
            temp_dir=self._temp_dir,
            part_factory=self._part_factory,
            frame_offset=self._frame_offset,
            transcoder=self._transcoder,
        )

    def _finalize_part(self, concatenated: bool) -> None:
        """Finalize the current part, if one is open, and advance the frame offset
        and per-part accounting for the next one."""
        if self._sink is None:
            return
        self._filepaths.append(self._sink.finalize(concatenated))
        self._frame_offset += self._part_frames
        self._part_frames = 0
        self._did_split = True
        self._sink = None


class BaseFileWriter(metaclass=ABCMeta):
    """Shared state and helpers for PyramidFileWriter and GroupFileWriter."""

    def __init__(
        self,
        output_path: str | Path | UPath,
        uid_generator: UidGenerator,
        transcoder: Encoder | None,
        force_transcoding: bool,
        offset_table: OffsetTableType | None,
        file_options: dict[str, Any] | None,
        instance_number_start: int,
        metadata: WsiMetadata | None = None,
        replace_metadata: bool = True,
        instance_split: InstanceSplit = InstanceSplit.NONE,
    ):
        """Initialize shared writer state.

        Parameters
        ----------
        output_path: Union[str, Path, UPath]
            Directory to write output files to.
        uid_generator: UidGenerator
            UID generator for new instances.
        transcoder: Optional[Encoder]
            Encoder for transcoding. If None, source encoding is preserved.
        force_transcoding: bool
            If True, transcode even if transfer syntax matches.
        offset_table: Optional[OffsetTableType]
            Offset table type to use. If None, determined automatically.
        file_options: Optional[Dict[str, Any]]
            Keyword arguments for file operations.
        instance_number_start: int
            Starting instance number for output files.
        metadata: WsiMetadata | None = None
            Optional metadata to use for the written files. See
            `replace_metadata` for how it is applied. When None, the source
            datasets are used as is.
        replace_metadata: bool = True
            Only used when `metadata` is set. If True (default), the output
            datasets are rebuilt from `metadata` combined with the technical
            attributes of the source image data, so that attributes not modeled
            by the metadata schema (including private tags and other unhandled
            attributes) are dropped. If False, `metadata` is instead overlaid on
            top of the source datasets, preserving any attributes it does not
            set.
        instance_split: InstanceSplit = InstanceSplit.NONE
            Controls how optical paths and focal planes are split across output
            instances. See `InstanceSplit`.
        """
        self._output_path = UPath(output_path)
        self._uid_generator = uid_generator
        self._transcoder = transcoder
        self._force_transcoding = force_transcoding
        self._offset_table = offset_table
        self._file_options = file_options
        self._instance_number = instance_number_start
        self._metadata = metadata
        self._replace_metadata = replace_metadata
        self._instance_split = instance_split

    @abstractmethod
    def write(self) -> list[UPath]:
        """Write DICOM files and return paths to created files."""
        raise NotImplementedError()

    def _build_base_dataset(
        self,
        source_instance: WsiInstance,
        image_type: ImageType,
        pyramid_index: int | None = None,
    ) -> WsiDataset:
        """Return the base dataset for an output instance.

        When no metadata is set, the source dataset is returned unchanged.

        When metadata is set and `replace_metadata` is True, the dataset is
        rebuilt from that metadata combined with the technical attributes of the
        source image data. The rebuilt dataset only contains attributes emitted
        by the metadata schema and the technical attributes set from the image
        data; any other attributes present in the source dataset (e.g. private
        tags or unhandled attributes) are not carried over.

        When metadata is set and `replace_metadata` is False, the metadata is
        instead overlaid on top of a copy of the source dataset, preserving any
        attributes the metadata does not set.

        Parameters
        ----------
        source_instance: WsiInstance
            Instance to take the source dataset and image data from.
        image_type: ImageType
            Type of instance to create.
        pyramid_index: int | None = None
            Pyramid index of the image data, required for volume images.

        Returns
        -------
        WsiDataset
            Base dataset to create the output instance from.
        """
        if self._metadata is None:
            return source_instance.dataset
        image_data = source_instance.image_data
        # ICC Profile (0028,2000) is Type 1C in the Optical Path Module, required
        # when Photometric Interpretation is not MONOCHROME2 (DICOM PS3.3
        # C.8.12.5). Insert a default profile where it is required but missing; a
        # profile already present in the metadata is kept.
        require_icc_profile = image_data.photometric_interpretation != "MONOCHROME2"
        dumped = WsiMetadataDicomSchema().dump(
            self._metadata, image_type, require_icc_profile
        )
        if self._replace_metadata:
            return WsiDataset.create_instance_dataset(
                dumped, image_type, image_data, pyramid_index
            )
        # Overlay the supplied metadata on a copy of the source dataset. Wrapping
        # the copy in a new WsiDataset resets cached properties so they reflect
        # the updated attributes.
        dataset = WsiDataset(deepcopy(source_instance.dataset))
        dataset.update(dumped)
        return dataset

    def _resolve_transcoding(
        self, source_image_data: ImageData
    ) -> tuple[Encoder, bool]:
        """Resolve encoder and whether transcoding is needed.

        Returns
        -------
        Tuple[Encoder, bool]
            Encoder to use and whether transcoding from source is needed.
        """
        if self._transcoder is not None and (
            self._force_transcoding
            or source_image_data.transfer_syntax != self._transcoder.transfer_syntax
        ):
            transcoder = self._transcoder
        elif source_image_data.transcoder is not None:
            transcoder = source_image_data.transcoder
        else:
            transcoder = None

        if transcoder is not None:
            if (
                transcoder.bits != source_image_data.bits
                or transcoder.samples_per_pixel != source_image_data.samples_per_pixel
            ):
                raise ValueError(
                    "Transcode settings must match image data bits and "
                    "photometric interpretation."
                )
            return transcoder, True
        return source_image_data.encoder, False

    def _resolve_offset_table(self, transfer_syntax: UID) -> OffsetTableType:
        """Resolve offset table type from settings or transfer syntax."""
        if self._offset_table is not None:
            return self._offset_table
        if transfer_syntax.is_encapsulated:
            return OffsetTableType.BASIC
        return OffsetTableType.NONE

    def _split_planes_paths(
        self,
        focal_planes_by_optical_path: dict[str, list[float]],
    ) -> list[tuple[list[float], list[str]]]:
        """Split planes and paths into one (planes, paths) bucket per instance.

        The split is determined by the configured `InstanceSplit`, but each
        bucket is always a complete and encodable TILED_FULL grid: optical paths
        are split when they do not all share the same focal planes (a sparse
        grid), and focal planes are split when they are not equally spaced.

        Parameters
        ----------
        focal_planes_by_optical_path: dict[str, list[float]]
            The focal planes present for each optical path of the source group.

        Returns
        -------
        list[tuple[list[float], list[str]]]
            One (focal planes, optical paths) pair per instance to write.
        """
        optical_paths = list(focal_planes_by_optical_path)
        # A sparse grid (optical paths with differing focal planes) cannot share
        # one TILED_FULL instance, so split optical paths even if not requested.
        grid_is_sparse = any(
            focal_planes_by_optical_path[optical_path]
            != focal_planes_by_optical_path[optical_paths[0]]
            for optical_path in optical_paths
        )
        split_optical_paths = (
            bool(self._instance_split & InstanceSplit.OPTICAL_PATH) or grid_is_sparse
        )
        if split_optical_paths:
            path_groups = [
                ([optical_path], focal_planes_by_optical_path[optical_path])
                for optical_path in optical_paths
            ]
        else:
            path_groups = [
                (optical_paths, focal_planes_by_optical_path[optical_paths[0]])
            ]

        # Unequally spaced focal planes cannot share one TILED_FULL instance, so
        # split them per focal plane even if not requested.
        buckets: list[tuple[list[float], list[str]]] = []
        for paths, focal_planes in path_groups:
            if self._instance_split & InstanceSplit.FOCAL_PLANE or (
                not WsiDataset.focal_planes_equally_spaced(focal_planes)
            ):
                buckets.extend(([focal_plane], paths) for focal_plane in focal_planes)
            else:
                buckets.append((focal_planes, paths))
        return buckets


class PyramidFileWriter(BaseFileWriter):
    """Writes pyramid levels to DICOM files, generating missing levels on demand.

    Source levels are read from the input pyramid; missing dyadic levels are
    produced by cascading downsamples from the level below. Each level is
    written to its own DICOM file via a `WsiDicomWriter` sink.

    Level classification:
    - Isolated passthrough: not chained to another level, no transcoding.
      Sequential write, no TileSequencer.
    - Isolated transcoding: not chained to another level, but encoding needed.
      Submit to shared pool, with TileSequencer.
    - Pyramid chain: linked to the level above and/or below. TileSequencer on
      all levels, leaf driven via shared pool.
    """

    def __init__(
        self,
        pyramid: Pyramid,
        output_path: str | Path | UPath,
        uid_generator: UidGenerator,
        max_threads: int = 16,
        offset_table: OffsetTableType | None = None,
        transcoder: Encoder | None = None,
        force_transcoding: bool = False,
        include_levels: Sequence[int] | None = None,
        add_missing_levels: bool = True,
        regenerate_pyramid: bool = False,
        file_options: dict[str, Any] | None = None,
        instance_number_start: int = 1,
        queue_maxsize: int = 100,
        memory_budget_bytes: int | None = None,
        source_workers: int | None = None,
        chunk_size: int | None = None,
        metadata: WsiMetadata | None = None,
        replace_metadata: bool = True,
        instance_split: InstanceSplit = InstanceSplit.NONE,
        concatenation: ConcatenationByFrames | ConcatenationByBytes | None = None,
    ):
        """Create a pull-based pyramid writer.

        Parameters
        ----------
        pyramid: Pyramid
            Source pyramid to generate from.
        output_path: Union[str, Path, UPath]
            Directory to write output files to.
        uid_generator: UidGenerator
            UID generator for new instances.
        max_threads: int
            Maximum threads in the shared pool across all levels.
        offset_table: Optional[OffsetTableType]
            Offset table type to use. If None, determined automatically.
        transcoder: Optional[Encoder]
            Encoder for transcoding. If None, source encoding is preserved.
        force_transcoding: bool
            If True, transcode even if transfer syntax matches.
        include_levels: Optional[Sequence[int]]
            Optional indices of levels to include.
        add_missing_levels: bool
            If True, generate missing dyadic levels up to the single tile
            level. If False, only write levels present in the source pyramid.
        regenerate_pyramid: bool
            If True, only the base level is read from the source; every other
            written level is re-derived by downsampling from the base instead
            of being read from the source's stored pyramid. Orthogonal to
            `add_missing_levels`, which independently controls whether the
            output extends up to the single tile level.
        file_options: Optional[Dict[str, Any]]
            Keyword arguments for file operations.
        instance_number_start: int
            Starting instance number for output files.
        queue_maxsize: int
            Maximum size of the TileSequencer priority queue per level.
        memory_budget_bytes: Optional[int]
            Byte budget for the memory tier of the tile cache. When set,
            uses ByteBudgetTileCache instead of DictTileCache.
        source_workers: Optional[int]
            Number of worker threads for source tile reading. Defaults to
            None (= use max_threads). Set to 1 for thread-unsafe sources
            (e.g. iSyntax/Philips SDK) — this runs source reads inline on
            the calling thread while encoding and downsampling still run
            in parallel.
        chunk_size: Optional[int]
            Per-batch tile width hint for source tile reading. When None,
            each source's `ImageData.suggested_minimum_chunk_size` is used.
            Hard floor of 2.
        metadata: WsiMetadata | None = None
            Optional metadata to apply to the output datasets. See
            `BaseFileWriter`.
        replace_metadata: bool = True
            Whether to replace or overlay the source datasets with `metadata`.
            See `BaseFileWriter`.
        instance_split: InstanceSplit = InstanceSplit.NONE
            How optical paths and focal planes are split across output instances.
            See `BaseFileWriter`.
        concatenation: ConcatenationByFrames | ConcatenationByBytes | None = None
            If set, split each pyramid level into concatenated instances by frame
            count (`ConcatenationByFrames`) or byte size (`ConcatenationByBytes`).
            Orthogonal to `instance_split`. None (default) writes one instance
            per level.
        """
        super().__init__(
            output_path=output_path,
            uid_generator=uid_generator,
            transcoder=transcoder,
            force_transcoding=force_transcoding,
            offset_table=offset_table,
            file_options=file_options,
            instance_number_start=instance_number_start,
            metadata=metadata,
            replace_metadata=replace_metadata,
            instance_split=instance_split,
        )
        self._pyramid = pyramid
        self._concatenation = concatenation
        self._max_threads = max_threads
        self._include_levels = include_levels
        self._add_missing_levels = add_missing_levels
        self._regenerate_pyramid = regenerate_pyramid
        self._queue_maxsize = queue_maxsize
        self._memory_budget_bytes = memory_budget_bytes
        self._source_workers = source_workers
        self._chunk_size = chunk_size

    def write(self) -> list[UPath]:
        """Write the pyramid files.

        Returns
        -------
        List[UPath]
            Paths to created DICOM files, one per level.
        """
        # Get source info
        base_instance = list(self._pyramid.base_level.instances.values())[0]
        source_image_data = base_instance.image_data

        # Resolve encoder and whether transcoding is needed
        encoder, transcode = self._resolve_transcoding(source_image_data)

        downsampler = Downsampler.create_for_pyramid()
        stitcher = NumpyStitcher()
        temp_dir = UPath(tempfile.mkdtemp(prefix="wsidicom_pull_"))

        token = CancellationToken()
        encoder_pool = EncoderPool(
            encoder=encoder,
            num_workers=self._max_threads,
            downsampler=downsampler,
            stitcher=stitcher,
            tile_size=self._pyramid.base_level.tile_size,
            blank_tile=source_image_data.blank_tile,
            token=token,
        )
        try:
            level_writers: list[PyramidLevelWriter] = []
            file_writers: list[InstanceFileWriter] = []
            try:
                level_writers = self._build_level_writers(
                    self._pyramid.pyramid_indices,
                    encoder,
                    transcode,
                    encoder_pool,
                    temp_dir,
                    token,
                )
                instance_counter = itertools.count(self._instance_number)
                for level_writer in level_writers:
                    if (
                        isinstance(level_writer, SourcePyramidLevelWriter)
                        and not transcode
                    ):
                        level_transfer_syntax = source_image_data.transfer_syntax
                    else:
                        level_transfer_syntax = encoder.transfer_syntax
                    file_writer = self._open_writer(
                        level_writer,
                        instance_counter,
                        level_transfer_syntax,
                        self._resolve_offset_table(level_transfer_syntax),
                        encoder if transcode else None,
                        temp_dir,
                    )
                    file_writers.append(file_writer)
                    level_writer.start(file_writer)
                encoder_pool.start()

                source_level_writers = [
                    level_writer
                    for level_writer in level_writers
                    if isinstance(level_writer, SourcePyramidLevelWriter)
                ]
                source_pool_workers = self._source_workers or self._max_threads
                if not all(
                    image_data.thread_safe
                    for level_writer in source_level_writers
                    for image_data in level_writer.source_image_data
                ):
                    source_pool_workers = 1
                with ThreadPoolExecutor(
                    max_workers=source_pool_workers,
                    thread_name_prefix="SourceReader",
                ) as source_pool:
                    pool = ReadExecutor(source_pool_workers, source_pool)
                    if source_pool_workers == 1:
                        for level_writer in source_level_writers:
                            level_writer.run(pool, self._max_threads * 2)
                    else:
                        self._run_source_level_writers_in_threads(
                            source_level_writers, pool, token
                        )

                self._finalize_writers(
                    level_writers,
                    file_writers,
                    encoder_pool,
                    token,
                )
            except BaseException as error:
                # Cancel the shared token so every stage's blocking queue waits
                # unblock, then re-raise the root cause (the first failure
                # recorded by a worker, if any) rather than a downstream symptom.
                token.cancel(error)
                self._cleanup_writers(level_writers, file_writers, encoder_pool)
                raise token.exception or error from None

            filepaths: list[UPath] = []
            for file_writer in file_writers:
                filepaths.extend(file_writer.filepaths)
            return filepaths

        finally:
            self._cleanup_temp(temp_dir)

    def _validate_uniform_levels(self, present_levels: Sequence[int]) -> None:
        """Raise if present levels differ in optical paths / focal planes.

        Instance splitting builds one cascade chain per (optical path, focal
        plane) bucket spanning all levels, so every present level must share the
        same membership. Thumbnails, labels, and overviews are separate groups
        and are not affected.

        Parameters
        ----------
        present_levels: Sequence[int]
            Indices of the levels present in the source pyramid.
        """
        base_membership = self._pyramid.base_level.focal_planes_by_optical_path
        for level_index in present_levels:
            membership = self._pyramid.get(level_index).focal_planes_by_optical_path
            if membership != base_membership:
                raise NotImplementedError(
                    "Pyramid levels have differing optical paths / focal planes; "
                    "instance splitting across non-uniform levels is not supported."
                )

    def _build_level_writers(
        self,
        present_levels: Sequence[int],
        encoder: Encoder,
        transcode: bool,
        encoder_pool: EncoderPool,
        temp_dir: UPath,
        token: CancellationToken,
    ) -> list[PyramidLevelWriter]:
        """Build PyramidLevelWriters for all included levels.

        Iterates levels from highest to lowest. Generated levels are created
        first (with accumulator chain wiring), then source levels get tile
        readers referencing the next generated level above them.
        """
        self._validate_uniform_levels(present_levels)

        # Determine which levels to build
        highest_in_file = present_levels[-1]
        if self._add_missing_levels:
            lowest_single_tile = self._pyramid.lowest_single_tile_level
            highest_level = max(highest_in_file, lowest_single_tile)
            candidate_levels = list(range(highest_level + 1))
        else:
            highest_level = highest_in_file
            candidate_levels = list(present_levels)
        selected_levels = self._select_included_levels(
            candidate_levels, self._include_levels
        )
        included_levels = [
            level_index
            for level_index in range(highest_level + 1)
            if level_index in selected_levels
        ]

        # Which levels are read from the source. Normally every natively
        # present level; when regenerating, only the base, so all other written
        # levels are re-derived by downsampling from it. The base must be read
        # to feed that downsampling, so it has to be among the included levels
        # whenever any higher level is written (we only read what we write).
        base_level_index = present_levels[0]
        if self._regenerate_pyramid:
            source_levels = {base_level_index}
            if (
                any(level != base_level_index for level in included_levels)
                and base_level_index not in included_levels
            ):
                raise ValueError(
                    "regenerate_pyramid requires the base level to be included "
                    "(it is the source every other level is downsampled from); "
                    f"include_levels selected {included_levels} without the "
                    f"base level {base_level_index}."
                )
        else:
            source_levels = set(present_levels)

        # Levels actually read from the source: those it stores natively that are
        # also written. A stored level that is not written is not read either, and
        # is re-derived by downsampling like any other generated level.
        read_levels = source_levels & set(included_levels)

        # Each generated level is downsampled from the level directly below it, so
        # the cascade needs an accumulator for every level between a written level
        # and the level it is derived from. A pyramid whose levels are not
        # consecutive leaves gaps, and regenerating widens them further by always
        # deriving from the base. The levels in the gaps are built to bridge the
        # cascade but are not written.
        cascade_levels: set[int] = set()
        for level_index in included_levels:
            if level_index in read_levels:
                continue
            source_level = self._source_level_of(
                level_index, base_level_index, read_levels, self._regenerate_pyramid
            )
            cascade_levels.update(range(source_level + 1, level_index + 1))
        build_levels = sorted(set(included_levels) | cascade_levels)

        # One reversed pass per instance-split bucket. Each bucket holds a
        # subset of the focal planes / optical paths and gets its own
        # independent cascade chain, producing one instance per written level.
        tile_cache = self._create_tile_cache(temp_dir)
        level_writers: list[PyramidLevelWriter] = []
        base_group = self._pyramid.base_level
        for planes, paths in self._split_planes_paths(
            base_group.focal_planes_by_optical_path
        ):
            next_accumulator: PyramidTileAccumulator | None = None
            bucket_writers: dict[int, PyramidLevelWriter] = {}
            # Accumulators for the levels that are written, and for the levels that
            # only bridge a gap in the cascade to reach them.
            writing_accumulators: dict[int, WritingPyramidTileAccumulator] = {}
            intermediate_accumulators: dict[int, PyramidTileAccumulator] = {}
            # Written generated levels, as (level_index, dataset, tiled_size). The
            # writers are built after the loop, once every accumulator below them
            # exists.
            generated: list[tuple[int, WsiDataset, Size]] = []
            for level_index in reversed(build_levels):
                in_source = level_index in read_levels
                is_written = level_index in included_levels
                if in_source:
                    source_group = self._pyramid.get(level_index)
                    scale = 1
                else:
                    source_level = self._source_level_of(
                        level_index,
                        base_level_index,
                        read_levels,
                        self._regenerate_pyramid,
                    )
                    source_group = self._pyramid.get(source_level)
                    scale = int(2 ** (level_index - source_level))

                if not is_written:
                    # Bridges a gap in the cascade; no instance is written for it,
                    # so it needs neither a dataset nor an output queue. It is
                    # wired up below.
                    intermediate = PyramidTileAccumulator(
                        level_index=level_index,
                        input_tiled_size=source_group.tiled_size.ceil_div(
                            max(scale // 2, 1)
                        ),
                        encoder_pool_queue=encoder_pool.queue,
                        next_accumulator=next_accumulator,
                        is_chain_start=level_index - 1 in read_levels,
                        queue_maxsize=self._queue_maxsize,
                        token=token,
                    )
                    intermediate_accumulators[level_index] = intermediate
                    next_accumulator = intermediate
                    continue

                tiled_size = source_group.tiled_size.ceil_div(scale)
                base_instance = source_group.instance_at(paths[0], planes[0])
                base_dataset = self._build_base_dataset(
                    base_instance, ImageType.VOLUME, level_index
                )
                dataset = base_dataset.as_tiled_full(
                    planes,
                    paths,
                    source_group.tiled_size,
                    scale,
                )
                transcoder = encoder if transcode else None
                if transcoder is not None:
                    dataset.update_for_transcoding(transcoder, scale)

                if in_source:
                    # Source level: create tile reader, referencing next
                    # accumulator for cascading if present
                    tile_reader: TileReader
                    if transcode and next_accumulator is not None:
                        tile_reader = CascadingTranscodeTileReader(
                            level_index,
                            encoder_pool.queue,
                            next_accumulator,
                            token,
                        )
                    elif transcode:
                        tile_reader = TranscodeTileReader(
                            level_index,
                            encoder_pool.queue,
                            token,
                        )
                    elif next_accumulator is not None:
                        tile_reader = CascadingPassthroughTileReader(
                            level_index,
                            next_accumulator,
                            token,
                        )
                    else:
                        tile_reader = PassthroughTileReader(level_index, token)
                    bucket_writers[level_index] = SourcePyramidLevelWriter(
                        level_index=level_index,
                        dataset=dataset,
                        tile_cache=tile_cache,
                        source_group=source_group,
                        tiled_size=tiled_size,
                        tile_reader=tile_reader,
                        queue_maxsize=self._queue_maxsize,
                        chunk_size=self._chunk_size,
                        focal_planes=planes,
                        optical_paths=paths,
                        token=token,
                    )
                    next_accumulator = None
                else:
                    # Generated level: create the accumulator, chaining it to the
                    # previously-created one (the level above in the pyramid). The
                    # input tiled size is that of the level below (scale / 2). The
                    # writer is built after the loop, once the accumulators for any
                    # intermediate levels below it exist.
                    accumulator = WritingPyramidTileAccumulator(
                        level_index=level_index,
                        input_tiled_size=source_group.tiled_size.ceil_div(
                            max(scale // 2, 1)
                        ),
                        encoder_pool_queue=encoder_pool.queue,
                        next_accumulator=next_accumulator,
                        is_chain_start=level_index - 1 in read_levels,
                        queue_maxsize=self._queue_maxsize,
                        token=token,
                    )
                    writing_accumulators[level_index] = accumulator
                    generated.append((level_index, dataset, tiled_size))
                    next_accumulator = accumulator

            for generated_level, generated_dataset, generated_tiled_size in generated:
                bucket_writers[generated_level] = GeneratedPyramidLevelWriter(
                    level_index=generated_level,
                    dataset=generated_dataset,
                    tile_cache=tile_cache,
                    focal_planes=planes,
                    optical_paths=paths,
                    tiled_size=generated_tiled_size,
                    accumulator=writing_accumulators[generated_level],
                    token=token,
                    intermediate_accumulators=[
                        intermediate_accumulators[level]
                        for level in self._intermediate_levels_of(
                            generated_level, intermediate_accumulators
                        )
                    ],
                )

            level_writers.extend(
                bucket_writers[level_index] for level_index in sorted(bucket_writers)
            )

        return level_writers

    @staticmethod
    def _source_level_of(
        level_index: int,
        base_level_index: int,
        read_levels: set[int],
        regenerate_pyramid: bool,
    ) -> int:
        """Return the level a generated level is downsampled from.

        When regenerating, always the base. Otherwise the closest level below it
        that is actually read from the source: a level the source stores but that
        is not written is not read either, so it cannot be downsampled from.

        Parameters
        ----------
        level_index: int
            The generated level.
        base_level_index: int
            The lowest level present in the source.
        read_levels: set[int]
            Levels read from the source.
        regenerate_pyramid: bool
            If every level is re-derived from the base.

        Returns
        -------
        int
            Level to downsample `level_index` from.
        """
        if regenerate_pyramid:
            return base_level_index
        levels_below = [level for level in read_levels if level < level_index]
        if not levels_below:
            raise ValueError(
                f"Level {level_index} is generated by downsampling, but no level "
                "below it is read from the source to downsample from. Levels read: "
                f"{sorted(read_levels)}."
            )
        return max(levels_below)

    @staticmethod
    def _intermediate_levels_of(
        level_index: int,
        intermediate_accumulators: dict[int, PyramidTileAccumulator],
    ) -> list[int]:
        """Return the unwritten cascade levels directly below `level_index`.

        These bridge a gap between this level and the level it is derived from.
        They have no writer of their own, so the writer for `level_index` owns
        them. Returned lowest level first, which is the order they must be shut
        down in: the chain start injects the shutdown sentinel, and every
        accumulator above it waits for that sentinel to reach it.

        Parameters
        ----------
        level_index: int
            The written level to find the intermediate levels below.
        intermediate_accumulators: dict[int, PyramidTileAccumulator]
            Accumulators for the unwritten levels, by level.

        Returns
        -------
        list[int]
            Levels directly below `level_index` that are not written, lowest
            first.
        """
        intermediates: list[int] = []
        candidate = level_index - 1
        while candidate in intermediate_accumulators:
            intermediates.append(candidate)
            candidate -= 1
        intermediates.reverse()
        return intermediates

    def _create_tile_cache(self, temp_dir: UPath) -> TileCache:
        """Create a shared tile cache for all levels."""
        if self._memory_budget_bytes is not None:
            return ByteBudgetTileCache(
                temp_dir / "tile_cache",
                memory_budget_bytes=self._memory_budget_bytes,
            )
        return DictTileCache()

    def _open_writer(
        self,
        level_writer: PyramidLevelWriter,
        instance_counter: Iterator[int],
        transfer_syntax: UID,
        offset_table: OffsetTableType,
        transcoder: Encoder | None,
        temp_dir: UPath,
    ) -> InstanceFileWriter:
        """Open the instance file(s) for a level.

        With no concatenation (or when a frame/byte budget already fits the whole
        level) the level is written as one plain, non-concatenated instance.
        `ConcatenationByFrames` splits deterministically into parts of at most N
        frames each (opened up front). `ConcatenationByBytes` accumulates frames
        in raster order and splits off each part when it reaches the byte budget.
        In the split cases `NumberOfFrames` is per part and `TotalPixelMatrix*` is
        the whole-level value; parts share a `ConcatenationUID` and a notional
        source SOP Instance UID. Whether the first part is marked concatenated is
        decided when it is opened (streamed) or finalized (buffered), so a level
        that never splits yields a plain instance regardless of the concatenation
        mode.
        """
        base_dataset = level_writer.dataset
        total_frames = int(base_dataset.NumberOfFrames)
        concatenation = self._concatenation

        if isinstance(concatenation, ConcatenationByFrames):
            splitter: PartSplitter = FrameCountSplitter(concatenation.count)
            # Part count is deterministic for a frame budget, so emit it.
            part_total: int | None = (
                total_frames + concatenation.count - 1
            ) // concatenation.count
        elif isinstance(concatenation, ConcatenationByBytes):
            splitter = ByteSizeSplitter(concatenation.count)
            # Unknown while streaming; InConcatenationTotalNumber is omitted.
            part_total = None
        else:
            # No concatenation: one part, never split, not marked concatenated.
            splitter = NoSplitter()
            part_total = None

        part_factory = PartFactory(
            base_dataset,
            self._uid_generator,
            self._output_path,
            self._file_options,
            transfer_syntax,
            offset_table,
            instance_counter,
            part_total,
        )

        return InstanceFileWriter(
            part_factory=part_factory,
            splitter=splitter,
            total_frames=total_frames,
            transcoder=transcoder,
            temp_dir=temp_dir,
        )

    @staticmethod
    def _finalize_writers(
        level_writers: list[PyramidLevelWriter],
        file_writers: list[InstanceFileWriter],
        encoder_pool: EncoderPool,
        token: CancellationToken,
    ) -> None:
        """Shut down pipeline in correct order and finalize all levels."""
        for level_writer in level_writers:
            if isinstance(level_writer, GeneratedPyramidLevelWriter):
                level_writer.shutdown()

        encoder_pool.shutdown(wait=True)

        # Surface a failure that cancelled the token during shutdown before the
        # per-level finalize reports a (misleading) tile-count mismatch.
        token.raise_if_cancelled()

        for level_writer, concatenation_writer in zip(
            level_writers, file_writers, strict=True
        ):
            level_writer.finalize_writers()
            concatenation_writer.finalize()

    @staticmethod
    def _cleanup_writers(
        level_writers: list[PyramidLevelWriter],
        file_writers: list[InstanceFileWriter],
        encoder_pool: EncoderPool,
    ) -> None:
        """Clean up all resources on error."""
        with contextlib.suppress(Exception):
            encoder_pool.shutdown(wait=False)
        for level_writer in level_writers:
            level_writer.cleanup()
        for concatenation_writer in file_writers:
            concatenation_writer.close()

    def _run_source_level_writers_in_threads(
        self,
        source_level_writers: list[SourcePyramidLevelWriter],
        pool: ReadExecutor,
        token: CancellationToken,
    ) -> None:
        """Run each source level writer in its own thread.

        A producer that fails cancels the shared token (first failure wins); a
        producer that observes an already-cancelled token exits quietly via
        `Cancelled`. After all producers join, the first recorded cause is
        re-raised.
        """

        def make_wrapper(
            level_writer: SourcePyramidLevelWriter,
        ) -> Callable[[], None]:
            def wrapper() -> None:
                try:
                    level_writer.run(pool, self._max_threads * 2)
                except Cancelled:
                    # Token already cancelled by another worker; exit quietly.
                    pass
                except Exception as error:
                    logging.error(
                        f"Producer thread Producer-L{level_writer.level_index} "
                        f"failed: {error}"
                    )
                    token.cancel(error)

            return wrapper

        threads = [
            Thread(
                target=make_wrapper(level_writer),
                name=f"Producer-L{level_writer.level_index}",
            )
            for level_writer in source_level_writers
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        token.raise_if_cancelled()

    @staticmethod
    def _cleanup_temp(temp_dir: UPath) -> None:
        """Remove temporary directory."""
        if temp_dir.exists():
            with contextlib.suppress(OSError):
                shutil.rmtree(str(temp_dir), ignore_errors=True)

    @staticmethod
    def _select_included_levels(
        candidate_levels: Sequence[int],
        include_indices: Sequence[int] | None = None,
    ) -> set[int]:
        """Return the set of pyramid levels to include from candidate_levels.

        Parameters
        ----------
        candidate_levels: Sequence[int]
            Pyramid levels eligible for inclusion. When missing levels may be
            generated, this is the extended level list (``range(0,
            highest_level + 1)``) so that ``include_indices`` selects from
            generated levels too; otherwise it is the natively present levels.
        include_indices: Optional[Sequence[int]] = None
            Optional list of indices (into ``candidate_levels``) to include,
            e.g. ``[0, 1]`` includes the two lowest. Negative indices can be
            used, e.g. ``[-1, -2]`` includes the two highest. Out-of-range
            indices are silently ignored. ``None`` selects all candidates; an
            empty sequence selects none.

        Returns
        -------
        set[int]
            Set of pyramid levels to include.
        """
        if include_indices is None:
            return set(candidate_levels)
        candidate_count = len(candidate_levels)
        valid_indices = [
            index
            for index in include_indices
            if -candidate_count <= index < candidate_count
        ]
        return {candidate_levels[index] for index in valid_indices}


class GroupFileWriter(BaseFileWriter):
    """Writes Label, Overview, and Thumbnail groups to DICOM files.

    Handles transcoder resolution, dataset preparation, and sequential
    tile writing for non-pyramid image groups.
    """

    def __init__(
        self,
        group: Label | Level | Overview | Thumbnail,
        output_path: str | Path | UPath,
        uid_generator: UidGenerator,
        transcoder: Encoder | None,
        force_transcoding: bool,
        offset_table: OffsetTableType | None = None,
        file_options: dict[str, Any] | None = None,
        instance_number_start: int = 1,
        metadata: WsiMetadata | None = None,
        replace_metadata: bool = True,
        instance_split: InstanceSplit = InstanceSplit.NONE,
    ):
        """Create a GroupFileWriter.

        Parameters
        ----------
        group: Union[Label, Level, Overview, Thumbnail]
            Group to write.
        output_path: Union[str, Path, UPath]
            Folder path to save files to.
        uid_generator: UidGenerator
            UID generator to use.
        transcoder: Optional[Encoder]
            Encoder for transcoding. If None, source encoding is preserved.
        force_transcoding: bool
            If True, transcode even if transfer syntax matches.
        offset_table: Optional[OffsetTableType]
            Offset table type to use. If None, determined automatically.
        file_options: Optional[Dict[str, Any]]
            Keyword arguments for file operations.
        instance_number_start: int
            Starting instance number for output files.
        metadata: WsiMetadata | None = None
            Optional metadata to apply to the output datasets. See
            `BaseFileWriter`.
        replace_metadata: bool = True
            Whether to replace or overlay the source datasets with `metadata`.
            See `BaseFileWriter`.
        instance_split: InstanceSplit = InstanceSplit.NONE
            How optical paths and focal planes are split across output instances.
            See `BaseFileWriter`.
        """
        super().__init__(
            output_path=output_path,
            uid_generator=uid_generator,
            transcoder=transcoder,
            force_transcoding=force_transcoding,
            offset_table=offset_table,
            file_options=file_options,
            instance_number_start=instance_number_start,
            metadata=metadata,
            replace_metadata=replace_metadata,
            instance_split=instance_split,
        )
        self._group = group

    def write(self) -> list[UPath]:
        """Write group to DICOM files.

        Returns
        -------
        List[UPath]
            Paths to created DICOM files.
        """
        filepaths: list[UPath] = []
        for sub_group in self._group_instances_to_file(self._group):
            source_image_data = sub_group[0].image_data
            sub_instances = Instances(sub_group)

            encoder, transcode = self._resolve_transcoding(source_image_data)
            transcoder = encoder if transcode else None
            transfer_syntax = (
                encoder.transfer_syntax
                if transcode
                else source_image_data.transfer_syntax
            )
            offset_table = self._resolve_offset_table(transfer_syntax)

            for planes, paths in self._split_planes_paths(
                sub_instances.focal_planes_by_optical_path
            ):
                base_instance = sub_instances.instance_at(paths[0], planes[0])
                base_dataset = self._build_base_dataset(
                    base_instance, base_instance.dataset.image_type
                )
                dataset = base_dataset.as_tiled_full(
                    planes,
                    paths,
                    sub_instances.tiled_size,
                    1,
                )
                if transcoder is not None:
                    dataset.update_for_transcoding(transcoder, 1)

                instance_writer = GroupInstanceWriter(
                    dataset=dataset,
                    instances=sub_instances,
                    encoder=encoder,
                    transcode=transcode,
                    focal_planes=planes,
                    optical_paths=paths,
                )
                dataset.SOPInstanceUID = self._uid_generator.sop_uid(dataset)
                dataset.InstanceNumber = self._instance_number
                filepath = self._output_path.joinpath(
                    str(dataset.SOPInstanceUID) + ".dcm"
                )
                with WsiDicomWriter.open_instance(
                    filepath,
                    transfer_syntax,
                    offset_table,
                    self._file_options,
                    dataset,
                ) as file_writer:
                    instance_writer.write(file_writer)
                    file_writer.finalize(dataset, transcoder)
                self._instance_number += 1
                filepaths.append(filepath)
        return filepaths

    @staticmethod
    def _group_instances_to_file(
        group: Label | Level | Overview | Thumbnail,
    ) -> list[list[WsiInstance]]:
        """Group instances by properties that can't differ in a DICOM-file.

        Parameters
        ----------
        group: Union[Label, Level, Overview, Thumbnail]
            Group to partition.

        Returns
        -------
        List[List[WsiInstance]]
            Instances grouped by common properties.
        """
        groups: dict[
            tuple[str, UID, bool, int | None, float | None, str],
            list[WsiInstance],
        ] = defaultdict(list)

        for instance in group.instances.values():
            groups[
                instance.image_data.photometric_interpretation,
                instance.image_data.transfer_syntax,
                instance.ext_depth_of_field,
                instance.ext_depth_of_field_planes,
                instance.ext_depth_of_field_plane_distance,
                instance.focus_method,
            ].append(instance)
        return list(groups.values())
