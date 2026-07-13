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
import shutil
import tempfile
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Thread
from typing import Any

from pydicom.uid import UID
from upath import UPath

from wsidicom.codec import Encoder
from wsidicom.config import settings
from wsidicom.downsampler import PillowDownsampler
from wsidicom.file.instance_split import InstanceSplit
from wsidicom.file.io import OffsetTableType, WsiDicomWriter
from wsidicom.group import Instances, Label, Level, Overview, Thumbnail
from wsidicom.instance import ImageData, WsiDataset, WsiInstance
from wsidicom.metadata import ImageType, WsiMetadata
from wsidicom.metadata.schema.dicom.wsi import WsiMetadataDicomSchema
from wsidicom.metadata.uid_generator import UidGenerator
from wsidicom.series import Pyramid
from wsidicom.stitcher import PillowStitcher
from wsidicom.thread import CancellationToken, Cancelled, ReadExecutor
from wsidicom.writing.encoder_pool import EncoderPool
from wsidicom.writing.instance_writers import (
    GeneratedPyramidLevelWriter,
    GroupInstanceWriter,
    PyramidLevelWriter,
    SourcePyramidLevelWriter,
)
from wsidicom.writing.pyramid_tile_accumulator import PyramidTileAccumulator
from wsidicom.writing.tile_cache import ByteBudgetTileCache, DictTileCache, TileCache
from wsidicom.writing.tile_readers import (
    CascadingPassthroughTileReader,
    CascadingTranscodeTileReader,
    PassthroughTileReader,
    TileReader,
    TranscodeTileReader,
)


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

        downsampler = PillowDownsampler(resample=settings.pillow_resampling_filter)
        stitcher = PillowStitcher()
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
            file_writers: list[WsiDicomWriter] = []
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
                    file_writer = self._prepare_writer(
                        level_writer,
                        next(instance_counter),
                        level_transfer_syntax,
                        self._resolve_offset_table(level_transfer_syntax),
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
                    encoder if transcode else None,
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
                filepath = file_writer.filepath
                assert filepath is not None  # always set once the writer is opened
                filepaths.append(filepath)
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

        # One reversed pass per instance-split bucket. Each bucket holds a
        # subset of the focal planes / optical paths and gets its own
        # independent cascade chain, producing one instance per level.
        tile_cache = self._create_tile_cache(temp_dir)
        level_writers: list[PyramidLevelWriter] = []
        base_group = self._pyramid.base_level
        for planes, paths in self._split_planes_paths(
            base_group.focal_planes_by_optical_path
        ):
            next_accumulator: PyramidTileAccumulator | None = None
            bucket_writers: list[PyramidLevelWriter] = []
            for level_index in reversed(included_levels):
                in_source = level_index in source_levels
                if in_source:
                    source_group = self._pyramid.get(level_index)
                    scale = 1
                elif self._regenerate_pyramid:
                    # Re-derive from the base, not from any native intermediate
                    # level the source happens to store.
                    source_group = self._pyramid.get(base_level_index)
                    scale = int(2 ** (level_index - base_level_index))
                else:
                    source_group = self._pyramid.get_closest_by_level(level_index)
                    scale = int(2 ** (level_index - source_group.level))

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
                    bucket_writers.append(
                        SourcePyramidLevelWriter(
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
                    )
                    next_accumulator = None
                else:
                    # Generated level: create the accumulator first, then the
                    # level writer that composes it. Cascade chains to the
                    # previously-created accumulator (one level above in the
                    # pyramid). Input tiled size is the level below (scale / 2).
                    input_tiled_size = source_group.tiled_size.ceil_div(
                        max(scale // 2, 1)
                    )
                    is_chain_start = (
                        level_index - 1 not in included_levels
                        or level_index - 1 in source_levels
                    )
                    accumulator = PyramidTileAccumulator(
                        level_index=level_index,
                        input_tiled_size=input_tiled_size,
                        encoder_pool_queue=encoder_pool.queue,
                        next_accumulator=next_accumulator,
                        is_chain_start=is_chain_start,
                        queue_maxsize=self._queue_maxsize,
                        token=token,
                    )
                    generated_writer = GeneratedPyramidLevelWriter(
                        level_index=level_index,
                        dataset=dataset,
                        tile_cache=tile_cache,
                        focal_planes=planes,
                        optical_paths=paths,
                        tiled_size=tiled_size,
                        accumulator=accumulator,
                        token=token,
                    )
                    bucket_writers.append(generated_writer)
                    next_accumulator = accumulator

            bucket_writers.reverse()
            level_writers.extend(bucket_writers)

        return level_writers

    def _create_tile_cache(self, temp_dir: UPath) -> TileCache:
        """Create a shared tile cache for all levels."""
        if self._memory_budget_bytes is not None:
            return ByteBudgetTileCache(
                temp_dir / "tile_cache",
                memory_budget_bytes=self._memory_budget_bytes,
            )
        return DictTileCache()

    def _prepare_writer(
        self,
        level_writer: PyramidLevelWriter,
        instance_number: int,
        transfer_syntax: UID,
        offset_table: OffsetTableType,
    ) -> WsiDicomWriter:
        """Open a DICOM file and configure it for this level's dataset.

        Mutates `level_writer.dataset` to set `SOPInstanceUID` and
        `InstanceNumber`, then writes the file header and pixel-data preamble.
        """
        uid = self._uid_generator.sop_uid(level_writer.dataset)
        filepath = self._output_path.joinpath(uid + ".dcm")
        file_writer = WsiDicomWriter.open(
            filepath, transfer_syntax, offset_table, self._file_options
        )
        try:
            level_writer.dataset.SOPInstanceUID = uid
            level_writer.dataset.InstanceNumber = instance_number
            file_writer.write_header(level_writer.dataset)
            file_writer.start_pixel_data(level_writer.dataset)
        except BaseException:
            file_writer.close()
            raise
        return file_writer

    @staticmethod
    def _finalize_writers(
        level_writers: list[PyramidLevelWriter],
        file_writers: list[WsiDicomWriter],
        encoder_pool: EncoderPool,
        transcoder: Encoder | None,
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

        for level_writer, file_writer in zip(level_writers, file_writers, strict=True):
            level_writer.finalize_writers()
            file_writer.finalize(level_writer.dataset, transcoder)

    @staticmethod
    def _cleanup_writers(
        level_writers: list[PyramidLevelWriter],
        file_writers: list[WsiDicomWriter],
        encoder_pool: EncoderPool,
    ) -> None:
        """Clean up all resources on error."""
        with contextlib.suppress(Exception):
            encoder_pool.shutdown(wait=False)
        for level_writer in level_writers:
            level_writer.cleanup()
        for file_writer in file_writers:
            with contextlib.suppress(Exception):
                file_writer.close()

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
                uid = self._uid_generator.sop_uid(dataset)
                filepath = self._output_path.joinpath(uid + ".dcm")
                with WsiDicomWriter.open(
                    filepath, transfer_syntax, offset_table, self._file_options
                ) as file_writer:
                    dataset.SOPInstanceUID = uid
                    dataset.InstanceNumber = self._instance_number
                    file_writer.write_header(dataset)
                    file_writer.start_pixel_data(dataset)
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
