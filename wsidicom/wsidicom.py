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

import logging
import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Literal,
    Union,
    overload,
)

import numpy as np
from PIL.Image import Image, fromarray
from pydicom.uid import UID
from upath import UPath

from wsidicom.cache import lru_cached_method
from wsidicom.codec import Encoder
from wsidicom.codec import Settings as EncoderSettings
from wsidicom.config import Settings, use_settings
from wsidicom.errors import (
    WsiDicomMatchError,
    WsiDicomNotFoundError,
    WsiDicomOutOfBoundsError,
)
from wsidicom.file import (
    OffsetTableType,
    WsiDicomFileSource,
    WsiDicomFileTarget,
)
from wsidicom.geometry import Point, PointMm, Region, RegionMm, Size, SizeMm
from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.group import Level, Thumbnail
from wsidicom.instance import WsiDataset, WsiInstance
from wsidicom.metadata import WsiMetadata
from wsidicom.metadata.schema.dicom.label import LabelBaseDicomSchema
from wsidicom.metadata.schema.dicom.wsi import BaseWsiMetadataDicomSchema
from wsidicom.metadata.uid_generator import CallableUidGenerator, UidGenerator
from wsidicom.options import (
    ConcatenationByBytes,
    ConcatenationByFrames,
    InstanceSplit,
)
from wsidicom.series import Labels, Overviews, Pyramid, Pyramids
from wsidicom.source import Source
from wsidicom.stringprinting import list_pretty_str
from wsidicom.thread import ReadExecutor
from wsidicom.uid import SlideUids
from wsidicom.web import WsiDicomWebClient, WsiDicomWebSource


class WsiDicom:
    """A WSI containing pyramidal levels and optionally labels and/or overviews."""

    def __init__(
        self,
        source: Source,
        source_owned: bool = True,
        read_executor: Executor | None = None,
        *,
        settings: Settings | None = None,
    ):
        """Hold WSI DICOM levels, labels and overviews.

        Note that WsiDicom.open() should be used for opening DICOM WSI files.

        Parameters
        ----------
        source: Source
            A source providing instances for the wsi to open.
        source_owned: bool = True
            If source should be closed by this instance if used in a context manager.
        read_executor: Executor | None = None
            Optional shared, thread-based executor reused across reads.
            When supplied, reads parallelize across it by default unless ``threads=1``.
            When ``None`` each parallel read uses a per-read pool when ``threads>1``.
        settings: Settings | None = None
            Settings to use for this object instead of the process-wide default.
            Captured now (the per-instance downsampler and decoder are resolved
            here) and reapplied for operations that read settings at call time.
        """
        self._selected_pyramid = 0
        self._source = source
        self._source_owned = source_owned
        self._read_executor = read_executor
        self._settings = settings
        with use_settings(settings):
            self._pyramids = Pyramids.open(
                source.level_instances, source.thumbnail_instances
            )
            self._labels = Labels.open(source.label_instances)
            self._overviews = Overviews.open(source.overview_instances)
            self._annotations = list(source.annotation_instances)
            self._uids = self._validate_collection()

    @classmethod
    def open(
        cls,
        files: str | Path | UPath | Iterable[str | Path | UPath],
        file_options: dict[str, Any] | None = None,
        read_executor: Executor | None = None,
        *,
        settings: Settings | None = None,
    ) -> "WsiDicom":
        """Open valid WSI DICOM files and return a WsiDicom object.

        Non-valid files are ignored.

        Parameters
        ----------
        files: str | Path | UPath | Iterable[str | Path | UPath]
            Files to open. Can be a path for a single file, a list of paths for multiple
            files, or a path to a folder containing files. Path can be local or an URL
            supported by fsspec.
        file_options: dict[str, Any] | None = None
            Optional options for when opening files.
        read_executor: Executor | None = None
            Optional shared, thread-based executor reused across reads.
            When supplied, reads parallelize across it by default unless ``threads=1``.
            When ``None`` each parallel read uses a per-read pool when ``threads>1``.
        settings: Settings | None = None
            Settings to use for this object instead of the process-wide default.

        Returns
        -------
        WsiDicom
            WsiDicom created from WSI DICOM files in path.
        """
        with use_settings(settings):
            source = WsiDicomFileSource.open(files, file_options)
            return cls(source, True, read_executor, settings=settings)

    @classmethod
    def open_dicomdir(
        cls,
        path: UPath,
        file_options: dict[str, Any] | None = None,
        read_executor: Executor | None = None,
        *,
        settings: Settings | None = None,
    ) -> "WsiDicom":
        """Open WSI DICOM files in DICOMDIR and return a WsiDicom object.

        Parameters
        ----------
        path: UPath
            Path to DICOMDIR file or directory with a DICOMDIR file. Path can be local
            or an URL supported by fsspec.
        file_options: dict[str, Any] | None = None
            Optional options for when opening files.
        read_executor: Executor | None = None
            Optional shared, thread-based executor reused across reads.
            When supplied, reads parallelize across it by default unless ``threads=1``.
            When ``None`` each parallel read uses a per-read pool when ``threads>1``.

        settings: Settings | None = None
            Settings to use for this object instead of the process-wide default.

        Returns
        -------
        WsiDicom
            WsiDicom created from WSI DICOM files in DICOMDIR.
        """
        with use_settings(settings):
            source = WsiDicomFileSource.open_dicomdir(path, file_options)
            return cls(source, True, read_executor, settings=settings)

    @classmethod
    def open_streams(
        cls,
        streams: Iterable[BinaryIO],
        read_executor: Executor | None = None,
        *,
        settings: Settings | None = None,
    ) -> "WsiDicom":
        """Open valid WSI DICOM files in path or stream and return a WsiDicom object.

        Non-valid files are ignored. Only opened files (i.e. not streams) will e closed
        by WsiDicom.

        Parameters
        ----------
        streams: Iterable[BinaryIO],
            Streams to open.
        read_executor: Executor | None = None
            Optional shared, thread-based executor reused across reads.
            When supplied, reads parallelize across it by default unless ``threads=1``.
            When ``None`` each parallel read uses a per-read pool when ``threads>1``.

        settings: Settings | None = None
            Settings to use for this object instead of the process-wide default.

        Returns
        -------
        WsiDicom
            WsiDicom created from WSI DICOM files in path.
        """
        with use_settings(settings):
            source = WsiDicomFileSource.open_streams(streams)
            return cls(source, False, read_executor, settings=settings)

    @classmethod
    def open_web(
        cls,
        client: WsiDicomWebClient,
        study_uid: str | UID,
        series_uids: str | UID | Iterable[str | UID],
        requested_transfer_syntax: str | UID | Sequence[str | UID] | None = None,
        read_executor: Executor | None = None,
        *,
        settings: Settings | None = None,
    ) -> "WsiDicom":
        """Open WSI DICOM instances using DICOM web client.

        Parameters
        ----------
        client: WsiDicomWebClient
            Configured DICOM web client.
        study_uid: str | UID
            Study uid of wsi to open.
        series_uids: str | UID | Iterable[str | UID]
            Series uids of wsi to open
        requested_transfer_syntax:
            str | UID | Sequence[str | UID]
         | None = JPEGBaseline8Bit
            Transfer syntax to request for image data, for example
            "1.2.840.10008.1.2.4.50" for JPEGBaseline8Bit. By default the first
            supported transfer syntax is requested.
        read_executor: Executor | None = None
            Optional shared, thread-based executor reused across region reads.
            When supplied, reads parallelize across it by default (never shut
            down by the returned object); pass ``threads=1`` to a read to opt
            out. When ``None``, each parallel read uses a per-read pool.

        settings: Settings | None = None
            Settings to use for this object instead of the process-wide default.

        Returns
        -------
        WsiDicom
            WsiDicom created from WSI DICOM instances in study-series.
        """
        study_uid = UID(study_uid)
        if isinstance(series_uids, (str, UID)):
            series_uids = [UID(series_uids)]
        else:
            series_uids = [UID(series_uid) for series_uid in series_uids]
        if isinstance(requested_transfer_syntax, (str, UID)):
            requested_transfer_syntax = [UID(requested_transfer_syntax)]
        elif requested_transfer_syntax is not None:
            requested_transfer_syntax = [
                UID(transfer_syntax) for transfer_syntax in requested_transfer_syntax
            ]

        with use_settings(settings):
            source = WsiDicomWebSource(
                client, study_uid, series_uids, requested_transfer_syntax
            )
            return cls(source, True, read_executor, settings=settings)

    def save(
        self,
        output_path: str | Path | UPath,
        uid_generator: Callable[[], UID] | UidGenerator | None = None,
        workers: int | None = None,
        chunk_size: int | None = None,
        offset_table: Union["str", OffsetTableType] | None = None,
        include_pyramids: Sequence[int] | None = None,
        include_levels: Sequence[int] | None = None,
        include_labels: bool = True,
        include_overviews: bool = True,
        include_thumbnails: bool = True,
        add_missing_levels: bool = False,
        regenerate_pyramid: bool = False,
        label: Image | str | Path | UPath | None = None,
        transcoding: EncoderSettings | Encoder | None = None,
        force_transcoding: bool = False,
        file_options: dict[str, Any] | None = None,
        metadata: WsiMetadata | None = None,
        replace_metadata: bool = True,
        instance_split: InstanceSplit = InstanceSplit.NONE,
        concatenation: ConcatenationByFrames | ConcatenationByBytes | None = None,
    ) -> list[UPath]:
        """
        Save wsi as DICOM-files in path. By default instances for the same
        pyramid level are combined when possible to one file (e.g. not split
        for optical paths or focal planes); use `instance_split` to write a
        separate instance per focal plane and/or optical path instead. If
        instances are sparse tiled they will be converted to full tiled by
        inserting blank tiles. All instance uids will be changed.

        Parameters
        ----------
        output_path: str | Path | UPath
            Output folder to write files to. Should preferably be an dedicated folder
            for the wsi. Path can be local or an URL supported by fsspec.
        uid_generator: Callable[[], UID] | UidGenerator | None = None
            Zero-arg callable or `UidGenerator` instance for producing UIDs.
        workers: int | None = None
            Maximum number of thread workers to use.
        chunk_size: int | None = None
            Per-batch tile width hint for source tile reading. When None,
            each source's `ImageData.suggested_minimum_chunk_size` is used.
        offset_table: "str" | OffsetTableType | None = None,
            Offset table to use, defined either by string (`empty`, `bot`, `eot`, or
            `none`) or `OffsetTableType` enum. Default to None, which will use
            `bot` for encapsulated  syntaxes and `none` for non-encapsulated transfer
            syntaxes.
        include_pyramids: Sequence[int] | None = None
            Optional list of indices (in present pyramids) to include.
        include_levels: Sequence[int] | None = None
            Optional list of indices (in all pyramids) to include, e.g. [0, 1]
            includes the two lowest levels. Negative indices can be used,
            e.g. [-1, -2] includes the two highest levels.
        include_labels: bool = True
            If to include label series.
        include_overviews: bool = True
            If to include overview series.
        include_thumbnails: bool = True
            If to include thumbnail series.
        add_missing_levels: bool = False
            If to add missing dyadic levels up to the single tile level.
        regenerate_pyramid: bool = False
            If True, only the base level is read from the source and every
            other written level is re-derived by downsampling from the base,
            instead of being read from the source's stored pyramid. Orthogonal
            to `add_missing_levels`: combine with it to rebuild a complete
            pyramid up to the single tile level (e.g. to replace a source's
            pyramid with consistently downsampled levels). When set, the base
            level must be among the selected `include_levels`.
        label: Image | str | Path | UPath | None = None
            Optional label image to use instead of present label (if any).
        transcoding: EncoderSettings | Encoder | None = None,
            Optional settings or encoder for transcoding image data. If None, image data
            will be copied as is.
        force_transcoding: bool = False
            If to force transcoding even if transfer syntax already matches the encoding
            settings.
        file_options: dict[str, Any] | None = None
            Optional options for saving files to output path.
        metadata: WsiMetadata | None = None
            Optional metadata to use for the written files. See
            `replace_metadata` for how it is applied. When None, the source
            datasets are used as is.
        replace_metadata: bool = True
            Only used when `metadata` is set. If True (default), the output
            datasets are rebuilt from `metadata` combined with the technical
            attributes of the source image data, so that attributes not modeled
            by the metadata schema (including private tags and other unhandled
            attributes) are dropped. If False, `metadata` is overlaid on top of
            the source datasets, preserving any attributes it does not set.
        instance_split: InstanceSplit = InstanceSplit.NONE
            Controls how optical paths and focal planes are split across output
            instances. Default (`InstanceSplit.NONE`) combines all into one
            instance per level. `InstanceSplit.FOCAL_PLANE` and/or
            `InstanceSplit.OPTICAL_PATH` write a separate instance per focal
            plane and/or optical path. Unequally spaced focal planes cannot
            share one instance and are always split per focal plane, regardless
            of this setting.
        concatenation: ConcatenationByFrames | ConcatenationByBytes | None = None
            If set, split each pyramid level into concatenated instances (SOP
            Instances sharing a Concatenation UID). Use `ConcatenationByFrames(n)`
            to cut every n frames, or `ConcatenationByBytes(size)` to cut when a
            part's encapsulated pixel data reaches `size` bytes (an int, or a
            string with a binary suffix such as `"100M"` or `"2G"`). When None
            (default) one instance per level is written.

        Returns
        -------
        list[UPath]
            List of paths of created files.

        Notes
        -----
        To de-identify, compose `metadata` from the source metadata and replace
        the identity-bearing modules, e.g.::

            deid = dataclasses.replace(wsi.metadata, patient=Patient(name="Anon"))
            wsi.save(path, metadata=deid)
        """
        if workers is None:
            cpus = os.cpu_count()
            workers = 1 if cpus is None else cpus
        if isinstance(offset_table, str):
            offset_table = OffsetTableType.from_string(offset_table)
        if uid_generator is None:
            uid_generator = CallableUidGenerator()
        elif not isinstance(uid_generator, UidGenerator):
            uid_generator = CallableUidGenerator(uid_generator)
        output_path = self._normalize_path(output_path)
        if include_labels:
            if label is not None:
                label_instances = [
                    WsiInstance.create_label(
                        label,
                        self._source.base_dataset,
                    )
                ]
                labels = Labels.open(label_instances)
            else:
                labels = self.labels
        else:
            labels = None

        overviews = self.overviews if include_overviews else None

        with (
            use_settings(self._settings),
            WsiDicomFileTarget(
                output_path,
                uid_generator,
                workers,
                chunk_size,
                offset_table,
                include_pyramids,
                include_levels,
                add_missing_levels,
                regenerate_pyramid,
                transcoding,
                force_transcoding,
                file_options,
                metadata,
                replace_metadata,
                instance_split,
                concatenation,
            ) as target,
        ):
            target.save(self.pyramids, labels, overviews, include_thumbnails)
            return target.filepaths

    @staticmethod
    def _normalize_path(path: str | Path | UPath) -> Path | UPath:
        """Normalize local paths (including `file://` and `local://` URIs) to
        absolute; preserve remote fsspec paths."""
        if isinstance(path, str):
            path = UPath(path)
        if isinstance(path, Path) or (
            isinstance(path, UPath) and path.protocol in ("", "file", "local")
        ):
            return path.resolve()
        return path

    @classmethod
    def is_ready_for_viewing(
        cls,
        files: str | Path | UPath | Iterable[str | Path | UPath],
        file_options: dict[str, Any] | None = None,
    ) -> bool | None:
        """
        Return true if files in path are formatted for fast viewing, i.e.
        have TILED_FULL tile arrangement and have an offset table.

        Parameters
        ----------
        files: str | Path | UPath | Iterable[str | Path | UPath]
            Files to open. Can be a path for a single file, a list of paths for multiple
            files, or a path to a folder containing files. Path can be local or an URL
            supported by fsspec.
        file_options: dict[str, Any] | None = None
            Optional options for when opening files.

        Returns
            True if files in path are formatted for fast viewing, None if no DICOM WSI
            files are in the path.
        """
        source = WsiDicomFileSource.open(files, file_options)
        return source.is_ready_for_viewing

    @classmethod
    def is_supported(
        cls,
        files: str | Path | UPath | Iterable[str | Path | UPath],
        file_options: dict[str, Any] | None = None,
    ) -> bool:
        """Return true if files in path have at least one level that can be read with
        WsiDicom.

        Parameters
        ----------
        files: str | Path | UPath | Iterable[str | Path | UPath]
            Path to files to test. Path can be local or an URL supported by fsspec.
        file_options: dict[str, Any] | None = None
            Optional options for when opening files.

        Returns
            True if files in path have one level that can be read with WsiDicom.
        """
        source = WsiDicomFileSource.open(files, file_options)
        return source.contains_levels

    @property
    def size(self) -> Size:
        """Return pixel size of base level of selected pyramid."""
        return self.pyramids[self.selected_pyramid].size

    @property
    def mm_size(self) -> SizeMm:
        """Return image size in mm of selected pyramid."""
        return self.pyramids[self.selected_pyramid].mm_size

    @property
    def tile_size(self) -> Size:
        """Return tile size of selected pyramid."""
        return self.pyramids[self.selected_pyramid].tile_size

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm/pixel of base level of selected pyramid."""
        return self.pyramids[self.selected_pyramid].pixel_spacing

    @property
    def mpp(self) -> SizeMm:
        """Return pixel size in um/pixel of base level of selected pyramid."""
        return self.pyramids[self.selected_pyramid].mpp

    @property
    def selected_pyramid(self) -> int:
        """Return index of selected pyramid."""
        return self._selected_pyramid

    @property
    def uids(self) -> SlideUids | None:
        return self._uids

    @property
    def pyramids(self) -> Pyramids:
        """Return contained pyramids."""
        if self._pyramids is not None:
            return self._pyramids
        raise WsiDicomNotFoundError("pyramids", str(self))

    @property
    def pyramid(self) -> Pyramid:
        """Return contained pyramid for selected pyramid."""
        return self.pyramids[self.selected_pyramid]

    @property
    def levels(self) -> Pyramid:
        """Return contained levels for selected pyramid."""
        return self.pyramid

    @property
    def labels(self) -> Labels | None:
        """Return contained labels."""
        return self._labels

    @property
    def overviews(self) -> Overviews | None:
        """Return contained overviews."""
        return self._overviews

    @property
    def annotations(self) -> list[AnnotationInstance]:
        """Return contained annotations."""
        return self._annotations

    @property
    def collection(self) -> list[Pyramid | Labels | Overviews]:
        collection: list[Pyramid | Labels | Overviews | None] = [
            self._labels,
            self._overviews,
        ]
        collection.extend(pyramid for pyramid in self._pyramids)
        return [series for series in collection if series is not None]

    @property
    def metadata(self) -> WsiMetadata:
        return self._create_metadata(self._selected_pyramid)

    @property
    def files(self) -> list[UPath] | None:
        """Return opened files if source is file-based."""
        if isinstance(self._source, WsiDicomFileSource):
            return self._source.files
        return None

    def pretty_str(self, indent: int = 0, depth: int | None = None) -> str:
        string = self.__class__.__name__
        if depth is not None:
            depth -= 1
            if depth < 0:
                return string
        return (
            string
            + " of pyramids:\n"
            + list_pretty_str(list(self.pyramids), indent, depth, 0, 2)
        )

    def set_selected_pyramid(self, index: int) -> None:
        """Set selected pyramid.

        Parameters
        ----------
        index: int
            Index of pyramid to select.
        """
        if index < 0 or index >= len(self.pyramids):
            raise WsiDicomNotFoundError(f"Pyramid {index}", str(self.pyramids))
        self._selected_pyramid = index

    @overload
    def read_label(
        self, index: int = 0, *, as_array: Literal[False] = False
    ) -> Image: ...

    @overload
    def read_label(self, index: int = 0, *, as_array: Literal[True]) -> np.ndarray: ...

    def read_label(
        self, index: int = 0, *, as_array: bool = False
    ) -> Image | np.ndarray:
        """Read label image of the whole slide. If several label
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int = 0
            Index of the label image to read.
        as_array: bool = False
            If to return the image as a numpy array instead of a Pillow image.

        Returns
        -------
        Union[Image, np.ndarray]
            Label as Pillow image, or as a numpy array if `as_array`.
        """
        if self.labels is None:
            raise WsiDicomNotFoundError("label", str(self))
        label = self.labels.get(index)
        with ReadExecutor() as executor:
            array = label.get_default_full(executor=executor)
        if as_array:
            return array
        return self._to_image(array)

    @overload
    def read_overview(
        self, index: int = 0, *, as_array: Literal[False] = False
    ) -> Image: ...

    @overload
    def read_overview(
        self, index: int = 0, *, as_array: Literal[True]
    ) -> np.ndarray: ...

    def read_overview(
        self, index: int = 0, *, as_array: bool = False
    ) -> Image | np.ndarray:
        """Read overview image of the whole slide. If several overview
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int = 0
            Index of the overview image to read.
        as_array: bool = False
            If to return the image as a numpy array instead of a Pillow image.

        Returns
        -------
        Union[Image, np.ndarray]
            Overview as Pillow image, or as a numpy array if `as_array`.
        """
        if self.overviews is None:
            raise WsiDicomNotFoundError("overview", str(self))
        overview = self.overviews.get(index)
        with ReadExecutor() as executor:
            array = overview.get_default_full(executor=executor)
        if as_array:
            return array
        return self._to_image(array)

    @overload
    def read_thumbnail(
        self,
        size: int | tuple[int, int] = 512,
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        force_generate: bool = False,
        threads: int | None = None,
        *,
        as_array: Literal[False] = False,
    ) -> Image: ...

    @overload
    def read_thumbnail(
        self,
        size: int | tuple[int, int] = 512,
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        force_generate: bool = False,
        threads: int | None = None,
        *,
        as_array: Literal[True],
    ) -> np.ndarray: ...

    def read_thumbnail(
        self,
        size: int | tuple[int, int] = 512,
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        force_generate: bool = False,
        threads: int | None = None,
        *,
        as_array: bool = False,
    ) -> Image | np.ndarray:
        """Read thumbnail image of the whole slide with dimensions no larger than given
        size.

        Parameters
        ----------
        size: int | tuple[int, int] = 512
            Upper size limit for thumbnail.
        z: float | None = None
            Z coordinate, optional.
        path: str | None = None
            Optical path, optional.
        pyramid: int | None = None
            Pyramid to read thumbnail from. If `None` the index in `selected_pyramid` is
            used.
        force_generate: bool = False
            If to force generation of thumbnail from levels, even if thumbnail image is
            present for levels.
        threads: int | None = None
            Number of chunks to split the read across. ``None`` (default) reads
            single-threaded unless a ``read_executor`` was supplied at open, in
            which case the read is parallelized across it. Pass ``1`` to force a
            single-threaded read even when an executor is present.
        as_array: bool = False
            If to return the image as a numpy array instead of a Pillow image.

        Returns
        -------
        Union[Image, np.ndarray]
            Thumbnail as Pillow image, or as a numpy array if `as_array`.
        """
        if isinstance(size, int):
            size = (size, size)
        thumbnail_size = Size.from_tuple(size)
        selected_pyramid = self.pyramids.get(pyramid or self.selected_pyramid)
        thumbnail: Thumbnail | Level | None = None
        if not force_generate and selected_pyramid.thumbnails is not None:
            thumbnail = selected_pyramid.thumbnails.get_closest_by_size(thumbnail_size)
        if selected_pyramid.thumbnails is None or thumbnail is None:
            thumbnail = selected_pyramid.get_closest_by_size(thumbnail_size)
        if thumbnail is None:
            raise WsiDicomNotFoundError(
                f"Image for generating thumbnail of size {thumbnail_size}", "levels"
            )
        with ReadExecutor(threads, self._read_executor) as executor:
            array = thumbnail.get_thumbnail(thumbnail_size, z, path, executor=executor)
        if as_array:
            return array
        return self._to_image(array)

    @overload
    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        threads: int | None = None,
        *,
        as_array: Literal[False] = False,
    ) -> Image: ...

    @overload
    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        threads: int | None = None,
        *,
        as_array: Literal[True],
    ) -> np.ndarray: ...

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        threads: int | None = None,
        *,
        as_array: bool = False,
    ) -> Image | np.ndarray:
        """Read region defined by pixels.

        Parameters
        ----------
        location: tuple[int, int]
            Upper left corner of region in pixels.
        level: int
            Level in pyramid.
        size: tuple[int, int]
            Size of region in pixels.
        z: float | None = None
            Z coordinate, optional.
        path: str | None = None
            Optical path, optional.
        pyramid: int | None = None
            Pyramid to read region from. If `None` the index in `selected_pyramid` is
            used.
        threads: int | None = None
            Number of chunks to split the read across. ``None`` (default) reads
            single-threaded unless a ``read_executor`` was supplied at open, in
            which case the read is parallelized across it. Pass ``1`` to force a
            single-threaded read even when an executor is present.
        as_array: bool = False
            If to return the region as a numpy array instead of a Pillow image.

        Returns
        -------
        Union[Image, np.ndarray]
            Region as Pillow image, or as a numpy array if `as_array`.
        """
        if pyramid is None:
            pyramid = self.selected_pyramid
        wsi_level = self.pyramids.get(pyramid).get_closest_by_level(level)
        scale_factor = wsi_level.calculate_scale(level)
        scaled_region = (
            Region(position=Point.from_tuple(location), size=Size.from_tuple(size))
            * scale_factor
        )

        if not wsi_level.valid_pixels(scaled_region):
            raise WsiDicomOutOfBoundsError(
                f"Region {scaled_region}", f"level size {wsi_level.size}"
            )
        with ReadExecutor(threads, self._read_executor) as executor:
            array = wsi_level.get_region(
                scaled_region,
                z,
                path,
                output_size=Size.from_tuple(size),
                executor=executor,
            )
        if as_array:
            return array
        return self._to_image(array)

    @overload
    def read_region_mm(
        self,
        location: tuple[float, float],
        level: int,
        size: tuple[float, float],
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        slide_origin: bool = False,
        threads: int | None = None,
        *,
        as_array: Literal[False] = False,
    ) -> Image: ...

    @overload
    def read_region_mm(
        self,
        location: tuple[float, float],
        level: int,
        size: tuple[float, float],
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        slide_origin: bool = False,
        threads: int | None = None,
        *,
        as_array: Literal[True],
    ) -> np.ndarray: ...

    def read_region_mm(
        self,
        location: tuple[float, float],
        level: int,
        size: tuple[float, float],
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        slide_origin: bool = False,
        threads: int | None = None,
        *,
        as_array: bool = False,
    ) -> Image | np.ndarray:
        """Read image from region defined in mm.

        Parameters
        ----------
        location: tuple[float, float]
            Upper left corner of region in mm, or lower left corner if using slide
            origin.
        level: int
            Level in pyramid.
        size: tuple[float, float].
            Size of region in mm.
        z: float | None = None
            Z coordinate, optional.
        path: str | None = None
            Optical path, optional.
        pyramid: int | None = None
            Pyramid to read region from. If `None` the index in `selected_pyramid` is
            used.
        slide_origin: bool = False
            If to use the slide origin instead of image origin.
        threads: int | None = None
            Number of chunks to split the read across. ``None`` (default) reads
            single-threaded unless a ``read_executor`` was supplied at open, in
            which case the read is parallelized across it. Pass ``1`` to force a
            single-threaded read even when an executor is present.
        as_array: bool = False
            If to return the region as a numpy array instead of a Pillow image.

        Returns
        -------
        Union[Image, np.ndarray]
            Region as Pillow image, or as a numpy array if `as_array`.
        """
        if pyramid is None:
            pyramid = self.selected_pyramid
        wsi_level = self.pyramids.get(pyramid).get_closest_by_level(level)
        scale_factor = wsi_level.calculate_scale(level)
        region = RegionMm(PointMm.from_tuple(location), SizeMm.from_tuple(size))
        with ReadExecutor(threads, self._read_executor) as executor:
            array = wsi_level.get_region_mm(
                region,
                z,
                path,
                slide_origin,
                scale=scale_factor,
                executor=executor,
            )
        if as_array:
            return array
        return self._to_image(array)

    @overload
    def read_region_mpp(
        self,
        location: tuple[float, float],
        mpp: float,
        size: tuple[float, float],
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        slide_origin: bool = False,
        threads: int | None = None,
        *,
        as_array: Literal[False] = False,
    ) -> Image: ...

    @overload
    def read_region_mpp(
        self,
        location: tuple[float, float],
        mpp: float,
        size: tuple[float, float],
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        slide_origin: bool = False,
        threads: int | None = None,
        *,
        as_array: Literal[True],
    ) -> np.ndarray: ...

    def read_region_mpp(
        self,
        location: tuple[float, float],
        mpp: float,
        size: tuple[float, float],
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        slide_origin: bool = False,
        threads: int | None = None,
        *,
        as_array: bool = False,
    ) -> Image | np.ndarray:
        """Read image from region defined in mm with set pixel spacing.

        Parameters
        ----------
        location: tuple[float, float].
            Upper left corner of region in mm, or lower left corner if using slide
            origin.
        mpp: float
            Requested pixel spacing (um/pixel).
        size: tuple[float, float].
            Size of region in mm.
        z: float | None = None
            Z coordinate, optional.
        path: str | None = None
            Optical path, optional.
        pyramid: int | None = None
            Pyramid to read region from. If `None` the index in `selected_pyramid` is
            used.
        slide_origin: bool = False
            If to use the slide origin instead of image origin.
        threads: int | None = None
            Number of chunks to split the read across. ``None`` (default) reads
            single-threaded unless a ``read_executor`` was supplied at open, in
            which case the read is parallelized across it. Pass ``1`` to force a
            single-threaded read even when an executor is present.
        as_array: bool = False
            If to return the region as a numpy array instead of a Pillow image.

        Returns
        --------
        Union[Image, np.ndarray]
            Region as Pillow image, or as a numpy array if `as_array`.
        """
        pixel_spacing = mpp / 1000.0
        if pyramid is None:
            pyramid = self.selected_pyramid
        wsi_level = self.pyramids.get(pyramid).get_closest_by_pixel_spacing(
            SizeMm(pixel_spacing, pixel_spacing)
        )
        region = RegionMm(PointMm.from_tuple(location), SizeMm.from_tuple(size))
        with ReadExecutor(threads, self._read_executor) as executor:
            array = wsi_level.get_region_mpp(
                region,
                pixel_spacing,
                z,
                path,
                slide_origin,
                executor=executor,
            )
        if as_array:
            return array
        return self._to_image(array)

    @overload
    def read_tile(
        self,
        level: int,
        tile: tuple[int, int],
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        crop_to_image_boundary: bool = True,
        *,
        as_array: Literal[False] = False,
    ) -> Image: ...

    @overload
    def read_tile(
        self,
        level: int,
        tile: tuple[int, int],
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        crop_to_image_boundary: bool = True,
        *,
        as_array: Literal[True],
    ) -> np.ndarray: ...

    def read_tile(
        self,
        level: int,
        tile: tuple[int, int],
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        crop_to_image_boundary: bool = True,
        *,
        as_array: bool = False,
    ) -> Image | np.ndarray:
        """Read tile in pyramid level.

        Parameters
        ----------
        level: int
            Pyramid level.
        tile: int, int
            tile xy coordinate.
        z: float | None = None
            Z coordinate, optional.
        path: str | None = None
            Optical path, optional.
        pyramid: int | None = None
            Pyramid to read tile from. If `None` the index in `selected_pyramid` is
            used.
        crop_to_image_boundary: bool = True
            If to crop tile to image boundary.
        as_array: bool = False
            If to return the tile as a numpy array instead of a Pillow image.

        Returns
        -------
        Union[Image, np.ndarray]
            Tile as Pillow image, or as a numpy array if `as_array`.
        """
        tile_point = Point.from_tuple(tile)
        wsi_pyramid = self.pyramids.get(
            pyramid if pyramid is not None else self._selected_pyramid
        )
        try:
            wsi_level = wsi_pyramid.get(level)
            array = wsi_level.get_tile(tile_point, z, path, crop_to_image_boundary)
        except WsiDicomNotFoundError:
            # Scale from closest level, which reads a region and may fan out.
            wsi_level = wsi_pyramid.get_closest_by_level(level)
            with ReadExecutor(None, self._read_executor) as executor:
                array = wsi_level.get_scaled_tile(
                    tile_point,
                    level,
                    z,
                    path,
                    crop_to_image_boundary,
                    executor=executor,
                )
        if as_array:
            return array
        return self._to_image(array)

    def read_encoded_tile(
        self,
        level: int,
        tile: tuple[int, int],
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
        crop_to_image_boundary: bool = True,
    ) -> bytes:
        """Read tile in pyramid level as encoded bytes. For non-existing levels
        the tile is scaled down from a lower level, using the similar encoding.

        Parameters
        ----------
        level: int
            Pyramid level.
        tile: int, int
            Tile xy coordinate.
        z: float | None = None
            Z coordinate, optional.
        path: str | None = None
            Optical path, optional.
        pyramid: int | None = None
            Pyramid to read tile from. If `None` the index in `selected_pyramid` is
            used.
        crop_to_image_boundary: bool = True
            If to crop tile to image boundary.

        Returns
        -------
        bytes
            Tile in file encoding.
        """
        tile_point = Point.from_tuple(tile)
        wsi_pyramid = self.pyramids.get(
            pyramid if pyramid is not None else self._selected_pyramid
        )
        try:
            wsi_level = wsi_pyramid.get(level)
            return wsi_level.get_encoded_tile(
                tile_point, z, path, crop_to_image_boundary
            )
        except WsiDicomNotFoundError:
            # Scale from closest level, which reads a region and may fan out.
            wsi_level = wsi_pyramid.get_closest_by_level(level)
            with ReadExecutor(None, self._read_executor) as executor:
                return wsi_level.get_scaled_encoded_tile(
                    tile_point,
                    level,
                    z,
                    path,
                    crop_to_image_boundary,
                    executor=executor,
                )

    def get_instance(
        self,
        level: int,
        z: float | None = None,
        path: str | None = None,
        pyramid: int | None = None,
    ) -> WsiInstance:
        """Return instance fulfilling level, z and/or path.

        Parameters
        ----------
        level: int
            Pyramid level
        z: float | None = None
            Z coordinate, optional.
        path: str | None = None
            Optical path, optional.
        pyramid: int | None = None
            Pyramid to read tile from. If `None` the index in `selected_pyramid` is
            used.

        Returns
        -------
        WsiInstance:
            Instance
        """
        if pyramid is None:
            pyramid = self.selected_pyramid
        wsi_level = self.pyramids.get(pyramid).get(level)
        return wsi_level.get_instance(z, path)

    def close(self) -> None:
        """Close source if owned by this instance."""
        if self._source_owned:
            self._source.close()

    def clear_cache(self):
        """Clear cache of encoded and decoded tiles."""
        self._source.clear_cache()

    def resize_cache(self, size: int):
        """Resize cache of encoded and decoded tiles."""
        self._source.resize_cache(size)

    @staticmethod
    def _to_image(pixels: np.ndarray) -> Image:
        """Derive a Pillow image from assembled pixels."""
        image = fromarray(np.ascontiguousarray(pixels))
        if image.mode.startswith("I;16"):
            return image.convert("I")
        return image

    def _validate_collection(self) -> SlideUids:
        """
        Check that no files or instance in collection is duplicate, and, if
        strict, that all series have the same base uids.
        Raises WsiDicomMatchError otherwise. Returns base uid for collection.

        Returns
        -------
        SlideUids
            Matching uids
        """
        datasets = [
            dataset for collection in self.collection for dataset in collection.datasets
        ]
        WsiDataset.check_duplicate_dataset(datasets, self)

        instances = [
            instance
            for collection in self.collection
            for instance in collection.instances
        ]

        WsiInstance.check_duplicate_instance(instances, self)

        slide_uids = self.levels.uids
        if slide_uids is None:
            raise WsiDicomNotFoundError("Valid levels", str(self))
        for series in self.collection:
            if series.uids is not None and not series.uids.matches(slide_uids):
                raise WsiDicomMatchError(str(series), str(self))

        if self.annotations != []:
            for annotation in self.annotations:
                if not annotation.slide_uids.matches(slide_uids):
                    logging.warning("Annotations uids does not match.")
        return slide_uids

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.pyramids}, {self.labels}"
            f"{self.overviews}, {self.annotations})"
        )

    def __str__(self) -> str:
        return self.pretty_str()

    @lru_cached_method()
    def _create_metadata(self, pyramid_index: int) -> WsiMetadata:
        pyramid = self.pyramids.get(pyramid_index)

        if self.labels is not None:
            label_metadata = self.labels.metadata
        else:
            label_metadata = LabelBaseDicomSchema().load(pyramid.datasets[0])
        if self.overviews is not None:
            overview_metadata = self.overviews.metadata
        else:
            overview_metadata = None
        base = BaseWsiMetadataDicomSchema().load(pyramid.datasets[0])
        return WsiMetadata(
            study=base.study,
            series=base.series,
            patient=base.patient,
            equipment=base.equipment,
            slide=base.slide,
            pyramid=pyramid.metadata,
            label=label_metadata,
            overview=overview_metadata,
            frame_of_reference_uid=base.frame_of_reference_uid,
            dimension_organization_uids=base.dimension_organization_uids,
        )
