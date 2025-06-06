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
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from PIL.Image import Image
from pydicom.uid import UID, generate_uid
from upath import UPath

from wsidicom.codec import Encoder
from wsidicom.codec import Settings as EncoderSettings
from wsidicom.config import settings
from wsidicom.errors import (
    WsiDicomMatchError,
    WsiDicomNotFoundError,
    WsiDicomOutOfBoundsError,
)
from wsidicom.file import OffsetTableType, WsiDicomFileSource, WsiDicomFileTarget
from wsidicom.geometry import Point, PointMm, Region, RegionMm, Size, SizeMm
from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.group import Level, Thumbnail
from wsidicom.instance import WsiDataset, WsiInstance
from wsidicom.metadata import WsiMetadata
from wsidicom.series import Labels, Overviews, Pyramid, Pyramids
from wsidicom.source import Source
from wsidicom.stringprinting import list_pretty_str
from wsidicom.uid import SlideUids
from wsidicom.web import WsiDicomWebClient, WsiDicomWebSource


class WsiDicom:
    """A WSI containing pyramidal levels and optionally labels and/or overviews."""

    def __init__(
        self,
        source: Source,
        source_owned: bool = True,
    ):
        """Hold WSI DICOM levels, labels and overviews.

        Note that WsiDicom.open() should be used for opening DICOM WSI files.

        Parameters
        ----------
        source: Source
            A source providing instances for the wsi to open.
        source_owned: bool = True
            If source should be closed by this instance if used in a context manager.
        """
        self._selected_pyramid = 0
        self._source = source
        self._source_owned = source_owned
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
        files: Union[str, Path, UPath, Iterable[Union[str, Path, UPath]]],
        file_options: Optional[Dict[str, Any]] = None,
    ) -> "WsiDicom":
        """Open valid WSI DICOM files and return a WsiDicom object.

        Non-valid files are ignored.

        Parameters
        ----------
        files: Union[str, Path, UPath, Iterable[Union[str, Path, UPath]]]
            Files to open. Can be a path for a single file, a list of paths for multiple
            files, or a path to a folder containing files. Path can be local or an URL
            supported by fsspec.
        file_options: Optional[Dict[str, Any]] = None
            Optional options for when opening files.

        Returns
        -------
        WsiDicom
            WsiDicom created from WSI DICOM files in path.
        """
        source = WsiDicomFileSource.open(files, file_options)
        return cls(source, True)

    @classmethod
    def open_dicomdir(
        cls, path: UPath, file_options: Optional[Dict[str, Any]] = None
    ) -> "WsiDicom":
        """Open WSI DICOM files in DICOMDIR and return a WsiDicom object.

        Parameters
        ----------
        path: UPath
            Path to DICOMDIR file or directory with a DICOMDIR file. Path can be local
            or an URL supported by fsspec.
        file_options: Optional[Dict[str, Any]] = None
            Optional options for when opening files.

        Returns
        -------
        WsiDicom
            WsiDicom created from WSI DICOM files in DICOMDIR.
        """
        source = WsiDicomFileSource.open_dicomdir(path, file_options)
        return cls(source, True)

    @classmethod
    def open_streams(
        cls,
        streams: Iterable[BinaryIO],
    ) -> "WsiDicom":
        """Open valid WSI DICOM files in path or stream and return a WsiDicom object.

        Non-valid files are ignored. Only opened files (i.e. not streams) will e closed
        by WsiDicom.

        Parameters
        ----------
        streams: Iterable[BinaryIO],
            Streams to open.

        Returns
        -------
        WsiDicom
            WsiDicom created from WSI DICOM files in path.
        """
        source = WsiDicomFileSource.open_streams(streams)
        return cls(source, False)

    @classmethod
    def open_web(
        cls,
        client: WsiDicomWebClient,
        study_uid: Union[str, UID],
        series_uids: Union[str, UID, Iterable[Union[str, UID]]],
        requested_transfer_syntax: Optional[
            Union[str, UID, Sequence[Union[str, UID]]]
        ] = None,
    ) -> "WsiDicom":
        """Open WSI DICOM instances using DICOM web client.

        Parameters
        ----------
        client: WsiDicomWebClient
            Configured DICOM web client.
        study_uid: Union[str, UID]
            Study uid of wsi to open.
        series_uids: Union[str, UID, Iterable[Union[str, UID]]]
            Series uids of wsi to open
        requested_transfer_syntax: Optional[
            Union[str, UID, Sequence[Union[str, UID]]]
        ] = JPEGBaseline8Bit
            Transfer syntax to request for image data, for example
            "1.2.840.10008.1.2.4.50" for JPEGBaseline8Bit. By default the first
            supported transfer syntax is requested.

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

        source = WsiDicomWebSource(
            client, study_uid, series_uids, requested_transfer_syntax
        )
        return cls(source, True)

    def save(
        self,
        output_path: Union[str, Path, UPath],
        uid_generator: Callable[..., UID] = generate_uid,
        workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        offset_table: Optional[Union["str", OffsetTableType]] = None,
        include_pyramids: Optional[Sequence[int]] = None,
        include_levels: Optional[Sequence[int]] = None,
        include_labels: bool = True,
        include_overviews: bool = True,
        include_thumbnails: bool = True,
        add_missing_levels: bool = False,
        label: Optional[Union[Image, Union[str, Path, UPath]]] = None,
        transcoding: Optional[Union[EncoderSettings, Encoder]] = None,
        file_options: Optional[Dict[str, Any]] = None,
    ) -> List[UPath]:
        """
        Save wsi as DICOM-files in path. Instances for the same pyramid
        level will be combined when possible to one file (e.g. not split
        for optical paths or focal planes). If instances are sparse tiled they
        will be converted to full tiled by inserting blank tiles. All instance uids will
        be changed.

        Parameters
        ----------
        output_path: Union[str, Path, UPath]
            Output folder to write files to. Should preferably be an dedicated folder
            for the wsi. Path can be local or an URL supported by fsspec.
        uid_generator: Callable[..., UID] = pydicom.uid.generate_uid
            Function that can generate unique identifiers.
        workers: Optional[int] = None
            Maximum number of thread workers to use.
        chunk_size: Optional[int] = None
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.
        offset_table: Optional[Union["str", OffsetTableType]] = None,
            Offset table to use, defined either by string (`empty`, `bot`, `eot`, or
            `none`) or `OffsetTableType` enum. Default to None, which will use
            `bot` for encapsulated  syntaxes and `none` for non-encapsulated transfer
            syntaxes.
        include_pyramids: Optional[Sequence[int]] = None
            Optional list of indices (in present pyramids) to include.
        include_levels: Optional[Sequence[int]] = None
            Optional list of indices (in all pyramids) to include, e.g. [0, 1]
            includes the two lowest levels. Negative indicies can be used,
            e.g. [-1, -2] includes the two highest levels.
        include_labels: bool = True
            If to include label series.
        include_overviews: bool = True
            If to include overview series.
        include_thumbnails: bool = True
            If to include thumbnail series.
        add_missing_levels: bool = False
            If to add missing dyadic levels up to the single tile level.
        label: Optional[Union[Image, Union[str, Path, UPath]]] = None
            Optional label image to use instead of present label (if any).
        transcoding: Optional[Union[EncoderSettings, Encoder]] = None,
            Optional settings or encoder for transcoding image data. If None, image data
            will be copied as is.
        file_options: Optional[Dict[str, Any]] = None
            Optional options for saving files to output path.

        Returns
        -------
        List[UPath]
            List of paths of created files.
        """
        if workers is None:
            cpus = os.cpu_count()
            if cpus is None:
                workers = 1
            else:
                workers = cpus
        if chunk_size is None:
            chunk_size = 16
        if isinstance(offset_table, str):
            offset_table = OffsetTableType.from_string(offset_table)
        with WsiDicomFileTarget(
            output_path,
            uid_generator,
            workers,
            chunk_size,
            offset_table,
            include_pyramids,
            include_levels,
            add_missing_levels,
            transcoding,
            file_options,
        ) as target:
            target.save_pyramids(self.pyramids, include_thumbnails)
            if include_overviews and self.overviews is not None:
                target.save_overviews(self.overviews)
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
                if labels is not None:
                    target.save_labels(labels)
            return target.filepaths

    @classmethod
    def is_ready_for_viewing(
        cls,
        files: Union[str, Path, UPath, Iterable[Union[str, Path, UPath]]],
        file_options: Optional[Dict[str, Any]] = None,
    ) -> Optional[bool]:
        """
        Return true if files in path are formatted for fast viewing, i.e.
        have TILED_FULL tile arrangement and have an offset table.

        Parameters
        ----------
        files: Union[str, Path, UPath, Iterable[Union[str, Path, UPath]]]
            Files to open. Can be a path for a single file, a list of paths for multiple
            files, or a path to a folder containing files. Path can be local or an URL
            supported by fsspec.
        file_options: Optional[Dict[str, Any]] = None
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
        files: Union[str, Path, UPath, Iterable[Union[str, Path, UPath]]],
        file_options: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Return true if files in path have at least one level that can be read with
        WsiDicom.

        Parameters
        ----------
        files: Union[str, Path, UPath, Iterable[Union[str, Path, UPath]]]
            Path to files to test. Path can be local or an URL supported by fsspec.
        file_options: Optional[Dict[str, Any]] = None
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
    def uids(self) -> Optional[SlideUids]:
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
    def labels(self) -> Optional[Labels]:
        """Return contained labels."""
        return self._labels

    @property
    def overviews(self) -> Optional[Overviews]:
        """Return contained overviews."""
        return self._overviews

    @property
    def annotations(self) -> List[AnnotationInstance]:
        """Return contained annotations."""
        return self._annotations

    @property
    def collection(self) -> List[Union[Pyramid, Labels, Overviews]]:
        collection: List[Optional[Union[Pyramid, Labels, Overviews]]] = [
            self._labels,
            self._overviews,
        ]
        collection.extend(pyramid for pyramid in self._pyramids)
        return [series for series in collection if series is not None]

    @property
    def metadata(self) -> WsiMetadata:
        return self._create_metadata(self._selected_pyramid)

    @property
    def files(self) -> Optional[List[UPath]]:
        """Return opened files if source is file-based."""
        if isinstance(self._source, WsiDicomFileSource):
            return self._source.files
        return None

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
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

    def read_label(self, index: int = 0) -> Image:
        """Read label image of the whole slide. If several label
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int = 0
            Index of the label image to read.

        Returns
        -------
        Image
            Label as Pillow image.
        """
        if self.labels is None:
            raise WsiDicomNotFoundError("label", str(self))
        label = self.labels.get(index)
        return label.get_default_full()

    def read_overview(self, index: int = 0) -> Image:
        """Read overview image of the whole slide. If several overview
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int = 0
            Index of the overview image to read.

        Returns
        -------
        Image
            Overview as Pillow image.
        """
        if self.overviews is None:
            raise WsiDicomNotFoundError("overview", str(self))
        overview = self.overviews.get(index)
        return overview.get_default_full()

    def read_thumbnail(
        self,
        size: Union[int, Tuple[int, int]] = 512,
        z: Optional[float] = None,
        path: Optional[str] = None,
        pyramid: Optional[int] = None,
        force_generate: bool = False,
    ) -> Image:
        """Read thumbnail image of the whole slide with dimensions no larger than given
        size.

        Parameters
        ----------
        size: Union[int, Tuple[int, int]] = 512
            Upper size limit for thumbnail.
        z: Optional[float] = None
            Z coordinate, optional.
        path: Optional[str] = None
            Optical path, optional.
        pyramid: Optional[int] = None
            Pyramid to read thumbnail from. If `None` the index in `selected_pyramid` is
            used.
        force_generate: bool = False
            If to force generation of thumbnail from levels, even if thumbnail image is
            present for levels.

        Returns
        -------
        Image
            Thumbnail as Pillow image,
        """
        if isinstance(size, int):
            size = (size, size)
        thumbnail_size = Size.from_tuple(size)
        selected_pyramid = self.pyramids.get(pyramid or self.selected_pyramid)
        thumbnail: Optional[Union[Thumbnail, Level]] = None
        if not force_generate and selected_pyramid.thumbnails is not None:
            thumbnail = selected_pyramid.thumbnails.get_closest_by_size(thumbnail_size)
        if selected_pyramid.thumbnails is None or thumbnail is None:
            thumbnail = selected_pyramid.get_closest_by_size(thumbnail_size)
        if thumbnail is None:
            raise WsiDicomNotFoundError(
                f"Image for generating thumbnail of size {thumbnail_size}", "levels"
            )
        region = Region(position=Point(0, 0), size=thumbnail.size)
        image = thumbnail.get_region(region, z, path)
        image.thumbnail((size), resample=settings.pillow_resampling_filter)
        return image

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        z: Optional[float] = None,
        path: Optional[str] = None,
        pyramid: Optional[int] = None,
        threads: int = 1,
    ) -> Image:
        """Read region defined by pixels.

        Parameters
        ----------
        location: Tuple[int, int]
            Upper left corner of region in pixels.
        level: int
            Level in pyramid.
        size: Tuple[int, int]
            Size of region in pixels.
        z: Optional[float] = None
            Z coordinate, optional.
        path: Optional[str] = None
            Optical path, optional.
        pyramid: Optional[int] = None
            Pyramid to read region from. If `None` the index in `selected_pyramid` is
            used.
        threads: int = 1
            Number of threads to use for read.

        Returns
        -------
        Image
            Region as Pillow image.
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
        image = wsi_level.get_region(scaled_region, z, path, threads)
        if scale_factor != 1:
            image = image.resize((size), resample=settings.pillow_resampling_filter)
        return image

    def read_region_mm(
        self,
        location: Tuple[float, float],
        level: int,
        size: Tuple[float, float],
        z: Optional[float] = None,
        path: Optional[str] = None,
        pyramid: Optional[int] = None,
        slide_origin: bool = False,
        threads: int = 1,
    ) -> Image:
        """Read image from region defined in mm.

        Parameters
        ----------
        location: Tuple[float, float]
            Upper left corner of region in mm, or lower left corner if using slide
            origin.
        level: int
            Level in pyramid.
        size: Tuple[float, float].
            Size of region in mm.
        z: Optional[float] = None
            Z coordinate, optional.
        path: Optional[str] = None
            Optical path, optional.
        pyramid: Optional[int] = None
            Pyramid to read region from. If `None` the index in `selected_pyramid` is
            used.
        slide_origin: bool = False
            If to use the slide origin instead of image origin.
        threads: int = 1
            Number of threads to use for read.

        Returns
        -------
        Image
            Region as Pillow image.
        """
        if pyramid is None:
            pyramid = self.selected_pyramid
        wsi_level = self.pyramids.get(pyramid).get_closest_by_level(level)
        scale_factor = wsi_level.calculate_scale(level)
        region = RegionMm(PointMm.from_tuple(location), SizeMm.from_tuple(size))
        image = wsi_level.get_region_mm(region, z, path, slide_origin, threads)
        image_size = Size(width=image.size[0], height=image.size[1]) // scale_factor
        return image.resize(
            image_size.to_tuple(), resample=settings.pillow_resampling_filter
        )

    def read_region_mpp(
        self,
        location: Tuple[float, float],
        mpp: float,
        size: Tuple[float, float],
        z: Optional[float] = None,
        path: Optional[str] = None,
        pyramid: Optional[int] = None,
        slide_origin: bool = False,
        threads: int = 1,
    ) -> Image:
        """Read image from region defined in mm with set pixel spacing.

        Parameters
        ----------
        location: Tuple[float, float].
            Upper left corner of region in mm, or lower left corner if using slide
            origin.
        mpp: float
            Requested pixel spacing (um/pixel).
        size: Tuple[float, float].
            Size of region in mm.
        z: Optional[float] = None
            Z coordinate, optional.
        path: Optional[str] = None
            Optical path, optional.
        pyramid: Optional[int] = None
            Pyramid to read region from. If `None` the index in `selected_pyramid` is
            used.
        slide_origin: bool = False
            If to use the slide origin instead of image origin.
        threads: int = 1
            Number of threads to use for read.

        Returns
        --------
        Image
            Region as Pillow image.
        """
        pixel_spacing = mpp / 1000.0
        if pyramid is None:
            pyramid = self.selected_pyramid
        wsi_level = self.pyramids.get(pyramid).get_closest_by_pixel_spacing(
            SizeMm(pixel_spacing, pixel_spacing)
        )
        region = RegionMm(PointMm.from_tuple(location), SizeMm.from_tuple(size))
        image = wsi_level.get_region_mm(region, z, path, slide_origin, threads)
        image_size = SizeMm.from_tuple(size) // pixel_spacing
        return image.resize(
            image_size.to_tuple(), resample=settings.pillow_resampling_filter
        )

    def read_tile(
        self,
        level: int,
        tile: Tuple[int, int],
        z: Optional[float] = None,
        path: Optional[str] = None,
        pyramid: Optional[int] = None,
    ) -> Image:
        """Read tile in pyramid level as Pillow image.

        Parameters
        ----------
        level: int
            Pyramid level.
        tile: int, int
            tile xy coordinate.
        z: Optional[float] = None
            Z coordinate, optional.
        path: Optional[str] = None
            Optical path, optional.
        pyramid: Optional[int] = None
            Pyramid to read tile from. If `None` the index in `selected_pyramid` is
            used.

        Returns
        -------
        Image
            Tile as Pillow image.
        """
        tile_point = Point.from_tuple(tile)
        wsi_pyramid = self.pyramids.get(
            pyramid if pyramid is not None else self._selected_pyramid
        )
        try:
            wsi_level = wsi_pyramid.get(level)
            return wsi_level.get_tile(tile_point, z, path)
        except WsiDicomNotFoundError:
            # Scale from closest level
            wsi_level = wsi_pyramid.get_closest_by_level(level)
            return wsi_level.get_scaled_tile(tile_point, level, z, path)

    def read_encoded_tile(
        self,
        level: int,
        tile: Tuple[int, int],
        z: Optional[float] = None,
        path: Optional[str] = None,
        pyramid: Optional[int] = None,
    ) -> bytes:
        """Read tile in pyramid level as encoded bytes. For non-existing levels
        the tile is scaled down from a lower level, using the similar encoding.

        Parameters
        ----------
        level: int
            Pyramid level.
        tile: int, int
            Tile xy coordinate.
        z: Optional[float] = None
            Z coordinate, optional.
        path: Optional[str] = None
            Optical path, optional.
        pyramid: Optional[int] = None
            Pyramid to read tile from. If `None` the index in `selected_pyramid` is
            used.

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
            return wsi_level.get_encoded_tile(tile_point, z, path)
        except WsiDicomNotFoundError:
            # Scale from closest level
            wsi_level = wsi_pyramid.get_closest_by_level(level)
            return wsi_level.get_scaled_encoded_tile(tile_point, level, z, path)

    def get_instance(
        self,
        level: int,
        z: Optional[float] = None,
        path: Optional[str] = None,
        pyramid: Optional[int] = None,
    ) -> WsiInstance:
        """Return instance fulfilling level, z and/or path.

        Parameters
        ----------
        level: int
            Pyramid level
        z: Optional[float] = None
            Z coordinate, optional.
        path: Optional[str] = None
            Optical path, optional.
        pyramid: Optional[int] = None
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

    @lru_cache
    def _create_metadata(self, pyramid_index: int) -> WsiMetadata:
        if self.labels is not None:
            label_metadata = self.labels.metadata
        else:
            label_metadata = None
        if self.overviews is not None:
            overview_metadata = self.overviews.metadata
        else:
            overview_metadata = None
        pyramid = self.pyramids.get(pyramid_index)
        return WsiMetadata.merge_image_types(
            pyramid.metadata, label_metadata, overview_metadata
        )
