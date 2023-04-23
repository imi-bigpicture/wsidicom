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

import os
import warnings
from pathlib import Path
from typing import BinaryIO, Callable, List, Optional, Sequence, Tuple, Union

from PIL import Image
from PIL.Image import Image as PILImage
from pydicom.uid import UID, JPEGBaseline8Bit, generate_uid

from wsidicom.errors import (
    WsiDicomMatchError,
    WsiDicomNotFoundError,
    WsiDicomOutOfBoundsError,
)
from wsidicom.file import WsiDicomFileSource, WsiDicomFileTarget
from wsidicom.geometry import Point, PointMm, Region, RegionMm, Size, SizeMm
from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.instance import WsiDataset, WsiInstance
from wsidicom.optical import OpticalManager
from wsidicom.series import Labels, Levels, Overviews
from wsidicom.source import Source
from wsidicom.stringprinting import list_pretty_str
from wsidicom.uid import SlideUids
from wsidicom.web import WsiDicomWebClient, WsiDicomWebSource


class WsiDicom:
    """Represent a wsi slide containing pyramidal levels and optionally
    labels and/or overviews."""

    def __init__(
        self,
        source: Source,
        label: Optional[Union[PILImage, str, Path]] = None,
    ):
        """Holds wsi dicom levels, labels and overviews.

        Note that WsiDicom.open() should be used for opening dicom wsi files.

        Parameters
        ----------
        source: Source
            A source providing instances for the wsi to open.
        label: Optional[Union[PILImage, str, Path]] = None
            Optional label image to use instead of label found in source.
        """
        self._source = source
        if label is None:
            label_instances = source.label_instances
        else:
            label_instances = [
                WsiInstance.create_label(
                    label,
                    source.base_dataset,
                )
            ]
        self._levels = Levels.open(source.level_instances)
        self._labels = Labels.open(label_instances)
        self._overviews = Overviews.open(source.overview_instances)
        self._annotations = list(source.annotation_instances)
        self._uids = self._validate_collection()

        self.optical = OpticalManager.open(
            [instance for series in self.collection for instance in series.instances]
        )

        self.__enter__()

    @classmethod
    def open(
        cls,
        files: Union[str, Path, BinaryIO, Sequence[Union[str, Path, BinaryIO]]],
        label: Optional[Union[PILImage, str, Path]] = None,
    ) -> "WsiDicom":
        """Open valid wsi dicom files in path or stream and return a WsiDicom object.
        Non-valid files are ignored. Only opened files (i.e. not streams) will e closed
        by WsiDicom.

        Parameters
        ----------
        files: Union[str, Path, BinaryIO, Sequence[Union[str, Path, BinaryIO]]],
            Files to open. Can be a single file, a list of files, or a folder containing
            files.
        label: Optional[Union[PILImage, str, Path]] = None
            Optional label image to use instead of label found in path.

        Returns
        ----------
        WsiDicom
            WsiDicom created from wsi dicom files in path.
        """
        source = WsiDicomFileSource(files)
        return cls(source, label)

    @classmethod
    def open_web(
        cls,
        client: WsiDicomWebClient,
        study_uid: Union[str, UID],
        series_uid: Union[str, UID],
        requested_transfer_syntax: UID = JPEGBaseline8Bit,
    ) -> "WsiDicom":
        """Open wsi dicom instances using dicom web client.

        Parameters
        ----------
        client: WsiDicomWebClient
            Configured dicom web client.
        study_uid: Union[str, UID]
            Study uid of wsi to open.
        series_uid: Union[str, UID]
            Series uid of wsi to open
        transfer_syntax: UID
            Transfer syntax to request for image data, for example
            UID("1.2.840.10008.1.2.4.50") for JPEGBaseline8Bit.

        Returns
        ----------
        WsiDicom
            WsiDicom created from wsi dicom files in study-series.
        """
        source = WsiDicomWebSource(
            client, study_uid, series_uid, requested_transfer_syntax
        )
        return cls(source)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.levels}, {self.labels}"
            f"{self.overviews}, {self.annotations})"
        )

    def __str__(self) -> str:
        return self.pretty_str()

    @property
    def size(self) -> Size:
        """Return pixel size of base level."""
        return self.levels.size

    @property
    def mm_size(self) -> SizeMm:
        return self.levels.mm_size

    @property
    def tile_size(self) -> Size:
        """Return tile size of levels."""
        return self.levels.tile_size

    @property
    def pixel_spacing(self) -> SizeMm:
        return self.levels.pixel_spacing

    @property
    def mpp(self) -> SizeMm:
        return self.levels.mpp

    @property
    def uids(self) -> Optional[SlideUids]:
        return self._uids

    @property
    def levels(self) -> Levels:
        """Return contained levels."""
        if self._levels is not None:
            return self._levels
        raise WsiDicomNotFoundError("levels", str(self))

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
    def collection(self) -> List[Union[Levels, Labels, Overviews]]:
        collection: List[Optional[Union[Levels, Labels, Overviews]]] = [
            self._levels,
            self._labels,
            self._overviews,
        ]
        return [series for series in collection if series is not None]

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
        string = self.__class__.__name__
        if depth is not None:
            depth -= 1
            if depth < 0:
                return string
        return (
            string
            + " of levels:\n"
            + list_pretty_str(self.levels.groups, indent, depth, 0, 2)
        )

    def read_label(self, index: int = 0) -> PILImage:
        """Read label image of the whole slide. If several label
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int = 0
            Index of the label image to read.

        Returns
        ----------
        PILImage
            label as image.
        """
        if self.labels is None:
            raise WsiDicomNotFoundError("label", str(self))
        try:
            label = self.labels[index]
            return label.get_default_full()
        except IndexError as exception:
            raise WsiDicomNotFoundError("label", "series") from exception

    def read_overview(self, index: int = 0) -> PILImage:
        """Read overview image of the whole slide. If several overview
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int = 0
            Index of the overview image to read.

        Returns
        ----------
        PILImage
            Overview as image.
        """
        if self.overviews is None:
            raise WsiDicomNotFoundError("overview", str(self))
        try:
            overview = self.overviews[index]
            return overview.get_default_full()
        except IndexError as exception:
            raise WsiDicomNotFoundError("overview", "series") from exception

    def read_thumbnail(
        self,
        size: Tuple[int, int] = (512, 512),
        z: Optional[float] = None,
        path: Optional[str] = None,
    ) -> PILImage:
        """Read thumbnail image of the whole slide with dimensions
        no larger than given size.

        Parameters
        ----------
        size: Tuple[int, int] = (512, 512)
            Upper size limit for thumbnail.
        z: Optional[float] = None
            Z coordinate, optional.
        path: Optional[str] = None
            Optical path, optional.

        Returns
        ----------
        PILImage
            Thumbnail as image,
        """
        thumbnail_size = Size.from_tuple(size)
        level = self.levels.get_closest_by_size(thumbnail_size)
        region = Region(position=Point(0, 0), size=level.size)
        image = level.get_region(region, z, path)
        image.thumbnail((size), resample=Image.Resampling.BILINEAR)
        return image

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        z: Optional[float] = None,
        path: Optional[str] = None,
        threads: int = 1,
    ) -> PILImage:
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
        threads: int = 1
            Number of threads to use for read.

        Returns
        ----------
        PILImage
            Region as image
        """
        wsi_level = self.levels.get_closest_by_level(level)
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
            image = image.resize((size), resample=Image.Resampling.BILINEAR)
        return image

    def read_region_mm(
        self,
        location: Tuple[float, float],
        level: int,
        size: Tuple[float, float],
        z: Optional[float] = None,
        path: Optional[str] = None,
        slide_origin: bool = False,
        threads: int = 1,
    ) -> PILImage:
        """Read image from region defined in mm.

        Parameters
        ----------
        location: Tuple[float, float]
            Upper left corner of region in mm, or lower left corner if using
            slide origin.
        level: int
            Level in pyramid
        size: Tuple[float, float].
            Size of region in mm
        z: Optional[float] = None
            Z coordinate, optional
        path: Optional[str] = None
            optical path, optional
        slide_origin: bool = False
            If to use the slide origin instead of image origin.
        threads: int = 1
            Number of threads to use for read.

        Returns
        ----------
        PILImage
            Region as image
        """
        wsi_level = self.levels.get_closest_by_level(level)
        scale_factor = wsi_level.calculate_scale(level)
        region = RegionMm(PointMm.from_tuple(location), SizeMm.from_tuple(size))
        image = wsi_level.get_region_mm(region, z, path, slide_origin, threads)
        image_size = Size(width=image.size[0], height=image.size[1]) // scale_factor
        return image.resize(image_size.to_tuple(), resample=Image.Resampling.BILINEAR)

    def read_region_mpp(
        self,
        location: Tuple[float, float],
        mpp: float,
        size: Tuple[float, float],
        z: Optional[float] = None,
        path: Optional[str] = None,
        slide_origin: bool = False,
        threads: int = 1,
    ) -> PILImage:
        """Read image from region defined in mm with set pixel spacing.

        Parameters
        ----------
        location: Tuple[float, float].
            Upper left corner of region in mm, or lower left corner if using
            slide origin.
        mpp: float
            Requested pixel spacing (um/pixel).
        size: Tuple[float, float].
            Size of region in mm.
        z: Optional[float] = None
            Z coordinate, optional.
        path: Optional[str] = None
            Optical path, optional.
        slide_origin: bool = False
            If to use the slide origin instead of image origin.
        threads: int = 1
            Number of threads to use for read.

        Returns
        -----------
        PILImage
            Region as image
        """
        pixel_spacing = mpp / 1000.0
        wsi_level = self.levels.get_closest_by_pixel_spacing(
            SizeMm(pixel_spacing, pixel_spacing)
        )
        region = RegionMm(PointMm.from_tuple(location), SizeMm.from_tuple(size))
        image = wsi_level.get_region_mm(region, z, path, slide_origin, threads)
        image_size = SizeMm.from_tuple(size) // pixel_spacing
        return image.resize(image_size.to_tuple(), resample=Image.Resampling.BILINEAR)

    def read_tile(
        self,
        level: int,
        tile: Tuple[int, int],
        z: Optional[float] = None,
        path: Optional[str] = None,
    ) -> PILImage:
        """Read tile in pyramid level as image.

        Parameters
        ----------
        level: int
            Pyramid level
        tile: int, int
            tile xy coordinate
        z: Optional[float] = None
            Z coordinate, optional
        path: Optional[str] = None
            optical path, optional

        Returns
        ----------
        PILImage
            Tile as image
        """
        tile_point = Point.from_tuple(tile)
        try:
            wsi_level = self.levels.get_level(level)
            return wsi_level.get_tile(tile_point, z, path)
        except WsiDicomNotFoundError:
            # Scale from closest level
            wsi_level = self.levels.get_closest_by_level(level)
            return wsi_level.get_scaled_tile(tile_point, level, z, path)

    def read_encoded_tile(
        self,
        level: int,
        tile: Tuple[int, int],
        z: Optional[float] = None,
        path: Optional[str] = None,
    ) -> bytes:
        """Read tile in pyramid level as encoded bytes. For non-existing levels
        the tile is scaled down from a lower level, using the similar encoding.

        Parameters
        ----------
        level: int
            Pyramid level
        tile: int, int
            tile xy coordinate
        z: Optional[float] = None
            Z coordinate, optional
        path: Optional[str] = None
            optical path, optional

        Returns
        ----------
        bytes
            Tile in file encoding.
        """
        tile_point = Point.from_tuple(tile)
        try:
            wsi_level = self.levels.get_level(level)
            return wsi_level.get_encoded_tile(tile_point, z, path)
        except WsiDicomNotFoundError:
            # Scale from closest level
            wsi_level = self.levels.get_closest_by_level(level)
            return wsi_level.get_scaled_encoded_tile(tile_point, level, z, path)

    def get_instance(
        self, level: int, z: Optional[float] = None, path: Optional[str] = None
    ) -> WsiInstance:
        """Return instance fullfilling level, z and/or path.

        Parameters
        ----------
        level: int
            Pyramid level
        z: Optional[float] = None
            Z coordinate, optional
        path: Optional[str] = None
            optical path, optional

        Returns
        ----------
        WsiInstance:
            Instance
        """
        wsi_level = self.levels.get_level(level)
        return wsi_level.get_instance(z, path)

    def close(self) -> None:
        """Close source."""
        self._source.close()

    def save(
        self,
        output_path: Union[str, Path],
        uid_generator: Callable[..., UID] = generate_uid,
        workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        offset_table: Optional[str] = "bot",
        add_missing_levels: bool = False,
    ) -> List[Path]:
        """
        Save wsi as DICOM-files in path. Instances for the same pyramid
        level will be combined when possible to one file (e.g. not split
        for optical paths or focal planes). If instances are sparse tiled they
        will be converted to full tiled by inserting blank tiles. The PixelData
        will contain a basic offset table. All instance uids will be changed.

        Parameters
        ----------
        output_path: Union[str, Path]
        uid_generator: Callable[..., UID] = pydicom.uid.generate_uid
             Function that can gernerate unique identifiers.
        workers: Optional[int] = None
            Maximum number of thread workers to use.
        chunk_size: Optional[int] = None
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.
        offset_table: Optional[str] = 'bot'
            Offset table to use, 'bot' basic offset table, 'eot' extended
            offset table, None - no offset table.
        add_missing_levels: bool = False
            If to add missing dyadic levels up to the single tile level.

        Returns
        ----------
        List[Path]
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
        if isinstance(output_path, str):
            output_path = Path(output_path)
        target = WsiDicomFileTarget(
            output_path,
            uid_generator,
            workers,
            chunk_size,
            offset_table,
            add_missing_levels,
        )
        target.save_levels(self.levels)
        if self.overviews is not None:
            target.save_overviews(self.overviews)
        if self.labels is not None:
            target.save_labels(self.labels)
        return target.filepaths

    def _validate_collection(self) -> SlideUids:
        """
        Check that no files or instance in collection is duplicate, and, if
        strict, that all series have the same base uids.
        Raises WsiDicomMatchError otherwise. Returns base uid for collection.

        Returns
        ----------
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

        try:
            slide_uids = next(
                series.uids for series in self.collection if series.uids is not None
            )
        except StopIteration as exception:
            raise WsiDicomNotFoundError("Valid series", "in collection") from exception
        for series in self.collection:
            if series.uids is not None and series.uids != slide_uids:
                raise WsiDicomMatchError(str(series), str(self))

        if self.annotations != []:
            for annotation in self.annotations:
                if annotation.slide_uids != slide_uids:
                    warnings.warn("Annotations uids does not match")
        return slide_uids

    @classmethod
    def is_ready_for_viewing(
        cls, path: Union[str, Sequence[str], Path, Sequence[Path]]
    ) -> Optional[bool]:
        """
        Return true if files in path are formated for fast viewing, i.e.
        have TILED_FULL tile arrangement and have an offset table.

        Parameters
        ----------
        path: Union[str, Sequence[str], Path, Sequence[Path]]
            Path to files to test.

        Returns
            True if files in path are formated for fast viewing, None if no DICOM WSI
            files are in the path.
        """
        source = WsiDicomFileSource(path)
        return source.is_ready_for_viewing

    @classmethod
    def is_supported(
        cls, path: Union[str, Sequence[str], Path, Sequence[Path]]
    ) -> bool:
        """Return true if files in path have at least one level that can be read with
        WsiDicom.

        Parameters
        ----------
        path: Union[str, Sequence[str], Path, Sequence[Path]]
            Path to files to test.

        Returns
            True if files in path have one level that can be read with WsiDicom.
        """
        source = WsiDicomFileSource(path)
        return source.contains_levels
