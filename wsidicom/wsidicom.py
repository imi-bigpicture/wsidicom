#    Copyright 2021, 2022 SECTRA AB
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
from typing import Callable, List, Optional, Sequence, Tuple, Union

from PIL import Image
from pydicom.dataset import FileMetaDataset
from pydicom.filereader import read_file_meta_info
from pydicom.misc import is_dicom
from pydicom.uid import UID, generate_uid

from wsidicom.dataset import WsiDataset
from wsidicom.errors import (WsiDicomMatchError, WsiDicomNotFoundError,
                             WsiDicomOutOfBoundsError)
from wsidicom.file import WsiDicomFile
from wsidicom.geometry import Point, PointMm, Region, RegionMm, Size, SizeMm
from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.instance import WsiDicomLevel, WsiInstance
from wsidicom.optical import OpticalManager
from wsidicom.series import (WsiDicomLabels, WsiDicomLevels, WsiDicomOverviews,
                             WsiDicomSeries)
from wsidicom.stringprinting import list_pretty_str
from wsidicom.uid import ANN_SOP_CLASS_UID, WSI_SOP_CLASS_UID, SlideUids


class WsiDicom:
    """Represent a wsi slide containing pyramidal levels and optionally
    labels and/or overviews."""
    def __init__(
        self,
        levels: WsiDicomLevels,
        labels: WsiDicomLabels,
        overviews: WsiDicomOverviews,
        annotations: Optional[Sequence[AnnotationInstance]] = None
    ):
        """Holds wsi dicom levels, labels and overviews.

        Parameters
        ----------
        levels: WsiDicomLevels
            Series of pyramidal levels.
        labels: WsiDicomLabels
            Series of label images.
        overviews: WsiDicomOverviews
            Series of overview images
        annotations: Optional[Sequence[AnnotationInstance]] = None
            Sup-222 annotation instances.
        """
        if annotations is None:
            annotations = []
        self._levels = levels
        self._labels = labels
        self._overviews = overviews
        self.annotations = annotations

        self._uids = self._validate_collection(
            [self.levels, self.labels, self.overviews]
        )

        self.optical = OpticalManager.open(
            levels.instances + labels.instances + overviews.instances
        )

        if self.annotations != []:
            for annotation in self.annotations:
                if annotation.slide_uids != self.uids:
                    warnings.warn("Annotations uids does not match")

        self.__enter__()

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
    def base_level(self) -> WsiDicomLevel:
        return self.levels.base_level

    @property
    def size(self) -> Size:
        """Return pixel size of base level."""
        return self.base_level.size

    @property
    def tile_size(self) -> Size:
        """Return tile size of levels."""
        return self.base_level.tile_size

    @property
    def levels(self) -> WsiDicomLevels:
        """Return contained levels"""
        if self._levels is not None:
            return self._levels
        raise WsiDicomNotFoundError("levels", str(self))

    @property
    def labels(self) -> WsiDicomLabels:
        """Return contained labels"""
        if self._labels is not None:
            return self._labels
        raise WsiDicomNotFoundError("labels", str(self))

    @property
    def overviews(self) -> WsiDicomOverviews:
        """Return contained overviews"""
        if self._overviews is not None:
            return self._overviews
        raise WsiDicomNotFoundError("overviews", str(self))

    @property
    def files(self) -> List[Path]:
        """Return contained files"""
        return self.levels.files + self.labels.files + self.overviews.files

    @property
    def datasets(self) -> List[WsiDataset]:
        """Return contained datasets."""
        return (
            self.levels.datasets
            + self.labels.datasets
            + self.overviews.datasets
        )

    @property
    def instances(self) -> List[WsiInstance]:
        """Return contained instances"""
        return (
            self.levels.instances
            + self.labels.instances
            + self.overviews.instances
        )

    @property
    def image_size(self) -> Size:
        """Size of base leve."""
        return self.base_level.size

    @property
    def mm_size(self) -> SizeMm:
        return self.levels.mm_size

    @property
    def uids(self) -> Optional[SlideUids]:
        return self._uids

    def pretty_str(
        self,
        indent: int = 0,
        depth: Optional[int] = None
    ) -> str:
        string = self.__class__.__name__
        if depth is not None:
            depth -= 1
            if depth < 0:
                return string
        return (
            string + ' of levels:\n'
            + list_pretty_str(self.levels.groups, indent, depth, 0, 2)
        )

    @classmethod
    def open(
        cls,
        path: Union[str, Sequence[str], Path, Sequence[Path]]
    ) -> 'WsiDicom':
        """Open valid wsi dicom files in path and return a WsiDicom object.
        Non-valid files are ignored.

        Parameters
        ----------
        path: Union[str, Sequence[str], Path, Sequence[Path]]
            Path to files to open.

        Returns
        ----------
        WsiDicom
            Object created from wsi dicom files in path.
        """
        filepaths = cls._get_filepaths(path)
        level_files: List[WsiDicomFile] = []
        label_files: List[WsiDicomFile] = []
        overview_files: List[WsiDicomFile] = []
        annotation_files: List[Path] = []

        for filepath in cls._filter_paths(filepaths):
            sop_class_uid = cls._get_sop_class_uid(filepath)
            if sop_class_uid == WSI_SOP_CLASS_UID:
                wsi_file = WsiDicomFile(filepath)
                if wsi_file.wsi_type == WsiDicomLevels.WSI_TYPE:
                    level_files.append(wsi_file)
                elif wsi_file.wsi_type == WsiDicomLabels.WSI_TYPE:
                    label_files.append(wsi_file)
                elif wsi_file.wsi_type == WsiDicomOverviews.WSI_TYPE:
                    overview_files.append(wsi_file)
                else:
                    wsi_file.close()
            elif sop_class_uid == ANN_SOP_CLASS_UID:
                annotation_files.append(filepath)
        if len(level_files) == 0:
            raise WsiDicomNotFoundError("Level files", str(path))
        base_dataset = cls._get_base_dataset(level_files)
        slide_uids = base_dataset.uids.slide
        base_tile_size = base_dataset.tile_size
        level_instances = WsiInstance.open(
            level_files,
            slide_uids,
            base_tile_size
        )
        label_instances = WsiInstance.open(label_files, slide_uids)
        overview_instances = WsiInstance.open(overview_files, slide_uids)

        levels = WsiDicomLevels.open(level_instances)
        labels = WsiDicomLabels.open(label_instances)
        overviews = WsiDicomOverviews.open(overview_instances)
        annotations = AnnotationInstance.open(annotation_files)

        return cls(levels, labels, overviews, annotations)

    def read_label(self, index: int = 0) -> Image.Image:
        """Read label image of the whole slide. If several label
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int = 0
            Index of the label image to read.

        Returns
        ----------
        Image.Image
            label as image.
        """
        try:
            label = self.labels[index]
            return label.get_default_full()
        except IndexError as exception:
            raise WsiDicomNotFoundError("label", "series") from exception

    def read_overview(self, index: int = 0) -> Image.Image:
        """Read overview image of the whole slide. If several overview
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int = 0
            Index of the overview image to read.

        Returns
        ----------
        Image.Image
            Overview as image.
        """
        try:
            overview = self.overviews[index]
            return overview.get_default_full()
        except IndexError as exception:
            raise WsiDicomNotFoundError("overview", "series") from exception

    def read_thumbnail(
        self,
        size: Tuple[int, int] = (512, 512),
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> Image.Image:
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
        Image.Image
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
        path: Optional[str] = None
    ) -> Image.Image:
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

        Returns
        ----------
        Image.Image
            Region as image
        """
        wsi_level = self.levels.get_closest_by_level(level)
        scale_factor = wsi_level.calculate_scale(level)
        scaled_region = Region(
            position=Point.from_tuple(location),
            size=Size.from_tuple(size)
        ) * scale_factor

        if not wsi_level.valid_pixels(scaled_region):
            raise WsiDicomOutOfBoundsError(
                f"Region {scaled_region}", f"level size {wsi_level.size}"
            )
        image = wsi_level.get_region(scaled_region, z, path)
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
        slide_origin: bool = False
    ) -> Image.Image:
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

        Returns
        ----------
        Image.Image
            Region as image
        """
        wsi_level = self.levels.get_closest_by_level(level)
        scale_factor = wsi_level.calculate_scale(level)
        region = RegionMm(
            PointMm.from_tuple(location),
            SizeMm.from_tuple(size)
        )
        image = wsi_level.get_region_mm(region, z, path, slide_origin)
        image_size = (
            Size(width=image.size[0], height=image.size[1]) // scale_factor
        )
        return image.resize(
            image_size.to_tuple(),
            resample=Image.Resampling.BILINEAR
        )

    def read_region_mpp(
        self,
        location: Tuple[float, float],
        mpp: float,
        size: Tuple[float, float],
        z: Optional[float] = None,
        path: Optional[str] = None,
        slide_origin: bool = False
    ) -> Image.Image:
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

        Returns
        ----------
        Image.Image
            Region as image
        """
        pixel_spacing = mpp/1000.0
        wsi_level = self.levels.get_closest_by_pixel_spacing(
            SizeMm(pixel_spacing, pixel_spacing)
        )
        region = RegionMm(
            PointMm.from_tuple(location),
            SizeMm.from_tuple(size)
        )
        image = wsi_level.get_region_mm(region, z, path, slide_origin)
        image_size = SizeMm.from_tuple(size) // pixel_spacing
        return image.resize(
            image_size.to_tuple(),
            resample=Image.Resampling.BILINEAR
        )

    def read_tile(
        self,
        level: int,
        tile: Tuple[int, int],
        z: Optional[float] = None,
        path: Optional[str] = None
    ) -> Image.Image:
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
        Image.Image
            Tile as image
        """
        tile_point = Point.from_tuple(tile)
        try:
            wsi_level = self.levels.get_level(level)
            return wsi_level.get_tile(tile_point, z, path)
        except WsiDicomNotFoundError:
            # Scale from closest level
            wsi_level = self.levels.get_closest_by_level(level)
            return wsi_level.get_scaled_tile(
                tile_point,
                level,
                z,
                path)

    def read_encoded_tile(
        self,
        level: int,
        tile: Tuple[int, int],
        z: Optional[float] = None,
        path: Optional[str] = None
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
            return wsi_level.get_scaled_encoded_tile(
                tile_point,
                level,
                z,
                path
            )

    def get_instance(
        self,
        level: int,
        z: Optional[float] = None,
        path: Optional[str] = None
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
        """Close all files."""
        for series in [self.levels, self.overviews, self.labels]:
            series.close()

    def save(
        self,
        output_path: str,
        uid_generator: Callable[..., UID] = generate_uid,
        workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        offset_table: Optional[str] = 'bot'
    ) -> List[Path]:
        """Save wsi as DICOM-files in path. Instances for the same pyramid
        level will be combined when possible to one file (e.g. not split
        for optical paths or focal planes). If instances are sparse tiled they
        will be converted to full tiled by inserting blank tiles. The PixelData
        will contain a basic offset table. All instance uids will be changed.

        Parameters
        ----------
        output_path: str
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

        collections: List[WsiDicomSeries] = [
            self.levels, self.labels, self.overviews
        ]

        filepaths: List[Path] = []
        for collection in collections:
            collection_filepaths = collection.save(
                output_path,
                uid_generator,
                workers,
                chunk_size,
                offset_table
            )
            filepaths.extend(collection_filepaths)
        return filepaths

    @staticmethod
    def _get_sop_class_uid(path: Path) -> UID:
        metadata: FileMetaDataset = read_file_meta_info(path)
        return metadata.MediaStorageSOPClassUID

    @staticmethod
    def _get_filepaths(path: Union[str, Sequence[str], Path, Sequence[Path]]):
        """Return file paths to files in path.
        If path is folder, return list of folder files in path.
        If path is single file, return list of that path.
        If path is list, return list of paths that are files.
        Raises WsiDicomNotFoundError if no files found

        Parameters
        ----------
        path: path: Union[str, Sequence[str], Path, Sequence[Path]]
            Path to folder, file or list of files

        Returns
        ----------
        List[Path]
            List of found file paths
        """
        if isinstance(path, (str, Path)):
            single_path = Path(path)
            if single_path.is_dir():
                return list(single_path.iterdir())
            elif single_path.is_file():
                return [single_path]
        elif isinstance(path, list):
            multiple_paths = [
                Path(file_path) for file_path in path
                if Path(file_path).is_file()
            ]
            if multiple_paths != []:
                return multiple_paths

        raise WsiDicomNotFoundError("No files found", str(path))

    @staticmethod
    def _get_base_dataset(
        files: Sequence[WsiDicomFile]
    ) -> WsiDataset:
        """Return file with largest image (width) from list of files.

        Parameters
        ----------
        files: Sequence[WsiDicomFile]
           List of files.

        Returns
        ----------
        WsiDataset
            Base layer dataset.
        """
        base_size = Size(0, 0)
        base_dataset = files[0].dataset
        for file in files[1:]:
            if file.dataset.image_size.width > base_size.width:
                base_dataset = file.dataset
                base_size = file.dataset.image_size
        return base_dataset

    @staticmethod
    def _filter_paths(filepaths: Sequence[Path]) -> List[Path]:
        """Filter list of paths to only include valid dicom files.

        Parameters
        ----------
        filepaths: Sequence[Path]
            Paths to filter

        Returns
        ----------
        List[Path]
            List of paths with dicom files
        """
        return [
            path for path in filepaths if path.is_file() and is_dicom(path)
        ]

    def _validate_collection(
        self,
        series: Sequence[WsiDicomSeries]
    ) -> SlideUids:
        """Check that no files or instance in collection is duplicate, and, if
        strict, that all series have the same base uids.
        Raises WsiDicomMatchError otherwise. Returns base uid for collection.

        Parameters
        ----------
        series: Sequence[WsiDicomSeries]
            List of series to check.

        Returns
        ----------
        SlideUids
            Matching uids
        """
        WsiDataset.check_duplicate_dataset(self.datasets, self)
        WsiInstance.check_duplicate_instance(self.instances, self)

        try:
            slide_uids = next(
                item.uids for item in series if item.uids is not None
            )
        except StopIteration as exception:
            raise WsiDicomNotFoundError(
                "Valid series",
                "in collection"
            ) from exception
        for item in series:
            if item.uids is not None and item.uids != slide_uids:
                raise WsiDicomMatchError(str(item), str(self))
        return slide_uids

    @classmethod
    def ready_for_viewing(
        cls,
        path: Union[str, Sequence[str], Path, Sequence[Path]]
    ) -> bool:
        """Return true if files in path are formated for fast viewing, i.e.
        have TILED_FULL tile arrangement and have an offset table.

        Parameters
        ----------
        path: Union[str, Sequence[str], Path, Sequence[Path]]
            Path to files to test.

        Returns
            True if files in path are formated for fast viewing.
        """
        filepaths = cls._get_filepaths(path)
        # Sort files by file size to test smallest file first.
        filepaths.sort(key=os.path.getsize)
        for filepath in cls._filter_paths(filepaths):
            file = WsiDicomFile(filepath, parse_pixel_data=False)
            if file.wsi_type is None:
                continue
            if (
                file.dataset.tile_type != 'TILED_FULL'
                or file.offset_table_type is None
            ):
                return False

        return True
