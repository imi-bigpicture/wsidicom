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

from collections import defaultdict
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

from pydicom.dataset import FileMetaDataset
from pydicom.uid import UID

from wsidicom.errors import (
    WsiDicomNotFoundError,
    WsiDicomNotSupportedError,
)
from wsidicom.file.wsidicom_file import WsiDicomFile
from wsidicom.file.wsidicom_file_image_data import WsiDicomFileImageData
from wsidicom.geometry import Size
from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.instance import ImageType, TileType, WsiDataset, WsiInstance
from wsidicom.source import Source
from wsidicom.uid import ANN_SOP_CLASS_UID, WSI_SOP_CLASS_UID, SlideUids
from pydicom.filereader import read_preamble, _read_file_meta_info
from pydicom.errors import InvalidDicomError

"""A source for reading WSI DICOM files."""


class WsiDicomFileSource(Source):
    """Source reading WSI DICOM instances."""

    def __init__(
        self,
        files: Union[str, Path, BinaryIO, Sequence[Union[str, Path, BinaryIO]]],
    ) -> None:
        """Create a WsiDicomFileSource.

        Parameters
        ----------
        files: Union[str, Path, BinaryIO, Sequence[Union[str, Path, BinaryIO]]],
            Files to open. Can be a single file, a list of files, or a folder containing
            files.
        """
        self._level_files: List[WsiDicomFile] = []
        self._label_files: List[WsiDicomFile] = []
        self._overview_files: List[WsiDicomFile] = []
        self._annotation_files: List[BinaryIO] = []
        for file, sop_class_uid, filepath in self._get_sop_class_uids(files):
            if sop_class_uid == WSI_SOP_CLASS_UID:
                try:
                    wsi_file = WsiDicomFile(file, filepath)
                    if wsi_file.image_type == ImageType.VOLUME:
                        self._level_files.append(wsi_file)
                    elif wsi_file.image_type == ImageType.LABEL:
                        self._label_files.append(wsi_file)
                    elif wsi_file.image_type == ImageType.OVERVIEW:
                        self._overview_files.append(wsi_file)
                except WsiDicomNotSupportedError:
                    warnings.warn(f"Non-supported file {filepath}")
                    if filepath is not None:
                        file.close()
            elif sop_class_uid == ANN_SOP_CLASS_UID:
                self._annotation_files.append(file)
            elif filepath is not None:
                # File was opened but not supported
                file.close()
        if len(self._level_files) == 0:
            raise WsiDicomNotFoundError("Level files", str(files))
        self._base_dataset = self._get_base_dataset(self._level_files)
        self._slide_uids = self._base_dataset.uids.slide
        self._base_tile_size = self._base_dataset.tile_size

    @property
    def base_dataset(self) -> WsiDataset:
        """The dataset of the base level instance."""
        return self._base_dataset

    @property
    def level_instances(self) -> Iterable[WsiInstance]:
        """The level instances parsed from the source."""
        return self._open_files(
            self._level_files, self._slide_uids, self._base_tile_size
        )

    @property
    def label_instances(self) -> Iterable[WsiInstance]:
        """The label instances parsed from the source."""
        return self._open_files(self._label_files, self._slide_uids)

    @property
    def overview_instances(self) -> Iterable[WsiInstance]:
        """The overview instances parsed from the source."""
        return self._open_files(self._overview_files, self._slide_uids)

    @property
    def annotation_instances(self) -> Iterable[AnnotationInstance]:
        """The annotation instances parsed from the source."""
        return AnnotationInstance.open(self._annotation_files)

    def close(self) -> None:
        """Close all opened files in the source."""
        for image_file in self.image_files:
            image_file.close()

    @property
    def image_files(self) -> List[WsiDicomFile]:
        """Return the image files in the source."""
        file_lists: List[List[WsiDicomFile]] = [
            self._level_files,
            self._label_files,
            self._overview_files,
        ]
        return [file for file_list in file_lists for file in file_list]

    @property
    def is_ready_for_viewing(self) -> Optional[bool]:
        """
        Returns True if files in source are formated for fast viewing.

        Returns None if no files are in source.
        """
        if not self.contains_levels:
            return None
        files = sorted(self.image_files, key=lambda file: file.frame_count)
        for file in files:
            if file.image_type is None:
                continue
            if (
                file.dataset.tile_type != TileType.SPARSE
                or file.offset_table_type is None
            ):
                return False

        return True

    @property
    def contains_levels(self) -> bool:
        """Returns true source has one level that can be read with WsiDicom."""
        return len(self.image_files) > 0

    @staticmethod
    def _open_inputs(
        files: Union[str, Path, BinaryIO, Sequence[Union[str, Path, BinaryIO]]],
    ) -> Iterable[Tuple[BinaryIO, Optional[Path]]]:
        """
        Return streams for input files.

        Parameters
        ----------
        files: Union[str, Path, BinaryIO, Sequence[Union[str, Path, BinaryIO]]],
            Files to open

        Returns
        ----------
        Iterable[Tuple[BinaryIO, Optional[Path]]]:
            Iterable of streams and optional filename (if opened file).
        """

        def open_item(
            item: Union[str, Path, BinaryIO]
        ) -> Tuple[BinaryIO, Optional[Path]]:
            if isinstance(item, (str, Path)):
                return open(item, "rb"), Path(item)
            return item, None

        if isinstance(files, (str, Path)):
            single_path = Path(files)
            if single_path.is_dir():
                return [open_item(file) for file in single_path.iterdir()]
            return [open_item(single_path)]
        return (
            open_item(file)
            for file in files
            if isinstance(file, BinaryIO)
            or (isinstance(file, (str, Path)) and Path(file).is_file())
        )

    @staticmethod
    def _get_base_dataset(files: Sequence[WsiDicomFile]) -> WsiDataset:
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
        return next(
            file.dataset
            for file in sorted(
                files, reverse=True, key=lambda file: file.dataset.image_size.width
            )
        )

    @classmethod
    def _get_sop_class_uids(
        cls, files: Union[str, Path, BinaryIO, Sequence[Union[str, Path, BinaryIO]]]
    ) -> Iterable[Tuple[BinaryIO, Optional[UID], Optional[Path]]]:
        """Read SOP class uids from inputs if DICOM.

        Parameters
        ----------
        files: Union[str, Path, BinaryIO, Sequence[Union[str, Path, BinaryIO]]]
            Files to get SOP class uids from.

        Returns
        ----------
        Iterable[Tuple[BinaryIO, Optional[UID], Optional[Path]]]
            Iterable of streams with dicom files
        """
        opened_files = cls._open_inputs(files)
        return (
            (file, cls._get_sop_class_uid(file), path) for (file, path) in opened_files
        )

    @staticmethod
    def _get_sop_class_uid(file: BinaryIO) -> Optional[UID]:
        """Return the SOP class UID from the file metadata."""
        try:
            file.seek(0)
            read_preamble(file, False)
            metadata: FileMetaDataset = _read_file_meta_info(file)
            file.seek(0)
            return metadata.MediaStorageSOPClassUID
        except InvalidDicomError:
            return None

    @classmethod
    def _open_files(
        cls,
        files: Sequence[WsiDicomFile],
        series_uids: SlideUids,
        series_tile_size: Optional[Size] = None,
    ) -> Iterable["WsiInstance"]:
        """
        Create instances from Dicom files.

        Only files with matching series uid and tile size, if defined, are used. Other
        files are closed.

        Parameters
        ----------
        files: Sequence[WsiDicomFile]
            Files to create instances from.
        series_uids: SlideUids
            Uid to match against.
        series_tile_size: Optional[Size]
            Tile size to match against (for level instances).

        Returns
        ----------
        Iterable[WsiInstancece]
            Iterable of created instances.
        """
        filtered_files = cls._filter_files(files, series_uids, series_tile_size)
        files_grouped_by_instance = cls._group_files(filtered_files)
        return (
            WsiInstance(
                [file.dataset for file in instance_files],
                WsiDicomFileImageData(instance_files),
            )
            for instance_files in files_grouped_by_instance.values()
        )

    @staticmethod
    def _filter_files(
        files: Iterable[WsiDicomFile],
        series_uids: SlideUids,
        series_tile_size: Optional[Size] = None,
    ) -> List[WsiDicomFile]:
        """
        Filter list of wsi dicom files on uids and tile size if defined.

        Parameters
        ----------
        files: Iterable['WsiDicomFile']
            Wsi files to filter.
        series_uids: Uids
            Uids to check against.
        series_tile_size: Optional[Size] = None
            Tile size to check against.

        Returns
        ----------
        List['WsiDicomFile']
            List of matching wsi dicom files.
        """
        valid_files: List[WsiDicomFile] = []

        for file in files:
            if file.dataset.matches_series(series_uids, series_tile_size):
                valid_files.append(file)
            else:
                warnings.warn(
                    f"{file.filepath} with uids {file.uids.slide} "
                    f"did not match series with {series_uids} "
                    f"and tile size {series_tile_size}"
                )
                file.close()

        return valid_files

    @staticmethod
    def _group_files(
        files: Iterable["WsiDicomFile"],
    ) -> Dict[str, List["WsiDicomFile"]]:
        """
        Return files grouped by instance identifier (instances).

        Parameters
        ----------
        files: Iterable[WsiDicomFile]
            Files to group into instances

        Returns
        ----------
        Dict[str, List[WsiDicomFile]]
            Files grouped by instance, with instance identifier as key.
        """
        grouped_files: Dict[str, List[WsiDicomFile]] = defaultdict(list)
        for file in files:
            grouped_files[file.uids.identifier].append(file)
        return grouped_files
