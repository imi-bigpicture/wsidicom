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
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union
import warnings

from pydicom.dataset import FileMetaDataset
from pydicom.filereader import read_file_meta_info
from pydicom.misc import is_dicom
from pydicom.uid import UID

from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.file.wsidicom_file import WsiDicomFile
from wsidicom.file.wsidicom_file_image_data import WsiDicomFileImageData
from wsidicom.geometry import Size
from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.instance import ImageType, TileType, WsiDataset, WsiInstance
from wsidicom.source import Source
from wsidicom.uid import ANN_SOP_CLASS_UID, WSI_SOP_CLASS_UID, SlideUids

"""A source for reading WSI DICOM files from disk."""


class WsiDicomFileSource(Source):
    """Source reading WSI DICOM instances from disk."""

    def __init__(
        self,
        path: Union[str, Sequence[str], Path, Sequence[Path]],
    ) -> None:
        """Create a WsiDicomFileSource.

        Parameters
        ----------
        path: Union[str, Sequence[str], Path, Sequence[Path]]
            A path to WSI DICOM files for a slide image.
        """
        filepaths = self._get_filepaths(path)
        self._level_files: List[WsiDicomFile] = []
        self._label_files: List[WsiDicomFile] = []
        self._overview_files: List[WsiDicomFile] = []
        self._annotation_files: List[Path] = []
        for filepath in self._filter_paths(filepaths):
            sop_class_uid = self._get_sop_class_uid(filepath)
            if sop_class_uid == WSI_SOP_CLASS_UID:
                try:
                    wsi_file = WsiDicomFile(filepath)
                    if wsi_file.image_type == ImageType.VOLUME:
                        self._level_files.append(wsi_file)
                    elif wsi_file.image_type == ImageType.LABEL:
                        self._label_files.append(wsi_file)
                    elif wsi_file.image_type == ImageType.OVERVIEW:
                        self._overview_files.append(wsi_file)
                except Exception:
                    warnings.warn(f"Non-supported file {filepath}")
            elif sop_class_uid == ANN_SOP_CLASS_UID:
                self._annotation_files.append(filepath)
        if len(self._level_files) == 0:
            raise WsiDicomNotFoundError("Level files", str(path))
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
        """Returns rue if files in source are formated for fast viewing, None if no
        files are in source"""
        if not self.contains_levels:
            return None
        files = sorted(
            self.image_files, key=lambda file: os.path.getsize(file.filepath)
        )
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
        """Returns true if files in source have one level that can be read with
        WsiDicom."""
        return len(self.image_files) > 0

    @staticmethod
    def _get_filepaths(
        path: Union[str, Sequence[str], Path, Sequence[Path]]
    ) -> Iterable[Path]:
        """Return file paths to files in path.

        If path is folder, return list of folder files in path.
        If path is single file, return list of that path.
        If path is list, return list of paths that are files.

        Parameters
        ----------
        path: path: Union[str, Sequence[str], Path, Sequence[Path]]
            Path to folder, file or list of files

        Returns
        ----------
        Iterable[Path]
            Iterable of found file paths
        """
        if isinstance(path, (str, Path)):
            single_path = Path(path)
            if single_path.is_dir():
                return single_path.iterdir()
            return [single_path]
        return (Path(file_path) for file_path in path if Path(file_path).is_file())

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

    @staticmethod
    def _filter_paths(filepaths: Iterable[Path]) -> Iterable[Path]:
        """Filter list of paths to only include valid dicom files.

        Parameters
        ----------
        filepaths: Iterable[Path]
            Paths to filter

        Returns
        ----------
        Iterable[Path]
            Iterable of paths with dicom files
        """
        return (path for path in filepaths if path.is_file() and is_dicom(path))

    @staticmethod
    def _get_sop_class_uid(path: Path) -> UID:
        """Return the SOP class UID from the file metadata."""
        metadata: FileMetaDataset = read_file_meta_info(path)
        return metadata.MediaStorageSOPClassUID

    @classmethod
    def _open_files(
        cls,
        files: Sequence[WsiDicomFile],
        series_uids: SlideUids,
        series_tile_size: Optional[Size] = None,
    ) -> Iterable["WsiInstance"]:
        """Create instances from Dicom files. Only files with matching series
        uid and tile size, if defined, are used. Other files are closed.

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
        """Filter list of wsi dicom files to only include matching uids and
        tile size if defined.

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
        """Return files grouped by instance identifier (instances).

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
