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

"""A source for reading WSI DICOM files."""

import io
import logging
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional, Union

from pydicom import dcmread
from pydicom.errors import InvalidDicomError
from pydicom.fileset import FileSet
from pydicom.uid import UID, MediaStorageDirectoryStorage

from wsidicom.errors import (
    WsiDicomNotFoundError,
    WsiDicomNotSupportedError,
)
from wsidicom.file.io import WsiDicomReader
from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.file.wsidicom_file_image_data import WsiDicomFileImageData
from wsidicom.geometry import Size
from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.instance import ImageType, TileType, WsiDataset, WsiInstance
from wsidicom.source import Source
from wsidicom.uid import ANN_SOP_CLASS_UID, WSI_SOP_CLASS_UID, SlideUids


class WsiDicomFileSource(Source):
    """Source reading WSI DICOM file instances."""

    def __init__(
        self,
        files: Union[str, Path, BinaryIO, Iterable[Union[str, Path, BinaryIO]]],
    ) -> None:
        """Create a WsiDicomFileSource.

        Parameters
        ----------
        files: Union[str, Path, BinaryIO, Iterable[Union[str, Path, BinaryIO]]],
            Files to open. Can be a path or stream for a single file, a list of paths or
            streams for multiple files, or a path to a folder containing files.
        """
        self._levels: List[WsiDicomReader] = []
        self._labels: List[WsiDicomReader] = []
        self._overviews: List[WsiDicomReader] = []
        self._annotations: List[WsiDicomIO] = []
        for file in self._list_input_files(files):
            try:
                stream = self._open_file(file)
                sop_class_uid = self._get_sop_class_uid(stream)
                if sop_class_uid == WSI_SOP_CLASS_UID:
                    try:
                        reader = WsiDicomReader(stream)
                        if reader.image_type == ImageType.VOLUME:
                            self._levels.append(reader)
                        elif reader.image_type == ImageType.LABEL:
                            self._labels.append(reader)
                        elif reader.image_type == ImageType.OVERVIEW:
                            self._overviews.append(reader)
                    except WsiDicomNotSupportedError:
                        logging.info(f"Non-supported file {stream}.")
                        if stream.filepath is not None:
                            stream.close()
                elif sop_class_uid == ANN_SOP_CLASS_UID:
                    self._annotations.append(stream)
                elif stream.filepath is not None:
                    logging.debug(
                        f"Non-supported SOP class {sop_class_uid} "
                        f"for file {stream}."
                    )
                    # File was opened but not supported SOP class.
                    stream.close()
            except Exception:
                logging.error(
                    f"Failed to open file {file.name} due to exception", exc_info=True
                )
        if len(self._levels) == 0:
            raise WsiDicomNotFoundError("Level files", str(files))
        self._base_dataset = self._get_base_dataset(self._levels)
        self._slide_uids = self._base_dataset.uids.slide
        self._base_tile_size = self._base_dataset.tile_size

    @property
    def base_dataset(self) -> WsiDataset:
        """The dataset of the base level instance."""
        return self._base_dataset

    @property
    def level_instances(self) -> Iterable[WsiInstance]:
        """The level instances parsed from the source."""
        return self._create_instances(
            self._levels, self._slide_uids, self._base_tile_size
        )

    @property
    def label_instances(self) -> Iterable[WsiInstance]:
        """The label instances parsed from the source."""
        return self._create_instances(self._labels, self._slide_uids)

    @property
    def overview_instances(self) -> Iterable[WsiInstance]:
        """The overview instances parsed from the source."""
        return self._create_instances(self._overviews, self._slide_uids)

    @property
    def annotation_instances(self) -> Iterable[AnnotationInstance]:
        """The annotation instances parsed from the source."""
        return [
            AnnotationInstance.open_dataset(file.read_dataset())
            for file in self._annotations
        ]

    def close(self) -> None:
        """Close all opened readers in the source. Does not close provided streams."""
        for reader in self.readers:
            reader.close()

    @property
    def readers(self) -> List[WsiDicomReader]:
        """Return the readers in the source."""
        reader_lists: List[List[WsiDicomReader]] = [
            self._levels,
            self._labels,
            self._overviews,
        ]
        return [reader for reader_list in reader_lists for reader in reader_list]

    @property
    def is_ready_for_viewing(self) -> Optional[bool]:
        """
        Returns True if files in source are formatted for fast viewing.

        Returns None if no files are in source.
        """
        if not self.contains_levels:
            return None
        files = sorted(self.readers, key=lambda file: file.frame_count)
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
        return len(self._levels) > 0

    @classmethod
    def open_dicomdir(cls, path: Union[str, Path]):
        """Open a DICOMDIR file and return a WsiDicomFileSource for contained files.

        Parameters
        ----------
        path: Union[str, Path]
            Path to DICOMDIR file.

        Returns
        ----------
        WsiDicomFileSource
            Source for files in DICOMDIR.
        """
        dicomdir = dcmread(path)
        if dicomdir.file_meta.MediaStorageSOPClassUID != MediaStorageDirectoryStorage:
            raise ValueError()
        fileset = FileSet(dicomdir)
        files = [file.path for file in fileset]
        return cls(files)

    @staticmethod
    def _open_file(file: Union[Path, BinaryIO]) -> WsiDicomIO:
        """Open stream if file is path. Return stream and optional filepath."""
        if isinstance(file, Path):
            return WsiDicomIO.open(file, "rb")
        return WsiDicomIO(file)

    @staticmethod
    def _list_input_files(
        files: Union[str, Path, BinaryIO, Iterable[Union[str, Path, BinaryIO]]],
    ) -> Iterable[Union[Path, BinaryIO]]:
        """List input files. Iterate directory content if directory.

        Parameters
        ----------
        files: Union[str, Path, BinaryIO, Iterable[Union[str, Path, BinaryIO]]],
            Files or directory to list.

        Returns
        ----------
        Iterable[Tuple[BinaryIO, Optional[Path]]]:
            Iterable files to open.
        """

        if isinstance(files, (str, Path)):
            # Path to single file or folder with files.
            single_path = Path(files)
            if single_path.is_dir():
                return (file for file in single_path.iterdir() if file.is_file())
            if single_path.is_file():
                return [single_path]
            raise ValueError(f"File in path {single_path} was not a file or directory.")

        if isinstance(files, BinaryIO):
            # Single stream.
            return [files]

        # Multiple paths or streams.
        return (
            Path(file) if isinstance(file, str) else file
            for file in files
            if isinstance(file, io.IOBase)
            or (isinstance(file, (str, Path)) and Path(file).is_file())
        )

    @staticmethod
    def _get_base_dataset(files: Iterable[WsiDicomReader]) -> WsiDataset:
        """Return file with largest image (width) from list of files.

        Parameters
        ----------
        files: Iterable[WsiDicomReader]
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
    def _get_sop_class_uid(stream: WsiDicomIO) -> Optional[UID]:
        """Return the SOP class UID from file metadata or None if invalid DICOM."""
        try:
            return stream.read_media_storage_sop_class_uid()
        except InvalidDicomError as exception:
            logging.debug(
                f"Failed to parse DICOM file metadata for file {stream}, not DICOM? "
                f"Got exception {exception}"
            )
            return None

    @classmethod
    def _create_instances(
        cls,
        files: Iterable[WsiDicomReader],
        series_uids: SlideUids,
        series_tile_size: Optional[Size] = None,
    ) -> Iterable["WsiInstance"]:
        """
        Create instances from Dicom files.

        Only files with matching series uid and tile size, if defined, are used. Other
        files are closed.

        Parameters
        ----------
        files: Sequence[WsiDicomReader]
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
        files: Iterable[WsiDicomReader],
        series_uids: SlideUids,
        series_tile_size: Optional[Size] = None,
    ) -> Iterable[WsiDicomReader]:
        """
        Filter list of wsi dicom files on uids and tile size if defined.

        Parameters
        ----------
        files: Iterable[WsiDicomReader]
            Wsi files to filter.
        series_uids: Uids
            Uids to check against.
        series_tile_size: Optional[Size] = None
            Tile size to check against.

        Returns
        ----------
        List[WsiDicomReader]
            List of matching wsi dicom files.
        """
        for file in files:
            if file.dataset.matches_series(series_uids, series_tile_size):
                yield file
            else:
                logging.warning(
                    f"{file.filepath} with uids {file.uids.slide} "
                    f"did not match series with {series_uids} "
                    f"and tile size {series_tile_size}"
                )
                file.close()

    @staticmethod
    def _group_files(
        files: Iterable[WsiDicomReader],
    ) -> Dict[str, List[WsiDicomReader]]:
        """
        Return files grouped by instance identifier (instances).

        Parameters
        ----------
        files: Iterable[WsiDicomReader]
            Files to group into instances

        Returns
        ----------
        Dict[str, List[WsiDicomReader]]
            Files grouped by instance, with instance identifier as key.
        """
        grouped_files: Dict[str, List[WsiDicomReader]] = defaultdict(list)
        for file in files:
            grouped_files[file.uids.identifier].append(file)
        return grouped_files
