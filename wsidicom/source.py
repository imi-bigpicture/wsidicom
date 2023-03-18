from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Sequence, Union
from pydicom import Dataset

from pydicom.dataset import FileMetaDataset
from pydicom.filereader import read_file_meta_info
from pydicom.misc import is_dicom
from pydicom.uid import UID

from wsidicom.dataset import ImageType, WsiDataset
from wsidicom.errors import (
    WsiDicomNotFoundError,
)
from wsidicom.file import WsiDicomFile
from wsidicom.file.file import WsiDicomFile
from wsidicom.geometry import Size
from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.image_data.dicom_web_image_data import DicomWebImageData
from wsidicom.instance import WsiInstance
from wsidicom.uid import ANN_SOP_CLASS_UID, WSI_SOP_CLASS_UID
from wsidicom.web.web import DicomWebClient, WsiDicomWeb


class Source(metaclass=ABCMeta):
    @property
    @abstractmethod
    def base_dataset(self) -> Dataset:
        raise NotImplementedError()

    @property
    @abstractmethod
    def level_instances(self) -> List[WsiInstance]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def label_instances(self) -> List[WsiInstance]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def overview_instances(self) -> List[WsiInstance]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def annotation_instances(self) -> List[AnnotationInstance]:
        raise NotImplementedError()


class DicomFileSource(Source):
    def __init__(
        self,
        path: Union[str, Sequence[str], Path, Sequence[Path]],
        parse_pixel_data: bool = True,
    ) -> None:
        filepaths = self._get_filepaths(path)
        self._level_files: List[WsiDicomFile] = []
        self._label_files: List[WsiDicomFile] = []
        self._overview_files: List[WsiDicomFile] = []
        self._annotation_files: List[Path] = []

        for filepath in self._filter_paths(filepaths):
            sop_class_uid = self._get_sop_class_uid(filepath)
            if sop_class_uid == WSI_SOP_CLASS_UID:
                wsi_file = WsiDicomFile(filepath, parse_pixel_data=parse_pixel_data)
                if wsi_file.image_type == ImageType.VOLUME:
                    self._level_files.append(wsi_file)
                elif wsi_file.image_type == ImageType.LABEL:
                    self._label_files.append(wsi_file)
                elif wsi_file.image_type == ImageType.OVERVIEW:
                    self._overview_files.append(wsi_file)
                else:
                    wsi_file.close()
            elif sop_class_uid == ANN_SOP_CLASS_UID:
                self._annotation_files.append(filepath)
        if len(self._level_files) == 0:
            raise WsiDicomNotFoundError("Level files", str(path))
        self._base_dataset = self._get_base_dataset(self._level_files)
        self._slide_uids = self._base_dataset.uids.slide
        self._base_tile_size = self._base_dataset.tile_size

    @property
    def base_dataset(self) -> Dataset:
        return self._base_dataset

    @property
    def level_instances(self) -> List[WsiInstance]:
        return WsiInstance.open(
            self._level_files, self._slide_uids, self._base_tile_size
        )

    @property
    def label_instances(self) -> List[WsiInstance]:
        return WsiInstance.open(self._label_files, self._slide_uids)

    @property
    def overview_instances(self) -> List[WsiInstance]:
        return WsiInstance.open(self._overview_files, self._slide_uids)

    @property
    def annotation_instances(self) -> List[AnnotationInstance]:
        return AnnotationInstance.open(self._annotation_files)

    @property
    def image_files(self) -> List[WsiDicomFile]:
        file_lists: List[List[WsiDicomFile]] = [
            self._level_files,
            self._label_files,
            self._overview_files,
        ]
        return [file for file_list in file_lists for file in file_list]

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
                Path(file_path) for file_path in path if Path(file_path).is_file()
            ]
            if multiple_paths != []:
                return multiple_paths

        raise WsiDicomNotFoundError("No files found", str(path))

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
        return [path for path in filepaths if path.is_file() and is_dicom(path)]

    @staticmethod
    def _get_sop_class_uid(path: Path) -> UID:
        metadata: FileMetaDataset = read_file_meta_info(path)
        return metadata.MediaStorageSOPClassUID


class DicomWebSource(Source):
    def __init__(
        self,
        client: DicomWebClient,
        study_uid: Union[str, UID],
        series_uid: Union[str, UID],
    ):
        if not isinstance(study_uid, UID):
            study_uid = UID(study_uid)
        if not isinstance(series_uid, UID):
            series_uid = UID(series_uid)
        self._level_instances: List[WsiInstance] = []
        self._label_instances: List[WsiInstance] = []
        self._overview_instances: List[WsiInstance] = []
        for instance_uid in client.get_wsi_instances(study_uid, series_uid):
            web_instance = WsiDicomWeb(client, study_uid, series_uid, instance_uid)
            image_data = DicomWebImageData(web_instance)
            instance = WsiInstance(web_instance.dataset, image_data)
            if instance.image_type == ImageType.VOLUME:
                self._level_instances.append(instance)
            elif instance.image_type == ImageType.LABEL:
                self._label_instances.append(instance)
            elif instance.image_type == ImageType.OVERVIEW:
                self._overview_instances.append(instance)
        self._annotation_instances: List[AnnotationInstance] = []
        for instance_uid in client.get_ann_instances(study_uid, series_uid):
            instance = client.get_instance(study_uid, series_uid, instance_uid)
            annotation_instance = AnnotationInstance.open_dataset(instance)
            self._annotation_instances.append(annotation_instance)

        self._base_dataset = self._level_instances[0].dataset

    @property
    def base_dataset(self) -> Dataset:
        return self._base_dataset

    @property
    def level_instances(self) -> List[WsiInstance]:
        return self._level_instances

    @property
    def label_instances(self) -> List[WsiInstance]:
        return self._label_instances

    @property
    def overview_instances(self) -> List[WsiInstance]:
        return self._overview_instances

    @property
    def annotation_instances(self) -> List[AnnotationInstance]:
        return self._annotation_instances
