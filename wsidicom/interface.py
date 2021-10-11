import io
import math
import os
import threading
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import (Callable, DefaultDict, Dict, List, Optional, OrderedDict,
                    Set, Tuple, Union)

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import UID as Uid

from wsidicom.errors import (WsiDicomError, WsiDicomFileError,
                             WsiDicomMatchError, WsiDicomNotFoundError,
                             WsiDicomOutOfBoundsError,
                             WsiDicomUidDuplicateError)
from wsidicom.geometry import Point, PointMm, Region, RegionMm, Size, SizeMm
from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.optical import OpticalManager
from wsidicom.stringprinting import (dict_pretty_str, list_pretty_str,
                                     str_indent)
from wsidicom.uid import (ANN_SOP_CLASS_UID, WSI_SOP_CLASS_UID, BaseUids,
                          FileUids)


class WsiDataset(Dataset):
    """Extend pydicom.dataset.Dataset (containing WSI metadata) with simple
    parsers for attributes specific for WSI. Use snake case to avoid name
    collision with dicom fields (that are handled by pydicom.dataset.Dataset).
    """
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self._instance_uid = Uid(self.SOPInstanceUID)
        self._concatenation_uid = getattr(
            self, 'SOPInstanceUIDOfConcatenationSource', None
        )
        self._base_uids = BaseUids(
            self.StudyInstanceUID,
            self.SeriesInstanceUID,
            self.FrameOfReferenceUID,
        )
        self._uids = FileUids(
            self.instance_uid,
            self.concatenation_uid,
            self.base_uids
        )
        if self.concatenation_uid is None:
            self._frame_offset = 0
        else:
            try:
                self._frame_offset = int(self.ConcatenationFrameOffsetNumber)
            except AttributeError:
                raise WsiDicomError(
                    'Concatenated file missing concatenation frame offset'
                    'number'
                )
        self._frame_count = int(getattr(self, 'NumberOfFrames', 1))
        if(getattr(self, 'DimensionOrganizationType', '') == 'TILED_FULL'):
            self._tile_type = 'TILED_FULL'
        elif 'PerFrameFunctionalGroupsSequence' in self:
            self._tile_type = 'TILED_SPARSE'
        else:
            WsiDicomError("undetermined tile type")
        self._pixel_measure = (
            self.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
        )
        pixel_spacing: Tuple[float, float] = self.pixel_measure.PixelSpacing
        if any([spacing == 0 for spacing in pixel_spacing]):
            raise WsiDicomError("Pixel spacing is zero")
        self._pixel_spacing = SizeMm(pixel_spacing[0], pixel_spacing[1])
        self._spacing_between_slices = getattr(
            self.pixel_measure, 'SpacingBetweenSlices', 0.0
        )
        self._number_of_focal_planes = getattr(
            self, 'TotalPixelMatrixFocalPlanes', 1
        )
        if (
            'PerFrameFunctionalGroupsSequence' in self and
            (
                'PlanePositionSlideSequence' in
                self.PerFrameFunctionalGroupsSequence[0]
            )
        ):
            self._frame_sequence = self.PerFrameFunctionalGroupsSequence
        else:
            self._frame_sequence = self.SharedFunctionalGroupsSequence
        self._ext_depth_of_field = self.ExtendedDepthOfField == 'YES'
        self._ext_depth_of_field_planes = getattr(
            self, 'NumberOfFocalPlanes', None
        )
        self._ext_depth_of_field_plane_distance = getattr(
            self, 'DistanceBetweenFocalPlanes', None
        )
        self._focus_method = str(self.FocusMethod)
        self._image_size = Size(
            self.TotalPixelMatrixColumns,
            self.TotalPixelMatrixRows
        )
        if self.image_size.width == 0 or self.image_size.height == 0:
            raise WsiDicomFileError(self.filepath, "Image size is zero")

        self._mm_size = SizeMm(self.ImagedVolumeWidth, self.ImagedVolumeHeight)
        self._mm_depth = self.ImagedVolumeDepth
        self._tile_size = Size(self.Columns, self.Rows)
        self._samples_per_pixel = self.SamplesPerPixel
        self._photometric_interpretation = self.PhotometricInterpretation
        self._instance_number = self.InstanceNumber
        self._optical_path_sequence = self.OpticalPathSequence
        try:
            self._slice_thickness = self.pixel_measure.SliceThickness
        except AttributeError:
            # This might not be correct if multiple focal planes
            self._slice_thickness = self.mm_depth

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of dataset {self.instance_uid}"

    @classmethod
    def is_wsi_dicom(self, datset: Dataset) -> bool:
        """Check if dataset is dicom wsi type and that required attributes
        (for the function of the library) is available.
        Warn if attribute listed as requierd in the library or required in the
        standard is missing.

        Returns
        ----------
        bool
            True if file is wsi dicom SOP class and all required attributes
            are available
        """
        REQURED_GENERAL_STUDY_MODULE_ATTRIBUTES = [
            "StudyInstanceUID"
        ]
        REQURED_GENERAL_SERIES_MODULE_ATTRIBUTES = [
            "SeriesInstanceUID"
        ]
        STANDARD_GENERAL_SERIES_MODULE_ATTRIBUTES = [
            "Modality"
        ]
        REQURED_FRAME_OF_REFERENCE_MODULE_ATTRIBUTES = [
            "FrameOfReferenceUID"
        ]
        STANDARD_ENHANCED_GENERAL_EQUIPMENT_MODULE_ATTRIBUTES = [
            "Manufacturer",
            "ManufacturerModelName",
            "DeviceSerialNumber",
            "SoftwareVersions"
        ]
        REQURED_IMAGE_PIXEL_MODULE_ATTRIBUTES = [
            "Rows",
            "Columns",
            "SamplesPerPixel",
            "PhotometricInterpretation"
        ]
        STANDARD_IMAGE_PIXEL_MODULE_ATTRIBUTES = [
            "BitsAllocated",
            "BitsStored",
            "HighBit",
            "PixelRepresentation"

        ]
        REQURED_WHOLE_SLIDE_MICROSCOPY_MODULE_ATTRIBUTES = [
            "ImageType",
            "TotalPixelMatrixColumns",
            "TotalPixelMatrixRows"
        ]
        STANDARD_WHOLE_SLIDE_MICROSCOPY_MODULE_ATTRIBUTES = [
            "TotalPixelMatrixOriginSequence",
            "FocusMethod",
            "ExtendedDepthOfField",
            "ImageOrientationSlide",
            "AcquisitionDateTime",
            "LossyImageCompression",
            "VolumetricProperties",
            "SpecimenLabelInImage",
            "BurnedInAnnotation"
        ]
        REQURED_MULTI_FRAME_FUNCTIONAL_GROUPS_MODULE_ATTRIBUTES = [
            "NumberOfFrames",
            "SharedFunctionalGroupsSequence"
        ]
        STANDARD_MULTI_FRAME_FUNCTIONAL_GROUPS_MODULE_ATTRIBUTES = [
            "ContentDate",
            "ContentTime",
            "InstanceNumber"
        ]
        STANDARD_MULTI_FRAME_DIMENSIONAL_GROUPS_MODULE_ATTRIBUTES = [
            "DimensionOrganizationSequence"
        ]
        STANDARD_SPECIMEN_MODULE_ATTRIBUTES = [
            "ContainerIdentifier",
            "SpecimenDescriptionSequence"
        ]
        REQUIRED_OPTICAL_PATH_MODULE_ATTRIBUTES = [
            "OpticalPathSequence"
        ]
        STANDARD_SOP_COMMON_MODULE_ATTRIBUTES = [
            "SOPClassUID",
            "SOPInstanceUID"
        ]

        REQUIRED_MODULE_ATTRIBUTES = [
            REQURED_GENERAL_STUDY_MODULE_ATTRIBUTES,
            REQURED_GENERAL_SERIES_MODULE_ATTRIBUTES,
            REQURED_FRAME_OF_REFERENCE_MODULE_ATTRIBUTES,
            REQURED_IMAGE_PIXEL_MODULE_ATTRIBUTES,
            REQURED_WHOLE_SLIDE_MICROSCOPY_MODULE_ATTRIBUTES,
            REQURED_MULTI_FRAME_FUNCTIONAL_GROUPS_MODULE_ATTRIBUTES,
            REQUIRED_OPTICAL_PATH_MODULE_ATTRIBUTES
        ]

        STANDARD_MODULE_ATTRIBUTES = [
            STANDARD_GENERAL_SERIES_MODULE_ATTRIBUTES,
            STANDARD_ENHANCED_GENERAL_EQUIPMENT_MODULE_ATTRIBUTES,
            STANDARD_IMAGE_PIXEL_MODULE_ATTRIBUTES,
            STANDARD_WHOLE_SLIDE_MICROSCOPY_MODULE_ATTRIBUTES,
            STANDARD_MULTI_FRAME_FUNCTIONAL_GROUPS_MODULE_ATTRIBUTES,
            STANDARD_MULTI_FRAME_DIMENSIONAL_GROUPS_MODULE_ATTRIBUTES,
            STANDARD_SPECIMEN_MODULE_ATTRIBUTES,
            STANDARD_SOP_COMMON_MODULE_ATTRIBUTES
        ]
        TO_TEST = {
            'required': REQUIRED_MODULE_ATTRIBUTES,
            'standard': STANDARD_MODULE_ATTRIBUTES
        }
        passed = {
            'required': True,
            'standard': True
        }
        for key, module_attributes in TO_TEST.items():
            for module in module_attributes:
                for attribute in module:
                    if attribute not in datset:
                        warnings.warn(
                            f' is missing {key} attribute {attribute}'
                        )
                        passed[key] = False

        sop_class_uid = getattr(datset, "SOPClassUID", "")
        sop_class_uid_check = (sop_class_uid == WSI_SOP_CLASS_UID)
        return passed['required'] and sop_class_uid_check

    @staticmethod
    def check_duplicate_dataset(
        datasets: List[Dataset],
        caller: object
    ) -> None:
        """Check for duplicates in a list of datasets. Datasets are duplicate
        if instance uids match. Stops at first found duplicate and raises
        WsiDicomUidDuplicateError.

        Parameters
        ----------
        datasets: List[Dataset]
            List of datasets to check.
        caller: Object
            Object that the files belongs to.
        """
        instance_uids: List[Uid] = []

        for dataset in datasets:
            instance_uid = Uid(dataset.SOPInstanceUID)
            if instance_uid not in instance_uids:
                instance_uids.append(instance_uid)
            else:
                raise WsiDicomUidDuplicateError(str(dataset), str(caller))

    def matches_instance(self, other_dataset: 'WsiDataset') -> bool:
        """Return true if other file is of the same instance as self.

        Parameters
        ----------
        other_dataset: 'WsiDataset
            Dataset to check.

        Returns
        ----------
        bool
            True if same instance.
        """
        return (
            self.uids.match(other_dataset.uids) and
            self.image_size == other_dataset.image_size and
            self.tile_size == other_dataset.tile_size and
            self.tile_type == other_dataset.tile_type
        )

    def matches_series(self, uids: BaseUids, tile_size: Size = None) -> bool:
        """Check if instance is valid (Uids and tile size match).
        Base uids should match for instances in all types of series,
        tile size should only match for level series.
        """
        if tile_size is not None and tile_size != self.tile_size:
            return False
        return uids == self.base_uids

    def get_supported_wsi_dicom_type(
        self,
        transfer_syntax_uid: Uid
    ) -> str:
        """Check image flavor and transfer syntax of dicom dataset.
        Return image flavor if file valid.

        Parameters
        ----------
        transfer_syntax_uid: Uid'
            Transfer syntax uid for file.

        Returns
        ----------
        str
            WSI image flavor
        """
        SUPPORTED_IMAGE_TYPES = ['VOLUME', 'LABEL', 'OVERVIEW']
        IMAGE_FLAVOR_INDEX_IN_IMAGE_TYPE = 2
        image_type: str = self.ImageType[IMAGE_FLAVOR_INDEX_IN_IMAGE_TYPE]
        image_type_supported = image_type in SUPPORTED_IMAGE_TYPES
        if not image_type_supported:
            warnings.warn(f"Non-supported image type {image_type}")

        syntax_supported = (
            pydicom.pixel_data_handlers.pillow_handler.
            supports_transfer_syntax(transfer_syntax_uid)
        )
        if not syntax_supported:
            warnings.warn(
                "Non-supported transfer syntax "
                f"{transfer_syntax_uid}"
            )
        if image_type_supported and syntax_supported:
            return image_type
        return ""

    def read_optical_path_identifier(self, frame: Dataset) -> str:
        """Return optical path identifier from frame, or from self if not
        found."""
        optical_sequence = getattr(
            frame,
            'OpticalPathIdentificationSequence',
            self.OpticalPathSequence
        )
        return getattr(optical_sequence[0], 'OpticalPathIdentifier', '0')

    @property
    def instance_uid(self) -> Uid:
        """Return instance uid from dataset."""
        return self._instance_uid

    @property
    def concatenation_uid(self) -> Optional[Uid]:
        """Return concatenation uid, if defined, from dataset. An instance that
        is concatenated (split into several files) should have the same
        concatenation uid."""
        return self._concatenation_uid

    @property
    def base_uids(self) -> BaseUids:
        """Return base uids (study, series, and frame of reference Uids)."""
        return self._base_uids

    @property
    def uids(self) -> FileUids:
        """Return instance, concatenation, and base Uids."""
        return self._uids

    @property
    def frame_offset(self) -> int:
        """Return frame offset (offset to first frame in instance if
        concatenated). Is zero if non-catenated instance or first instance
        in concatenated instance."""
        return self._frame_offset

    @property
    def frame_count(self) -> int:
        """Return number of frames in instance."""
        return self._frame_count

    @property
    def tile_type(self) -> str:
        """Return tiling type from dataset. Raises WsiDicomError if type
        is undetermined.

        Parameters
        ----------
        ds: Dataset
            Pydicom dataset.

        Returns
        ----------
        str
            Tiling type
        """
        return self._tile_type

    @property
    def pixel_measure(self) -> Dataset:
        """Return pixel measure from dataset."""
        return self._pixel_measure

    @property
    def pixel_spacing(self) -> SizeMm:
        """Read pixel spacing from dicom dataset.

        Parameters
        ----------
        ds: Dataset
            Pydicom dataset

        Returns
        ----------
        SizeMm
            The pixel spacing in mm/pixel.
        """
        return self._pixel_spacing

    @property
    def spacing_between_slices(self) -> float:
        """Return spacing between slices."""
        return self._spacing_between_slices

    @property
    def number_of_focal_planes(self) -> int:
        """Return number of focal planes."""
        return self._number_of_focal_planes

    @property
    def frame_sequence(self) -> DicomSequence:
        """Return frame sequence from dataset."""
        return self._frame_sequence

    @property
    def ext_depth_of_field(self) -> bool:
        """Return true if instance has extended depth of field
        (several focal planes are combined to one plane)."""
        return self._ext_depth_of_field

    @property
    def ext_depth_of_field_planes(self) -> Optional[int]:
        """Return number of focal planes used for extended depth of
        field."""
        return self._ext_depth_of_field_planes

    @property
    def ext_depth_of_field_plane_distance(self) -> Optional[int]:
        """Return total focal depth used for extended depth of field.
        """
        return self._ext_depth_of_field_plane_distance

    @property
    def focus_method(self) -> str:
        """Return focus method."""
        return self._focus_method

    @property
    def image_size(self) -> Size:
        """Read total pixel size from dataset.

        Returns
        ----------
        Size
            The image size
        """
        return self._image_size

    @property
    def mm_size(self) -> SizeMm:
        """Read mm size from dataset.

        Returns
        ----------
        SizeMm
            The size of the image in mm
        """
        return self._mm_size

    @property
    def mm_depth(self) -> float:
        """Return depth of image in mm."""
        return self._mm_depth

    @property
    def tile_size(self) -> Size:
        """Read tile size from from dataset.

        Returns
        ----------
        Size
            The tile size
        """
        return self._tile_size

    @property
    def samples_per_pixel(self) -> int:
        """Return samples per pixel (3 for RGB)."""
        return self._samples_per_pixel

    @property
    def photometric_interpretation(self) -> str:
        """Return photometric interpretation."""
        return self._photometric_interpretation

    @property
    def instance_number(self) -> str:
        """Return instance number."""
        return self._instance_number

    @property
    def optical_path_sequence(self) -> DicomSequence:
        """Return optical path sequence from dataset."""
        return self._optical_path_sequence

    @property
    def slice_thickness(self) -> float:
        """Return slice thickness."""
        return self._slice_thickness


class WsiDicomFile:
    """Represents a DICOM file (potentially) containing WSI image and metadata.
    """
    def __init__(self, filepath: Path):
        """Open dicom file in filepath. If valid wsi type read required
        parameters. Parses frames in pixel data but does not read the frames.

        Parameters
        ----------
        filepath: Path
            Path to file to open
        """
        self._filepath = filepath
        self._lock = threading.Lock()

        if not pydicom.misc.is_dicom(self.filepath):
            raise WsiDicomFileError(self.filepath, "is not a DICOM file")

        file_meta = pydicom.filereader.read_file_meta_info(self.filepath)
        self._transfer_syntax_uid = Uid(file_meta.TransferSyntaxUID)

        self._fp = pydicom.filebase.DicomFile(self.filepath, mode='rb')
        self._fp.is_little_endian = self._transfer_syntax_uid.is_little_endian
        self._fp.is_implicit_VR = self._transfer_syntax_uid.is_implicit_VR

        dataset = pydicom.dcmread(self._fp, stop_before_pixels=True)
        self._pixel_data_position = self._fp.tell()

        if WsiDataset.is_wsi_dicom(dataset):
            self._dataset = WsiDataset(dataset)
            self._wsi_type = self.dataset.get_supported_wsi_dicom_type(
                self.transfer_syntax
            )
            instance_uid = self.dataset.instance_uid
            concatenation_uid = self.dataset.concatenation_uid
            base_uids = self.dataset.base_uids
            self._uids = FileUids(instance_uid, concatenation_uid, base_uids)
            # If supported wsi type and transfer syntax, parse pixel data.
            if self._wsi_type != '':
                self._frame_offset = self.dataset.frame_offset
                self._frame_count = self.dataset.frame_count
                self._frame_positions = self._parse_pixel_data()
        else:
            self._wsi_type = ''
            warnings.warn(f"Non-supported file {filepath}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.filepath})"

    def __str__(self) -> str:
        return self.pretty_str()

    @property
    def dataset(self) -> WsiDataset:
        """Return pydicom dataset of file."""
        return self._dataset

    @property
    def filepath(self) -> Path:
        """Return filepath"""
        return self._filepath

    @property
    def wsi_type(self) -> str:
        return self._wsi_type

    @property
    def uids(self) -> FileUids:
        """Return uids"""
        return self._uids

    @property
    def transfer_syntax(self) -> Uid:
        """Return transfer syntax uid"""
        return self._transfer_syntax_uid

    @property
    def frame_offset(self) -> int:
        """Return frame offset (for concatenated file, 0 otherwise)"""
        return self._frame_offset

    @property
    def frame_positions(self) -> List[Tuple[int, int]]:
        """Return frame positions and lengths"""
        return self._frame_positions

    @property
    def frame_count(self) -> int:
        """Return number of frames"""
        return self._frame_count

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        return f"File with path: {self.filepath}"

    def get_filepointer(
        self,
        frame_index: int
    ) -> Tuple[pydicom.filebase.DicomFileLike, int, int]:
        """Return file pointer, frame position, and frame lenght for frame
        number.

        Parameters
        ----------
        frame_index: int
            Frame, including concatenation offset, to get.

        Returns
        ----------
        Tuple[pydicom.filebase.DicomFileLike, int, int]:
            File pointer, frame offset and frame lenght in number of bytes
        """
        frame_index -= self.frame_offset
        frame_position, frame_length = self.frame_positions[frame_index]
        return self._fp, frame_position, frame_length

    def close(self) -> None:
        """Close the file."""
        self._fp.close()

    def _read_bot(self) -> None:
        """Read (skips over) basic table offset (BOT). The BOT can contain
        frame positions, and the BOT should always be present but does not need
        to contain any data (exept lenght), and sometimes the content is
        corrupt. As we in that case anyway need to check the validity of the
        BOT by checking the actual frame sequence, we just skip over it.
        """
        # Jump to BOT
        offset_to_value = pydicom.filereader.data_element_offset_to_value(
            self._fp.is_implicit_VR,
            'OB'
        )
        self._fp.seek(offset_to_value, 1)
        # has_BOT, offset_table = pydicom.encaps.get_frame_offsets(self._fp)
        # Read the BOT lenght and skip over the BOT
        if(self._fp.read_tag() != pydicom.tag.ItemTag):
            raise WsiDicomFileError(self.filepath, 'No item tag after BOT')
        length = self._fp.read_UL()
        self._fp.seek(length, 1)

    def _read_positions(self) -> List[Tuple[int, int]]:
        """Get frame positions and length from sequence of frames that ends
        with a tag not equal to itemtag. fp needs to be positioned after the
        BOT.
        Each frame contains:
        item tag (4 bytes)
        item lenght (4 bytes)
        item data (item length)
        The position of item data and the item lenght is stored.

        Returns
        ----------
        list[tuple[int, int]]
            A list with frame positions and frame lengths
        """
        TAG_BYTES = 4
        LENGHT_BYTES = 4
        positions: List[Tuple[int, int]] = []
        frame_position = self._fp.tell()
        # Read items until sequence delimiter
        while(self._fp.read_tag() == pydicom.tag.ItemTag):
            # Read item length
            length = self._fp.read_UL()
            if length == 0 or length % 2:
                raise WsiDicomFileError(self.filepath, 'Invalid frame length')
            # Frame position
            position = frame_position
            positions.append((position+TAG_BYTES+LENGHT_BYTES, length))
            # Jump to end of item
            self._fp.seek(length, 1)
            frame_position = self._fp.tell()
        return positions

    def _read_sequence_delimeter(self):
        """Check if last read tag was a sequence delimter.
        Raises WsiDicomFileError otherwise.
        """
        TAG_BYTES = 4
        self._fp.seek(-TAG_BYTES, 1)
        if(self._fp.read_tag() != pydicom.tag.SequenceDelimiterTag):
            raise WsiDicomFileError(self.filepath, 'No sequence delimeter tag')

    def _read_frame_positions(self) -> List[Tuple[int, int]]:
        """Parse pixel data to get frame positions (relative to end of BOT)
        and frame lenght.

        Returns
        ----------
        list[tuple[int, int]]
            A list with frame positions and frame lengths
        """
        self._read_bot()
        positions = self._read_positions()
        self._read_sequence_delimeter()
        self._fp.seek(self._pixel_data_position, 0)  # Wind back to start
        return positions

    def _read_frame(self, frame_index: int) -> bytes:
        """Return frame data from pixel data by frame index.

        Parameters
        ----------
        frame_index: int
            Frame, including concatenation offset, to get.

        Returns
        ----------
        bytes
            The frame as bytes
        """
        fp, frame_position, frame_length = self.get_filepointer(frame_index)
        with self._lock:
            fp.seek(frame_position, 0)
            frame: bytes = fp.read(frame_length)
        return frame

    def _parse_pixel_data(self) -> List[Tuple[int, int]]:
        """Parse file pixel data, reads frame positions.
        Note that fp needs to be positionend at pixel data.

        Returns
        ----------
        List[Tuple[int, int]]
            List of frame positions and lenghts
        """
        frame_positions = self._read_frame_positions()
        fragment_count = len(frame_positions)
        if(self.frame_count != len(frame_positions)):
            raise WsiDicomFileError(
                self.filepath,
                (
                    f"Frames {self.frame_count} != Fragments {fragment_count}."
                    " Fragmented frames are not supported"
                )
            )
        return frame_positions

    @staticmethod
    def _filter_files(
        files: List['WsiDicomFile'],
        series_uids: BaseUids,
        series_tile_size: Size = None
    ) -> List['WsiDicomFile']:
        """Filter list of wsi dicom files to only include matching uids and
        tile size if defined.

        Parameters
        ----------
        files: List['WsiDicomFile']
            Wsi files to filter.
        series_uids: Uids
            Uids to check against.
        series_tile_size: Size
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
                    f'{file.filepath} with uids {file.uids.base} '
                    f'did not match series with {series_uids} '
                    f'and tile size {series_tile_size}'
                )
                file.close()

        return valid_files

    @classmethod
    def _group_files(
        cls,
        files: List['WsiDicomFile']
    ) -> Dict[str, List['WsiDicomFile']]:
        """Return files grouped by instance identifier (instances).

        Parameters
        ----------
        files: List[WsiDicomFile]
            Files to group into instances

        Returns
        ----------
        Dict[str, List[WsiDicomFile]]
            Files grouped by instance, with instance identifier as key.
        """
        grouped_files: Dict[str, List[WsiDicomFile]] = {}
        for file in files:
            try:
                grouped_files[file.uids.identifier].append(file)
            except KeyError:
                grouped_files[file.uids.identifier] = [file]
        return grouped_files


class ImageData(metaclass=ABCMeta):
    """Generic class for image data that can be inherited to implement support
    for other image/file formats. Subclasses should implement properties to get
    transfer_syntax, image_size, tile_size, pixel_spacing,  samples_per_pixel,
    and photometric_interpretation and methods get_tile() and close().
    Additionally properties focal_planes and/or optical_paths should be
    overridden if multiple focal planes or optical paths are implemented."""
    default_z: float = None

    @property
    @abstractmethod
    def transfer_syntax(self) -> Uid:
        """Should return the uid of the transfer syntax of the image."""
        raise NotImplementedError

    @property
    @abstractmethod
    def image_size(self) -> Size:
        """Should return the pixel size of the image."""
        raise NotImplementedError

    @property
    @abstractmethod
    def tile_size(self) -> Size:
        """Should return the pixel tile size of the image, or pixel size of
        the image if not tiled."""
        raise NotImplementedError

    @property
    @abstractmethod
    def pixel_spacing(self) -> SizeMm:
        """Should return the size of the pixels in mm/pixel."""
        raise NotImplementedError

    @property
    @abstractmethod
    def samples_per_pixel(self) -> int:
        """Should return number of samples per pixel (e.g. 3 for RGB."""
        raise NotImplementedError

    @property
    @abstractmethod
    def photometric_interpretation(self) -> str:
        """Should return the photophotometric interpretation of the image
        data."""
        raise NotImplementedError

    @abstractmethod
    def get_decoded_tile(
        self,
        tile: Point,
        z: float,
        path: str
    ) -> Image.Image:
        """Should return Image for tile defined by tile (x, y), z,
        and optical path."""
        raise NotImplementedError

    @abstractmethod
    def get_encoded_tile(
        self,
        tile: Point,
        z: float,
        path: str
    ) -> bytes:
        """Should return image bytes for tile defined by tile (x, y), z,
        and optical path."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Should close any open files."""
        raise NotImplementedError

    @property
    def tiled_size(self) -> Size:
        """The size of the image when divided into tiles, e.g. number of
        columns and rows of tiles. Equals (1, 1) if image is not tiled."""
        return self.image_size / self.tile_size

    @property
    def focal_planes(self) -> List[float]:
        """Focal planes avaiable in the image defined in um."""
        return [0.0]

    @property
    def optical_paths(self) -> List[str]:
        """Optical paths avaiable in the image."""
        return ['0']

    @property
    def image_mode(self) -> str:
        """Return Pillow image mode (e.g. RGB) for image data"""
        if(self.samples_per_pixel == 1):
            return "L"
        elif(self.samples_per_pixel == 3):
            return "RGB"
        raise NotImplementedError

    @property
    def blank_color(self) -> Tuple[int, int, int]:
        """Return RGB background color."""
        return self._get_blank_color(self.photometric_interpretation)

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        return str(self)

    @property
    def default_z(self) -> float:
        """Return single defined focal plane (in um) if only one focal plane
        defined. Return the middle focal plane if several focal planes are
        defined."""
        if self._default_z is None:
            default = 0
            if(len(self.focal_planes) > 1):
                smallest = min(self.focal_planes)
                largest = max(self.focal_planes)
                middle = (largest - smallest)/2
                default = min(range(len(self.focal_planes)),
                              key=lambda i: abs(self.focal_planes[i]-middle))

            self._default_z = self.focal_planes[default]

        return self._default_z

    @property
    def default_path(self) -> str:
        """Return the first defined optical path as default optical path
        identifier."""
        return self.optical_paths[0]

    @property
    def plane_region(self) -> Region:
        return Region(position=Point(0, 0), size=self.tiled_size - 1)

    def get_decoded_tiles(
        self,
        tiles: List[Point],
        z: float,
        path: str
    ) -> List[Image.Image]:
        """Return Images for tile defined by tile (x, y), z, and optical
        path."""
        return [
            self.get_decoded_tile(tile, z, path) for tile in tiles
        ]

    def get_encoded_tiles(
        self,
        tiles: List[Point],
        z: float,
        path: str
    ) -> List[bytes]:
        """Return image bytes for tile defined by tile (x, y), z, and optical
        path."""
        return [
            self.get_encoded_tile(tile, z, path) for tile in tiles
        ]

    def valid_tiles(self, region: Region, z: float, path: str) -> bool:
        """Check if tile region is inside tile geometry and z coordinate and
        optical path exists.

        Parameters
        ----------
        region: Region
            Tile region.
        z: float
            Z coordinate.
        path: str
            Optical path.
        """
        return (
            region.is_inside(self.plane_region) and
            (z in self.focal_planes) and
            (path in self.optical_paths)
        )

    def encode(self, image: Image.Image) -> bytes:
        """Encode image using transfer syntax.

        Parameters
        ----------
        image: Image.Image
            Image to encode

        Returns
        ----------
        bytes
            Encoded image as bytes

        """
        image_format, image_options = self._image_settings(
            self.transfer_syntax
        )
        with io.BytesIO() as buffer:
            image.save(buffer, format=image_format, **image_options)
            return buffer.getvalue()

    @staticmethod
    def _image_settings(
        transfer_syntax: pydicom.uid
    ) -> Tuple[str, Dict[str, int]]:
        """Return image format and options for creating encoded tiles as in the
        used transfer syntax.

        Parameters
        ----------
        transfer_syntax: pydicom.uid
            Transfer syntax to match image format and options to

        Returns
        ----------
        tuple[str, dict[str, int]]
            image format and image options

        """
        if(transfer_syntax == pydicom.uid.JPEGBaseline8Bit):
            image_format = 'jpeg'
            image_options = {'quality': 95}
        elif(transfer_syntax == pydicom.uid.JPEG2000):
            image_format = 'jpeg2000'
            image_options = {'irreversible': True}
        elif(transfer_syntax == pydicom.uid.JPEG2000Lossless):
            image_format = 'jpeg2000'
            image_options = {'irreversible': False}
        else:
            raise NotImplementedError(
                "Only supports jpeg and jpeg2000"
            )
        return (image_format, image_options)

    @staticmethod
    def _get_blank_color(
        photometric_interpretation: str
    ) -> Tuple[int, int, int]:
        """Return color to use blank tiles.

        Parameters
        ----------
        photometric_interpretation: str
            The photomoetric interpretation of the dataset

        Returns
        ----------
        int, int, int
            RGB color

        """
        BLACK = 0
        WHITE = 255
        if(photometric_interpretation == "MONOCHROME2"):
            return (BLACK, BLACK, BLACK)  # Monocrhome2 is black
        return (WHITE, WHITE, WHITE)


class DicomImageData(ImageData):
    """Represents image data read from dicom file(s). Image data can
    be sparsly or fully tiled and/or concatenated."""
    def __init__(self, files: Union[WsiDicomFile, List[WsiDicomFile]]) -> None:
        """Create DicomImageData from frame data in files.

        Parameters
        ----------
        files: Union[WsiDicomFile, List[WsiDicomFile]]
            Single or list of WsiDicomFiles containing frame data.
        """
        if not isinstance(files, list):
            files = [files]

        # Key is frame offset
        self._files = OrderedDict(
            (file.frame_offset, file) for file
            in sorted(files, key=lambda file: file.frame_offset)
        )

        base_file = files[0]
        datasets = [file.dataset for file in self._files.values()]
        if base_file.dataset.tile_type == 'TILED_FULL':
            self.tiles = FullTileIndex(datasets)
        else:
            self.tiles = SparseTileIndex(datasets)

        self._pixel_spacing = datasets[0].pixel_spacing
        self._transfer_syntax = base_file.transfer_syntax
        self._default_z: float = None
        self._photometric_interpretation = (
            datasets[0].photometric_interpretation
        )
        self._samples_per_pixel = datasets[0].samples_per_pixel

        self._blank_tile = None
        self._encoded_blank_tile = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._files.values()})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of files {self._files.values()}"

    @property
    def transfer_syntax(self) -> Uid:
        """The uid of the transfer syntax of the image."""
        return self._transfer_syntax

    @property
    def image_size(self) -> Size:
        """The pixel size of the image."""
        return self.tiles.image_size

    @property
    def tile_size(self) -> Size:
        """The pixel tile size of the image."""
        return self.tiles.tile_size

    @property
    def focal_planes(self) -> List[float]:
        """Focal planes avaiable in the image defined in um."""
        return self.tiles.focal_planes

    @property
    def optical_paths(self) -> List[str]:
        """Optical paths avaiable in the image."""
        return self.tiles.optical_paths

    @property
    def pixel_spacing(self) -> SizeMm:
        """Size of the pixels in mm/pixel."""
        return self._pixel_spacing

    @property
    def photometric_interpretation(self) -> str:
        """Return photometric interpretation."""
        return self._photometric_interpretation

    @property
    def samples_per_pixel(self) -> int:
        """Return samples per pixel (1 or 3)."""
        return self._samples_per_pixel

    @property
    def blank_tile(self) -> Image.Image:
        """Return background tile."""
        if self._blank_tile is None:
            self._blank_tile = self._create_blank_tile()
        return self._blank_tile

    @property
    def blank_encoded_tile(self) -> bytes:
        """Return encoded background tile."""
        if self._encoded_blank_tile is None:
            self._encoded_blank_tile = self.encode(self.blank_tile)
        return self._encoded_blank_tile

    def get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        frame_index = self._get_frame_index(tile, z, path)
        if frame_index == -1:
            return self.blank_encoded_tile
        return self._get_tile_frame(frame_index)

    def get_decoded_tile(
        self,
        tile: Point,
        z: float,
        path: str
    ) -> Image.Image:
        frame_index = self._get_frame_index(tile, z, path)
        if frame_index == -1:
            return self.blank_tile
        frame = self._get_tile_frame(frame_index)
        return Image.open(io.BytesIO(frame))

    def get_filepointer(
        self,
        tile: Point,
        z: float,
        path: str
    ) -> Optional[Tuple[pydicom.filebase.DicomFileLike, int, int]]:
        """Return file pointer, frame position, and frame lenght for tile with
        z and path. If frame is inside tile geometry but no tile exists in
        frame data None is returned.

        Parameters
        ----------
        tile: Point
            Tile coordinate to get.
        z: float
            z coordinate to get tile for.
        path: str
            Optical path to get tile for.

        Returns
        ----------
        Optional[Tuple[pydicom.filebase.DicomFileLike, int, int]]:
            File pointer, frame offset and frame lenght in number of bytes.
        """
        frame_index = self._get_frame_index(tile, z, path)
        if frame_index == -1:
            return None
        file = self._get_file(frame_index)
        return file.get_filepointer(frame_index)

    def _get_file(self, frame_index: int) -> WsiDicomFile:
        """Return file contaning frame index. Raises WsiDicomNotFoundError if
        frame is not found.

        Parameters
        ----------
        frame_index: int
             Frame index to get

        Returns
        ----------
        WsiDicomFile
            File containing the frame
        """
        for frame_offset, file in self._files.items():
            if (frame_index < frame_offset + file.frame_count and
                    frame_index >= frame_offset):
                return file

        raise WsiDicomNotFoundError(f"Frame index {frame_index}", "instance")

    def _get_tile_frame(self, frame_index: int) -> bytes:
        """Return tile frame for frame index.

        Parameters
        ----------
        frame_index: int
             Frame index to get

        Returns
        ----------
        bytes
            The frame in bytes
        """
        file = self._get_file(frame_index)
        tile_frame = file._read_frame(frame_index)
        return tile_frame

    def _get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Return frame index for tile. Raises WsiDicomOutOfBoundsError if
        tile, z, or path is not valid.

        Parameters
        ----------
        tile: Point
             Tile coordinate
        z: float
            Z coordinate
        path: str
            Optical identifier

        Returns
        ----------
        int
            Tile frame index
        """
        tile_region = Region(position=tile, size=Size(0, 0))
        if not self.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBoundsError(
                f"Tile region {tile_region}",
                f"plane {self.tiles.tiled_size}"
            )
        frame_index = self.tiles.get_frame_index(tile, z, path)
        return frame_index

    def is_sparse(self, tile: Point, z: float, path: str) -> bool:
        return (self.tiles.get_frame_index(tile, z, path) == -1)

    def _create_blank_tile(self) -> Image.Image:
        """Create blank tile for instance.

        Returns
        ----------
        Image.Image
            Blank tile image
        """
        return Image.new(
            mode=self.image_mode,
            size=self.tile_size.to_tuple(),
            color=self.blank_color[:self.samples_per_pixel]
        )

    def close(self) -> None:
        for file in self._files.values():
            file.close()


class DicomWsiFileWriter:
    def __init__(self, path: Path) -> None:
        """Return a dicom filepointer.

        Parameters
        ----------
        path: Path
            Path to filepointer.

        """
        self._fp = pydicom.filebase.DicomFile(path, mode='wb')
        self._fp.is_little_endian = True
        self._fp.is_implicit_VR = False

        self.__enter__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write_preamble(self) -> None:
        """Writes file preamble to file."""
        preamble = b'\x00' * 128
        self._fp.write(preamble)
        self._fp.write(b'DICM')

    def write_file_meta(self, uid: Uid, transfer_syntax: Uid) -> None:
        """Writes file meta dataset to file.

        Parameters
        ----------
        uid: Uid
            SOP instance uid to include in file.
        transfer_syntax: Uid
            Transfer syntax used in file.
        """
        meta_ds = pydicom.dataset.FileMetaDataset()
        meta_ds.TransferSyntaxUID = transfer_syntax
        meta_ds.MediaStorageSOPInstanceUID = uid
        meta_ds.MediaStorageSOPClassUID = WSI_SOP_CLASS_UID
        pydicom.dataset.validate_file_meta(meta_ds)
        pydicom.filewriter.write_file_meta_info(self._fp, meta_ds)

    def write_base(self, dataset: WsiDataset) -> None:
        """Writes base dataset to file.

        Parameters
        ----------
        dataset: WsiDataset

        """
        now = datetime.now()
        dataset.ContentDate = datetime.date(now).strftime('%Y%m%d')
        dataset.ContentTime = datetime.time(now).strftime('%H%M%S.%f')
        pydicom.filewriter.write_dataset(self._fp, dataset)

    def write_pixel_data_start(self) -> None:
        """Writes tags starting pixel data."""
        pixel_data_element = pydicom.dataset.DataElement(
            0x7FE00010,
            'OB',
            0,
            is_undefined_length=True
            )

        # Write pixel data tag
        self._fp.write_tag(pixel_data_element.tag)

        if not self._fp.is_implicit_VR:
            # Write pixel data VR (OB), two empty bytes (PS3.5 7.1.2)
            self._fp.write(bytes(pixel_data_element.VR, "iso8859"))
            self._fp.write_US(0)
        # Write unspecific length
        self._fp.write_UL(0xFFFFFFFF)

        # Write item tag and (empty) length for BOT
        self._fp.write_tag(pydicom.tag.ItemTag)
        self._fp.write_UL(0)

    def write_pixel_data(
        self,
        image_data: ImageData,
        z: float,
        path: str
    ) -> None:
        """Writes pixel data to file.

        Parameters
        ----------
        image_data: ImageData
            Image data to read pixel tiles from.
        z: float
            Focal plane to write.
        path: str
            Optical path to write.
        """
        # Single get_tile method
        # tile_points = Region(
        #     Point(0, 0),
        #     image_data.tiled_size
        # ).iterate_all()
        # for tile_point in tile_points:
        #     tile = image_data.get_tile(tile_point, z, path)
        #     for frame in pydicom.encaps.itemize_frame(tile, 1):
        #         fp.write(frame)
        minimum_chunk_size = getattr(
            image_data,
            'suggested_minimum_chunk_size',
            1
        )
        #
        number_of_chunks = 10

        chunk_size = number_of_chunks*minimum_chunk_size

        # Divide the image tiles up into chunk_size chunks (up to tiled size)
        chunked_tile_points = (
            Region(
                Point(x, y),
                Size(min(chunk_size, image_data.tiled_size.width - x), 1)
            ).iterate_all()
            for y in range(image_data.tiled_size.height)
            for x in range(0, image_data.tiled_size.width, chunk_size)
        )
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            def thread(tile_points: List[Point]) -> List[bytes]:
                # Thread that takes a chunk of tile points and returns list of
                # tile bytes
                return image_data.get_encoded_tiles(tile_points, z, path)

            # Each thread result is a list of tiles that is itemized and writen
            for thread_result in pool.map(thread, chunked_tile_points):
                for tile in thread_result:
                    for frame in pydicom.encaps.itemize_frame(tile, 1):
                        self._fp.write(frame)

    def write_pixel_data_end(self) -> None:
        """Writes tags ending pixel data."""
        self._fp.write_tag(pydicom.tag.SequenceDelimiterTag)
        self._fp.write_UL(0)

    def close(self) -> None:
        self._fp.close()


class SparseTilePlane:
    """Hold frame indices for the tiles in a sparse tiled file. Empty (sparse)
    frames are represented by -1."""
    def __init__(self, tiled_size: Size):
        """Create a SparseTilePlane of specified size.

        Parameters
        ----------
        tiled_size: Size
            Size of the tiling
        """
        self.plane = np.full(tiled_size.to_tuple(), -1, dtype=int)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({Size.from_tuple(self.plane.shape)})"

    def __str__(self) -> str:
        return self.pretty_str()

    def __getitem__(self, position: Point) -> int:
        """Get frame index from tile index at plane_position.

        Parameters
        ----------
        plane_position: Point
            Position in plane to get the frame index from

        Returns
        ----------
        int
            Frame index
        """
        frame_index = int(self.plane[position.x, position.y])
        return frame_index

    def __setitem__(self, position: Point, frame_index: int):
        """Add frame index to tile index.

        Parameters
        ----------
        plane_position: Point
            Position in plane to add the frame index
        frame_index: int
            Frame index to add to the index
        """
        self.plane[position.x, position.y] = frame_index

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        return "Sparse tile plane"


class TileIndex(metaclass=ABCMeta):
    """Index for mapping tile position to frame number. Is subclassed into
    FullTileIndex and SparseTileIndex."""
    def __init__(
        self,
        datasets: List[WsiDataset]
    ):
        """Create tile index for frames in datasets. Requires equal tile
        size for all tile planes.

        Parameters
        ----------
        datasets: List[WsiDataset]
            List of datasets containing tiled image data.

        """
        base_dataset = datasets[0]
        self._image_size = base_dataset.image_size
        self._tile_size = base_dataset.tile_size
        self._frame_count = self._read_frame_count_from_datasets(datasets)
        self._optical_paths = self._read_optical_paths_from_datasets(datasets)
        self._tiled_size = self.image_size / self.tile_size

    def __str__(self) -> str:
        return (
            f"{type(self).__name__} with image size {self.image_size}, "
            f"tile size {self.tile_size}, tiled size {self.tiled_size}, "
            f"optical paths {self.optical_paths}, "
            f"focal planes {self.focal_planes}, "
            f"and frame count {self.frame_count}"
        )

    @property
    @abstractmethod
    def focal_planes(self) -> List[float]:
        """Return list of focal planes in index."""
        raise NotImplementedError

    @property
    def image_size(self) -> Size:
        """Return image size in pixels."""
        return self._image_size

    @property
    def tile_size(self) -> Size:
        """Return tile size in pixels."""
        return self._tile_size

    @property
    def tiled_size(self) -> Size:
        """Return size of tiling (columns x rows)."""
        return self._tiled_size

    @property
    def frame_count(self) -> int:
        """Return total number of frames in index."""
        return self._frame_count

    @property
    def optical_paths(self) -> List[str]:
        """Return list of optical paths in index."""
        return self._optical_paths

    @abstractmethod
    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Abstract method for getting the frame index for a tile"""
        raise NotImplementedError

    @staticmethod
    def _read_frame_count_from_datasets(
        datasets: List[WsiDataset]
    ) -> int:
        """Return total frame count from files.

        Parameters
        ----------
        datasets: List[WsiDataset]
           List of datasets.

        Returns
        ----------
        int
            Total frame count.

        """
        count = 0
        for dataset in datasets:
            count += dataset.frame_count
        return count

    @staticmethod
    def _read_optical_paths_from_datasets(
        datasets: List[WsiDataset]
    ) -> List[str]:
        """Return list of optical path identifiers from files.

        Parameters
        ----------
        datasets: List[WsiDataset]
           List of datasets.

        Returns
        ----------
        List[str]
            Optical identifiers.

        """
        paths: Set[str] = set()
        for dataset in datasets:
            paths.update(OpticalManager.get_path_identifers(
                dataset.optical_path_sequence
            ))
        return list(paths)


class FullTileIndex(TileIndex):
    """Index for mapping tile position to frame number for datasets containing
    full tiles. Pixel data tiles are ordered by colum, row, z and path, thus
    the frame index for a tile can directly be calculated."""
    def __init__(
        self,
        datasets: List[WsiDataset]
    ):
        """Create full tile index for frames in datasets. Requires equal tile
        size for all tile planes.

        Parameters
        ----------
        datasets: List[WsiDataset]
            List of datasets containing full tiled image data.
        """
        super().__init__(datasets)
        self._focal_planes = self._read_focal_planes_from_datasets(datasets)

    @property
    def focal_planes(self) -> List[float]:
        return self._focal_planes

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f"Full tile index tile size: {self.tile_size}"
            f", plane size: {self.tiled_size}"
        )
        if depth is not None:
            depth -= 1
            if(depth < 0):
                return string
        string += (
            f" of z: {self.focal_planes} and path: {self.optical_paths}"
        )

        return string

    def get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Return frame index for a Point tile, z coordinate, and optical path
        from full tile index. Assumes that tile, z, and path are valid.

        Parameters
        ----------
        tile: Point
            Tile xy to get.
        z: float
            Z coordinate to get.
        path: str
            ID of optical path to get.

        Returns
        ----------
        int
            Frame index.
        """
        plane_offset = tile.x + self.tiled_size.width * tile.y
        z_offset = self._get_focal_plane_index(z) * self.tiled_size.area
        path_offset = (
            self._get_optical_path_index(path)
            * len(self._focal_planes) * self.tiled_size.area
        )
        return plane_offset + z_offset + path_offset

    def _read_focal_planes_from_datasets(
        self,
        datasets: List[WsiDataset]
    ) -> List[float]:
        """Return list of focal planes in datasets. Values in Pixel Measures
        Sequene are in mm.

        Parameters
        ----------
        datasets: List[WsiDataset]
           List of datasets to read focal planes from.

        Returns
        ----------
        List[float]
            Focal planes, specified in um.

        """
        MM_TO_MICRON = 1000.0
        DECIMALS = 3
        focal_planes: Set[float] = set()
        for dataset in datasets:
            slice_spacing = dataset.spacing_between_slices
            number_of_focal_planes = dataset.number_of_focal_planes
            if slice_spacing == 0 and number_of_focal_planes != 1:
                raise ValueError
            for plane in range(number_of_focal_planes):
                z = round(plane * slice_spacing * MM_TO_MICRON, DECIMALS)
                focal_planes.add(z)
        return list(focal_planes)

    def _get_optical_path_index(self, path: str) -> int:
        """Return index of the optical path in instance.
        This assumes that all files in a concatenated set contains all the
        optical path identifiers of the set.

        Parameters
        ----------
        path: str
            Optical path identifier to search for.

        Returns
        ----------
        int
            The index of the optical path identifier in the optical path
            sequence.
        """
        try:
            return next(
                (index for index, plane_path in enumerate(self._optical_paths)
                 if plane_path == path)
            )
        except StopIteration:
            raise WsiDicomNotFoundError(f"Optical path {path}", self)

    def _get_focal_plane_index(self, z: float) -> int:
        """Return index of the focal plane of z.

        Parameters
        ----------
        z: float
            The z coordinate (in um) to search for.

        Returns
        ----------
        int
            Focal plane index for z coordinate.
        """
        try:
            return next(index for index, plane in enumerate(self.focal_planes)
                        if plane == z)
        except StopIteration:
            raise WsiDicomNotFoundError(f"Z {z} in instance", self)


class SparseTileIndex(TileIndex):
    """Index for mapping tile position to frame number for datasets containing
    sparse tiles. Frame indices are retrieved from tile position, z, and path
    by finding the corresponding matching SparseTilePlane (z and path) and
    returning the frame index at tile position. If the tile is missing (due to
    the sparseness), -1 is returned."""
    def __init__(
        self,
        datasets: List[WsiDataset]
    ):
        """Create sparse tile index for frames in datasets. Requires equal tile
        size for all tile planes. Pixel data tiles are identified by the Per
        Frame Functional Groups Sequence that contains tile colum, row, z,
        path, and frame index. These are stored in a SparseTilePlane
        (one plane for every combination of z and path).

        Parameters
        ----------
        datasets: List[WsiDataset]
            List of datasets containing sparse tiled image data.
        """
        super().__init__(datasets)
        self._planes = self._read_planes_from_datasets(datasets)
        self._focal_planes = self._get_focal_planes()

    @property
    def focal_planes(self) -> List[float]:
        return self._focal_planes

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        return (
            f"Sparse tile index tile size: {self.tile_size}, "
            f"plane size: {self.tiled_size}"
        )

    def get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Return frame index for a Point tile, z coordinate, and optical
        path.

        Parameters
        ----------
        tile: Point
            Tile xy to get.
        z: float
            Z coordinate to get.
        path: str
            ID of optical path to get.

        Returns
        ----------
        int
            Frame index.
        """
        try:
            plane = self._planes[(z, path)]
        except KeyError:
            raise WsiDicomNotFoundError(
                f"Plane with z {z}, path {path}", self
            )
        frame_index = plane[tile]
        return frame_index

    def _get_focal_planes(self) -> List[float]:
        """Return list of focal planes defiend in planes.

        Returns
        ----------
        List[float]
            Focal planes, specified in um.
        """
        focal_planes: Set[float] = set()
        for z, path in self._planes.keys():
            focal_planes.add(z)
        return list(focal_planes)

    def _read_planes_from_datasets(
        self,
        datasets: List[WsiDataset]
    ) -> Dict[Tuple[float, str], SparseTilePlane]:
        """Return SparseTilePlane from planes in datasets.

        Parameters
        ----------
        datasets: List[WsiDataset]
           List of datasets to read planes from.

        Returns
        ----------
        Dict[Tuple[float, str], SparseTilePlane]
            Dict of planes with focal plane and optical identifier as key.
        """
        planes: Dict[Tuple[float, str], SparseTilePlane] = {}

        for dataset in datasets:
            frame_sequence = dataset.frame_sequence
            for i, frame in enumerate(frame_sequence):
                (tile, z) = self._read_frame_coordinates(frame)
                identifier = dataset.read_optical_path_identifier(frame)

                try:
                    plane = planes[(z, identifier)]
                except KeyError:
                    plane = SparseTilePlane(self.tiled_size)
                    planes[(z, identifier)] = plane
                plane[tile] = i + dataset.frame_offset

        return planes

    def _read_frame_coordinates(
            self,
            frame: DicomSequence

    ) -> Tuple[Point, float]:
        """Return frame coordinate (Point(x, y) and float z) of the frame.
        In the Plane Position Slide Sequence x and y are defined in mm and z in
        um.

        Parameters
        ----------
        frame: DicomSequence
            Pydicom frame sequence.

        Returns
        ----------
        Point, float
            The frame xy coordinate and z coordinate
        """
        DECIMALS = 3
        position = frame.PlanePositionSlideSequence[0]
        y = int(position.RowPositionInTotalImagePixelMatrix) - 1
        x = int(position.ColumnPositionInTotalImagePixelMatrix) - 1
        z = round(float(position.ZOffsetInSlideCoordinateSystem), DECIMALS)
        tile = Point(x=x, y=y) // self.tile_size
        return tile, z


class WsiInstance:
    """Represents a level, label, or overview wsi image, containing image data
    and datasets with metadata."""
    def __init__(
        self,
        datasets: Union[WsiDataset, List[WsiDataset]],
        image_data: ImageData
    ):
        """Create a WsiInstance from datasets with metadata and image data.

        Parameters
        ----------
        datasets: Union[WsiDataset, List[WsiDataset]]
            Single dataset or list of datasets.
        image_data: ImageData
            Image data.
        """
        if not isinstance(datasets, list):
            datasets = [datasets]
        self._datasets = datasets
        self._image_data = image_data
        self._identifier, self._uids = self._validate_instance(self.datasets)
        self._wsi_type = self.dataset.get_supported_wsi_dicom_type(
            self.image_data.transfer_syntax
        )

        if self.ext_depth_of_field:
            if self.ext_depth_of_field_planes is None:
                raise WsiDicomError("Instance Missing NumberOfFocalPlanes")
            if self.ext_depth_of_field_plane_distance is None:
                raise WsiDicomError(
                    "Instance Missing DistanceBetweenFocalPlanes"
                )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.dataset}, {self.image_data})"

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f"default z: {self.default_z} "
            f"default path: { self.default_path}"

        )
        if depth is not None:
            depth -= 1
            if(depth < 0):
                return string
        string += (
            ' ImageData ' + self._image_data.pretty_str(indent+1, depth)
        )
        return string

    @property
    def wsi_type(self) -> str:
        """Return wsi type."""
        return self._wsi_type

    @property
    def datasets(self) -> List[WsiDataset]:
        return self._datasets

    @property
    def dataset(self) -> WsiDataset:
        return self.datasets[0]

    @property
    def image_data(self) -> ImageData:
        return self._image_data

    @property
    def size(self) -> Size:
        """Return image size in pixels."""
        return self._image_data.image_size

    @property
    def tile_size(self) -> Size:
        """Return tile size in pixels."""
        return self._image_data.tile_size

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um/pixel."""
        return self.pixel_spacing*1000.0

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm/pixel."""
        return self._image_data.pixel_spacing

    @property
    def mm_size(self) -> SizeMm:
        """Return slide size in mm."""
        return self.dataset.mm_size

    @property
    def mm_depth(self) -> float:
        """Return imaged depth in mm."""
        return self.dataset.mm_depth

    @property
    def slice_thickness(self) -> float:
        """Return slice thickness."""
        return self.dataset.slice_thickness

    @property
    def slice_spacing(self) -> float:
        """Return slice spacing."""
        return self.dataset.spacing_between_slices

    @property
    def focus_method(self) -> str:
        return self.dataset.focus_method

    @property
    def ext_depth_of_field(self) -> bool:
        return self.dataset.ext_depth_of_field

    @property
    def ext_depth_of_field_planes(self) -> Optional[int]:
        return self.dataset.ext_depth_of_field_planes

    @property
    def ext_depth_of_field_plane_distance(self) -> Optional[float]:
        return self.dataset.ext_depth_of_field_plane_distance

    @property
    def identifier(self) -> Uid:
        """Return identifier (instance uid for single file instance or
        concatenation uid for multiple file instance)."""
        return self._identifier

    @property
    def instance_number(self) -> int:
        return self.dataset.instance_number

    @property
    def default_z(self) -> float:
        return self._image_data.default_z

    @property
    def default_path(self) -> str:
        return self._image_data.default_path

    @property
    def focal_planes(self) -> List[float]:
        return self._image_data.focal_planes

    @property
    def optical_paths(self) -> List[str]:
        return self._image_data.optical_paths

    @property
    def tiled_size(self) -> Size:
        return self._image_data.tiled_size

    @property
    def datasets(self) -> List[WsiDataset]:
        return self._datasets

    @property
    def uids(self) -> Optional[BaseUids]:
        """Return base uids"""
        return self._uids

    @classmethod
    def open(
        cls,
        files: List[WsiDicomFile],
        series_uids: BaseUids,
        series_tile_size: Size = None
    ) -> List['WsiInstance']:
        """Create instances from Dicom files. Only files with matching series
        uid and tile size, if defined, are used. Other files are closed.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Files to create instances from.
        series_uids: BaseUids
            Uid to match against.
        series_tile_size: Size
            Tile size to match against (for level instances).

        Returns
        ----------
        List[WsiInstancece]
            List of created instances.
        """
        filtered_files = WsiDicomFile._filter_files(
            files,
            series_uids,
            series_tile_size
        )
        files_grouped_by_instance = WsiDicomFile._group_files(filtered_files)
        return [
            cls(
                [file.dataset for file in instance_files],
                DicomImageData(instance_files)
            )
            for instance_files in files_grouped_by_instance.values()
        ]

    def stitch_tiles(
        self,
        region: Region,
        path: str = None,
        z: float = None
    ) -> Image.Image:
        """Stitches tiles together to form requested image.

        Parameters
        ----------
        region: Region
             Pixel region to stitch to image
        path: str
            Optical path
        z: float
            Z coordinate

        Returns
        ----------
        Image.Image
            Stitched image
        """
        if z is None:
            z = self.default_z
        if path is None:
            path = self.default_path

        image = Image.new(
            mode=self.image_data.image_mode,
            size=region.size.to_tuple()
        )
        stitching_tiles = self.get_tile_range(region, z, path)

        write_index = Point(x=0, y=0)
        tile = stitching_tiles.position
        for tile in stitching_tiles.iterate_all(include_end=True):
            tile_image = self.get_tile(tile, z, path)
            tile_crop = self.crop_tile(tile, region)
            tile_image = tile_image.crop(box=tile_crop.box)
            image.paste(tile_image, write_index.to_tuple())
            write_index = self._write_indexer(
                write_index,
                tile_crop.size,
                region.size
            )
        return image

    def get_tile_range(
        self,
        pixel_region: Region,
        z: float,
        path: str
    ) -> Region:
        """Return range of tiles to cover pixel region.

        Parameters
        ----------
        pixel_region: Region
            Pixel region of tiles to get
        z: float
            Z coordinate of tiles to get
        path: str
            Optical path identifier of tiles to get

        Returns
        ----------
        Region
            Region of tiles for stitching image
        """
        start = pixel_region.start // self._image_data.tile_size
        end = pixel_region.end / self._image_data.tile_size - 1
        tile_region = Region.from_points(start, end)
        if not self._image_data.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBoundsError(
                f"Tile region {tile_region}",
                f"tiled size {self._image_data.tiled_size}"
            )
        return tile_region

    def crop_encoded_tile_to_level(
        self,
        tile: Point,
        tile_frame: bytes
    ) -> bytes:
        cropped_tile_region = self.crop_to_level_size(tile)
        if cropped_tile_region.size != self.tile_size:
            image = Image.open(io.BytesIO(tile_frame))
            image.crop(box=cropped_tile_region.box_from_origin)
            tile_frame = self.encode(image)
        return tile_frame

    def crop_tile_to_level(
        self,
        tile: Point,
        tile_image: Image.Image
    ) -> Image.Image:
        cropped_tile_region = self.crop_to_level_size(tile)
        if cropped_tile_region.size != self.tile_size:
            tile_image = tile_image.crop(
                box=cropped_tile_region.box_from_origin
            )
        return tile_image

    def crop_tile(self, tile: Point, stitching: Region) -> Region:
        """Crop tile at edge of stitching region so that the tile after croping
        is inside the stitching region.

        Parameters
        ----------
        tile: Point
            Position of tile to crop
        stitching : Region
            Region of stitched image

        Returns
        ----------
        Region
            Region of tile inside stitching region
        """
        tile_region = Region(
            position=tile * self.tile_size,
            size=self.tile_size
        )
        cropped_tile_region = stitching.crop(tile_region)
        cropped_tile_region.position = (
            cropped_tile_region.position % self.tile_size
        )
        return cropped_tile_region

    def crop_to_level_size(self, item: Union[Point, Region]) -> Region:
        """Crop tile or region so that the tile (Point) or region (Region)
        after cropping is inside the image size of the level.

        Parameters
        ----------
        item: Union[Point, Region]
            Position of tile or region to crop

        Returns
        ----------
        Region
            Region of tile or region inside level image
        """
        level_region = Region(
            position=Point(x=0, y=0),
            size=self.size
        )
        if isinstance(item, Point):
            return self.crop_tile(item, level_region)
        return level_region.crop(item)

    @staticmethod
    def _write_indexer(
        index: Point,
        previous_size: Size,
        image_size: Size
    ) -> Point:
        """Increment index in x by previous width until index x exceds image
        size. Then resets index x to 0 and increments index y by previous
        height. Requires that tiles are scanned row by row.

        Parameters
        ----------
        index: Point
            The last write index position
        previouis_size: Size
            The size of the last written last tile
        image_size: Size
            The size of the image to be written

        Returns
        ----------
        Point
            The position (upper right) in image to insert the next tile into
        """
        index.x += previous_size.width
        if(index.x >= image_size.width):
            index.x = 0
            index.y += previous_size.height
        return index

    @staticmethod
    def _image_settings(
        transfer_syntax: pydicom.uid
    ) -> Tuple[str, Dict[str, int]]:
        """Return image format and options for creating encoded tiles as in the
        used transfer syntax.

        Parameters
        ----------
        transfer_syntax: pydicom.uid
            Transfer syntax to match image format and options to

        Returns
        ----------
        tuple[str, dict[str, int]]
            image format and image options

        """
        if(transfer_syntax == pydicom.uid.JPEGBaseline8Bit):
            image_format = 'jpeg'
            image_options = {'quality': 95}
        elif(transfer_syntax == pydicom.uid.JPEG2000):
            image_format = 'jpeg2000'
            image_options = {'irreversible': True}
        elif(transfer_syntax == pydicom.uid.JPEG2000Lossless):
            image_format = 'jpeg2000'
            image_options = {'irreversible': False}
        else:
            raise NotImplementedError(
                "Only supports jpeg and jpeg2000"
            )
        return (image_format, image_options)

    def get_tile(
        self,
        tile: Point,
        z: float = None,
        path: str = None
    ) -> Image.Image:
        """Get tile image at tile coordinate x, y.
        If frame is inside tile geometry but no tile exists in
        frame data (sparse) returns blank image.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        ----------
        Image.Image
            Tile image.
        """
        if z is None:
            z = self.default_z
        if path is None:
            path = self.default_path
        image = self.image_data.get_decoded_tile(tile, z, path)
        return self.crop_tile_to_level(tile, image)

    def get_encoded_tile(
        self,
        tile: Point,
        z: float,
        path: str
    ) -> bytes:
        """Get tile bytes at tile coordinate x, y
        If frame is inside tile geometry but no tile exists in
        frame data (sparse) returns encoded blank image.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate.
        z: float
            Z coordinate.
        path: str
            Optical path.
        crop: bool
            If the tile should be croped to image region.

        Returns
        ----------
        bytes
            Tile image as bytes.
        """
        if z is None:
            z = self.default_z
        if path is None:
            path = self.default_path
        tile_frame = self.image_data.get_encoded_tile(tile, z, path)

        # Check if tile is an edge tile that should be croped
        cropped_tile_region = self.crop_to_level_size(tile)
        if cropped_tile_region.size != self.tile_size:
            image = Image.open(io.BytesIO(tile_frame))
            image.crop(box=cropped_tile_region.box_from_origin)
            tile_frame = self.image_data.encode(image)
        return tile_frame

    @staticmethod
    def check_duplicate_instance(
        instances: List['WsiInstance'],
        self: object
    ) -> None:
        """Check for duplicates in list of instances. Instances are duplicate
        if instance identifier (file instance uid or concatenation uid) match.
        Stops at first found duplicate and raises WsiDicomUidDuplicateError.

        Parameters
        ----------
        instances: List['WsiInstance']
            List of instances to check.
        caller: Object
            Object that the instances belongs to.
        """
        instance_identifiers: List[str] = []
        for instance in instances:
            instance_identifier = instance.identifier
            if instance_identifier not in instance_identifiers:
                instance_identifiers.append(instance_identifier)
            else:
                raise WsiDicomUidDuplicateError(str(instance), str(self))

    def _validate_instance(
        self,
        datasets: List[WsiDataset]
    ) -> Tuple[str, BaseUids, str]:
        """Check that no files in instance are duplicate, that all files in
        instance matches (uid, type and size).
        Raises WsiDicomMatchError otherwise.
        Returns the matching file uid.


        Returns
        ----------
        Tuple[str, BaseUids, str]
            Instance identifier uid and base uids
        """
        WsiDataset.check_duplicate_dataset(datasets, self)

        base_dataset = datasets[0]
        for dataset in datasets[1:]:
            if not base_dataset.matches_instance(dataset):
                raise WsiDicomError()
        return (
            base_dataset.uids.identifier,
            base_dataset.uids.base,
        )

    def get_tile_range(
        self,
        pixel_region: Region,
        z: float,
        path: str
    ) -> Region:
        start = pixel_region.start // self.tile_size
        end = pixel_region.end / self.tile_size - 1
        tile_region = Region.from_points(start, end)
        return tile_region

    def matches(self, other_instance: 'WsiInstance') -> bool:
        """Return true if other instance is of the same group as self.

        Parameters
        ----------
        other_instance: WsiInstance
            Instance to check.

        Returns
        ----------
        bool
            True if instanes are of same group.

        """
        return (
            self.uids == other_instance.uids and
            self.size == other_instance.size and
            self.tile_size == other_instance.tile_size and
            self.wsi_type == other_instance.wsi_type
        )

    def close(self) -> None:
        self._image_data.close()


class WsiDicomGroup:
    """Represents a group of instances having the same size, but possibly
    different z coordinate and/or optical path."""
    def __init__(
        self,
        instances: List[WsiInstance]
    ):
        """Create a group of WsiInstances. Instances should match in the common
        uids, wsi type, and tile size.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to build the group.
        """
        self._instances = {  # key is identifier (Uid)
            instance.identifier: instance for instance in instances
        }
        self._validate_group()

        base_instance = instances[0]
        self._wsi_type = base_instance.wsi_type
        self._uids = base_instance.uids

        self._size = base_instance.size
        self._pixel_spacing = base_instance.pixel_spacing
        self._default_instance_uid: str = base_instance.identifier

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.instances.values()})"

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f'Image: size: {self.size} px, mpp: {self.mpp} um/px'
        )
        if depth is not None:
            depth -= 1
            if(depth < 0):
                return string
        string += (
            ' Instances: ' + dict_pretty_str(self.instances, indent, depth)
        )
        return string

    def __getitem__(self, index) -> WsiInstance:
        return self.instances[index]

    @property
    def uids(self) -> BaseUids:
        """Return uids"""
        return self._uids

    @property
    def wsi_type(self) -> str:
        """Return wsi type"""
        return self._wsi_type

    @property
    def size(self) -> Size:
        """Return image size in pixels"""
        return self._size

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um/pixel"""
        return self.pixel_spacing*1000.0

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm/pixel"""
        return self._pixel_spacing

    @property
    def instances(self) -> Dict[str, WsiInstance]:
        """Return contained instances"""
        return self._instances

    @property
    def default_instance(self) -> WsiInstance:
        """Return default instance"""
        return self.instances[self._default_instance_uid]

    @property
    def files(self) -> List[WsiDicomFile]:
        """Return contained files"""
        instance_files = [
            instance.files for instance in self.instances.values()
        ]
        return [file for sublist in instance_files for file in sublist]

    @property
    def datasets(self) -> List[Dataset]:
        """Return contained datasets."""
        instance_datasets = [
            instance.datasets for instance in self.instances.values()
        ]
        return [
            dataset for sublist in instance_datasets for dataset in sublist
        ]

    @property
    def optical_paths(self) -> List[str]:
        return list({
            path
            for instance in self.instances.values()
            for path in instance.optical_paths
        })

    @property
    def focal_planes(self) -> List[float]:
        return list({
            focal_plane
            for innstance in self.instances.values()
            for focal_plane in innstance.focal_planes
        })

    @classmethod
    def open(
        cls,
        instances: List[WsiInstance],
    ) -> List['WsiDicomGroup']:
        """Return list of groups created from wsi instances.

        Parameters
        ----------
        files: List[WsiInstance]
            Instances to create groups from.

        Returns
        ----------
        List[WsiDicomGroup]
            List of created groups.

        """
        groups: List[cls] = []

        grouped_instances = cls._group_instances(instances)

        for group in grouped_instances.values():
            groups.append(cls(group))

        return groups

    def matches(self, other_group: 'WsiDicomGroup') -> bool:
        """Check if group is valid (Uids and tile size match).
        The common Uids should match for all series.
        """
        return (
            other_group.uids == self.uids and
            other_group.wsi_type == self.wsi_type
        )

    def valid_pixels(self, region: Region) -> bool:
        """Check if pixel region is withing the size of the group image size.

        Parameters
        ----------
        region: Region
            Pixel region to check

        Returns
        ----------
        bool
            True if pixel position and size is within image
        """
        # Return true if inside pixel plane.
        image_region = Region(Point(0, 0), self.size)
        return region.is_inside(image_region)

    def get_instance(
        self,
        z: float = None,
        path: str = None
    ) -> WsiInstance:
        """Search for instance fullfilling the parameters.
        The behavior when z and/or path is none could be made more
        clear.

        Parameters
        ----------
        z: float
            Z coordinate of the searched instance
        path: str
            Optical path of the searched instance

        Returns
        ----------
        WsiInstance
            The instance containing selected path and z coordinate
        """
        if z is None and path is None:
            instance = self.default_instance
            z = instance.default_z
            path = instance.default_path

            return self.default_instance

        # Sort instances by number of focal planes (prefer simplest instance)
        sorted_instances = sorted(
            list(self._instances.values()),
            key=lambda i: len(i.focal_planes)
        )
        try:
            if z is None:
                # Select the instance with selected optical path
                instance = next(i for i in sorted_instances if
                                path in i.optical_paths)
            elif path is None:
                # Select the instance with selected z
                instance = next(i for i in sorted_instances
                                if z in i.focal_planes)
            else:
                # Select by both path and z
                instance = next(i for i in sorted_instances
                                if (z in i.focal_planes and
                                    path in i.optical_paths))
        except StopIteration:
            raise WsiDicomNotFoundError(
                f"Instance for path: {path}, z: {z}", "group"
            )
        if z is None:
            z = instance.default_z
        if path is None:
            path = instance.default_path
        return instance

    def get_default_full(self) -> Image.Image:
        """Read full image using default z coordinate and path.

        Returns
        ----------
        Image.Image
            Full image of the group.
        """
        instance = self.default_instance
        z = instance.default_z
        path = instance.default_path
        region = Region(position=Point(x=0, y=0), size=self.size)
        image = self.get_region(region, z, path)
        return image

    def get_region(
        self,
        region: Region,
        z: float = None,
        path: str = None,
    ) -> Image.Image:
        """Read region defined by pixels.

        Parameters
        ----------
        location: int, int
            Upper left corner of region in pixels
        size: int
            Size of region in pixels
        z: float
            Z coordinate, optional
        path: str
            optical path, optional

        Returns
        ----------
        Image.Image
            Region as image
        """

        instance = self.get_instance(z, path)
        image = instance.stitch_tiles(region, path, z)
        return image

    def get_region_mm(
        self,
        region: RegionMm,
        z: float = None,
        path: str = None
    ) -> Image.Image:
        """Read region defined by mm.

        Parameters
        ----------
        location: float, float
            Upper left corner of region in mm
        size: float
            Size of region in mm
        z: float
            Z coordinate, optional
        path:
            optical path, optional

        Returns
        ----------
        Image.Image
            Region as image
        """
        pixel_region = self.mm_to_pixel(region)
        image = self.get_region(pixel_region, z, path)
        return image

    def get_tile(
        self,
        tile: Point,
        z: float = None,
        path: str = None
    ) -> Image.Image:
        """Return tile at tile coordinate x, y as image.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate
        z: float
            Z coordinate
        path: str
            Optical path

        Returns
        ----------
        Image.Image
            The tile as image
        """

        instance = self.get_instance(z, path)
        return instance.get_tile(tile, z, path)

    def get_encoded_tile(
        self,
        tile: Point,
        z: float = None,
        path: str = None
    ) -> bytes:
        """Return tile at tile coordinate x, y as bytes.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate
        z: float
            Z coordinate
        path: str
            Optical path

        Returns
        ----------
        bytes
            The tile as bytes
        """
        instance = self.get_instance(z, path)
        return instance.get_encoded_tile(tile, z, path)

    def mm_to_pixel(self, region: RegionMm) -> Region:
        """Convert region in mm to pixel region.

        Parameters
        ----------
        region: RegionMm
            Region in mm

        Returns
        ----------
        Region
            Region in pixels
        """
        pixel_region = Region(
            position=region.position // self.pixel_spacing,
            size=region.size // self.pixel_spacing
        )
        if not self.valid_pixels(pixel_region):
            raise WsiDicomOutOfBoundsError(
                f"Region {region}", f"level size {self.size}"
            )
        return pixel_region

    def close(self) -> None:
        """Close all instances on the group."""
        for instance in self._instances.values():
            instance.close()

    def _validate_group(self):
        """Check that no file or instance in group is duplicate, instances in
        group matches and that the optical manager matches by base uid.
        Raises WsiDicomMatchError otherwise.
        """
        WsiDataset.check_duplicate_dataset(self.datasets, self)
        instances = list(self.instances.values())
        base_instance = instances[0]
        for instance in instances[1:]:
            if not base_instance.matches(instance):
                raise WsiDicomMatchError(str(instance), str(self))
        WsiInstance.check_duplicate_instance(instances, self)

    @classmethod
    def _group_instances(
        cls,
        instances: List[WsiInstance]
    ) -> Dict[Size, List[WsiInstance]]:
        """Return instances grouped by image size.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to group by image size.

        Returns
        ----------
        Dict[Size, List[WsiInstance]]:
            Instances grouped by size, with size as key.

        """
        grouped_instances: Dict[Size, List[WsiInstance]] = {}
        for instance in instances:
            try:
                grouped_instances[instance.size].append(instance)
            except KeyError:
                grouped_instances[instance.size] = [instance]
        return grouped_instances

    def _group_instances_to_file(
        self,
    ) -> List[List[WsiInstance]]:
        """Group instances by properties that can't differ in a DICOM-file,
        i.e. the instances are grouped by output file.

        Returns
        ----------
        List[List[WsiInstance]]
            Instances grouped by common properties.
        """
        groups: DefaultDict[Union[str, Uid], List[str]] = defaultdict(list)
        for instance in self.instances.values():
            groups[
                instance.image_data.photometric_interpretation,
                instance.image_data.transfer_syntax,
                instance.ext_depth_of_field,
                instance.ext_depth_of_field_planes,
                instance.ext_depth_of_field_plane_distance,
                instance.focus_method,
                instance.slice_spacing
            ].append(
                instance
            )
        return list(groups.values())

    @staticmethod
    def _list_image_data(
        instances: List[WsiInstance]
    ) -> Tuple[Tuple[str, float], List[ImageData]]:
        """List and sort ImageData in instances by optical path and focal
        plane.

        Parameters
        ----------
        instances: List[WsiInstance]
            List of instances with optical paths and focal planes to list and
            sort.

        Returns
        ----------
        Tuple[Tuple[str, float], List[ImageData]]
            ImageData listed and sorted by optical path and focal plane.
        """
        output: Dict[Tuple[str, float], ImageData] = {}
        for instance in instances:
            for optical_path in instance.optical_paths:
                for z in instance.focal_planes:
                    if (optical_path, z) not in output:
                        output[optical_path, z] = instance._image_data
        return OrderedDict(output).items()

    def save(
        self,
        output_path: str,
        uid_generator: Callable[..., Uid] = pydicom.uid.generate_uid
    ) -> List[Path]:
        """Save a WsiDicomGroup to files in output_path. Instances are grouped
        by properties that can differ in the same file:
            - photometric interpretation
            - transfer syntax
            - extended depth of field (and planes and distance)
            - focus method
            - spacing between slices
        Other properties are assumed to be equal or to be updated.

        Parameters
        ----------
        output_path: str
            Folder path to save files to.

        uid_generator: Callable[..., Uid] = pydicom.uid.generate_uid
            Uid generator to use.

        Returns
        ----------
        List[str]
            List of paths of created files.
        """
        filepaths: List[Path] = []
        for instances in self._group_instances_to_file():
            uid = uid_generator()
            filepath = os.path.join(output_path, uid + '.dcm')
            transfer_syntax = instances[0]._image_data.transfer_syntax
            dataset = deepcopy(instances[0].dataset)
            with DicomWsiFileWriter(filepath) as wsi_file:
                wsi_file = DicomWsiFileWriter(filepath)
                wsi_file.write_preamble()
                wsi_file.write_file_meta(uid, transfer_syntax)
                dataset.SOPInstanceUID = uid
                wsi_file.write_base(dataset)
                wsi_file.write_pixel_data_start()
                for (path, z), image_data in self._list_image_data(instances):
                    wsi_file.write_pixel_data(image_data, z, path)
                wsi_file.write_pixel_data_end()
                wsi_file.close()
            filepaths.append(filepath)
        return filepaths


class WsiDicomLevel(WsiDicomGroup):
    """Represents a level in the pyramid and contains one or more instances
    having the same pyramid level index, pixel spacing, and size but possibly
    different focal planes and/or optical paths.
    """
    def __init__(
        self,
        instances: List[WsiInstance],
        base_pixel_spacing: SizeMm
    ):
        """Create a level from list of WsiInstances. Asign the pyramid level
        index from pixel spacing of base level.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to build the level.
        base_pixel_spacing: SizeMm
            Pixel spacing of base level.
        """
        super().__init__(instances)
        self._base_pixel_spacing = base_pixel_spacing
        self._level = self._assign_level(self._base_pixel_spacing)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.instances}, "
            f"{self._base_pixel_spacing})"
        )

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f'Level: {self.level}, size: {self.size} px, mpp: {self.mpp} um/px'
        )
        if depth is not None:
            depth -= 1
            if(depth < 0):
                return string
        string += (
            ' Instances: ' + dict_pretty_str(self.instances, indent, depth)
        )
        return string

    @property
    def pyramid(self) -> str:
        """Return string representation of the level"""
        return (
            f'Level [{self.level}]'
            f' tiles: {self.default_instance.tiled_size},'
            f' size: {self.size}, mpp: {self.mpp} um/px'
        )

    @property
    def tile_size(self) -> Size:
        return self.default_instance.tile_size

    @property
    def level(self) -> int:
        """Return pyramid level"""
        return self._level

    @classmethod
    def open(
        cls,
        instances: List[WsiInstance],
    ) -> List['WsiDicomLevel']:
        """Return list of levels created wsi files.

        Parameters
        ----------
        files: List[WsiInstance]
            Instances to create levels from.

        Returns
        ----------
        List[WsiDicomLevel]
            List of created levels.

        """
        levels: List[cls] = []
        instances_grouped_by_level = cls._group_instances(instances)
        largest_size = max(instances_grouped_by_level.keys())
        base_group = instances_grouped_by_level[largest_size]
        base_pixel_spacing = base_group[0].pixel_spacing
        for level in instances_grouped_by_level.values():
            levels.append(cls(level, base_pixel_spacing))

        return levels

    def matches(self, other_group: 'WsiDicomGroup') -> bool:
        """Check if group is valid (Uids and tile size match).
        The common Uids should match for all series. For level series the tile
        size should also match. It is assumed that the instances in the groups
        are matching each other.
        """
        other_instance = other_group.default_instance
        this_instance = self.default_instance
        return (
            other_group.uids == self.uids and
            other_instance.tile_size == this_instance.tile_size
        )

    def get_highest_level(self) -> int:
        """Return highest deep zoom scale that can be produced
        from the image in the level.

        Returns
        ----------
        int
            Relative level where the pixel size becomes 1x1
        """
        return math.ceil(math.log2(max(self.size.width, self.size.height)))

    def get_scaled_tile(
        self,
        tile: Point,
        level: int,
        z: float = None,
        path: str = None
    ) -> Image.Image:
        """Return tile in another level by scaling a region.
        If the tile is an edge tile, the resulting tile is croped
        to remove part outside of the image (as defiend by level size).

        Parameters
        ----------
        tile: Point
            Non scaled tile coordinate
        level: int
            Level to scale from
        z: float
            Z coordinate
        path: str
            Optical path

        Returns
        ----------
        Image.Image
            A tile image
        """
        scale = self.calculate_scale(level)
        instance = self.get_instance(z, path)
        scaled_region = Region.from_tile(tile, instance.tile_size) * scale
        cropped_region = instance.crop_to_level_size(scaled_region)
        if not self.valid_pixels(cropped_region):
            raise WsiDicomOutOfBoundsError(
                f"Region {cropped_region}", f"level size {self.size}"
            )
        image = self.get_region(cropped_region, z, path)
        tile_size = cropped_region.size/scale
        image = image.resize(
            tile_size.to_tuple(),
            resample=Image.BILINEAR
        )
        return image

    def get_scaled_encoded_tile(
        self,
        tile: Point,
        scale: int,
        z: float = None,
        path: str = None
    ) -> bytes:
        """Return encoded tile in another level by scaling a region.

        Parameters
        ----------
        tile: Point
            Non scaled tile coordinate
        level: int
           Level to scale from
        z: float
            Z coordinate
        path: str
            Optical path

        Returns
        ----------
        bytes
            A transfer syntax encoded tile
        """
        image = self.get_scaled_tile(tile, scale, z, path)
        instance = self.get_instance(z, path)
        return instance.image_data.encode(image)

    def calculate_scale(self, level_to: int) -> int:
        """Return scaling factor to given level.

        Parameters
        ----------
        level_to -- index of level to scale to

        Returns
        ----------
        int
            Scaling factor between this level and given level
        """
        return int(2 ** (level_to - self.level))

    def _assign_level(self, base_pixel_spacing: SizeMm) -> int:
        """Return (2^level scale factor) based on pixel spacing.
        Will round to closest integer. Raises NotImplementedError if level is
        to far from integer.

        Parameters
        ----------
        base_pixel_spacing: SizeMm
            The pixel spacing of the base lavel

        Returns
        ----------
        int
            The pyramid order of the level
        """
        float_level = math.log2(
            self.pixel_spacing.width/base_pixel_spacing.width
        )
        level = int(round(float_level))
        TOLERANCE = 1e-2
        if not math.isclose(float_level, level, rel_tol=TOLERANCE):
            raise NotImplementedError(f"Levels needs to be integer")
        return level


class WsiDicomSeries(metaclass=ABCMeta):
    """Represents a series of WsiDicomGroups with the same image flavor, e.g.
    pyramidal levels, lables, or overviews.
    """

    def __init__(self, groups: List[WsiDicomGroup]):
        """Create a WsiDicomSeries from list of WsiDicomGroups.

        Parameters
        ----------
        groups: List[WsiDicomGroup]
            List of groups to include in the series.
        """
        self._groups: List[WsiDicomGroup] = groups

        if len(self.groups) != 0 and self.groups[0].uids is not None:
            self._uids = self._validate_series(self.groups)
        else:
            self._uids = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.groups})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of groups {self.groups}"

    def __getitem__(self, index: int) -> WsiDicomGroup:
        """Get group by index.

        Parameters
        ----------
        index: int
            Index in series to get

        Returns
        ----------
        WsiDicomGroup
            The group at index in the series
        """
        return self.groups[index]

    def __len__(self) -> int:
        return len(self.stacks)

    @property
    @abstractmethod
    def wsi_type(self) -> str:
        """Should return the wsi type of the series ('VOLUME', 'LABEL', or
        'OVERVIEW'"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def group_open(instances: List[WsiInstance]) -> List[WsiDicomGroup]:
        """Should return WsiDicomGroups created from instances."""
        raise NotImplementedError

    @property
    def groups(self) -> List[WsiDicomGroup]:
        """Return contained groups."""
        return self._groups

    @property
    def uids(self) -> Optional[BaseUids]:
        """Return uids."""
        return self._uids

    @property
    def mpps(self) -> List[SizeMm]:
        """Return contained mpp (um/px)."""
        return [group.mpp for group in self.groups]

    @property
    def files(self) -> List[WsiDicomFile]:
        """Return contained files."""
        series_files = [series.files for series in self.groups]
        return [file for sublist in series_files for file in sublist]

    @property
    def datasets(self) -> List[Dataset]:
        """Return contained datasets."""

        series_datasets = [
            series.datasets for series in self.groups
        ]
        return [
            dataset for sublist in series_datasets for dataset in sublist
        ]

    @property
    def instances(self) -> List[WsiInstance]:
        """Return contained instances"""
        series_instances = [
            series.instances.values() for series in self.groups
        ]
        return [
            instance for sublist in series_instances for instance in sublist
        ]

    @classmethod
    def open(
        cls,
        instances: List[WsiInstance]
    ) -> 'WsiDicomSeries':
        """Return series created from wsi files.

        Parameters
        ----------
        instances: List[WsiInstance]
            Instances to create series from.

        Returns
        ----------
        WsiDicomSeries
            Created series.
        """
        labels = cls.group_open(instances)
        return cls(labels)

    def _validate_series(
            self,
            groups: Union[List[WsiDicomGroup], List[WsiDicomLevel]]
    ) -> Optional[BaseUids]:
        """Check that no files or instances in series is duplicate and that
        all groups in series matches.
        Raises WsiDicomMatchError otherwise.
        Returns the matching base uid. If list of groups is empty, return None.

        Parameters
        ----------
        groups: Union[List[WsiDicomGroup], List[WsiDicomLevel]]
            List of groups or levels to check

        Returns
        ----------
        Optional[BaseUids]:
            Matching uids
        """
        WsiDataset.check_duplicate_dataset(self.datasets, self)
        WsiInstance.check_duplicate_instance(self.instances, self)

        try:
            base_group = groups[0]
            if base_group.wsi_type != self.wsi_type:
                raise WsiDicomMatchError(
                    str(base_group), str(self)
                )
            for group in groups[1:]:
                if not group.matches(base_group):
                    raise WsiDicomMatchError(
                        str(group), str(self)
                    )
            return base_group.uids
        except IndexError:
            return None

    def close(self) -> None:
        """Close all groups in the series."""
        for group in self.groups:
            group.close()

    def save(
        self,
        output_path: str,
        uid_generator: Callable[..., Uid] = pydicom.uid.generate_uid
    ) -> List[str]:
        """Save WsiDicomSeries as DICOM-files in path.

        Parameters
        ----------
        output_path: str
        uid_generator: Callable[..., Uid] = pydicom.uid.generate_uid
             Function that can gernerate unique identifiers.

        Returns
        ----------
        List[str]
            List of paths of created files.
        """
        filepaths: List[str] = []
        for group in self.groups:
            group_file_paths = group.save(
                output_path,
                uid_generator
            )
            filepaths.extend(group_file_paths)
        return filepaths


class WsiDicomLabels(WsiDicomSeries):
    """Represents a series of WsiDicomGroups of the label wsi flavor."""
    @property
    def wsi_type(self) -> str:
        return 'LABEL'

    @staticmethod
    def group_open(instances: List[WsiInstance]) -> List[WsiDicomGroup]:
        return WsiDicomGroup.open(instances)


class WsiDicomOverviews(WsiDicomSeries):
    """Represents a series of WsiDicomGroups of the overview wsi flavor."""
    @property
    def wsi_type(self) -> str:
        return 'OVERVIEW'

    @staticmethod
    def group_open(instances: List[WsiInstance]) -> List[WsiDicomGroup]:
        return WsiDicomGroup.open(instances)


class WsiDicomLevels(WsiDicomSeries):
    """Represents a series of WsiDicomGroups of the volume (e.g. pyramidal
    level) wsi flavor."""
    @property
    def wsi_type(self) -> str:
        return 'VOLUME'

    @staticmethod
    def group_open(instances: List[WsiInstance]) -> List[WsiDicomGroup]:
        return WsiDicomLevel.open(instances)

    def __init__(self, levels: List[WsiDicomLevel]):
        """Holds a stack of levels.

        Parameters
        ----------
        levels: List[WsiDicomLevel]
            List of levels to include in series
        """
        self._levels = OrderedDict(
            (level.level, level)
            for level in sorted(levels, key=lambda level: level.level)
        )
        if len(self.groups) != 0 and self.groups[0].uids is not None:
            self._uids = self._validate_series(self.groups)
        else:
            self._uids = None

    @property
    def pyramid(self) -> str:
        """Return string representation of pyramid"""
        return (
            'Pyramid levels in file:\n'
            + '\n'.join(
                [str_indent(2) + level.pyramid
                 for level in self._levels.values()]
            )
        )

    @property
    def groups(self) -> List[WsiDicomGroup]:
        """Return contained groups"""
        return list(self._levels.values())

    @property
    def levels(self) -> List[int]:
        """Return contained levels"""
        return list(self._levels.keys())

    @property
    def highest_level(self) -> int:
        """Return highest valid pyramid level (which results in a 1x1 image)"""
        return self.base_level.get_highest_level()

    @property
    def base_level(self) -> WsiDicomLevel:
        """Return the base level of the pyramid"""
        return self._levels[0]

    def valid_level(self, level: int) -> bool:
        """Check that given level is less or equal to the highest level
        (1x1 pixel level).

        Parameters
        ----------
        level: int
            The level to check

        Returns
        ----------
        bool
            True if level is valid
        """
        return level <= self.highest_level

    def get_level(self, level: int) -> WsiDicomLevel:
        """Return wsi level.

        Parameters
        ----------
        level: int
            The level of the wsi level to return

        Returns
        ----------
        WsiDicomLevel
            The searched level
        """
        try:
            return self._levels[level]
        except KeyError:
            raise WsiDicomNotFoundError(
                f"Level of {level}", "level series"
            )

    def get_closest_by_level(self, level: int) -> WsiDicomLevel:
        """Search for level that is closest to and smaller than the given
        level.

        Parameters
        ----------
        level: int
            The level to search for

        Returns
        ----------
        WsiDicomLevel
            The level closest to searched level
        """
        if not self.valid_level(level):
            raise WsiDicomOutOfBoundsError(
                f"Level {level}", f"maximum level {self.highest_level}"
            )
        closest_level = 0
        closest = None
        for wsi_level in self._levels.values():
            if((level >= wsi_level.level) and
               (closest_level <= wsi_level.level)):
                closest_level = wsi_level.level
                closest = wsi_level
        if closest is None:
            raise WsiDicomNotFoundError(
                f"Level for {level}", "level series"
            )
        return closest

    def get_closest_by_size(self, size: Size) -> WsiDicomLevel:
        """Search for level that by size is closest to and larger than the
        given size.

        Parameters
        ----------
        size: Size
            The size to search for

        Returns
        ----------
        WsiDicomLevel
            The level with size closest to searched size
        """
        closest_size = self.groups[0].size
        closest = None
        for wsi_level in self._levels.values():
            if((size.width <= wsi_level.size.width) and
               (wsi_level.size.width <= closest_size.width)):
                closest_size = wsi_level.size
                closest = wsi_level
        if closest is None:
            raise WsiDicomNotFoundError(
                f"Level for size {size}", "level series"
            )
        return closest

    def get_closest_by_pixel_spacing(
        self,
        pixel_spacing: SizeMm
    ) -> WsiDicomLevel:
        """Search for level that by pixel spacing is closest to and smaller
        than the given pixel spacing. Only the spacing in x-axis is used.

        Parameters
        ----------
        pixel_spacing: SizeMm
            Pixel spacing to search for

        Returns
        ----------
        WsiDicomLevel
            The level with pixel spacing closest to searched spacing
        """
        closest_pixel_spacing: float = 0
        closest = None
        for wsi_level in self._levels.values():
            if((pixel_spacing.width >= wsi_level.pixel_spacing.width) and
               (closest_pixel_spacing <= wsi_level.pixel_spacing.width)):
                closest_pixel_spacing = wsi_level.pixel_spacing.width
                closest = wsi_level
        if closest is None:
            raise WsiDicomNotFoundError(
                f"Level for pixel spacing {pixel_spacing}", "level series")
        return closest


class WsiDicom:
    """Represent a wsi slide containing pyramidal levels and optionally
    labels and/or overviews."""
    def __init__(
        self,
        levels: WsiDicomLevels,
        labels: WsiDicomLabels,
        overviews: WsiDicomOverviews,
        annotations: AnnotationInstance = None

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
        annotations: AnnotationInstance = None
            Sup-222 annotation instance.
        """

        self._levels = levels
        self._labels = labels
        self._overviews = overviews
        self.annotations = annotations

        if self.levels.uids is not None:
            self.uids = self._validate_collection(
                [self.levels, self.labels, self.overviews],
            )
        else:
            self.uids = None

        self.optical = OpticalManager.open(
            levels.instances + labels.instances + overviews.instances
        )

        if (
            self.annotations is not None and
            self.annotations.frame_of_reference != self.uids.frame_of_reference
        ):
            warnings.warn("Annotations frame of referance does not match")

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
    def files(self) -> List[WsiDicomFile]:
        """Return contained files"""
        return self.levels.files + self.labels.files + self.overviews.files

    @property
    def datasets(self) -> List[Dataset]:
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
    def frame_of_reference(self) -> Uid:
        return self.uids.frame_of_reference

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = self.__class__.__name__
        if depth is not None:
            depth -= 1
            if(depth < 0):
                return string
        return (
            string + ' of levels:\n'
            + list_pretty_str(self.levels.groups, indent, depth, 0, 2)
        )

    @classmethod
    def open(cls, path: Union[str, List[str]]) -> 'WsiDicom':
        """Open valid wsi dicom files in path and return a WsiDicom object.
        Non-valid files are ignored.

        Parameters
        ----------
        paths: List[Path]
            Path to files to open

        Returns
        ----------
        WsiDicom
            Object created from wsi dicom files in path
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
                if(wsi_file.wsi_type == 'VOLUME'):
                    level_files.append(wsi_file)
                elif(wsi_file.wsi_type == 'LABEL'):
                    label_files.append(wsi_file)
                elif(wsi_file.wsi_type == 'OVERVIEW'):
                    overview_files.append(wsi_file)
                else:
                    wsi_file.close()
            elif sop_class_uid == ANN_SOP_CLASS_UID:
                annotation_files.append(filepath)

        base_dataset = cls._get_base_dataset(level_files)
        base_uids = base_dataset.base_uids
        base_tile_size = base_dataset.tile_size
        level_instances = WsiInstance.open(
            level_files,
            base_uids,
            base_tile_size
        )
        label_instances = WsiInstance.open(label_files, base_uids)
        overview_instances = WsiInstance.open(overview_files, base_uids)

        levels = WsiDicomLevels.open(level_instances)
        labels = WsiDicomLabels.open(label_instances)
        overviews = WsiDicomOverviews.open(overview_instances)
        annotations = AnnotationInstance.open(annotation_files)

        return cls(levels, labels, overviews, annotations)

    @staticmethod
    def _get_sop_class_uid(path: Path) -> Uid:
        metadata: pydicom.dataset.FileMetaDataset = (
            pydicom.filereader.read_file_meta_info(path)
        )
        return metadata.MediaStorageSOPClassUID

    @staticmethod
    def _get_filepaths(path: Union[str, List[str]]) -> List[Path]:
        """Return file paths to files in path.
        If path is folder, return list of folder files in path.
        If path is single file, return list of that path.
        If path is list, return list of paths that are files.
        Raises WsiDicomNotFoundError if no files found

        Parameters
        ----------
        path: Union[str, List[str]]
            Path to folder, file or list of files

        Returns
        ----------
        List[Path]
            List of found file paths
        """
        if isinstance(path, str):
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
        files: List[WsiDicomFile]
    ) -> WsiDataset:
        """Return file with largest image (width) from list of files.

        Parameters
        ----------
        files: List[WsiDicomFile]
           List of files.

        Returns
        ----------
        WsiDataset
            Base layer dataset.
        """
        base_size = Size(0, 0)
        base_dataset: WsiDataset
        for file in files:
            if file.dataset.image_size.width > base_size.width:
                base_dataset = file.dataset
                base_size = file.dataset.image_size
        return base_dataset

    @staticmethod
    def _filter_paths(filepaths: List[Path]) -> List[Path]:
        """Filter list of paths to only include valid dicom files.

        Parameters
        ----------
        filepaths: List[Path]
            Paths to filter

        Returns
        ----------
        List[Path]
            List of paths with dicom files
        """
        return [
            path for path in filepaths
            if path.is_file() and pydicom.misc.is_dicom(path)
        ]

    def _validate_collection(
        self,
        series: List[WsiDicomSeries]
    ) -> BaseUids:
        """Check that no files or instance in collection is duplicate, that all
        series have the same base uids. Raises WsiDicomMatchError otherwise.
        Returns base uid for collection.

        Parameters
        ----------
        series: List[WsiDicomSeries]
            List of series to check

        Returns
        ----------
        BaseUids
            Matching uids
        """
        WsiDataset.check_duplicate_dataset(self.datasets, self)
        WsiInstance.check_duplicate_instance(self.instances, self)

        try:
            base_uids = next(
                item.uids for item in series if item.uids is not None
            )
        except StopIteration:
            raise WsiDicomNotFoundError("Valid series", "in collection")
        for item in series:
            if item.uids is not None and item.uids != base_uids:
                raise WsiDicomMatchError(str(item), str(self))
        return base_uids

    def read_label(self, index: int = 0) -> Image.Image:
        """Read label image of the whole slide. If several label
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int
            Index of the label image to read

        Returns
        ----------
        Image.Image
            label as image
        """
        try:
            label = self.labels[index]
            return label.get_default_full()
        except IndexError:
            raise WsiDicomNotFoundError("label", "series")

    def read_overview(self, index: int = 0) -> Image.Image:
        """Read overview image of the whole slide. If several overview
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int
            Index of the overview image to read

        Returns
        ----------
        Image.Image
            Overview as image
        """
        try:
            overview = self.overviews[index]
            return overview.get_default_full()
        except IndexError:
            raise WsiDicomNotFoundError("overview", "series")

    def read_thumbnail(
        self,
        size: Tuple[int, int],
        z: float = None,
        path: str = None
    ) -> Image.Image:
        """Read thumbnail image of the whole slide with dimensions
        no larger than given size.

        Parameters
        ----------
        size: int, int
            Upper size limit for thumbnail
        z: float
            Z coordinate, optional
        path:
            optical path, optional

        Returns
        ----------
        Image.Image
            Thumbnail as image
        """
        thumbnail_size = Size.from_tuple(size)
        level = self.levels.get_closest_by_size(thumbnail_size)
        region = Region(position=Point(0, 0), size=level.size)
        image = level.get_region(region, z, path)
        image.thumbnail((size), resample=Image.BILINEAR)
        return image

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        z: float = None,
        path: str = None
    ) -> Image.Image:
        """Read region defined by pixels.

        Parameters
        ----------
        location: int, int
            Upper left corner of region in pixels
        level: int
            Level in pyramid
        size: int
            Size of region in pixels
        z: float
            Z coordinate, optional
        path:
            optical path, optional

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
        if(scale_factor != 1):
            image = image.resize((size), resample=Image.BILINEAR)
        return image

    def read_region_mm(
        self,
        location: Tuple[float, float],
        level: int,
        size: Tuple[float, float],
        z: float = None,
        path: str = None
    ) -> Image.Image:
        """Read image from region defined in mm.

        Parameters
        ----------
        location: float, float
            Upper left corner of region in mm
        level: int
            Level in pyramid
        size: float
            Size of region in mm
        z: float
            Z coordinate, optional
        path:
            optical path, optional

        Returns
        ----------
        Image.Image
            Region as image
        """
        wsi_level = self.levels.get_closest_by_level(level)
        scale_factor = wsi_level.calculate_scale(level)
        region = RegionMm(
            position=PointMm.from_tuple(location),
            size=SizeMm.from_tuple(size)
        )
        image = wsi_level.get_region_mm(region, z, path)
        image_size = (
            Size(width=image.size[0], height=image.size[1]) // scale_factor
        )
        return image.resize(image_size.to_tuple(), resample=Image.BILINEAR)

    def read_region_mpp(
        self,
        location: Tuple[float, float],
        mpp: float,
        size: Tuple[float, float],
        z: float = None,
        path: str = None
    ) -> Image.Image:
        """Read image from region defined in mm with set pixel spacing.

        Parameters
        ----------
        location: float, float
            Upper left corner of region in mm
        mpp: float
            Requested pixel spacing (um/pixel)
        size: float
            Size of region in mm
        z: float
            Z coordinate, optional
        path:
            optical path, optional

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
            position=PointMm.from_tuple(location),
            size=SizeMm.from_tuple(size)
        )
        image = wsi_level.get_region_mm(region, z, path)
        image_size = SizeMm(width=size[0], height=size[1]) // pixel_spacing
        return image.resize(image_size.to_tuple(), resample=Image.BILINEAR)

    def read_tile(
        self,
        level: int,
        tile: Tuple[int, int],
        z: float = None,
        path: str = None
    ) -> Image.Image:
        """Read tile in pyramid level as image.

        Parameters
        ----------
        level: int
            Pyramid level
        tile: int, int
            tile xy coordinate
        z: float
            Z coordinate, optional
        path:
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
        z: float = None,
        path: str = None
    ) -> bytes:
        """Read tile in pyramid level as encoded bytes. For non-existing levels
        the tile is scaled down from a lower level, using the similar encoding.

        Parameters
        ----------
        level: int
            Pyramid level
        tile: int, int
            tile xy coordinate
        z: float
            Z coordinate, optional
        path:
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
        z: float = None,
        path: str = None
    ) -> WsiInstance:

        """Return instance fullfilling level, z and/or path.

        Parameters
        ----------
        level: int
            Pyramid level
        z: float
            Z coordinate, optional
        path:
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
        uid_generator: Callable[..., Uid] = pydicom.uid.generate_uid
    ) -> List[str]:
        """Save wsi as DICOM-files in path.

        Parameters
        ----------
        output_path: str
        uid_generator: Callable[..., Uid] = pydicom.uid.generate_uid
             Function that can gernerate unique identifiers.

        Returns
        ----------
        List[str]
            List of paths of created files.
        """
        collections: List[WsiDicomSeries] = [
            self.levels, self.labels, self.overviews
        ]

        filepaths: List[str] = []
        for collection in collections:
            collection_filepaths = collection.save(
                output_path,
                uid_generator
            )
            filepaths.extend(collection_filepaths)
        return filepaths
