import io
import math
import os
import warnings
from abc import ABCMeta, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, OrderedDict, Tuple, Union

import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import UID as Uid

from .errors import (WsiDicomFileError, WsiDicomMatchError,
                     WsiDicomNotFoundError, WsiDicomOutOfBondsError,
                     WsiDicomSparse, WsiDicomUidDuplicateError)
from .geometry import Point, PointMm, Region, RegionMm, Size, SizeMm
from .optical import OpticalManager
from .stringprinting import dict_pretty_str, list_pretty_str, str_indent
from .uid import BaseUids, FileUids


class WsiDicomFile:
    def __init__(self, filepath: Path):
        """Open dicom file in filepath. If valid wsi type read required
        parameters. Parses frames in pixel data but does not read the frames.

        Parameters
        ----------
        filepath: Path
            Path to file to open
        """
        self._filepath = filepath

        if not pydicom.misc.is_dicom(self.filepath):
            raise WsiDicomFileError(self.filepath, "is not a DICOM file")

        file_meta = pydicom.filereader.read_file_meta_info(self.filepath)
        transfer_syntax_uid = Uid(file_meta.TransferSyntaxUID)

        self._fp = pydicom.filebase.DicomFile(self.filepath, mode='rb')
        self._fp.is_little_endian = transfer_syntax_uid.is_little_endian
        self._fp.is_implicit_VR = transfer_syntax_uid.is_implicit_VR
        ds = pydicom.dcmread(self._fp, stop_before_pixels=True)
        self._pixel_data_position = self._fp.tell()

        if self.is_wsi_dicom(self.filepath, ds):
            instance_uid = Uid(ds.SOPInstanceUID)
            concatenation_uid: Uid = getattr(
                ds,
                'SOPInstanceUIDOfConcatenationSource',
                None
            )
            base_uids = BaseUids(
                ds.StudyInstanceUID,
                ds.SeriesInstanceUID,
                ds.FrameOfReferenceUID,
            )
            self._uids = FileUids(instance_uid, concatenation_uid, base_uids)

            if concatenation_uid is not None:
                try:
                    self._frame_offset = int(ds.ConcatenationFrameOffsetNumber)
                except AttributeError:
                    raise WsiDicomFileError(
                        self.filepath,
                        'Concatenated file missing concatenation frame offset'
                        'number'
                    )
            else:
                self._frame_offset = 0
            self._wsi_type = self.get_supported_wsi_dicom_type(
                ds,
                transfer_syntax_uid
            )
            self._pixel_spacing = self._get_pixel_spacing(ds)
            self._image_size = self._get_image_size(ds)
            self._mm_size = self._get_mm_size(ds)
            self._frame_count = int(getattr(ds, 'NumberOfFrames', 1))
            self._tile_size = self._get_tile_size(ds)
            self._tile_type = self._get_tile_type(ds)
            self._samples_per_pixel = int(ds.SamplesPerPixel)
            self._transfer_syntax = Uid(ds.file_meta.TransferSyntaxUID)
            self._photometric_interpretation = str(
                ds.PhotometricInterpretation
                )
            self._optical_path_sequence = ds.OpticalPathSequence

            pixel_measure = (
                ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
            )
            # We assume that slice thickness is the same for all focal planes
            self._slice_spacing = getattr(
                pixel_measure,
                'SpacingBetweenSlices',
                0
            )
            self._slice_thickness = pixel_measure.SliceThickness

            if self.tile_type == 'TILED_FULL':
                try:
                    self._frame_sequence = ds.SharedFunctionalGroupsSequence
                except AttributeError:
                    raise WsiDicomFileError(
                        self.filepath,
                        'Tiled full file missing shared functional'
                        'group sequence'
                    )
                self._focal_planes = int(
                    getattr(ds, 'TotalPixelMatrixFocalPlanes', 1)
                )
            else:
                try:
                    self._frame_sequence = ds.PerFrameFunctionalGroupsSequence
                except AttributeError:
                    raise WsiDicomFileError(
                        self.filepath,
                        'Tiled sparse file missing per frame functional'
                        'group sequence'
                    )

            self._frame_positions = self._parse_pixel_data()

        else:
            self._wsi_type = "None"

    def __repr__(self) -> str:
        return f"WsiDicomFile('{self.filepath}')"

    def __str__(self) -> str:
        return self.pretty_str()

    @property
    def filepath(self) -> Path:
        """Return filepath"""
        return self._filepath

    @property
    def wsi_type(self) -> str:
        """Return wsi type"""
        return self._wsi_type

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um/pixel"""
        return self.pixel_spacing*1000.0

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm/pixel"""
        return self._pixel_spacing

    @property
    def image_size(self) -> Size:
        """Return image size in pixels"""
        return self._image_size

    @property
    def mm_size(self) -> SizeMm:
        """Return image size in mm"""
        return self._mm_size

    @property
    def tile_size(self) -> Size:
        """Return tile size in pixels"""
        return self._tile_size

    @property
    def tile_type(self) -> str:
        """Return tiling type (TILED_FULL or TILED_SPARSE)"""
        return self._tile_type

    @property
    def uids(self) -> FileUids:
        """Return uids"""
        return self._uids

    @property
    def samples_per_pixel(self) -> int:
        """Return samples per pixel (1 or 3)"""
        return self._samples_per_pixel

    @property
    def transfer_syntax(self) -> Uid:
        """Return transfer syntax uid"""
        return self._transfer_syntax

    @property
    def photometric_interpretation(self) -> str:
        """Return photometric interpretation"""
        return self._photometric_interpretation

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

    @property
    def optical_path_sequence(self) -> DicomSequence:
        """Return DICOM Optical path sequence"""
        return self._optical_path_sequence

    @property
    def frame_sequence(self) -> DicomSequence:
        """Return DICOM Shared functional group sequence if TILED_FULL or
        Per frame functional groups sequence if TILED_SPARSE.
        """
        return self._frame_sequence

    @property
    def focal_planes(self) -> int:
        """Return number of focal planes"""
        return self._focal_planes

    @property
    def slice_thickness(self) -> float:
        """Return slice thickness"""
        return self._slice_thickness

    @property
    def slice_spacing(self) -> float:
        """Return slice spacing"""
        return self._slice_spacing

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        return f"File with path: {self.filepath}"

    def matches_series(self, uids: BaseUids, tile_size: Size = None) -> bool:
        """Check if instance is valid (Uids and tile size match).
        Base uids should match for instances in all types of series,
        tile size should only match for level series.
        """
        if tile_size is not None and tile_size != self.tile_size:
            return False
        return uids == self.uids.base

    def matches_instance(self, other_file: 'WsiDicomFile') -> bool:
        """Return true if other file is of the same instance as self.

        Parameters
        ----------
        other_file: WsiDicomFile
            File to check.

        Returns
        ----------
        bool
            True if same instance.
        """
        return (
            self.uids.match(other_file.uids) and
            self.image_size == other_file.image_size and
            self.tile_size == other_file.tile_size and
            self.tile_type == other_file.tile_type and
            self.wsi_type == other_file.wsi_type
        )

    @staticmethod
    def check_duplicate_file(
        files: List['WsiDicomFile'],
        caller: object
    ) -> None:
        """Check for duplicates in list of files. Files are duplicate if file
        instance uids match. Stops at first found duplicate and raises
        WsiDicomUidDuplicateError.

        Parameters
        ----------
        files: List[WsiDicomFile]
            List of files to check.
        caller: Object
            Object that the files belongs to.
        """
        instance_uids: List[str] = []
        for file in files:
            instance_uid = file.uids.instance
            if instance_uid not in instance_uids:
                instance_uids.append(file.uids.identifier)
            else:
                raise WsiDicomUidDuplicateError(str(file), str(caller))

    @staticmethod
    def is_wsi_dicom(filepath: Path, ds: Dataset) -> bool:
        """Check if file is dicom wsi type and that required attributes
        (for the function of the library) is available.
        Warn if attribute listed as requierd in the library or required in the
        standard is missing.

        Parameters
        ----------
        filepath: Path
            Path to file
        ds: Dataset
            Dataset of file

        Returns
        ----------
        bool
            True if file is wsi dicom SOP class and all required attributes
            are available
        """
        WSI_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.77.1.6'

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
            "ImagedVolumeWidth",
            "ImagedVolumeHeight",
            "ImagedVolumeDepth",
            "TotalPixelMatrixColumns",
            "TotalPixelMatrixRows",
        ]
        STANDARD_WHOLE_SLIDE_MICROSCOPY_MODULE_ATTRIBUTES = [
            "TotalPixelMatrixOriginSequence",
            "FocusMethod",
            "ExtendedDepthOfField"
            "ImageOrientationSlide",
            "AcquisitionDateTime",
            "LossyImageCompression",
            "VolumetricProperties",
            "SpecimenLabelInImage",
            "BurnedInAnnotation",
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
                    if attribute not in ds:
                        warnings.warn(
                            f'File {filepath} is missing {key}'
                            f' attribute {attribute}'
                        )
                        passed[key] = False

        sop_class_uid = getattr(ds, "SOPClassUID", "")
        sop_class_uid_check = (sop_class_uid == WSI_SOP_CLASS_UID)
        return passed['required'] and sop_class_uid_check

    @staticmethod
    def get_supported_wsi_dicom_type(
        ds: Dataset,
        transfer_syntax_uid: Uid
    ) -> str:
        """Check image flavor and transfer syntax of dicom file.
        Return image flavor if file valid.

        Parameters
        ----------
        ds: Dataset
            Pydicom dataset to check.
        transfer_syntax_uid: Uid'
            Transfer syntax uid for file.

        Returns
        ----------
        str
            WSI image flavor
        """
        SUPPORTED_IMAGE_TYPES = ['VOLUME', 'LABEL', 'OVERVIEW']
        IMAGE_FLAVOR = 2
        image_type: str = ds.ImageType[IMAGE_FLAVOR]
        image_type_supported = image_type in SUPPORTED_IMAGE_TYPES
        if not image_type_supported:
            warnings.warn(f"Non-supported image type {image_type}")

        syntax_supported = (
            pydicom.pixel_data_handlers.pillow_handler.
            supports_transfer_syntax(transfer_syntax_uid)
        )
        if not syntax_supported:
            warnings.warn(
                "Non-supported transfer syntax"
                f"{transfer_syntax_uid}"
            )
        if image_type_supported and syntax_supported:
            return image_type
        return ""

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

    def _get_mm_size(self, ds: Dataset) -> SizeMm:
        """Read mm size from dicom file.

        Parameters
        ----------
        ds: Dataset
            Pydicom dataset

        Returns
        ----------
        SizeMm
            The size of the image in mm
        """
        width = float(ds.ImagedVolumeWidth)
        height = float(ds.ImagedVolumeHeight)
        if width == 0 or height == 0:
            raise WsiDicomFileError(self.filepath, "Image mm size is zero")
        return SizeMm(width=width, height=height)

    def _get_pixel_spacing(self, ds: Dataset) -> SizeMm:
        """Read pixel spacing from dicom file.

        Parameters
        ----------
        ds: Dataset
            Pydicom dataset

        Returns
        ----------
        SizeMm
            The pixel spacing
        """
        pixel_spacing: Tuple[float, float] = (
            ds.SharedFunctionalGroupsSequence[0].
            PixelMeasuresSequence[0].PixelSpacing
        )
        if any([spacing == 0 for spacing in pixel_spacing]):
            raise WsiDicomFileError(self.filepath, "Pixel spacing is zero")
        return SizeMm(width=pixel_spacing[0], height=pixel_spacing[1])

    def _get_tile_size(self, ds: Dataset) -> Size:
        """Read tile size from from dicom file.

        Parameters
        ----------
        ds: Dataset
            Pydicom dataset

        Returns
        ----------
        Size
            The tile size
        """
        width = int(ds.Columns)
        height = int(ds.Rows)
        if width == 0 or height == 0:
            raise WsiDicomFileError(self.filepath, "Tile size is zero")
        return Size(width=width, height=height)

    def _get_image_size(self, ds: Dataset) -> Size:
        """Read total pixel size from dicom file.

        Parameters
        ----------
        ds: Dataset
            Pydicom dataset

        Returns
        ----------
        Size
            The image size
        """
        width = int(ds.TotalPixelMatrixColumns)
        height = int(ds.TotalPixelMatrixRows)
        if width == 0 or height == 0:
            raise WsiDicomFileError(self.filepath, "Image size is zero")
        return Size(width=width, height=height)

    def _get_tile_type(self, ds: Dataset) -> str:
        """Return tiling type from dataset. Raises WsiDicomFileError if type
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
        if(getattr(ds, 'DimensionOrganizationType', '') == 'TILED_FULL'):
            return 'TILED_FULL'
        elif 'PerFrameFunctionalGroupsSequence' in ds:
            return 'TILED_SPARSE'
        raise WsiDicomFileError(self.filepath, "undetermined tile type")

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


class SparseTilePlane:
    def __init__(self, plane_size: Size):
        """Hold frame indices for the tiles in a sparse tiled file.
        Empty (sparse) frames are represented by -1.

        Parameters
        ----------
        plane_size: Size
            Size of the tiling
        """
        self.plane = np.full(plane_size.to_tuple(), -1, dtype=int)

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
        if frame_index == -1:
            raise WsiDicomSparse(position)
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
        return ("Sparse tile plane")


class TileIndex(metaclass=ABCMeta):
    def __init__(
        self,
        files: Dict[int, WsiDicomFile],
        optical: OpticalManager
    ):
        """Index for the tiling of pixel data.
        Requires same tile size for all tile planes

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
            Dict of file frame offset and wsi file.
        Optical: OpticalManager
            Optical manager to add optical paths to.

        """
        self._image_size = files[0].image_size
        self._tile_size = files[0].tile_size
        self._plane_size: Size = self.image_size / self.tile_size
        self._frame_count = self._get_frame_count_from_files(files)
        self._focal_planes = self._get_focal_planes_from_files(files)
        self._optical_paths = self._get_optical_paths_from_files(
            files,
            optical
        )
        self._default_z: float = self._select_default_z(self.focal_planes)
        self._default_path = self.optical_paths[0]

    @property
    def tile_size(self) -> Size:
        """Return tile size in pixels"""
        return self._tile_size

    @property
    def plane_size(self) -> Size:
        """Return size of tiling (columns x rows)"""
        return self._plane_size

    @property
    def default_z(self) -> float:
        """Return default focal plane in um"""
        return self._default_z

    @property
    def default_path(self) -> str:
        """Return default optical path identifier"""
        return self._default_path

    @property
    def frame_count(self) -> int:
        """Return total number of frames"""
        return self._frame_count

    @property
    def focal_planes(self) -> List[float]:
        """Return total number of focal planes"""
        return self._focal_planes

    @property
    def optical_paths(self) -> List[str]:
        """Return total number of optical paths"""
        return self._optical_paths

    @property
    def image_size(self) -> Size:
        """Return image size in pixels"""
        return self._image_size

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

    @abstractmethod
    def _get_focal_planes_from_files(
        self,
        files: Dict[int, WsiDicomFile]
    ) -> List[float]:
        """Abstract method for getting focal planes from file"""
        raise NotImplementedError

    def valid_tiles(self, region: Region, z: float, path: str) -> bool:
        """Check if tile region is inside tile geometry and z coordinate and
        optical path exists.

        Parameters
        ----------
        region: Region
            Tile region
        z: float
            z coordiante
        path: str
            optical path
        """
        plane_region = Region(
            position=Point(0, 0),
            size=self.plane_size - 1
        )
        return (
            region.is_inside(plane_region) and
            (z in self.focal_planes) and
            (path in self.optical_paths)
        )

    @staticmethod
    def _select_default_z(focal_planes: List[float]) -> float:
        """Select default z coordinate to use if specific plane not set.
        If more than one focal plane available the middle one is selected.

        Parameters
        ----------
        focal_planes: List[float]
           List of focal planes to select from

        Returns
        ----------
        float
            Default z coordinate

        """
        default = 0
        if(len(focal_planes) > 1):
            smallest = min(focal_planes)
            largest = max(focal_planes)
            middle = (largest - smallest)/2
            default = min(range(len(focal_planes)),
                          key=lambda i: abs(focal_planes[i]-middle))

        return focal_planes[default]

    @staticmethod
    def _get_frame_count_from_files(
        files: Dict[int, WsiDicomFile]
    ) -> int:
        """Return total frame count from files.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
           Dict of wsi dicom files

        Returns
        ----------
        int
            Total frame count

        """
        count = 0
        for file in files.values():
            count += file.frame_count
        return count

    @staticmethod
    def _get_optical_paths_from_files(
        files: Dict[int, WsiDicomFile],
        optical: OpticalManager
    ) -> List[str]:
        """Return list of optical path identifiers from files.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
           Dict of wsi dicom files
        optical: OpticalManager
            Optical manager to add optical paths to.

        Returns
        ----------
        List[str]
            Optical identifiers

        """
        paths: List[str] = []
        for file in files.values():
            paths += optical.add(file.optical_path_sequence)
        return paths


class FullTileIndex(TileIndex):
    """Index for full tiled pixel data.
    Requires same tile size for all tile planes.
    Pixel data tiles are ordered by colum, row, z and path, thus
    the frame index for a tile can directly be calculated.
    """
    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f"Full tile index tile size: {self.tile_size}"
            f", plane size: {self.plane_size}"
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
        """Return frame index for a Point tile, z coordinate,
        and optical path from full tile index. Assumes that tile, z, and path
        are valid.

        Parameters
        ----------
        tile: Point
            tile xy to get
        z: float
            z coordinate to get
        path: str
            ID of optical path to get

        Returns
        ----------
        int
            Frame index
        """
        z_index = self._focal_plane_index(z)
        plane_offset = tile.x + self.plane_size.width*tile.y
        tiles_in_plane = self.plane_size.width * self.plane_size.height
        z_offset = z_index * tiles_in_plane
        path_index = self._optical_path_index(path)
        path_offset = (
            path_index * len(self._focal_planes) * tiles_in_plane
        )
        return plane_offset + z_offset + path_offset

    def _get_focal_planes_from_files(
        self,
        files: Dict[int, WsiDicomFile]
    ) -> List[float]:
        """Return list of focal planes from files.
        Values in Pixel Measures Sequene are in mm.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
           Dict of wsi dicom files

        Returns
        ----------
        List[float]
            Focal planes

        """
        MM_TO_MICRON = 1000.0
        DECIMALS = 3
        focal_planes: List[float] = []
        for file in files.values():
            sequence = file.frame_sequence[0]
            spacing = getattr(
                sequence.PixelMeasuresSequence[0],
                'SpacingBetweenSlices',
                0
            )
            if spacing == 0 and file.focal_planes != 1:
                raise WsiDicomFileError(
                    file.filepath,
                    "Multipe focal planes and zero plane spacing"
                )
            for plane in range(file.focal_planes):
                z = round(plane * spacing * MM_TO_MICRON, DECIMALS)
                if z not in focal_planes:
                    focal_planes.append(z)
        return focal_planes

    def _optical_path_index(self, path: str) -> int:
        """Return index of the optical path in instance.
        This assumes that all files in a concatenated set contains all the
        optical path identifiers of the set.

        Parameters
        ----------
        path: str
            Optical path identifier

        Returns
        ----------
        int
            The index of the optical path identifier in the optical path
            sequence
        """
        try:
            return next(
                (index for index, plane_path in enumerate(self._optical_paths)
                 if plane_path == path)
            )
        except StopIteration:
            raise WsiDicomNotFoundError(
                f"Optical path {path}",
                "full tile index"
            )

    def _focal_plane_index(self, z: float) -> int:
        """Return index of the focal plane of z.

        Parameters
        ----------
        z: float
            The z coordinate to search for

        Returns
        ----------
        int
            Focal plane index
        """
        try:
            return next(index for index, plane in enumerate(self.focal_planes)
                        if plane == z)
        except StopIteration:
            raise WsiDicomNotFoundError(
                f"Z {z} in instance", "full tile index"
            )


class SparseTileIndex(TileIndex):
    def __init__(
        self,
        files: Dict[int, WsiDicomFile],
        optical: OpticalManager
    ):
        """Index for sparse tiled pixel data.
        Requires same tile size for all tile planes.
        Pixel data tiles are identified by the Per Frame Functional Groups
        Sequence that contains tile colum, row, z, path, and frame index. These
        are stored in a SparseTilePlane (one plane for every combination of z
        and path). Frame indices are retrieved from tile position, z, and path
        by finding the corresponding matching SparseTilePlane (z and path) and
        returning the frame index at tile position. If the tile is missing (due
        to the sparseness), -1 is returned.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
            Dict of file frame offset and wsi file.
        Optical: OpticalManager
            Optical manager to add optical paths to.

        """
        super().__init__(files, optical)
        self._planes = self._get_planes_from_files(files)

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f"Sparse tile index tile size: {self.tile_size}"
            f", plane size: {self.plane_size}"
        )
        return string

    def get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Return frame index for a Point tile, z coordinate, and optical
        path.

        Parameters
        ----------
        tile: Point
            tile xy to get
        z: float
            z coordinate to get
        path: str
            ID of optical path to get

        Returns
        ----------
        int
            Frame index
        """
        plane = self._get_plane(z, path)
        frame_index = plane[tile]
        return frame_index

    def _get_focal_planes_from_files(
        self,
        files: Dict[int, WsiDicomFile]
    ) -> List[float]:
        """Return list of focal planes from files.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
           Dict of wsi dicom files

        Returns
        ----------
        List[float]
            Focal planes
        """
        focal_planes: List[float] = []
        for file in files.values():
            for frame in file.frame_sequence:
                try:
                    (tile, z) = self._get_frame_coordinates(frame)
                except AttributeError:
                    raise WsiDicomFileError(
                        file.filepath,
                        "Invalid plane position slide sequence"
                    )
                if z not in focal_planes:
                    focal_planes.append(z)
        return focal_planes

    def _get_planes_from_files(
        self,
        files: Dict[int, WsiDicomFile]
    ) -> Dict[Tuple[float, str], SparseTilePlane]:
        """Return SparseTilePlane from planes read from files.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
           Dict of wsi dicom files

        Returns
        ----------
        Dict[Tuple[float, str], SparseTilePlane]
            Dict of planes with focal plane and optical identifier as key.
        """
        planes: Dict[Tuple[float, str], SparseTilePlane] = {}

        for file_offset, file in files.items():
            for i, frame in enumerate(file.frame_sequence):
                (tile, z) = self._get_frame_coordinates(frame)
                optical_sequence = getattr(
                    frame,
                    'OpticalPathIdentificationSequence',
                    file.optical_path_sequence
                )
                identifier = getattr(
                    optical_sequence[0],
                    'OpticalPathIdentifier',
                    '0'
                )
                try:
                    plane = planes[(z, identifier)]
                except KeyError:
                    plane = SparseTilePlane(self.plane_size)
                    planes[(z, identifier)] = plane
                plane[tile] = i + file_offset
        return planes

    def _get_plane(self, z: float, path: str) -> SparseTilePlane:
        """Return plane with z coordinate and optical path.

        Parameters
        ----------
        z: float
            Z coordinate to search for
        path: str
            Optical path identifer to search for

        Returns
        ----------
        SparseTilePlane
            The plane for z coordinate and path
        """
        try:
            return self._planes[(z, path)]
        except KeyError:
            raise WsiDicomNotFoundError(
                f"Plane with z {z}, path {path}", "sparse tile index"
            )

    def _get_frame_coordinates(
            self,
            frame: DicomSequence
    ) -> Tuple[Point, float]:
        """Return frame coordinate (Point(x, y) and float z) of the frame.
        In the Plane Position Slide Sequence x and y are defined in mm and z in
        um.

        Parameters
        ----------
        frame: DicomSequence
            Pydicom frame sequence

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


class WsiDicomInstance:
    def __init__(self, files: List[WsiDicomFile], optical: OpticalManager):
        """Represents a single SOP instance or a concatenated SOP instance.
        The instance can contain multiple focal planes and optical paths.

        Files needs to match in UIDs and have the same image and tile size.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Files to include in the instance.
        optical: OpticalManager
            Optical manager to add optical paths to.
        """
        # Key is frame offset
        self._files = OrderedDict(
            (file.frame_offset, file) for file
            in sorted(files, key=lambda file: file.frame_offset)
        )
        (
            self._identifier,
            self._uids,
            self._wsi_type
        ) = self._validate_instance(optical.uids)

        self.tiles = self._create_tileindex(self._files, optical)

        base_file = files[0]
        self._size = base_file.image_size
        self._pixel_spacing = base_file.pixel_spacing
        self._tile_size = base_file.tile_size
        self._samples_per_pixel = base_file.samples_per_pixel
        self._transfer_syntax = base_file.transfer_syntax
        self._photometric_interpretation = base_file.photometric_interpretation
        self._slice_thickness = base_file.slice_thickness
        self._slice_spacing = base_file.slice_spacing

        self._blank_color = self._get_blank_color(
            self._photometric_interpretation
        )
        if(self._samples_per_pixel == 1):
            self._image_mode = "L"
        else:
            self._image_mode = "RGB"
        self._blank_tile = self._create_blank_tile()
        self._blank_encoded_tile = self._encode_tile(self._blank_tile)

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        string = (
            f"Instance default z: {self.tiles.default_z,}"
            f" default path: { self.tiles.default_path}"
        )
        if depth is not None:
            depth -= 1
            if(depth < 0):
                return string
        string += (
            '\n' + str_indent(indent) + 'Files'
            + dict_pretty_str(self._files, indent+1, depth, 1, 1) + '\n'
            + str_indent(indent) + self.tiles.pretty_str(indent+1, depth)
        )
        return string

    @property
    def files(self) -> List[WsiDicomFile]:
        """Return list of files"""
        return list(self._files.values())

    @property
    def wsi_type(self) -> str:
        """Return wsi type"""
        return self._wsi_type

    @property
    def blank_color(self) -> Tuple[int, int, int]:
        """Return RGB background color"""
        return self._blank_color

    @property
    def blank_tile(self) -> Image:
        """Return background tile"""
        return self._blank_tile

    @property
    def blank_encoded_tile(self) -> bytes:
        """Return encoded background tile"""
        return self._blank_encoded_tile

    @property
    def uids(self) -> BaseUids:
        """Return base uids"""
        return self._uids

    @property
    def size(self) -> Size:
        """Return image size in pixels"""
        return self._size

    @property
    def tile_size(self) -> Size:
        """Return tile size in pixels"""
        return self._tile_size

    @property
    def mpp(self) -> SizeMm:
        """Return pixel spacing in um/pixel"""
        return self.pixel_spacing*1000.0

    @property
    def pixel_spacing(self) -> SizeMm:
        """Return pixel spacing in mm/pixel"""
        return self._pixel_spacing

    @property
    def slice_thickness(self) -> float:
        """Return slice thickness"""
        return self._slice_thickness

    @property
    def slice_spacing(self) -> float:
        """Return slice spacing"""
        return self._slice_spacing

    @property
    def identifier(self) -> Uid:
        """Return identifier (instance uid for single file instance or
        concatenation uid for multiple file instance)"""
        return self._identifier

    @staticmethod
    def check_duplicate_instance(
        instances: List['WsiDicomInstance'],
        self: object
    ) -> None:
        """Check for duplicates in list of instances. Instances are duplicate
        if instance identifier (file instance uid or concatenation uid) match.
        Stops at first found duplicate and raises WsiDicomUidDuplicateError.

        Parameters
        ----------
        instances: List['WsiDicomInstance']
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

    @classmethod
    def open(
        cls,
        files: List[WsiDicomFile],
        optical: OpticalManager,
        series_uids: BaseUids,
        series_tile_size: Size = None
    ) -> List['WsiDicomInstance']:
        """Return list of instances created from files. Only files matching
        series uid and tile_size, if defined, are opened.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Files to create instances from.
        optical: OpticalManager
            Optical manager to add optical paths to.
        series_uids: BaseUids
            Uid to match against.
        series_tile_size: Size
            Tile size to match against (for level instances)

        Returns
        ----------
        List[WsiDicomInstance]
            List of created instances.
        """
        filtered_files = cls._filter_files(
            files,
            series_uids,
            series_tile_size
        )
        instances: List[WsiDicomInstance] = []
        files_grouped_by_instance = cls._group_files(filtered_files)

        for instance_files in files_grouped_by_instance.values():
            new_instance = WsiDicomInstance(instance_files, optical)
            instances.append(new_instance)

        return instances

    def matches(self, other_instance: 'WsiDicomInstance') -> bool:
        """Return true if other instance is of the same stack as self.

        Parameters
        ----------
        other_instance: WsiDicomInstance
            Instance to check

        Returns
        ----------
        bool
            True if instanes are of same stack.

        """
        return (
            self.uids == other_instance.uids and
            self.size == other_instance.size and
            self.tile_size == other_instance.tile_size and
            self.wsi_type == other_instance.wsi_type
        )

    def create_ds(self) -> Dataset:
        """Create a base pydicom dataset based on first file in instance.

        Returns
        ----------
        pydicom.dataset
            Pydicom dataset with common attributes for the levels.
        """
        INCLUDE = [0x0002, 0x0008, 0x0010, 0x0018, 0x0020, 0x0040]
        DELETE = ['ImageType', 'SOPInstanceUID', 'ContentTime',
                  'InstanceNumber', 'DimensionOrganizationSequence']

        base_file_path = self._files[0].filepath
        base_ds = pydicom.dcmread(base_file_path, stop_before_pixels=True)
        ds = pydicom.Dataset()
        for group in INCLUDE:
            group_ds = base_ds.group_dataset(group)
            for element in group_ds.iterall():
                ds.add(element)
        for delete in DELETE:
            ds.pop(delete)
        return ds

    def encode(self, image: Image) -> bytes:
        """Encode image using transfer syntax.

        Parameters
        ----------
        image: Image
            Image to encode

        Returns
        ----------
        bytes
            Encoded image as bytes

        """
        (image_format, image_options) = self._image_settings(
            self._transfer_syntax
        )
        with io.BytesIO() as buffer:
            image.save(buffer, format=image_format, **image_options)
            return buffer.getvalue()

    def stitch_tiles(self, region: Region, path: str, z: float) -> Image:
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
        Image
            Stitched image
        """
        stitching_tiles = self.get_tile_range(region, z, path)
        image = Image.new(mode=self._image_mode, size=region.size.to_tuple())
        write_index = Point(x=0, y=0)
        tile = stitching_tiles.position
        for x, y in stitching_tiles.iterate_all(include_end=True):
            tile = Point(x=x, y=y)
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
        start = pixel_region.start // self.tiles.tile_size
        end = pixel_region.end / self.tiles.tile_size - 1
        tile_region = Region.from_points(start, end)
        if not self.tiles.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBondsError(
                f"Tile region {tile_region}", f"plane {self.tiles.plane_size}"
            )
        return tile_region

    def get_tile(self, tile: Point, z: float, path: str) -> Image:
        """Get tile image at tile coordinate x, y.
        If frame is inside tile geometry but no tile exists in
        frame data (sparse) returns blank image

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
        Image
            Tile image
        """
        try:
            frame_index = self._get_frame_index(tile, z, path)
            tile_frame = self._get_tile_frame(frame_index)
            image = Image.open(io.BytesIO(tile_frame))
        except WsiDicomSparse:
            image = self.blank_tile

        # Check if tile is an edge tile that should be croped
        cropped_tile_region = self.crop_to_level_size(tile)
        if cropped_tile_region.size != self.tile_size:
            image = image.crop(box=cropped_tile_region.box_from_origin)
        return image

    def get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        """Get tile bytes at tile coordinate x, y
        If frame is inside tile geometry but no tile exists in
        frame data (sparse) returns encoded blank image.

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
            Tile image as bytes
        """

        try:
            frame_index = self._get_frame_index(tile, z, path)
            tile_frame = self._get_tile_frame(frame_index)
        except WsiDicomSparse:
            tile_frame = self.blank_encoded_tile

        # Check if tile is an edge tile that should be croped
        cropped_tile_region = self.crop_to_level_size(tile)
        if cropped_tile_region.size != self.tile_size:
            image = Image.open(io.BytesIO(tile_frame))
            image.crop(box=cropped_tile_region.box_from_origin)
            tile_frame = self.encode(image)
        return tile_frame

    def get_filepointer(
        self,
        tile: Point,
        z: float,
        path: str
    ) -> Tuple[pydicom.filebase.DicomFileLike, int, int]:
        """Return file pointer, frame position, and frame lenght for tile with
        z and path. If frame is inside tile geometry but no tile exists in
        frame data (sparse) WsiDicomSparse is raised.

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
        Tuple[pydicom.filebase.DicomFileLike, int, int]:
            File pointer, frame offset and frame lenght in number of bytes
        """
        frame_index = self._get_frame_index(tile, z, path)
        file = self._get_file(frame_index)
        return file.get_filepointer(frame_index)

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

    def close(self) -> None:
        """Close all files in the instance."""
        for file in self._files.values():
            file.close()

    def _validate_instance(
        self,
        optical_uids: BaseUids
    ) -> Tuple[str, BaseUids, str]:
        """Check that no files in instance are duplicate, that all files in
        instance matches (uid, type and size) and that the optical manager
        matches by base uid. Raises WsiDicomMatchError otherwise.
        Returns the matching file uid.

        Parameters
        ----------
        optical: OpticalManager
            Optical manager to check

        Returns
        ----------
        Tuple[str, BaseUids, str]
            Instance identifier uid, base uids and wsi type
        """
        WsiDicomFile.check_duplicate_file(self.files, self)
        base_file = self.files[0]
        if not base_file.uids.base == optical_uids:
            raise WsiDicomMatchError(
                str(base_file.filepath), str(self)
            )
        for file in self.files[1:]:
            if not base_file.matches_instance(file):
                raise WsiDicomMatchError(
                    str(file.filepath), str(self)
                )
        return (
            base_file.uids.identifier,
            base_file.uids.base,
            base_file.wsi_type
        )

    @staticmethod
    def _filter_files(
        files: List[WsiDicomFile],
        series_uids: BaseUids,
        series_tile_size: Size = None
    ) -> List[WsiDicomFile]:
        """Filter list of wsi dicom files to only include matching uids and
        tile size if defined.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Wsi files to filter
        series_uids: Uids
            Uids to check against
        series_tile_size: Size
            Tile size to check against

        Returns
        ----------
        List[WsiDicomFile]
            List of matching wsi dicom files
        """
        valid_files: List[WsiDicomFile] = []
        for file in files:
            if file.matches_series(series_uids, series_tile_size):
                valid_files.append(file)
            else:
                file.close()
        return valid_files

    @classmethod
    def _group_files(
        cls,
        files: List[WsiDicomFile]
    ) -> Dict[str, List[WsiDicomFile]]:
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

    def _create_blank_tile(self) -> Image:
        """Create blank tile for instance.

        Returns
        ----------
        Image
            Blank tile image
        """
        if(self._samples_per_pixel == 1):
            self._image_mode = "L"
        else:
            self._image_mode = "RGB"
        return Image.new(
            mode=self._image_mode,
            size=self.tile_size.to_tuple(),
            color=self.blank_color[:self._samples_per_pixel]
        )

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

    @staticmethod
    def _create_tileindex(
        files: Dict[int, WsiDicomFile],
        optical: OpticalManager
    ) -> TileIndex:
        """Return a tile index created from files. Add optical paths to optical
        manager.

        Parameters
        ----------
        files: Dict[int, WsiDicomFile]
            Files to add
        optical: OpticalManager
            Optical manager to add new optical paths to

        Returns
        ----------w
        TileIndex
            Created tile index
        """
        base_file = files[0]
        if(base_file.tile_type == 'TILED_FULL'):
            return FullTileIndex(files, optical)
        return SparseTileIndex(files, optical)

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

    def _encode_tile(
        self,
        tile: Image,
    ) -> bytes:
        """Encode tile using transfer syntax.

        Parameters
        ----------
        tile: Image
            Tile image to encode

        Returns
        ----------
        bytes
            Encoded tile as bytes
        """
        (image_format, image_options) = self._image_settings(
            self._transfer_syntax
        )

        with io.BytesIO() as buffer:
            tile.save(buffer, format=image_format, **image_options)
            return buffer.getvalue()

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
        """Return frame index for tile. Raises WsiDicomOutOfBondsError if
        tile, z, or path is not valid. Raises WsiDicomSparse if index is sparse
        and tile is not in frame data.

        Parameters
        ----------
        tile: Point
             Tile coordiante
        z: float
            Z coordiante
        path: str
            Optical identifier

        Returns
        ----------
        int
            Tile frame index
        """
        tile_region = Region(position=tile, size=Size(0, 0))
        if not self.tiles.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBondsError(
                f"Tile region {tile_region}",
                f"plane {self.tiles.plane_size}"
            )
        frame_index = self.tiles.get_frame_index(tile, z, path)
        return frame_index


class WsiDicomStack(metaclass=ABCMeta):
    def __init__(
        self,
        instances: List[WsiDicomInstance]
    ):
        """Represents a stack of instances having the same size,
        but possibly different z coordinate and/or optical path.
        Instances should match in the common uids, wsi type, and tile size.

        Parameters
        ----------
        instances: List[WsiDicomInstance]
            Instances to build the stack
        """
        self._instances = {  # key is identifier (str)
            instance.identifier: instance for instance in instances
        }
        self._uids, self._wsi_type = self._validate_stack()

        base_instance = instances[0]
        self._size = base_instance.size
        self._pixel_spacing = base_instance.pixel_spacing
        self._default_instance_uid: str = base_instance.identifier

    def __getitem__(self, index) -> WsiDicomInstance:
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
    def instances(self) -> Dict[str, WsiDicomInstance]:
        """Return contained instances"""
        return self._instances

    @property
    def default_instance(self) -> WsiDicomInstance:
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
    def optical_paths(self) -> List[str]:
        return list({
            path
            for instance in self.instances.values()
            for path in instance.tiles.optical_paths
        })

    @property
    def focal_planes(self) -> List[float]:
        return list({
            focal_plane
            for innstance in self.instances.values()
            for focal_plane in innstance.tiles.focal_planes
        })

    @classmethod
    def open(
        cls,
        instances: List[WsiDicomInstance],
    ) -> List['WsiDicomStack']:
        """Return list of stacks created from wsi files.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Files to create stacks from.
        optical: OpticalManager
            Optical manager to add optical paths to.

        Returns
        ----------
        List[WsiDicomStack]
            List of created stacks.

        """
        stacks: List[WsiDicomStack] = []

        instances_grouped_by_stack = cls._group_instances(instances)

        for instance_group in instances_grouped_by_stack.values():
            new_stack = WsiDicomStack(instance_group)
            stacks.append(new_stack)

        return stacks

    def matches(self, other_stack: 'WsiDicomStack') -> bool:
        """Check if stack is valid (Uids and tile size match).
        The common Uids should match for all series.
        """
        return (
            other_stack.uids == self.uids and
            other_stack.wsi_type == self.wsi_type
        )

    def valid_pixels(self, region: Region) -> bool:
        """Check if pixel region is withing the size of the stack image size.

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
    ) -> Tuple[WsiDicomInstance, float, str]:
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
        WsiDicomInstance, float, str
            The instance containing selected path and z coordinate,
            selected or default focal plane and optical path
        """
        if z is None and path is None:
            instance = self.default_instance
            z = instance.tiles.default_z
            path = instance.tiles.default_path
            return self.default_instance, z, path

        # Sort instances by number of focal planes (prefer simplest instance)
        sorted_instances = sorted(
            list(self._instances.values()),
            key=lambda i: len(i.tiles.focal_planes)
        )
        try:
            if z is None:
                # Select the instance with selected optical path
                instance = next(i for i in sorted_instances if
                                path in i.tiles.optical_paths)
            elif path is None:
                # Select the instance with selected z
                instance = next(i for i in sorted_instances
                                if z in i.tiles.focal_planes)
            else:
                # Select by both path and z
                instance = next(i for i in sorted_instances
                                if (z in i.tiles.focal_planes and
                                    path in i.tiles.optical_paths))
        except StopIteration:
            raise WsiDicomNotFoundError(
                f"Instance for path: {path}, z: {z}",
                "stack"
            )
        if z is None:
            z = instance.tiles.default_z
        if path is None:
            path = instance.tiles.default_path
        return instance, z, path

    def get_default_full(self) -> Image:
        """Read full image using default z coordinate and path.

        Returns
        ----------
        Image
            Full image of the stack
        """
        instance = self.default_instance
        z = instance.tiles.default_z
        path = instance.tiles.default_path
        region = Region(position=Point(x=0, y=0), size=self.size)
        image = self.get_region(region, z, path)
        return image

    def get_region(
        self,
        region: Region,
        z: float = None,
        path: str = None,
    ) -> Image:
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
        Image
            Region as image
        """

        (instance, z, path) = self.get_instance(z, path)
        image = instance.stitch_tiles(region, path, z)
        return image

    def get_region_mm(
        self,
        region: RegionMm,
        z: float = None,
        path: str = None
    ) -> Image:
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
        Image
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
    ) -> Image:
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
        Image
            The tile as image
        """

        (instance, z, path) = self.get_instance(z, path)
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
        (instance, z, path) = self.get_instance(z, path)
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
            raise WsiDicomOutOfBondsError(
                f"Region {region}", f"level size {self.size}"
            )
        return pixel_region

    def close(self) -> None:
        """Close all instances on the stack."""
        for instance in self._instances.values():
            instance.close()

    def _create_image_type_attribute(self) -> List[str]:
        value_1 = 'DERIVED'
        value_4 = 'RESAMPLED'
        if isinstance(self, WsiDicomLevel):
            if self.level == 0:
                value_1 = 'ORGINAL'
                value_4 = 'None'
        value_2 = 'PRIMARY'
        value_3 = self.wsi_type
        return [value_1, value_2, value_3, value_4]

    def _create_shared_functional_groups_sequence(self) -> DicomSequence:
        pixel_measure_item = Dataset()
        instance = self.default_instance
        pixel_measure_item.SliceThickness = instance.slice_thickness
        if instance.slice_spacing != 0:
            pixel_measure_item.SpacingBetweenSlices = instance.slice_spacing
        pixel_measure_item.PixelSpacing = [
            self.pixel_spacing.width,
            self.pixel_spacing.height
        ]
        pixel_measure_sequence = DicomSequence([pixel_measure_item])
        frame_type_item = Dataset()
        frame_type_item.FrameType = self._create_image_type_attribute()
        frame_type_sequence = DicomSequence([frame_type_item])
        sequence_item = Dataset()
        sequence_item.PixelMeasuresSequence = pixel_measure_sequence
        sequence_item.WholeSlideMicroscopyImageFrameTypeSequence = (
            frame_type_sequence
        )
        return DicomSequence([sequence_item])

    @staticmethod
    def write_preamble(fp: pydicom.filebase.DicomFileLike):
        """Writes file preamble to file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomFileLike
            Filepointer to file to write.
        """
        preamble = b'\x00' * 128
        fp.write(preamble)
        fp.write(b'DICM')

    def write_file_meta(self, fp: pydicom.filebase.DicomFileLike, uid: Uid):
        """Writes file meta dataset to file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomFileLike
            Filepointer to file to write.
        uid: Uid
            SOP instance uid to include in file.
        """
        WSI_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.77.1.6'
        meta_ds = pydicom.dataset.FileMetaDataset()
        meta_ds.TransferSyntaxUID = self.default_instance._transfer_syntax
        meta_ds.MediaStorageSOPInstanceUID = uid
        meta_ds.MediaStorageSOPClassUID = WSI_SOP_CLASS_UID
        pydicom.dataset.validate_file_meta(meta_ds)
        pydicom.filewriter.write_file_meta_info(fp, meta_ds)

    def write_base(
        self,
        fp: pydicom.filebase.DicomFileLike,
        uid: Uid,
        focal_planes: List[str],
        optical_paths: List[float]
    ):
        """Writes base dataset to file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomFileLike
            Filepointer to file to write.
        uid: Uid
            SOP instance uid to include in file.
        focal_planes: List[float]
            List of focal planes to include in file.
        optical_paths: List[str]
            List of optical paths to include in file.
        """
        ds = self.default_instance.create_ds()
        ds.ImageType = self._create_image_type_attribute()
        ds.SOPInstanceUID = uid
        ds.DimensionOrganizationType = 'TILED_FULL'
        plane_size = self.default_instance.tiles.plane_size
        number_of_frames = (
            plane_size.width
            * plane_size.height
            * len(optical_paths)
            * len(focal_planes)
        )
        ds.NumberOfFrames = number_of_frames
        now = datetime.now()
        ds.ContentDate = datetime.date(now).strftime('%Y%m%d')
        ds.ContentTime = datetime.time(now).strftime('%H%M%S.%f')
        ds.TotalPixelMatrixFocalPlanes = len(focal_planes)
        ds.SharedFunctionalGroupsSequence = (
            self._create_shared_functional_groups_sequence()
        )
        ds.NumberOfOpticalPaths = len(optical_paths)

        # We need to add to ds:
        # Optical path sequence

        pydicom.filewriter.write_dataset(fp, ds)

    def write_pixel_data(
        self,
        fp: pydicom.filebase.DicomFileLike,
        focal_planes: List[float],
        optical_paths: List[str]
    ):
        """Writes pixel data to file.

        Parameters
        ----------
        fp: pydicom.filebase.DicomFileLike
            Filepointer to file to write.
        focal_planes: List[float]
            List of focal planes to include in file.
        optical_paths: List[str]
            List of optical paths to include in file.
        """
        plane_size = self.default_instance.tiles.plane_size
        tile_geometry = Region(Point(0, 0), plane_size)
        # Generator for the tiles
        tiles = (
            self.get_encoded_tile(Point(x_tile, y_tile), z, path)
            for path in optical_paths
            for z in focal_planes
            for x_tile, y_tile in tile_geometry.iterate_all()
        )

        pixel_data_element = pydicom.dataset.DataElement(
            0x7FE00010,
            'OB',
            0,
            is_undefined_length=True
            )

        # Write pixel data tag
        fp.write_tag(pixel_data_element.tag)

        if not fp.is_implicit_VR:
            # Write pixel data VR (OB), two empty bytes (PS3.5 7.1.2)
            fp.write(bytes(pixel_data_element.VR, "iso8859"))
            fp.write_US(0)
        # Write unspecific length
        fp.write_UL(0xFFFFFFFF)

        # Write item tag and (empty) length for BOT
        fp.write_tag(pydicom.tag.ItemTag)
        fp.write_UL(0)

        # itemize and and write the tiles
        for tile in tiles:
            for frame in pydicom.encaps.itemize_frame(tile, 1):
                fp.write(frame)

        # end sequence
        fp.write_tag(pydicom.tag.SequenceDelimiterTag)
        fp.write_UL(0)

    @staticmethod
    def create_filepointer(path: Path) -> pydicom.filebase.DicomFileLike:
        """Return a dicom filepointer.

        Parameters
        ----------
        path: Path
            Path to filepointer.
        Returns
        ----------
        pydicom.filebase.DicomFileLike
            Created filepointer.
        """
        fp = pydicom.filebase.DicomFile(path, mode='wb')
        fp.is_little_endian = True
        fp.is_implicit_VR = False
        return fp

    def save(
        self,
        path: Path,
        focal_planes: List[float] = None,
        optical_paths: List[str] = None
    ):
        """Writes stack to to file. File is written as TILED_FULL.
        Writing of optical path sequence is not yet implemented.

        Parameters
        ----------
        path: Path
            Path to directory to write to.
        optical_paths: List[str]
            List of optical paths to include in file.
        focal_planes: List[float]
            List of focal planes to include in file.
        """
        uid = pydicom.uid.generate_uid()
        file_path = os.path.join(path, uid+'.dcm')

        fp = self.create_filepointer(file_path)
        self.write_preamble(fp)
        self.write_file_meta(fp, uid)

        if optical_paths is not None:
            if not all(path in self.optical_paths for path in optical_paths):
                raise ValueError("Requested optical paths not found")
        else:
            optical_paths = self.optical_paths

        if focal_planes is not None:
            if not all(plane in self.focal_planes for plane in focal_planes):
                raise ValueError("Requested focal planes not found")
        else:
            focal_planes = self.focal_planes

        self.write_base(fp, uid, focal_planes, optical_paths)
        self.write_pixel_data(fp, focal_planes, optical_paths)

        # close the file
        fp.close()

    def _validate_stack(
        self,
    ) -> Tuple[BaseUids, str]:
        """Check that no file or instance in stack is duplicate, instances in
        stack matches and that the optical manager matches by base uid.
        Raises WsiDicomMatchError otherwise.
        Returns the matching base uid.

        Returns
        ----------
        Tuple[BaseUids, str]
            Matching uids and wsi type
        """
        instances = list(self.instances.values())
        WsiDicomFile.check_duplicate_file(self.files, self)
        WsiDicomInstance.check_duplicate_instance(instances, self)
        base_instance = instances[0]
        for instance in instances[1:]:
            if not base_instance.matches(instance):
                raise WsiDicomMatchError(str(instance), str(self))
        return base_instance.uids, base_instance.wsi_type

    @classmethod
    def _group_instances(
        cls,
        instances: List[WsiDicomInstance]
    ) -> Dict[Size, List[WsiDicomInstance]]:
        """Return instances grouped by image size (stacks).

        Parameters
        ----------
        instances: List[WsiDicomInstance]
            Instances to group into stacks.

        Returns
        ----------
        Dict[Size, List[WsiDicomInstance]]:
            Instances grouped by size, with size as key.

        """
        grouped_instances: Dict[Size, List[WsiDicomInstance]] = {}
        for instance in instances:
            try:
                grouped_instances[instance.size].append(instance)
            except KeyError:
                grouped_instances[instance.size] = [instance]
        return grouped_instances


class WsiDicomLevel(WsiDicomStack):
    def __init__(
        self,
        instances: List[WsiDicomInstance],
        base_pixel_spacing: SizeMm
    ):
        """Represents a level in the pyramid and contains one or more
        instances having the same level, pixel spacing, and size but possibly
        different focal planes and/or optical paths and present in
        different files.

        Parameters
        ----------
        instances: List[WsiDicomInstance]
            Instances to build the stack.
        base_pixel_spacing: SizeMm
            Pixel spacing of base level.
        """
        super().__init__(instances)
        self._level = self._assign_level(base_pixel_spacing)

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
        string += dict_pretty_str(self.instances, indent, depth)
        return string

    @property
    def pyramid(self) -> str:
        """Return string representatin of the level"""
        return (
            f'Level [{self.level}]'
            f' tiles: {self.default_instance.tiles.plane_size},'
            f' size: {self.size}, mpp: {self.mpp} um/px'
        )

    @property
    def level(self) -> int:
        """Return pyramid level"""
        return self._level

    @classmethod
    def open_levels(
        cls,
        instances: List[WsiDicomInstance],
    ) -> List['WsiDicomLevel']:
        """Return list of level stacks created wsi files.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Files to create stacks from
        optical: OpticalManager
            Optical manager to add optical paths to.

        Returns
        ----------
        List[WsiDicomLevel]
            List of created level stacks.

        """
        levels: List[WsiDicomLevel] = []
        instances_grouped_by_stack = cls._group_instances(instances)
        largest_size = max(instances_grouped_by_stack.keys())
        base_group = instances_grouped_by_stack[largest_size]
        base_pixel_spacing = base_group[0].pixel_spacing
        for level_group in instances_grouped_by_stack.values():
            new_level = WsiDicomLevel(level_group, base_pixel_spacing)
            levels.append(new_level)

        return levels

    def matches(self, other_stack: 'WsiDicomStack') -> bool:
        """Check if stack is valid (Uids and tile size match).
        The common Uids should match for all series. For level series the tile
        size should also match. It is assumed that the instances in the stacks
        are matching each other.
        """
        other_instance = other_stack.default_instance
        this_instance = self.default_instance
        return (
            other_stack.uids == self.uids and
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
    ) -> Image:
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
        Image
            A tile image
        """
        scale = self.calculate_scale(level)
        (instance, z, path) = self.get_instance(z, path)
        scaled_region = Region.from_tile(tile, instance.tile_size) * scale
        cropped_region = instance.crop_to_level_size(scaled_region)
        if not self.valid_pixels(cropped_region):
            raise WsiDicomOutOfBondsError(
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
        (instance, z, path) = self.get_instance(z, path)
        return instance.encode(image)

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
            raise NotImplementedError("Levels needs to be integer")
        return level


class WsiDicomSeries(metaclass=ABCMeta):
    wsi_type: str

    def __init__(self, stacks: List[WsiDicomStack]):
        """Holds a series of stacks of same image flavor

        Parameters
        ----------
        stacks: List[WsiDicomStack]
            List of stacks to include in the series
        """
        self._stacks: List[WsiDicomStack] = stacks
        self._uids = self._validate_series(self.stacks)

    def __getitem__(self, index: int) -> WsiDicomStack:
        """Get stack by index.

        Parameters
        ----------
        index: int
            Index in series to get

        Returns
        ----------
        WsiDicomStack
            The stack at index in the series
        """
        return self.stacks[index]

    @property
    def stacks(self) -> List[WsiDicomStack]:
        """Return contained stacks"""
        return self._stacks

    @property
    def uids(self) -> Optional[BaseUids]:
        """Return uids"""
        return self._uids

    @property
    def mpps(self) -> List[SizeMm]:
        """Return contained mpp (um/px)"""
        return [stack.mpp for stack in self.stacks]

    @property
    def files(self) -> List[WsiDicomFile]:
        """Return contained files"""
        series_files = [series.files for series in self.stacks]
        return [file for sublist in series_files for file in sublist]

    @property
    def instances(self) -> List[WsiDicomInstance]:
        """Return contained instances"""
        series_instances = [
            series.instances.values() for series in self.stacks
        ]
        return [
            instance for sublist in series_instances for instance in sublist
        ]

    def _validate_series(
            self,
            stacks: Union[List[WsiDicomStack], List[WsiDicomLevel]]
    ) -> Optional[BaseUids]:
        """Check that no files or instances in series is duplicate, all stacks
        in series matches and that the optical manager matches by base uid.
        Raises WsiDicomMatchError otherwise.
        Returns the matching base uid. If list of stacks is empty, return None.

        Parameters
        ----------
        stacks: Union[List[WsiDicomStack], List[WsiDicomLevel]]
            List of stacks to check
        optical: OpticalManager
            Optical manager to check

        Returns
        ----------
        Optional[BaseUids]:
            Matching uids
        """
        WsiDicomFile.check_duplicate_file(self.files, self)
        WsiDicomInstance.check_duplicate_instance(self.instances, self)

        try:
            base_stack = stacks[0]
            if base_stack.wsi_type != self.wsi_type:
                raise WsiDicomMatchError(
                    str(base_stack), str(self)
                )
            for stack in stacks[1:]:
                if not stack.matches(base_stack):
                    raise WsiDicomMatchError(
                        str(stack), str(self)
                    )
            return base_stack.uids
        except IndexError:
            return None

    def close(self) -> None:
        """Close all stacks in the series."""
        for stack in self.stacks:
            stack.close()

    def save(
        self,
        path: Path,
    ):
        for stack in self.stacks:
            stack.save(path)


class WsiDicomLabels(WsiDicomSeries):
    wsi_type = 'LABEL'

    @classmethod
    def open(
        cls,
        instances: List[WsiDicomInstance]
    ) -> 'WsiDicomLevels':
        """Return label series created from wsi files.

        Parameters
        ----------
        instances: List[WsiDicomInstance]
            Instances to create levels from.

        Returns
        ----------
        WsiDicomLabels
            Created label series
        """
        labels = WsiDicomStack.open(instances)
        return WsiDicomLabels(labels)


class WsiDicomOverviews(WsiDicomSeries):
    wsi_type = 'OVERVIEW'

    @classmethod
    def open(
        cls,
        instances: List[WsiDicomInstance]
    ) -> 'WsiDicomLevels':
        """Return overview series created from wsi files.

        Parameters
        ----------
        instances: List[WsiDicomInstance]
            Instances to create levels from.

        Returns
        ----------
        WsiDicomOverviews
            Created overview series
        """
        overviews = WsiDicomStack.open(instances)
        return WsiDicomOverviews(overviews)


class WsiDicomLevels(WsiDicomSeries):
    wsi_type = 'VOLUME'

    def __init__(self, levels: List[WsiDicomLevel]):
        """Holds a stack of levels.

        Parameters
        ----------
        stacks: List[WsiDicomLevel]
            List of level stacks to include in series
        """
        self._levels = OrderedDict(
            (level.level, level)
            for level in sorted(levels, key=lambda level: level.level)
        )
        self._uids = self._validate_series(self.stacks)

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
    def stacks(self) -> List[WsiDicomStack]:
        """Return contained stacks"""
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

    @classmethod
    def open(
        cls,
        instances: List[WsiDicomInstance]
    ) -> 'WsiDicomLevels':
        """Return level series created from wsi files.

        Parameters
        ----------
        instances: List[WsiDicomInstance]
            Instances to create levels from.

        Returns
        ----------
        WsiDicomLevels
            Created level series
        """
        levels = WsiDicomLevel.open_levels(instances)
        return WsiDicomLevels(levels)

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
            raise WsiDicomOutOfBondsError(
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
        closest_size = self.stacks[0].size
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
    def __init__(
        self,
        series: List[WsiDicomSeries],
        optical: OpticalManager
    ):
        """Holds wsi dicom levels, labels and overviews.

        Parameters
        ----------
        series: List[WsiDicomSeries]
            List of series (levels, labels, overviews).
        optical: OpticalManager
            Manager for optical paths.
        """

        self._levels: WsiDicomLevels = self._assign(series, 'levels')
        self._labels: WsiDicomLabels = self._assign(series, 'labels')
        self._overviews: WsiDicomOverviews = self._assign(series, 'overviews')

        self.uids = self._validate_collection(
            [self.levels, self.labels, self.overviews],
            optical
        )
        self.optical = optical
        base = self.levels.base_level.default_instance
        self.ds = base.create_ds()
        self.__enter__()

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def __str__(self) -> str:
        return self.pretty_str()

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
    def instances(self) -> List[WsiDicomInstance]:
        """Return contained instances"""
        return (
            self.levels.instances
            + self.labels.instances
            + self.overviews.instances
        )

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
            + list_pretty_str(self.levels.stacks, indent, depth, 0, 2)
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

        for filepath in cls._filter_paths(filepaths):
            dicom_file = WsiDicomFile(filepath)
            if(dicom_file.wsi_type == 'VOLUME'):
                level_files.append(dicom_file)
            elif(dicom_file.wsi_type == 'LABEL'):
                label_files.append(dicom_file)
            elif(dicom_file.wsi_type == 'OVERVIEW'):
                overview_files.append(dicom_file)
            else:
                dicom_file.close()

        base_file = cls._get_base_file(level_files)
        optical = OpticalManager(base_file.uids.base)

        level_instances = WsiDicomInstance.open(
            level_files,
            optical,
            base_file.uids.base,
            base_file.tile_size
        )
        label_instances = WsiDicomInstance.open(
            label_files,
            optical,
            base_file.uids.base,
        )
        overview_instances = WsiDicomInstance.open(
            overview_files,
            optical,
            base_file.uids.base,
        )

        levels = WsiDicomLevels.open(level_instances)
        labels = WsiDicomLabels.open(label_instances)
        overviews = WsiDicomOverviews.open(overview_instances)

        return WsiDicom([levels, labels, overviews], optical=optical)

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
    def _get_base_file(
        files: List[WsiDicomFile]
    ) -> WsiDicomFile:
        """Return file with largest image (width) from list of files.

        Parameters
        ----------
        files: List[WsiDicomFile]
            Wsi files to check.

        Returns
        ----------
        WsiDicomFile
            Base layer file.
        """
        base_size = Size(0, 0)
        base_file: WsiDicomFile
        for file in files:
            if file.image_size.width > base_size.width:
                base_file = file
                base_size = file.image_size
        return base_file

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
        series: List[WsiDicomSeries],
        optical: OpticalManager
    ) -> BaseUids:
        """Check that no files or instance in collection is duplicate, that all
        series and optical manager all have the same base uids.
        Raises WsiDicomMatchError otherwise.
        Returns the matching base uid.

        Parameters
        ----------
        series: List[WsiDicomSeries]
            List of series to check
        optical: OpticalManager
            Optical manager to check

        Returns
        ----------
        BaseUids
            Matching uids
        """
        WsiDicomFile.check_duplicate_file(self.files, self)
        WsiDicomInstance.check_duplicate_instance(self.instances, self)

        try:
            base_uids = next(
                item.uids for item in series if item.uids is not None
            )
        except StopIteration:
            raise WsiDicomNotFoundError("Valid series", "in collection")
        for i, item in enumerate(series):
            if item.uids is not None and item.uids != base_uids:
                raise WsiDicomMatchError(str(item), str(self))
        if base_uids != optical.uids:
            raise WsiDicomMatchError(str(optical), str(self))
        return base_uids

    @staticmethod
    def _assign(series: List[WsiDicomSeries], assign_as: str):
        """Returns first series in list that matches assign as parameter.
        Returns None if none found.

        Parameters
        ----------
        series: List[WsiDicomSeries]
            List of series to check
        assign_as: str
            Type of series to get

        Returns
        ----------
        Optional[
            WsiDicomLevels,
            WsiDicomLabels,
            WsiDicomOverviews
        ]
            Series of assign as type
        """
        series_types = {
            'levels': WsiDicomLevels,
            'labels': WsiDicomLabels,
            'overviews': WsiDicomOverviews
        }
        return next(
            (item for item in series
             if type(item) == series_types[assign_as]), None
        )

    def save(
        self,
        path: Path,
        levels: Tuple[int, int] = None,
    ):
        series = [self.levels, self.labels, self.overviews]
        for item in series:
            item.save(path)

    def read_label(self, index: int = 0) -> Image:
        """Read label image of the whole slide. If several label
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int
            Index of the label image to read

        Returns
        ----------
        Image
            label as image
        """
        try:
            label = self.labels[index]
            return label.get_default_full()
        except IndexError:
            raise WsiDicomNotFoundError("label", "series")

    def read_overview(self, index: int = 0) -> Image:
        """Read overview image of the whole slide. If several overview
        images are present, index can be used to select a specific image.

        Parameters
        ----------
        index: int
            Index of the overview image to read

        Returns
        ----------
        Image
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
    ) -> Image:
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
        Image
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
    ) -> Image:
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
        Image
            Region as image
        """
        wsi_level = self.levels.get_closest_by_level(level)
        scale_factor = wsi_level.calculate_scale(level)
        scaled_region = Region(
            position=Point.from_tuple(location),
            size=Size.from_tuple(size)
        ) * scale_factor

        if not wsi_level.valid_pixels(scaled_region):
            raise WsiDicomOutOfBondsError(
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
    ) -> Image:
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
        Image
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
    ) -> Image:
        """Read image from region defined in mm with set pixel spacing.

        Parameters
        ----------
        location: float, float
            Upper left corner of region in mm
        mpp: float
            Requested pixel spacing (um/mm)
        size: float
            Size of region in mm
        z: float
            Z coordinate, optional
        path:
            optical path, optional

        Returns
        ----------
        Image
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
        return image.resize(image_size.to_int_tuple(), resample=Image.BILINEAR)

    def read_tile(
        self,
        level: int,
        tile: Tuple[int, int],
        z: float = None,
        path: str = None
    ) -> Image:
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
        Image
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
    ) -> Tuple[WsiDicomInstance, float, str]:
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
        Tuple[WsiDicomInstance, float, str]:
            Instance, selected z and path
        """
        wsi_level = self.levels.get_level(level)
        return wsi_level.get_instance(z, path)

    def close(self) -> None:
        """Close all files."""
        for series in [self.levels, self.overviews, self.labels]:
            series.close()
