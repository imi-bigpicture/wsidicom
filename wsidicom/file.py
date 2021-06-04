import warnings
from pathlib import Path
from typing import List, Tuple

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import UID as Uid

from .errors import WsiDicomFileError, WsiDicomUidDuplicateError
from .geometry import Size, SizeMm
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
                        # warnings.warn(
                        #     f'File {filepath} is missing {key}'
                        #     f' attribute {attribute}'
                        # )
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
