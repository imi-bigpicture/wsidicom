import warnings
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Tuple

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import UID as Uid

from .errors import WsiDicomError, WsiDicomFileError, WsiDicomUidDuplicateError
from .geometry import Size, SizeMm
from .uid import BaseUids, FileUids


class WsiDataset(Dataset):
    """Extend pydicom dataset with simple parsers for attributes. Use snake
    case to avoid name collision with dicom fields."""

    @staticmethod
    def is_wsi_dicom(ds: pydicom.Dataset) -> bool:
        """Check if dataset is dicom wsi type and that required attributes
        (for the function of the library) is available.
        Warn if attribute listed as requierd in the library or required in the
        standard is missing.

        Parameters
        ----------
        ds: pydicom.Dataset
            Dataset

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
                    if attribute not in ds:
                        # warnings.warn(
                        #     f' is missing {key} attribute {attribute}'
                        # )
                        passed[key] = False

        sop_class_uid = getattr(ds, "SOPClassUID", "")
        sop_class_uid_check = (sop_class_uid == WSI_SOP_CLASS_UID)
        return passed['required'] and sop_class_uid_check

    @staticmethod
    def check_duplicate_dataset(
        datasets: List[Dataset],
        caller: object
    ) -> None:
        """Check for duplicates in list of datasets. Datasets are duplicate if
        instance uids match. Stops at first found duplicate and raises
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
            self.tile_type == other_dataset.tile_type and
            (
                self.get_supported_wsi_dicom_type()
                == other_dataset.get_supported_wsi_dicom_type()
            )
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
        IMAGE_FLAVOR = 2
        image_type: str = self.ImageType[IMAGE_FLAVOR]
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

    def get_optical_path_identifier(self, frame: Dataset) -> str:
        """Return optical path identifier from frame, or from self if not
        found."""
        optical_sequence = getattr(
            frame,
            'OpticalPathIdentificationSequence',
            self.OpticalPathSequence
        )
        return getattr(optical_sequence[0], 'OpticalPathIdentifier', '0')

    @cached_property
    def instance_uid(self) -> Uid:
        return Uid(self.SOPInstanceUID)

    @cached_property
    def concatenation_uid(self) -> Optional[Uid]:
        return getattr(
            self,
            'SOPInstanceUIDOfConcatenationSource',
            None
        )

    @cached_property
    def base_uids(self) -> BaseUids:
        return BaseUids(
            self.StudyInstanceUID,
            self.SeriesInstanceUID,
            self.FrameOfReferenceUID,
        )

    @cached_property
    def uids(self) -> FileUids:
        return FileUids(
            self.instance_uid,
            self.concatenation_uid,
            self.base_uids
        )

    @cached_property
    def frame_offset(self) -> int:
        if self.concatenation_uid is None:
            return 0
        try:
            return int(self.ConcatenationFrameOffsetNumber)
        except AttributeError:
            raise WsiDicomError(
                'Concatenated file missing concatenation frame offset'
                'number'
            )

    @cached_property
    def frame_count(self) -> int:
        return int(getattr(self, 'NumberOfFrames', 1))

    @cached_property
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
        if(getattr(self, 'DimensionOrganizationType', '') == 'TILED_FULL'):
            return 'TILED_FULL'
        elif 'PerFrameFunctionalGroupsSequence' in self:
            return 'TILED_SPARSE'
        raise WsiDicomError("undetermined tile type")

    @cached_property
    def frame_count(self) -> int:
        return int(getattr(self, 'NumberOfFrames', 1))

    @cached_property
    def pixel_measure(self) -> Dataset:
        return self.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]

    @cached_property
    def pixel_spacing(self) -> SizeMm:
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
        pixel_measure = self.pixel_measure
        pixel_spacing: Tuple[float, float] = pixel_measure.PixelSpacing
        if any([spacing == 0 for spacing in pixel_spacing]):
            raise WsiDicomError("Pixel spacing is zero")
        return SizeMm(width=pixel_spacing[0], height=pixel_spacing[1])

    @cached_property
    def spacing_between_slices(self) -> float:
        pixel_measure = self.pixel_measure
        return getattr(pixel_measure, 'SpacingBetweenSlices', 0.0)

    @cached_property
    def number_of_focal_planes(self) -> int:
        return getattr(self, 'TotalPixelMatrixFocalPlanes', 1)

    @cached_property
    def file_offset(self) -> int:
        return int(getattr(self, 'ConcatenationFrameOffsetNumber', 0))

    @cached_property
    def frame_sequence(self) -> DicomSequence:
        if (
            'PerFrameFunctionalGroupsSequence' in self and
            (
                'PlanePositionSlideSequence' in
                self.PerFrameFunctionalGroupsSequence[0]
            )
        ):
            return self.PerFrameFunctionalGroupsSequence

        return self.SharedFunctionalGroupsSequence

    @cached_property
    def ext_depth_of_field(self) -> bool:
        return self.ExtendedDepthOfField == 'YES'

    @cached_property
    def ext_depth_of_field_planes(self) -> Optional[int]:
        return getattr(self, 'NumberOfFocalPlanes', None)

    @cached_property
    def ext_depth_of_field_plane_distance(self) -> Optional[int]:
        return getattr(self, 'DistanceBetweenFocalPlanes', None)

    @cached_property
    def focus_method(self) -> str:
        return str(self.FocusMethod)

    @cached_property
    def image_size(self) -> Size:
        """Read total pixel size from dataset.

        Returns
        ----------
        Size
            The image size
        """
        width = int(self.TotalPixelMatrixColumns)
        height = int(self.TotalPixelMatrixRows)
        if width == 0 or height == 0:
            raise WsiDicomFileError(self.filepath, "Image size is zero")
        return Size(width=width, height=height)

    @cached_property
    def mm_size(self) -> SizeMm:
        """Read mm size from dataset.

        Returns
        ----------
        SizeMm
            The size of the image in mm
        """
        width = float(self.ImagedVolumeWidth)
        height = float(self.ImagedVolumeHeight)
        return SizeMm(width=width, height=height)

    @cached_property
    def mm_depth(self) -> float:
        return self.ImagedVolumeDepth

    @cached_property
    def tile_size(self) -> Size:
        """Read tile size from from dataset.

        Returns
        ----------
        Size
            The tile size
        """
        width = int(self.Columns)
        height = int(self.Rows)
        return Size(width=width, height=height)

    @cached_property
    def samples_per_pixel(self) -> int:
        return int(self.SamplesPerPixel)

    @cached_property
    def photophotometric_interpretation(self) -> str:
        return self.PhotometricInterpretation

    @cached_property
    def instance_number(self) -> str:
        return self.InstanceNumber

    @cached_property
    def optical_path_sequence(self) -> DicomSequence:
        return self.OpticalPathSequence

    @cached_property
    def slice_thickness(self) -> float:
        try:
            pixel_measure = self.pixel_measure
            return float(pixel_measure.SliceThickness)
        except AttributeError:
            # This might not be correct if multiple focal planes
            return self.mm_depth

    @staticmethod
    def create_test_base_dataset() -> Dataset:
        ds = Dataset()
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
        ds.Modality = 'SM'
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.6'
        ds.Manufacturer = 'Manufacturer'
        ds.ManufacturerModelName = 'ManufacturerModelName'
        ds.DeviceSerialNumber = 'DeviceSerialNumber'
        ds.SoftwareVersions = ['SoftwareVersions']

        ds.ContainerIdentifier = 'ContainerIdentifier'
        specimen_description_sequence = Dataset()
        specimen_description_sequence.SpecimenIdentifier = 'SpecimenIdentifier'
        specimen_description_sequence.SpecimenUID = pydicom.uid.generate_uid()
        ds.SpecimenDescriptionSequence = DicomSequence(
            [specimen_description_sequence]
        )
        optical_path_sequence = Dataset()
        optical_path_sequence.OpticalPathIdentifier = '1'
        illumination_type_code_sequence = Dataset()
        illumination_type_code_sequence.CodeValue = '111744'
        illumination_type_code_sequence.CodingSchemeDesignator = 'DCM'
        illumination_type_code_sequence.CodeMeaning = (
            'Brightfield illumination'
        )
        optical_path_sequence.IlluminationTypeCodeSequence = DicomSequence(
            [illumination_type_code_sequence]
        )

        illumination_color_code_sequence = Dataset()
        illumination_color_code_sequence.CodeValue = 'R-102C0'
        illumination_color_code_sequence.CodingSchemeDesignator = 'SRT'
        illumination_color_code_sequence.CodeMeaning = 'Full Spectrum'
        optical_path_sequence.IlluminationColorCodeSequence = DicomSequence(
            [illumination_color_code_sequence]
        )

        ds.OpticalPathSequence = DicomSequence([optical_path_sequence])

        return ds


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
        self._transfer_syntax_uid = Uid(file_meta.TransferSyntaxUID)

        self._fp = pydicom.filebase.DicomFile(self.filepath, mode='rb')
        self._fp.is_little_endian = self._transfer_syntax_uid.is_little_endian
        self._fp.is_implicit_VR = self._transfer_syntax_uid.is_implicit_VR

        dataset = pydicom.dcmread(self._fp, stop_before_pixels=True)

        if WsiDataset.is_wsi_dicom(dataset):
            self._pixel_data_position = self._fp.tell()
            self._dataset = WsiDataset(dataset)
            self._wsi_type = self.dataset.get_supported_wsi_dicom_type(
                self.transfer_syntax
            )
            instance_uid = self.dataset.instance_uid
            concatenation_uid = self.dataset.concatenation_uid
            base_uids = self.dataset.base_uids
            self._uids = FileUids(instance_uid, concatenation_uid, base_uids)
            self._frame_offset = self.dataset.frame_offset
            self._frame_count = self.dataset.frame_count
            self._frame_positions = self._parse_pixel_data()
        else:
            self._wsi_type = "None"

    def __repr__(self) -> str:
        return f"WsiDicomFile('{self.filepath}')"

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
