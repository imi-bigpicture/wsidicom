import copy
import warnings
from functools import cached_property
from typing import Callable, List, Optional, Tuple

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import UID as Uid

from .errors import WsiDicomError, WsiDicomFileError, WsiDicomUidDuplicateError
from .geometry import Size, SizeMm
from .uid import WSI_SOP_CLASS_UID, BaseUids, FileUids


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
                        # print(f' is missing {key} attribute {attribute}')
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
            self.tile_type == other_dataset.tile_type
            # and
            # (
            #     self.get_supported_wsi_dicom_type()
            #     == other_dataset.get_supported_wsi_dicom_type()
            # )
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

    def read_optical_path_identifier(self, frame: Dataset) -> str:
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
    def create_test_base_dataset(
        uid_generator: Callable[..., Uid] = pydicom.uid.generate_uid
    ) -> Dataset:
        dataset = Dataset()
        dataset.StudyInstanceUID = uid_generator()
        dataset.SeriesInstanceUID = uid_generator()
        dataset.FrameOfReferenceUID = uid_generator()
        dataset.Modality = 'SM'
        dataset.SOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.6'
        dataset.Manufacturer = 'Manufacturer'
        dataset.ManufacturerModelName = 'ManufacturerModelName'
        dataset.DeviceSerialNumber = 'DeviceSerialNumber'
        dataset.SoftwareVersions = ['SoftwareVersions']

        # Generic specimen sequence
        dataset.ContainerIdentifier = 'ContainerIdentifier'
        specimen_description_sequence = Dataset()
        specimen_description_sequence.SpecimenIdentifier = 'SpecimenIdentifier'
        specimen_description_sequence.SpecimenUID = uid_generator()
        dataset.SpecimenDescriptionSequence = DicomSequence(
            [specimen_description_sequence]
        )

        # Generic optical path sequence
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
        dataset.OpticalPathSequence = DicomSequence([optical_path_sequence])

        # Generic dimension organization sequence
        dimension_organization_uid = uid_generator()
        dimension_organization_sequence = Dataset()
        dimension_organization_sequence.DimensionOrganizationUID = (
            dimension_organization_uid
        )
        dataset.DimensionOrganizationSequence = DicomSequence(
            [dimension_organization_sequence]
        )

        # Generic dimension index sequence
        dimension_index_sequence = Dataset()
        dimension_index_sequence.DimensionOrganizationUID = (
            dimension_organization_uid
        )
        dimension_index_sequence.DimensionIndexPointer = (
            pydicom.tag.Tag('PlanePositionSlideSequence')
        )
        dataset.DimensionIndexSequence = DicomSequence(
            [dimension_index_sequence]
        )

        dataset.BurnedInAnnotation = 'NO'
        dataset.BurnedInAnnotation = 'NO'
        dataset.SpecimenLabelInImage = 'NO'
        dataset.VolumetricProperties = 'VOLUME'
        return dataset

    @staticmethod
    def _get_image_type(image_flavor: str, level_index: int) -> List[str]:
        if image_flavor == 'VOLUME' and level_index == 0:
            resampled = 'NONE'
        else:
            resampled = 'RESAMPLED'

        return ['ORGINAL', 'PRIMARY', image_flavor, resampled]

    @classmethod
    def create_instance_dataset(
        cls,
        base_dataset: Dataset,
        image_flavour: str,
        level_index: int,
        image_size: Size,
        tile_size: Size,
        mpp: SizeMm,
        uid_generator: Callable[..., Uid] = pydicom.uid.generate_uid
    ) -> Dataset:
        dataset = copy.deepcopy(base_dataset)
        dataset.ImageType = cls._get_image_type(image_flavour, level_index)
        dataset.SOPInstanceUID = uid_generator()

        shared_functional_group_sequence = Dataset()
        pixel_measure_sequence = Dataset()
        pixel_measure_sequence.PixelSpacing = [mpp.width, mpp.height]
        pixel_measure_sequence.SpacingBetweenSlices = 0.0
        pixel_measure_sequence.SliceThickness = 0.0
        shared_functional_group_sequence.PixelMeasuresSequence = (
            DicomSequence([pixel_measure_sequence])
        )
        dataset.SharedFunctionalGroupsSequence = DicomSequence(
            [shared_functional_group_sequence]
        )
        dataset.TotalPixelMatrixColumns = image_size.width
        dataset.TotalPixelMatrixRows = image_size.height
        dataset.Columns = tile_size.width
        dataset.Rows = tile_size.height
        dataset.ImagedVolumeWidth = image_size.width * mpp.width
        dataset.ImagedVolumeHeight = image_size.height * mpp.height
        dataset.ImagedVolumeDepth = 0.0
        # If PhotometricInterpretation is YBR and no subsampling
        dataset.SamplesPerPixel = 3
        dataset.PhotometricInterpretation = 'YBR_FULL'
        # If transfer syntax pydicom.uid.JPEGBaseline8Bit
        dataset.BitsAllocated = 8
        dataset.BitsStored = 8
        dataset.HighBit = 8
        dataset.PixelRepresentation = 0
        dataset.LossyImageCompression = '01'
        dataset.LossyImageCompressionRatio = 1
        dataset.LossyImageCompressionMethod = 'ISO_10918_1'

        # Should be incremented
        dataset.InstanceNumber = 0
        dataset.FocusMethod = 'AUTO'
        dataset.ExtendedDepthOfField = 'NO'
        return dataset
