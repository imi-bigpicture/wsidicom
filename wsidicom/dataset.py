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

import warnings
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Any, List, Optional, Sequence, Tuple, cast

from pydicom.dataset import Dataset
from pydicom.pixel_data_handlers import pillow_handler
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import UID
from pydicom.valuerep import DSfloat

from wsidicom.config import settings
from wsidicom.errors import (WsiDicomError, WsiDicomFileError,
                             WsiDicomRequirementError,
                             WsiDicomStrictRequirementError,
                             WsiDicomUidDuplicateError)
from wsidicom.geometry import Orientation, PointMm, RegionMm, Size, SizeMm
from wsidicom.uid import WSI_SOP_CLASS_UID, FileUids, SlideUids


class Requirement(IntEnum):
    ALWAYS = auto()  # Always required, even if not in strict mode
    STRICT = auto()  # Required if in strict mode
    STANDARD = auto()  # Required or optional in standard, not (yet) needed


@dataclass
class WsiAttributeRequirement:
    """Class for defining requirement and optionally default value for DICOM
    WSI attribute. If image types is given, requirement is only checked for
    images of matching image type.
    """
    requirement: Requirement
    image_types: Tuple[str, ...]
    default: Any = None

    def __init__(
        self,
        requirement: Requirement,
        image_types: Optional[Sequence[str]] = None,
        default: Any = None
    ) -> None:
        self.requirement = requirement
        if image_types is not None:
            self.image_types = tuple(image_types)
        else:
            self.image_types = ()
        self.default = default

    def get_default(self, image_type: str) -> Any:
        """Get default value for attribute. Raises WsiDicomRequirementError if
        attribute is set as always required for image type. Raises
        WsiDicomStrictRequirementError if attribute is set as always required
        in strict mode."""
        if (
            self.requirement == Requirement.ALWAYS
            and image_type in self.image_types
        ):
            raise WsiDicomRequirementError(
                f'Attribute is set as always required for {image_type}')
        elif (
            settings.strict_attribute_check
            and self.requirement == Requirement.STRICT
            and image_type in self.image_types
        ):
            raise WsiDicomStrictRequirementError(
                f'Attribute is set as required for {image_type} '
                'and mode is strict'
            )
        return self.default

    def evaluate(self, image_type: str) -> bool:
        """Evaluate if attribute is required for image type."""
        if (
            self.requirement == Requirement.ALWAYS
            or (
                self.requirement == Requirement.STRICT
                and settings.strict_attribute_check
            )
        ):
            if image_type in self.image_types:
                return True
        return False


WSI_ATTRIBUTES = {
    'SOPClassUID': WsiAttributeRequirement(Requirement.ALWAYS),
    'SOPInstanceUID': WsiAttributeRequirement(Requirement.ALWAYS),
    'StudyInstanceUID': WsiAttributeRequirement(Requirement.ALWAYS),
    'SeriesInstanceUID': WsiAttributeRequirement(Requirement.ALWAYS),
    'Rows': WsiAttributeRequirement(Requirement.ALWAYS),
    'Columns': WsiAttributeRequirement(Requirement.ALWAYS),
    'SamplesPerPixel': WsiAttributeRequirement(Requirement.ALWAYS),
    'PhotometricInterpretation': WsiAttributeRequirement(Requirement.ALWAYS),
    'TotalPixelMatrixColumns': WsiAttributeRequirement(Requirement.ALWAYS),
    'TotalPixelMatrixRows': WsiAttributeRequirement(Requirement.ALWAYS),
    'ImageType': WsiAttributeRequirement(Requirement.ALWAYS),

    'SharedFunctionalGroupsSequence': WsiAttributeRequirement(
        Requirement.STRICT,
        ['VOLUME']
    ),
    'FrameOfReferenceUID': WsiAttributeRequirement(Requirement.STRICT),
    'FocusMethod': WsiAttributeRequirement(
        Requirement.STRICT,
        ['VOLUME'],
        'AUTO'
    ),
    'ExtendedDepthOfField': WsiAttributeRequirement(
        Requirement.STRICT,
        ['VOLUME'],
        'NO'
    ),
    'OpticalPathSequence': WsiAttributeRequirement(Requirement.STRICT),
    'ImagedVolumeWidth': WsiAttributeRequirement(
        Requirement.STRICT,
        ['VOLUME']
    ),
    'ImagedVolumeHeight': WsiAttributeRequirement(
        Requirement.STRICT,
        ['VOLUME']
    ),
    'ImagedVolumeDepth': WsiAttributeRequirement(
        Requirement.STRICT,
        ['VOLUME']
    ),
    'TotalPixelMatrixFocalPlanes': WsiAttributeRequirement(
        Requirement.STANDARD,
        default=1
    ),
    'NumberOfOpticalPaths': WsiAttributeRequirement(
        Requirement.STANDARD,
        default=1
    ),
    'NumberOfFrames': WsiAttributeRequirement(Requirement.STANDARD, default=1),
    'Modality': WsiAttributeRequirement(Requirement.STANDARD, default='SM'),
    'Manufacturer': WsiAttributeRequirement(Requirement.STANDARD),
    'ManufacturerModelName': WsiAttributeRequirement(Requirement.STANDARD),
    'DeviceSerialNumber': WsiAttributeRequirement(Requirement.STANDARD),
    'SoftwareVersions': WsiAttributeRequirement(Requirement.STANDARD),
    'BitsAllocated': WsiAttributeRequirement(Requirement.STANDARD),
    'BitsStored': WsiAttributeRequirement(Requirement.STANDARD),
    'HighBit': WsiAttributeRequirement(Requirement.STANDARD),
    'PixelRepresentation': WsiAttributeRequirement(Requirement.STANDARD),
    'TotalPixelMatrixOriginSequence': WsiAttributeRequirement(
        Requirement.STANDARD
    ),
    'ImageOrientationSlide': WsiAttributeRequirement(
        Requirement.STANDARD,
        default=[0, 1, 0, 1, 0, 0]
    ),
    'AcquisitionDateTime': WsiAttributeRequirement(Requirement.STANDARD),
    'LossyImageCompression': WsiAttributeRequirement(Requirement.STANDARD),
    'VolumetricProperties': WsiAttributeRequirement(
        Requirement.STANDARD,
        default='VOLUME'
    ),
    'SpecimenLabelInImage': WsiAttributeRequirement(
        Requirement.STANDARD,
        default='NO'
    ),
    'BurnedInAnnotation': WsiAttributeRequirement(
        Requirement.STANDARD,
        default='NO'
    ),
    'ContentDate': WsiAttributeRequirement(Requirement.STANDARD),
    'ContentTime': WsiAttributeRequirement(Requirement.STANDARD),
    'InstanceNumber': WsiAttributeRequirement(Requirement.STANDARD),
    'DimensionOrganizationSequence': WsiAttributeRequirement(
        Requirement.STANDARD
    ),
    'ContainerIdentifier': WsiAttributeRequirement(Requirement.STANDARD),
    'SpecimenDescriptionSequence': WsiAttributeRequirement(
        Requirement.STANDARD
    ),
    'SpacingBetweenSlices': WsiAttributeRequirement(
        Requirement.STANDARD,
        default=.0
    ),
    'SOPInstanceUIDOfConcatenationSource': WsiAttributeRequirement(
        Requirement.STANDARD),
    'DimensionOrganizationType': WsiAttributeRequirement(
        Requirement.STANDARD,
        default='TILED_SPARSE'
    ),
    'NumberOfFocalPlanes': WsiAttributeRequirement(
        Requirement.STANDARD,
        default=1
    ),
    'DistanceBetweenFocalPlanes': WsiAttributeRequirement(
        Requirement.STANDARD,
        default=0.0
    ),
    'SliceThickness': WsiAttributeRequirement(Requirement.STANDARD)
}


class ImageOrgin:
    def __init__(
        self,
        origin: PointMm,
        orientation: Orientation
    ):
        self._origin = origin
        self._orientation = orientation

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset
    ):
        try:
            origin = PointMm(
                dataset.TotalPixelMatrixOriginSequence[0].
                XOffsetInSlideCoordinateSystem,
                dataset.TotalPixelMatrixOriginSequence[0].
                YOffsetInSlideCoordinateSystem
            )
        except (AttributeError, IndexError):
            warnings.warn(
                "Using default image origin as TotalPixelMatrixOriginSequence "
                "not set in file"
            )
            origin = PointMm(0, 0)
        try:
            orientation = Orientation(dataset.ImageOrientationSlide)
        except AttributeError:
            warnings.warn(
                "Using default image orientation as ImageOrientationSlide "
                "not set in file"
            )
            orientation = Orientation([0, 1, 0, 1, 0, 0])
        return cls(origin, orientation)

    @property
    def rotation(self) -> float:
        return self._orientation.rotation

    def transform_region(
        self,
        region: RegionMm
    ) -> 'RegionMm':
        region.position = region.position - self._origin
        return self._orientation.apply(region)


class WsiDicomDataset(Dataset):
    """Extend pydicom.dataset.Dataset (containing WSI metadata) with simple
    parsers for attributes specific for WSI. Use snake case to avoid name
    collision with dicom fields (that are handled by pydicom.dataset.Dataset).
    """
    def __init__(
        self,
        dataset: Dataset
    ):
        """A WsiDicomDataset wrapping a pydicom Dataset.

        Parameters
        ----------
        dataset: Dataset
            Pydicom dataset containing WSI data.

        Returns
        ----------
        bool
            True if same instance.
        """
        super().__init__(dataset)
        self._uids = self._get_uids()
        self._frame_offset = self._get_concatenation_offset()
        self._frame_count = self._get_dicom_attribute('NumberOfFrames')
        self._tile_type = self._get_tile_organization_type()
        self._pixel_measure = self._get_pixel_measure()
        self._pixel_spacing, self._spacing_between_slices = (
            self._get_spacings(self.pixel_measure)
        )
        self._number_of_focal_planes = self._get_dicom_attribute(
            'TotalPixelMatrixFocalPlanes'
        )

        self._frame_sequence = self._get_frame_sequence()
        (
            self._ext_depth_of_field,
            self._ext_depth_of_field_planes,
            self._ext_depth_of_field_plane_distance
        ) = self._get_ext_depth_of_field()

        self._focus_method = self._get_dicom_attribute('FocusMethod')
        (
            self._image_size,
            self._mm_size,
            self._mm_depth
        ) = self._get_image_size()
        self._tile_size = Size(self.Columns, self.Rows)
        self._samples_per_pixel = self.SamplesPerPixel
        self._photometric_interpretation = self.PhotometricInterpretation
        self._instance_number = self.InstanceNumber
        self._optical_path_sequence = self._get_dicom_attribute(
            'OpticalPathSequence'
        )
        self._slice_thickness = self._get_slice_thickness(self.pixel_measure)
        self._image_origin = ImageOrgin.from_dataset(dataset)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of dataset {self.uids.instance}"

    @property
    def instance_uid(self) -> UID:
        """Return instance uid from dataset."""
        return self.uids.instance

    @property
    def concatenation_uid(self) -> Optional[UID]:
        """Return concatenation uid, if defined, from dataset. An instance that
        is concatenated (split into several files) should have the same
        concatenation uid."""
        return self.uids.concatenation

    @property
    def slide_uids(self) -> SlideUids:
        """Return base uids (study, series, and frame of reference Uids)."""
        return self.uids.slide

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
        return cast(int, self._frame_count)

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
    def pixel_measure(self) -> Optional[Dataset]:
        """Return pixel measure from dataset."""
        return self._pixel_measure

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
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
    def spacing_between_slices(self) -> Optional[float]:
        """Return spacing between slices."""
        if self._spacing_between_slices is None:
            return None
        return cast(float, self._spacing_between_slices)

    @property
    def number_of_focal_planes(self) -> int:
        """Return number of focal planes."""
        return cast(int, self._number_of_focal_planes)

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
    def ext_depth_of_field_plane_distance(self) -> Optional[float]:
        """Return total focal depth used for extended depth of field.
        """
        return self._ext_depth_of_field_plane_distance

    @property
    def focus_method(self) -> str:
        """Return focus method."""
        return str(self._focus_method)

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
    def mm_size(self) -> Optional[SizeMm]:
        """Read mm size from dataset.

        Returns
        ----------
        SizeMm
            The size of the image in mm
        """
        return self._mm_size

    @property
    def mm_depth(self) -> Optional[float]:
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
    def optical_path_sequence(self) -> Optional[DicomSequence]:
        """Return optical path sequence from dataset."""
        return self._optical_path_sequence

    @property
    def slice_thickness(self) -> Optional[float]:
        """Return slice thickness."""
        return self._slice_thickness

    @property
    def wsi_type(self) -> str:
        return self._get_wsi_flavor(self.ImageType)

    @property
    def image_origin(self) -> ImageOrgin:
        return self._image_origin

    @classmethod
    def is_supported_wsi_dicom(
        cls,
        dataset: Dataset,
        transfer_syntax: UID
    ) -> Optional[str]:
        """Check if dataset is dicom wsi type and that required attributes
        (for the function of the library) is available.
        Warn if attribute listed as requierd in the library or required in the
        standard is missing.

        Parameters
        ----------
        dataset: Dataset
            Pydicom dataset to check if is a WSI dataset.
        transfer_syntax: UID
            Transfer syntax of dataset.

        Returns
        ----------
        Optional[str]
            WSI image flavor
        """

        sop_class_uid: UID = getattr(dataset, "SOPClassUID")
        if sop_class_uid != WSI_SOP_CLASS_UID:
            warnings.warn(f"Non-wsi image, SOP class {sop_class_uid.name}")
            return None

        SUPPORTED_IMAGE_TYPES = ['VOLUME', 'LABEL', 'OVERVIEW']
        image_flavor = cls._get_wsi_flavor(dataset.ImageType)
        image_flavor_supported = image_flavor in SUPPORTED_IMAGE_TYPES
        if not image_flavor_supported:
            warnings.warn(f"Non-supported image type {image_flavor}")
            return None

        for name, attribute in WSI_ATTRIBUTES.items():
            if (
                name not in dataset
                and attribute.evaluate(image_flavor)
            ):
                warnings.warn(f"Missing required attribute {name}")
                return None

        syntax_supported = (
            pillow_handler.supports_transfer_syntax(transfer_syntax)
        )
        if not syntax_supported:
            warnings.warn(f"Non-supported transfer syntax {transfer_syntax}")
            return None

        return image_flavor

    @staticmethod
    def check_duplicate_dataset(
        datasets: Sequence['WsiDicomDataset'],
        caller: object
    ) -> None:
        """Check for duplicates in a list of datasets. Datasets are duplicate
        if instance uids match. Stops at first found duplicate and raises
        WsiDicomUidDuplicateError.

        Parameters
        ----------
        datasets: Sequence[Dataset]
            List of datasets to check.
        caller: Object
            Object that the files belongs to.
        """
        instance_uids: List[UID] = []

        for dataset in datasets:
            instance_uid = UID(dataset.SOPInstanceUID)
            if instance_uid not in instance_uids:
                instance_uids.append(instance_uid)
            else:
                raise WsiDicomUidDuplicateError(str(dataset), str(caller))

    def matches_instance(self, other_dataset: 'WsiDicomDataset') -> bool:
        """Return true if other file is of the same instance as self.

        Parameters
        ----------
        other_dataset: 'WsiDicomDataset'
            Dataset to check.

        Returns
        ----------
        bool
            True if same instance.
        """
        return (
            self.uids == other_dataset.uids and
            self.image_size == other_dataset.image_size and
            self.tile_size == other_dataset.tile_size and
            self.tile_type == other_dataset.tile_type
        )

    def matches_series(
        self,
        uids: SlideUids,
        tile_size: Optional[Size] = None
    ) -> bool:
        """Check if instance is valid (Uids and tile size match).
        Base uids should match for instances in all types of series,
        tile size should only match for level series.
        """
        if tile_size is not None and tile_size != self.tile_size:
            return False

        return self.slide_uids.matches(uids)

    def read_optical_path_identifier(self, frame: Dataset) -> str:
        """Return optical path identifier from frame, or from self if not
        found."""
        optical_sequence = getattr(
            frame,
            'OpticalPathIdentificationSequence',
            self.optical_path_sequence
        )
        if optical_sequence is None:
            return '0'
        return getattr(optical_sequence[0], 'OpticalPathIdentifier', '0')

    def as_tiled_full(
        self,
        focal_planes: Sequence[float],
        optical_paths: Sequence[str],
        tiled_size: Size,
        scale: int = 1
    ) -> 'WsiDicomDataset':
        """Return copy of dataset with properties set to reflect a tiled full
        arrangement of the listed image data. Optionally set properties to
        reflect scaled data.

        Parameters
        ----------
        focal_planes: Sequence[float]
            Focal planes that should be encoded into dataset.
        optical_paths: Sequence[str]
            Optical paths that should be encoded into dataset.
        tiled_size: Size
            Tiled size of image.
        scale: int = 1
            Optionally scale data.

        Returns
        ----------
        WsiDicomDataset
            Copy of dataset set as tiled full.

        """

        dataset = deepcopy(self)
        dataset.DimensionOrganizationType = 'TILED_FULL'

        # Make a new Shared functional group sequence and Pixel measure
        # sequence if not in dataset, otherwise update the Pixel measure
        # sequence
        shared_functional_group = getattr(
            dataset,
            'SharedFunctionalGroupsSequence',
            DicomSequence([Dataset()])
        )
        plane_position_slide = Dataset()
        plane_position_slide.ZOffsetInSlideCoordinateSystem = (
            focal_planes[0]
        )
        shared_functional_group[0].PlanePositionSlideSequence = (
            DicomSequence([plane_position_slide])
        )

        pixel_measure = getattr(
            shared_functional_group[0],
            'PixelMeasuresSequence',
            DicomSequence([Dataset()])
        )
        if dataset.pixel_spacing is not None:
            pixel_measure[0].PixelSpacing = [
                DSfloat(dataset.pixel_spacing.width * scale, True),
                DSfloat(dataset.pixel_spacing.height * scale, True)
            ]
        pixel_measure[0].SpacingBetweenSlices = (
            self._get_spacing_between_slices_for_focal_planes(focal_planes)
        )

        if dataset.slice_thickness is not None:
            pixel_measure[0].SliceThickness = dataset.slice_thickness

        shared_functional_group[0].PixelMeasuresSequence = pixel_measure
        dataset.SharedFunctionalGroupsSequence = shared_functional_group

        # Remove Per Frame functional groups sequence
        if 'PerFrameFunctionalGroupsSequence' in dataset:
            del dataset['PerFrameFunctionalGroupsSequence']

        dataset.TotalPixelMatrixFocalPlanes = len(focal_planes)
        dataset.NumberOfOpticalPaths = len(optical_paths)
        dataset.NumberOfFrames = max(
            tiled_size.ceil_div(scale).area,
            1
        ) * len(focal_planes) * len(optical_paths)
        scaled_size = dataset.image_size.ceil_div(scale)
        dataset.TotalPixelMatrixColumns = max(scaled_size.width, 1)
        dataset.TotalPixelMatrixRows = max(scaled_size.height, 1)
        return dataset

    def _get_dicom_attribute(
        self,
        name: str,
        dataset: Optional[Dataset] = None
    ) -> Optional[Any]:
        """Return value for DICOM attribute by name. Optionally get it from
        another dataset. Returns default value if possible when attribute
        is missing.

        Parameters
        ----------
        name: str
            Name of attribute to get.
        dataset: Optional[Dataset] = None
            Optional other dataset to get attribute from.

        Returns
        ----------
        Optional[Any]
            Value of attribute, or None.
        """
        if dataset is None:
            dataset = self
        value = getattr(dataset, name, None)
        if value is None:
            return WSI_ATTRIBUTES[name].get_default(self.wsi_type)
        return value

    def _get_uids(self) -> FileUids:
        """Return UIDs from dataset.

        Returns
        ----------
        FileUids
            Found UIDs from dataset.
        """
        instance_uid = UID(self.SOPInstanceUID)
        concatenation_uid = self._get_dicom_attribute(
            'SOPInstanceUIDOfConcatenationSource'
        )
        frame_of_reference_uid = self._get_dicom_attribute(
            'FrameOfReferenceUID'
        )

        slide_uids = SlideUids(
            self.StudyInstanceUID,
            self.SeriesInstanceUID,
            frame_of_reference_uid,
        )
        file_uids = FileUids(
            instance_uid,
            concatenation_uid,
            slide_uids
        )
        return file_uids

    def _get_concatenation_offset(self) -> int:
        """Return concatenation offset (number of frames). Will be 0 if file
        is not concatentated or first in concatenation.

        Returns
        ----------
        int
            Concatenation offset in number of frames.
        """
        if self.uids.concatenation is None:
            return 0
        try:
            return int(self.ConcatenationFrameOffsetNumber)
        except AttributeError:
            raise WsiDicomError(
                'Concatenated file missing concatenation frame offset'
                'number'
            )

    def _get_tile_organization_type(self) -> str:
        """Return tile organization type ('TILLED_SPARSE' or 'TILED_FULL').

        Returns
        ----------
        str
            Tile organization type.
        """
        tile_type = self._get_dicom_attribute('DimensionOrganizationType')
        if(tile_type == 'TILED_FULL'):
            return 'TILED_FULL'
        elif 'PerFrameFunctionalGroupsSequence' in self:
            return 'TILED_SPARSE'
        raise WsiDicomError("undetermined tile type")

    def _get_pixel_measure(self) -> Optional[Dataset]:
        """Return Pixel measure (sub)-dataset from dataset if found.

        Returns
        ----------
        Optional[Dataset]
            Found Pixel measure dataset.
        """
        shared_functional_group = self._get_dicom_attribute(
            'SharedFunctionalGroupsSequence'
        )
        if shared_functional_group is None:
            return None
        pixel_measure_sequence = self._get_dicom_attribute(
            'PixelMeasuresSequence',
            shared_functional_group[0]
        )
        if pixel_measure_sequence is None:
            return None
        return pixel_measure_sequence[0]

    @staticmethod
    def _get_spacings(
        pixel_measure: Optional[Dataset]
    ) -> Tuple[Optional[SizeMm], Optional[float]]:
        """Return Pixel and slice spacing from pixel measure dataset.

        Parameters
        ----------
        pixel_measure: Optional[Dataset]
            Pixel measure dataset.

        Returns
        ----------
        Tuple[Optional[SizeMm], Optional[float]]
            Pixel spacing and slice spacing, or None.
        """
        if pixel_measure is None:
            return None, None
        pixel_spacing_values = getattr(
            pixel_measure,
            'PixelSpacing',
            None
        )
        if pixel_spacing_values is not None:
            if any([spacing == 0 for spacing in pixel_spacing_values]):
                raise WsiDicomError("Pixel spacing is zero")
            pixel_spacing = SizeMm.from_tuple(pixel_spacing_values)
        else:
            pixel_spacing = None
        spacing_between_slices = getattr(
            pixel_measure,
            'SpacingBetweenSlices',
            None
        )
        return pixel_spacing, spacing_between_slices

    def _get_ext_depth_of_field(
        self
    ) -> Tuple[bool, Optional[int], Optional[float]]:
        """Return extended depth of field (enabled, number of focal planes,
        distance between focal planes) from dataset.

        Returns
        ----------
        Tuple[bool, Optional[int], Optional[float]]
            If extended depth of field is used, and if used number of focal
            planes and distance between focal planes.
        """
        if self._get_dicom_attribute('ExtendedDepthOfField') != 'YES':
            return False, None, None

        planes = self._get_dicom_attribute('NumberOfFocalPlanes')
        distance = self._get_dicom_attribute('DistanceBetweenFocalPlanes')
        if planes is None or distance is None:
            raise WsiDicomFileError(
                self.filepath,
                'Missing NumberOfFocalPlanes or DistanceBetweenFocalPlanes'
            )
        return True, planes, distance

    def _get_image_size(
        self
    ) -> Tuple[Size, Optional[SizeMm], Optional[float]]:
        """Return image size and physical image size from dataset.

        Returns
        ----------
        Tuple[Size, Optional[SizeMm], Optional[float]]:
            Pixel image size, physical image size and physical depth.
        """
        image_size = Size(
            self.TotalPixelMatrixColumns,
            self.TotalPixelMatrixRows
        )
        if image_size.width == 0 or image_size.height == 0:
            raise WsiDicomFileError(self.filepath, "Image size is zero")

        mm_width = self._get_dicom_attribute('ImagedVolumeWidth')
        mm_height = self._get_dicom_attribute('ImagedVolumeHeight')
        if mm_width is None or mm_height is None:
            mm_size = None
        else:
            mm_size = SizeMm(mm_width, mm_height)
        mm_depth = self._get_dicom_attribute('ImagedVolumeDepth')
        return image_size, mm_size, mm_depth

    def _get_slice_thickness(
        self,
        pixel_measure: Optional[Dataset]
    ) -> Optional[float]:
        """Return slice thickness spacing from pixel measure dataset.

        Parameters
        ----------
        pixel_measure: Optional[Dataset]
            Pixel measure dataset.

        Returns
        ----------
        Optional[float]
            Slice thickess or None if unkown.
        """
        try:
            return self._get_dicom_attribute(
                'SliceThickness',
                pixel_measure
            )
        except AttributeError:
            if self.mm_depth is not None:
                return self.mm_depth / self.number_of_focal_planes
        return None

    def _get_frame_sequence(self) -> DicomSequence:
        """Return per frame functional group sequene if present, otherwise
        shared functional group sequence.

        Returns
        ----------
        DicomSequence
            Per frame or shared functional group sequence.
        """
        if (
            'PerFrameFunctionalGroupsSequence' in self and
            (
                'PlanePositionSlideSequence' in
                self.PerFrameFunctionalGroupsSequence[0]
            )
        ):
            return self.PerFrameFunctionalGroupsSequence
        elif 'SharedFunctionalGroupsSequence' in self:
            return self.SharedFunctionalGroupsSequence
        return DicomSequence([])

    @staticmethod
    def _get_wsi_flavor(wsi_type: Tuple[str, str, str, str]) -> str:
        """Return wsi flavour from wsi type tuple.

        Returns
        ----------
        str
            Wsi flavour.
        """
        IMAGE_FLAVOR_INDEX_IN_IMAGE_TYPE = 2
        return wsi_type[IMAGE_FLAVOR_INDEX_IN_IMAGE_TYPE]

    @staticmethod
    def _get_spacing_between_slices_for_focal_planes(
        focal_planes: Sequence[float]
    ) -> float:
        """Return spacing between slices in mm for focal planes (defined in
        um). Spacing must be the same between all focal planes for TILED_FULL
        arrangement.

        Parameters
        ----------
        focal_planes: Sequence[float]
            Focal planes to calculate spacing for.

        Returns
        ----------
        float
            Spacing between focal planes.

        """
        if len(focal_planes) == 1:
            return 0.0
        spacing: Optional[float] = None
        sorted_focal_planes = sorted(focal_planes)
        for index in range(len(sorted_focal_planes)-1):
            this_spacing = (
                sorted_focal_planes[index + 1]
                - sorted_focal_planes[index]
            )
            if spacing is None:
                spacing = this_spacing
            elif (
                abs(spacing - this_spacing)
                > settings.focal_plane_distance_threshold
            ):
                raise NotImplementedError(
                    "Image data has non-equal spacing between slices: "
                    f"{spacing, this_spacing}, difference threshold: "
                    f"{settings.focal_plane_distance_threshold}, "
                    "not possible to encode as TILED_FULL"
                )
        if spacing is None:
            raise ValueError("Could not calculate spacings.")
        return spacing / 1000.0
