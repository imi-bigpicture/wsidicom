#    Copyright 2021 SECTRA AB
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

import io
from optparse import Option
import threading
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, auto
from pathlib import Path
from struct import pack, unpack
from typing import (Any, BinaryIO, Dict, Generator, List, Optional,
                    OrderedDict, Sequence, Set, Tuple, Union, cast)

import numpy as np
from PIL import Image
from pydicom import FileDataset
from pydicom.dataset import Dataset, FileMetaDataset, validate_file_meta
from pydicom.encaps import itemize_frame
from pydicom.filebase import DicomFile, DicomFileLike
from pydicom.filereader import read_file_meta_info, read_partial
from pydicom.filewriter import write_dataset, write_file_meta_info
from pydicom.misc import is_dicom
from pydicom.pixel_data_handlers import pillow_handler
from pydicom.sequence import Sequence as DicomSequence
from pydicom.tag import BaseTag, ItemTag, SequenceDelimiterTag, Tag
from pydicom.uid import JPEG2000, UID, JPEG2000Lossless, JPEGBaseline8Bit
from pydicom.valuerep import DSfloat

from wsidicom.config import settings
from wsidicom.errors import (WsiDicomError, WsiDicomFileError,
                             WsiDicomNotFoundError, WsiDicomOutOfBoundsError,
                             WsiDicomRequirementError,
                             WsiDicomStrictRequirementError,
                             WsiDicomUidDuplicateError)
from wsidicom.geometry import Point, Region, Size, SizeMm
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
        image_types: Sequence[str] = None,
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


class WsiDataset(Dataset):
    """Extend pydicom.dataset.Dataset (containing WSI metadata) with simple
    parsers for attributes specific for WSI. Use snake case to avoid name
    collision with dicom fields (that are handled by pydicom.dataset.Dataset).
    """
    def __init__(
        self,
        dataset: Dataset
    ):
        """A WsiDataset wrapping a pydicom Dataset.

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
        datasets: Sequence['WsiDataset'],
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

    def matches_instance(self, other_dataset: 'WsiDataset') -> bool:
        """Return true if other file is of the same instance as self.

        Parameters
        ----------
        other_dataset: 'WsiDataset'
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
        image_data: OrderedDict[Tuple[str, float], 'ImageData'],
        scale: int = 1
    ) -> 'WsiDataset':
        """Return copy of dataset with properties set to reflect a tiled full
        arrangement of the listed image data. Optionally set properties to
        reflect scaled data.

        Parameters
        ----------
        image_data: OrderedDict[Tuple[str, float], ImageData]
            List of image data that should be encoded into dataset. Each
            element is a tuple of (optical path, focal plane) and ImageData.
        scale: int = 1
            Optionally scale data.

        Returns
        ----------
        WsiDataset
            Copy of dataset set as tiled full.

        """
        focal_planes, optical_paths, tiled_size = (
            self._get_frame_information(image_data)
        )

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
        distances: Set[float] = set()
        sorted_focal_planes = sorted(focal_planes)
        for index in range(len(sorted_focal_planes)-1):
            distance = (
                sorted_focal_planes[index + 1]
                - sorted_focal_planes[index]
            )
            distances.add(distance)
        if len(distances) != 1:
            raise NotImplementedError(
                "Image data has non-equal spacing between slices "
                f"{distances}, not possible to encode as TILED_FULL"
            )
        return distances.pop() / 1000.0

    @staticmethod
    def _get_frame_information(
        data: OrderedDict[Tuple[str, float], 'ImageData']
    ) -> Tuple[List[float], List[str], Size]:
        """Return optical_paths, focal planes, and tiled size.
        """
        focal_planes_by_optical_path: Dict[str, List[float]] = (
            defaultdict(list)
        )
        all_focal_planes: Set[float] = set()
        tiled_sizes: Set[Size] = set()
        for (optical_path, focal_plane), image_data in data.items():
            focal_planes_by_optical_path[optical_path].append(focal_plane)
            all_focal_planes.add(focal_plane)
            tiled_sizes.add(image_data.tiled_size)

        focal_planes_sparse_by_optical_path = any(
            optical_path_focal_planes != list(all_focal_planes)
            for optical_path_focal_planes
            in focal_planes_by_optical_path.values()
        )

        if focal_planes_sparse_by_optical_path:
            raise ValueError(
                'Each optical path must have the same focal planes.'
            )

        if len(tiled_sizes) != 1:
            raise ValueError('Expected only one tiled size')
        tiled_size = list(tiled_sizes)[0]
        return (
            sorted(list(all_focal_planes)),
            sorted(list(focal_planes_by_optical_path.keys())),
            tiled_size
        )


class WsiDicomFileBase():
    def __init__(self, filepath: Path, mode: str):
        """Base class for reading or writing DICOM WSI file.

        Parameters
        ----------
        filepath: Path
            Filepath to file to read or write.
        mode: str
            Mode for opening file.
        """
        self._filepath = filepath
        self._fp = DicomFile(filepath, mode=mode)
        self.__enter__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.filepath})"

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: Optional[int] = None
    ) -> str:
        return f"File with path: {self.filepath}"

    @property
    def filepath(self) -> Path:
        """Return filepath"""
        return self._filepath

    def _read_tag_length(self, with_vr: bool = True) -> int:
        if (not self._fp.is_implicit_VR) and with_vr:
            VR = self._fp.read_UL()
        return self._fp.read_UL()

    def _check_tag_and_length(
        self,
        tag: BaseTag,
        length: int,
        with_vr: bool = True
    ) -> None:
        """Check if tag at position is expected tag with expected length.

        Parameters
        ----------
        tag: BaseTag
            Expected tag.
        length: int
            Expected length.

        """
        read_tag = self._fp.read_tag()
        if tag != read_tag:
            raise ValueError(f"Found tag {read_tag} expected {tag}")
        read_length = self._read_tag_length(with_vr)
        if length != read_length:
            raise ValueError(f"Found length {read_length} expected {length}")

    def _read_sequence_delimeter(self):
        """Check if last read tag was a sequence delimter.
        Raises WsiDicomFileError otherwise.
        """
        TAG_BYTES = 4
        self._fp.seek(-TAG_BYTES, 1)
        if(self._fp.read_tag() != SequenceDelimiterTag):
            raise WsiDicomFileError(self.filepath, 'No sequence delimeter tag')

    def close(self) -> None:
        """Close the file."""
        self._fp.close()


class WsiDicomFile(WsiDicomFileBase):
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
        self._lock = threading.Lock()

        if not is_dicom(filepath):
            raise WsiDicomFileError(filepath, "is not a DICOM file")

        file_meta = read_file_meta_info(filepath)
        self._transfer_syntax_uid = UID(file_meta.TransferSyntaxUID)

        super().__init__(filepath, mode='rb')
        self._fp.is_little_endian = self._transfer_syntax_uid.is_little_endian
        self._fp.is_implicit_VR = self._transfer_syntax_uid.is_implicit_VR
        pixel_data_tags = {
            Tag('PixelData'),
            Tag('ExtendedOffsetTable')
        }

        def _stop_at(
            tag: BaseTag,
            VR: Optional[str],
            length: int
        ) -> bool:
            return tag in pixel_data_tags
        dataset = read_partial(
            cast(BinaryIO, self._fp),
            _stop_at,
            defer_size=None,
            force=False,
            specific_tags=None,
        )
        self._pixel_data_position = self._fp.tell()

        self._wsi_type = WsiDataset.is_supported_wsi_dicom(
            dataset,
            self.transfer_syntax
        )
        if self._wsi_type is not None:
            self._dataset = WsiDataset(dataset)
            instance_uid = self.dataset.uids.instance
            concatenation_uid = self.dataset.uids.concatenation
            slide_uids = self.dataset.uids.slide
            self._uids = FileUids(instance_uid, concatenation_uid, slide_uids)

            self._frame_offset = self.dataset.frame_offset
            self._frame_count = self.dataset.frame_count
            self._frame_positions = self._parse_pixel_data()
        else:
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
    def wsi_type(self) -> Optional[str]:
        return self._wsi_type

    @property
    def uids(self) -> FileUids:
        """Return uids"""
        return self._uids

    @property
    def transfer_syntax(self) -> UID:
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

    def get_filepointer(
        self,
        frame_index: int
    ) -> Tuple[DicomFileLike, int, int]:
        """Return file pointer, frame position, and frame lenght for frame
        number.

        Parameters
        ----------
        frame_index: int
            Frame, including concatenation offset, to get.

        Returns
        ----------
        Tuple[DicomFileLike, int, int]:
            File pointer, frame offset and frame lenght in number of bytes
        """
        frame_index -= self.frame_offset
        frame_position, frame_length = self.frame_positions[frame_index]
        return self._fp, frame_position, frame_length

    def _read_bot(self) -> Optional[bytes]:
        """Read basic table offset (BOT). Returns None if BOT is empty

        Returns
        ----------
        Optional[bytes]
            BOT in bytes.
        """
        BOT_BYTES = 4
        if self._fp.read_tag() != ItemTag:
            raise WsiDicomFileError(
                self.filepath,
                "Basic offset table did not start with an ItemTag"
            )
        bot_length = self._fp.read_UL()
        if bot_length == 0:
            return None
        elif bot_length % BOT_BYTES:
            raise WsiDicomFileError(
                self.filepath,
                f"Basic offset table should be a multiple of {BOT_BYTES} bytes"
            )
        # Read the BOT into bytes
        bot = self._fp.read(bot_length)
        return bot

    def _read_eot(self) -> bytes:
        """Read extended table offset (EOT) and EOT lengths.

        Returns
        ----------
        bytes
            EOT in bytes.
        """
        EOT_BYTES = 8

        eot_length = self._read_tag_length()
        if eot_length == 0:
            raise WsiDicomFileError(
                self.filepath,
                "Expected Extended offset table present but empty"
            )
        elif eot_length % EOT_BYTES:
            raise WsiDicomFileError(
                self.filepath,
                "Extended offset table should be a multiple of "
                f"{EOT_BYTES} bytes"
            )
        # Read the EOT into bytes
        eot = self._fp.read(eot_length)
        # Read EOT lengths tag
        tag = self._fp.read_tag()
        if tag != Tag('ExtendedOffsetTableLengths'):
            raise WsiDicomFileError(
                self.filepath,
                "Expected Extended offset table lengths tag after reading "
                f"Extended offset table, found {tag}"
            )
        length = self._read_tag_length()
        # Jump over EOT lengths for now
        self._fp.seek(length, 1)
        return eot

    def _parse_table(
        self,
        table: bytes,
        table_type: str,
        pixels_start: int
    ) -> List[Tuple[int, int]]:
        """Parse table with offsets (BOT or EOT).

        Parameters
        ----------
        table: bytes
            BOT or EOT as bytes
        table_type: str
            Type of table, 'bot' or 'eot'.
        pixels_start: int
            Position of pixel start.

        Returns
        ----------
        List[Tuple[int, int]]
            A list with frame positions and frame lengths.
        """
        if self._fp.is_little_endian:
            mode = '<'
        else:
            mode = '>'
        if table_type == 'bot':
            bytes_per_item = 4
            mode += 'L'
        elif table_type == 'eot':
            bytes_per_item = 8
            mode = 'Q'
        else:
            raise ValueError("table type should be 'bot' or 'eot'")
        table_length = len(table)
        TAG_BYTES = 4
        LENGHT_BYTES = 4
        positions: List[Tuple[int, int]] = []
        # Read through table to get offset and length for all but last item
        this_offset: int = unpack(mode, table[0:bytes_per_item])[0]
        for index in range(bytes_per_item, table_length, bytes_per_item):
            next_offset = unpack(mode, table[index:index+bytes_per_item])[0]
            offset = this_offset + TAG_BYTES + LENGHT_BYTES
            length = next_offset - offset
            if length == 0 or length % 2:
                raise WsiDicomFileError(self.filepath, 'Invalid frame length')
            positions.append((pixels_start+offset, length))
            this_offset = next_offset

        # Go to last frame in pixel data and read the length of the frame
        self._fp.seek(pixels_start+this_offset)
        if self._fp.read_tag() != ItemTag:
            raise WsiDicomFileError(
                self.filepath,
                "Excepcted ItemTag in PixelData"
            )
        length: int = self._fp.read_UL()
        if length == 0 or length % 2:
            raise WsiDicomFileError(self.filepath, 'Invalid frame length')
        offset = this_offset+TAG_BYTES+LENGHT_BYTES
        positions.append((pixels_start+offset, length))

        return positions

    def _read_positions_from_pixeldata(self) -> List[Tuple[int, int]]:
        """Get frame positions and length from sequence of frames that ends
        with a tag not equal to ItemTag. fp needs to be positioned after the
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
        while(self._fp.read_tag() == ItemTag):
            # Read item length
            length: int = self._fp.read_UL()
            if length == 0 or length % 2:
                raise WsiDicomFileError(self.filepath, 'Invalid frame length')
            positions.append((frame_position+TAG_BYTES+LENGHT_BYTES, length))
            # Jump to end of frame
            self._fp.seek(length, 1)
            frame_position = self._fp.tell()

        self._read_sequence_delimiter()
        return positions

    def _read_sequence_delimiter(self):
        """Check if last read tag was a sequence delimter.
        Raises WsiDicomFileError otherwise.
        """
        TAG_BYTES = 4
        self._fp.seek(-TAG_BYTES, 1)
        if(self._fp.read_tag() != SequenceDelimiterTag):
            raise WsiDicomFileError(self.filepath, 'No sequence delimeter tag')

    def read_frame(self, frame_index: int) -> bytes:
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
        Note that fp needs to be positionend at Extended offset table (EOT)
        or Pixel data. An EOT can be present before the pixel data, and must
        then not be empty. A BOT most always be the first item in the Pixel
        data, but can be empty (zero length). If EOT is used BOT must be empty.

        Returns
        ----------
        List[Tuple[int, int]]
            List of frame positions and lenghts
        """

        table = None
        table_type = 'bot'
        pixel_data_or_eot_tag = self._fp.read_tag()
        if pixel_data_or_eot_tag == Tag('ExtendedOffsetTable'):
            table_type = 'eot'
            table = self._read_eot()
            pixel_data_tag = self._fp.read_tag()
        else:
            pixel_data_tag = pixel_data_or_eot_tag

        if pixel_data_tag != Tag('PixelData'):
            WsiDicomFileError(
                self.filepath,
                "Expected PixelData tag"
            )
        length = self._read_tag_length()
        if length != 0xFFFFFFFF:
            raise WsiDicomFileError(
                self.filepath,
                "Expected undefined length when reading Pixel data"
            )
        bot = self._read_bot()

        if bot is not None:
            if table is not None:
                raise WsiDicomFileError(
                    self.filepath,
                    "Both BOT and EOT present"
                )
            table = bot

        frame_positions = []
        if table is None:
            frame_positions = self._read_positions_from_pixeldata()
        else:
            frame_positions = self._parse_table(
                table,
                table_type,
                self._fp.tell()
            )

        if(self.frame_count != len(frame_positions)):
            raise WsiDicomFileError(
                self.filepath,
                (
                    f"Frame count {self.frame_count} "
                    f"!= Fragments {len(frame_positions)}."
                    " Fragmented frames are not supported"
                )
            )

        return frame_positions

    @staticmethod
    def filter_files(
        files: Sequence['WsiDicomFile'],
        series_uids: SlideUids,
        series_tile_size: Optional[Size] = None
    ) -> List['WsiDicomFile']:
        """Filter list of wsi dicom files to only include matching uids and
        tile size if defined.

        Parameters
        ----------
        files: Sequence['WsiDicomFile']
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
            if file.dataset.matches_series(
                series_uids,
                series_tile_size
            ):
                valid_files.append(file)
            else:
                warnings.warn(
                    f'{file.filepath} with uids {file.uids.slide} '
                    f'did not match series with {series_uids} '
                    f'and tile size {series_tile_size}'
                )
                file.close()

        return valid_files

    @classmethod
    def group_files(
        cls,
        files: Sequence['WsiDicomFile']
    ) -> Dict[str, List['WsiDicomFile']]:
        """Return files grouped by instance identifier (instances).

        Parameters
        ----------
        files: Sequence[WsiDicomFile]
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
    _default_z: Optional[float] = None
    _blank_tile: Optional[Image.Image] = None
    _encoded_blank_tile: Optional[bytes] = None

    @property
    @abstractmethod
    def files(self) -> List[Path]:
        raise NotImplementedError()

    @property
    @abstractmethod
    def transfer_syntax(self) -> UID:
        """Should return the uid of the transfer syntax of the image."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def image_size(self) -> Size:
        """Should return the pixel size of the image."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def tile_size(self) -> Size:
        """Should return the pixel tile size of the image, or pixel size of
        the image if not tiled."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def pixel_spacing(self) -> SizeMm:
        """Should return the size of the pixels in mm/pixel."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def samples_per_pixel(self) -> int:
        """Should return number of samples per pixel (e.g. 3 for RGB."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def photometric_interpretation(self) -> str:
        """Should return the photophotometric interpretation of the image
        data."""
        raise NotImplementedError()

    @abstractmethod
    def _get_decoded_tile(
        self,
        tile_point: Point,
        z: float,
        path: str
    ) -> Image.Image:
        """Should return Image for tile defined by tile (x, y), z,
        and optical path."""
        raise NotImplementedError()

    @abstractmethod
    def _get_encoded_tile(
        self,
        tile: Point,
        z: float,
        path: str
    ) -> bytes:
        """Should return image bytes for tile defined by tile (x, y), z,
        and optical path."""
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Should close any open files."""
        raise NotImplementedError()

    @property
    def tiled_size(self) -> Size:
        """The size of the image when divided into tiles, e.g. number of
        columns and rows of tiles. Equals (1, 1) if image is not tiled."""
        return self.image_size.ceil_div(self.tile_size)

    @property
    def image_region(self) -> Region:
        return Region(Point(0, 0), self.image_size)

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
            return 'L'
        elif(self.samples_per_pixel == 3):
            return 'RGB'
        raise NotImplementedError

    @property
    def blank_color(self) -> Tuple[int, int, int]:
        """Return RGB background color."""
        return self._get_blank_color(self.photometric_interpretation)

    def pretty_str(
        self,
        indent: int = 0,
        depth: Optional[int] = None
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

    def get_decoded_tiles(
        self,
        tiles: Sequence[Point],
        z: float,
        path: str
    ) -> List[Image.Image]:
        """Return tiles for tile defined by tile (x, y), z, and optical
        path.

        Parameters
        ----------
        tiles: Sequence[Point]
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        ----------
        List[Image.Image]
            Tiles as Images.
        """
        return [
            self._get_decoded_tile(tile, z, path) for tile in tiles
        ]

    def get_encoded_tiles(
        self,
        tiles: Sequence[Point],
        z: float,
        path: str
    ) -> List[bytes]:
        """Return tiles for tile defined by tile (x, y), z, and optical
        path.

        Parameters
        ----------
        tiles: Sequence[Point]
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        ----------
        List[bytes]
            Tiles in bytes.
        """
        return [
            self._get_encoded_tile(tile, z, path) for tile in tiles
        ]

    def get_scaled_tile(
        self,
        scaled_tile_point: Point,
        z: float,
        path: str,
        scale: int
    ) -> Image.Image:
        """Return scaled tile defined by tile (x, y), z, optical
        path and scale.

        Parameters
        ----------
        scaled_tile_point: Point,
            Scaled position of tile to get.
        z: float
            Z coordinate.
        path: str
            Optical path.
        Scale: int
            Scale to use for downscaling.

        Returns
        ----------
        Image.Image
            Scaled tiled as Image.
        """
        image = Image.new(
            mode=self.image_mode,  # type: ignore
            size=(self.tile_size * scale).to_tuple(),
            color=self.blank_color[:self.samples_per_pixel]
        )
        # Get decoded tiles for the region covering the scaled tile
        # in the image data
        tile_points = Region(scaled_tile_point*scale, Size(1, 1)*scale)
        origin = tile_points.start
        for tile_point in tile_points.iterate_all():
            if (
                (tile_point.x < self.tiled_size.width) and
                (tile_point.y < self.tiled_size.height)
            ):
                tile = self._get_decoded_tile(tile_point, z, path)
                image_coordinate = (tile_point - origin) * self.tile_size
                image.paste(tile, image_coordinate.to_tuple())

        return image.resize(self.tile_size.to_tuple(), resample=Image.BILINEAR)

    def get_scaled_encoded_tile(
        self,
        scaled_tile_point: Point,
        z: float,
        path: str,
        scale: int,
        image_format: str,
        image_options: Dict[str, Any]
    ) -> bytes:
        """Return scaled encoded tile defined by tile (x, y), z, optical
        path and scale.

        Parameters
        ----------
        scaled_tile_point: Point,
            Scaled position of tile to get.
        z: float
            Z coordinate.
        path: str
            Optical path.
        Scale: int
            Scale to use for downscaling.
        image_format: str
            Image format, e.g. 'JPEG', for encoding.
        image_options: Dict[str, Any].
            Dictionary of options for encoding.

        Returns
        ----------
        bytes
            Scaled tile as bytes.
        """
        image = self.get_scaled_tile(scaled_tile_point, z, path, scale)
        with io.BytesIO() as buffer:
            image.save(
                buffer,
                format=image_format,
                **image_options
            )
            return buffer.getvalue()

    def get_scaled_encoded_tiles(
        self,
        scaled_tile_points: Sequence[Point],
        z: float,
        path: str,
        scale: int,
        image_format: str,
        image_options: Dict[str, Any]
    ) -> List[bytes]:
        """Return scaled encoded tiles defined by tile (x, y) positions, z,
        optical path and scale.

        Parameters
        ----------
        scaled_tile_points: Sequence[Point],
            Scaled position of tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.
        Scale: int
            Scale to use for downscaling.
        image_format: str
            Image format, e.g. 'JPEG', for encoding.
        image_options: Dict[str, Any].
            Dictionary of options for encoding.

        Returns
        ----------
        List[bytes]
            Scaled tiles as bytes.
        """
        return [
            self.get_scaled_encoded_tile(
                scaled_tile_point,
                z,
                path,
                scale,
                image_format,
                image_options
            )
            for scaled_tile_point in scaled_tile_points
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
        transfer_syntax: UID
    ) -> Tuple[str, Dict[str, Any]]:
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
        if(transfer_syntax == JPEGBaseline8Bit):
            image_format = 'jpeg'
            image_options = {'quality': 95}
        elif(transfer_syntax == JPEG2000):
            image_format = 'jpeg2000'
            image_options = {"irreversible": True}
        elif(transfer_syntax == JPEG2000Lossless):
            image_format = 'jpeg2000'
            image_options = {"irreversible": False}
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
        Tuple[int, int, int]
            RGB color,

        """
        BLACK = 0
        WHITE = 255
        if(photometric_interpretation == "MONOCHROME2"):
            return (BLACK, BLACK, BLACK)  # Monocrhome2 is black
        return (WHITE, WHITE, WHITE)

    def _create_blank_tile(self) -> Image.Image:
        """Create blank tile for instance.

        Returns
        ----------
        Image.Image
            Blank tile image
        """
        return Image.new(
            mode=self.image_mode,  # type: ignore
            size=self.tile_size.to_tuple(),
            color=self.blank_color[:self.samples_per_pixel]
        )

    def stitch_tiles(
        self,
        region: Region,
        path: str,
        z: float
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

        image = Image.new(
            mode=self.image_mode,  # type: ignore
            size=region.size.to_tuple()
        )
        stitching_tiles = self.get_tile_range(region, z, path)

        write_index = Point(x=0, y=0)
        tile = stitching_tiles.position
        for tile in stitching_tiles.iterate_all(include_end=True):
            tile_image = self.get_tile(tile, z, path, region)
            image.paste(tile_image, write_index.to_tuple())
            write_index = self._write_indexer(
                write_index,
                Size.from_tuple(tile_image.size),
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
        start = pixel_region.start // self.tile_size
        end = pixel_region.end.ceil_div(self.tile_size) - 1
        tile_region = Region.from_points(start, end)
        if not self.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBoundsError(
                f"Tile region {tile_region}",
                f"tiled size {self.tiled_size}"
            )
        return tile_region

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

    def get_tile(
        self,
        tile: Point,
        z: float,
        path: str,
        crop: Union[bool, Region] = True
    ) -> Image.Image:
        """Get tile image at tile coordinate x, y. If frame is inside tile
        geometry but no tile exists in frame data (sparse) returns blank image.
        Optional crop tile to crop_region.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate.
        z: float
            Z coordinate.
        path: str
            Optical path.
        crop: Union[bool, Region] = True
            If to crop tile to image size (True, default) or to region.

        Returns
        ----------
        Image.Image
            Tile image.
        """
        image = self._get_decoded_tile(tile, z, path)
        if crop is False:
            return image

        if isinstance(crop, bool):
            crop = self.image_region
        tile_crop = crop.inside_crop(tile, self.tile_size)
        if tile_crop.size == self.tile_size:
            return image

        return image.crop(box=tile_crop.box)

    def get_encoded_tile(
        self,
        tile: Point,
        z: float,
        path: str,
        crop: Union[bool, Region] = True
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
        crop: Union[bool, Region] = True
            If to crop tile to image size (True, default) or to region.

        Returns
        ----------
        bytes
            Tile image as bytes.
        """
        tile_frame = self._get_encoded_tile(tile, z, path)
        if crop is False:
            return tile_frame

        if isinstance(crop, bool):
            crop = self.image_region
        # Check if tile is an edge tile that should be croped
        cropped_tile_region = crop.inside_crop(tile, self.tile_size)
        if cropped_tile_region.size != self.tile_size:
            image = Image.open(io.BytesIO(tile_frame))
            image.crop(box=cropped_tile_region.box_from_origin)
            tile_frame = self.encode(image)
        return tile_frame


class WsiDicomImageData(ImageData):
    """Represents image data read from dicom file(s). Image data can
    be sparsly or fully tiled and/or concatenated."""
    def __init__(
        self,
        files: Union[WsiDicomFile, Sequence[WsiDicomFile]]
    ) -> None:
        """Create WsiDicomImageData from frame data in files.

        Parameters
        ----------
        files: Union[WsiDicomFile, Sequence[WsiDicomFile]]
            Single or list of WsiDicomFiles containing frame data.
        """
        if not isinstance(files, Sequence):
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
        self._default_z: Optional[float] = None
        self._photometric_interpretation = (
            datasets[0].photometric_interpretation
        )
        self._samples_per_pixel = datasets[0].samples_per_pixel

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._files.values()})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of files {self._files.values()}"

    @property
    def files(self) -> List[Path]:
        return [file.filepath for file in self._files.values()]

    @property
    def transfer_syntax(self) -> UID:
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
    def pixel_spacing(self) -> Optional[SizeMm]:
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

    def _get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        frame_index = self._get_frame_index(tile, z, path)
        if frame_index == -1:
            return self.blank_encoded_tile
        return self._get_tile_frame(frame_index)

    def _get_decoded_tile(
        self,
        tile_point: Point,
        z: float,
        path: str
    ) -> Image.Image:
        frame_index = self._get_frame_index(tile_point, z, path)
        if frame_index == -1:
            return self.blank_tile
        frame = self._get_tile_frame(frame_index)
        return Image.open(io.BytesIO(frame))

    def get_filepointer(
        self,
        tile: Point,
        z: float,
        path: str
    ) -> Optional[Tuple[DicomFileLike, int, int]]:
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
        tile_frame = file.read_frame(frame_index)
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

    def close(self) -> None:
        for file in self._files.values():
            file.close()


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
        self._shape = tiled_size
        self.plane = np.full(tiled_size.to_tuple(), -1, dtype=np.dtype(int))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._shape})"

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
        depth: Optional[int] = None
    ) -> str:
        return "Sparse tile plane"


class TileIndex(metaclass=ABCMeta):
    """Index for mapping tile position to frame number. Is subclassed into
    FullTileIndex and SparseTileIndex."""
    def __init__(
        self,
        datasets: Sequence[WsiDataset]
    ):
        """Create tile index for frames in datasets. Requires equal tile
        size for all tile planes.

        Parameters
        ----------
        datasets: Sequence[WsiDataset]
            List of datasets containing tiled image data.

        """
        base_dataset = datasets[0]
        self._image_size = base_dataset.image_size
        self._tile_size = base_dataset.tile_size
        self._frame_count = self._read_frame_count_from_datasets(datasets)
        self._optical_paths = self._read_optical_paths_from_datasets(datasets)
        self._tiled_size = self.image_size.ceil_div(self.tile_size)

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
        depth: Optional[int] = None
    ) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Abstract method for getting the frame index for a tile"""
        raise NotImplementedError

    @staticmethod
    def _read_frame_count_from_datasets(
        datasets: Sequence[WsiDataset]
    ) -> int:
        """Return total frame count from files.

        Parameters
        ----------
        datasets: Sequence[WsiDataset]
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

    @classmethod
    def _read_optical_paths_from_datasets(
        cls,
        datasets: Sequence[WsiDataset]
    ) -> List[str]:
        """Return list of optical path identifiers from files.

        Parameters
        ----------
        datasets: Sequence[WsiDataset]
           List of datasets.

        Returns
        ----------
        List[str]
            Optical identifiers.

        """
        paths: Set[str] = set()
        for dataset in datasets:
            paths.update(cls._get_path_identifers(
                dataset.optical_path_sequence
            ))
        if len(paths) == 0:
            return ['0']
        return list(paths)

    @staticmethod
    def _get_path_identifers(
        optical_path_sequence: Optional[DicomSequence]
    ) -> List[str]:
        """Parse optical path sequence and return list of optical path
        identifiers

        Parameters
        ----------
        optical_path_sequence: DicomSequence
            Optical path sequence.

        Returns
        ----------
        List[str]
            List of optical path identifiers.
        """
        if optical_path_sequence is None:
            return ['0']
        return list({
            str(optical_ds.OpticalPathIdentifier)
            for optical_ds in optical_path_sequence
        })


class FullTileIndex(TileIndex):
    """Index for mapping tile position to frame number for datasets containing
    full tiles. Pixel data tiles are ordered by colum, row, z and path, thus
    the frame index for a tile can directly be calculated."""
    def __init__(
        self,
        datasets: Sequence[WsiDataset]
    ):
        """Create full tile index for frames in datasets. Requires equal tile
        size for all tile planes.

        Parameters
        ----------
        datasets: Sequence[WsiDataset]
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
        depth: Optional[int] = None
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
        datasets: Sequence[WsiDataset]
    ) -> List[float]:
        """Return list of focal planes in datasets. Values in Pixel Measures
        Sequene are in mm.

        Parameters
        ----------
        datasets: Sequence[WsiDataset]
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
            if slice_spacing is None:
                if number_of_focal_planes == 1:
                    slice_spacing = 0.0
                else:
                    raise ValueError(
                        "Slice spacing must be known if multple focal planes."
                    )
            elif slice_spacing == 0 and number_of_focal_planes != 1:
                raise ValueError(
                    "Slice spacing must be non-zero if multiple focal planes."
                )

            for plane in range(number_of_focal_planes):
                z = round(plane * slice_spacing * MM_TO_MICRON, DECIMALS)
                focal_planes.add(z)
        return sorted(list(focal_planes))

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
            raise WsiDicomNotFoundError(f"Optical path {path}", str(self))

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
            raise WsiDicomNotFoundError(f"Z {z} in instance", str(self))


class SparseTileIndex(TileIndex):
    """Index for mapping tile position to frame number for datasets containing
    sparse tiles. Frame indices are retrieved from tile position, z, and path
    by finding the corresponding matching SparseTilePlane (z and path) and
    returning the frame index at tile position. If the tile is missing (due to
    the sparseness), -1 is returned."""
    def __init__(
        self,
        datasets: Sequence[WsiDataset]
    ):
        """Create sparse tile index for frames in datasets. Requires equal tile
        size for all tile planes. Pixel data tiles are identified by the Per
        Frame Functional Groups Sequence that contains tile colum, row, z,
        path, and frame index. These are stored in a SparseTilePlane
        (one plane for every combination of z and path).

        Parameters
        ----------
        datasets: Sequence[WsiDataset]
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
        depth: Optional[int] = None
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
                f"Plane with z {z}, path {path}", str(self)
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
        for z, _ in self._planes.keys():
            focal_planes.add(z)
        return sorted(list(focal_planes))

    def _read_planes_from_datasets(
        self,
        datasets: Sequence[WsiDataset]
    ) -> Dict[Tuple[float, str], SparseTilePlane]:
        """Return SparseTilePlane from planes in datasets.

        Parameters
        ----------
        datasets: Sequence[WsiDataset]
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
            frame: Dataset
    ) -> Tuple[Point, float]:
        """Return frame coordinate (Point(x, y) and float z) of the frame.
        In the Plane Position Slide Sequence x and y are defined in mm and z in
        um.

        Parameters
        ----------
        frame: Dataset
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
        z_offset = getattr(position, 'ZOffsetInSlideCoordinateSystem', 0.0)
        z = round(float(z_offset), DECIMALS)
        tile = Point(x=x, y=y) // self.tile_size
        return tile, z


class WsiDicomFileWriter(WsiDicomFileBase):
    def __init__(self, filepath: Path) -> None:
        """Return a dicom filepointer.

        Parameters
        ----------
        filepath: Path
            Path to filepointer.

        """
        super().__init__(filepath, mode='w+b')
        self._fp.is_little_endian = True
        self._fp.is_implicit_VR = False

    def write(
        self,
        uid: UID,
        transfer_syntax: UID,
        dataset: Dataset,
        data: OrderedDict[Tuple[str, float], ImageData],
        workers: int,
        chunk_size: int,
        offset_table: Optional[str],
        scale: int = 1
    ) -> None:
        """Writes data to file.

        Parameters
        ----------
        uid: UID
            Instance UID for file.
        transfer_syntax: UID.
            Transfer syntax for file
        dataset: Dataset
            Dataset to write (exluding pixel data).
        data: OrderedDict[Tuple[str, float], ImageData],
            Pixel data to write.
        workers: int
            Number of workers to use for writing pixel data.
        chunk_size: int
            Number of frames to give each worker.
        offset_table: Optional[str] = 'bot'
            Offset table to use, 'bot' basic offset table, 'eot' extended
            offset table, None - no offset table.
        scale: int = 1
            Scale factor.

        """
        self._write_preamble()
        self._write_file_meta(uid, transfer_syntax)
        dataset.SOPInstanceUID = uid
        self._write_base(dataset)
        table_start, pixels_start = self._write_pixel_data_start(
            dataset.NumberOfFrames,
            offset_table
        )
        frame_positions: List[int] = []
        for (path, z), image_data in data.items():
            frame_positions += self._write_pixel_data(
                image_data,
                z,
                path,
                workers,
                chunk_size,
                scale
            )
        pixels_end = self._fp.tell()
        self._write_pixel_data_end()

        if offset_table is not None:
            if table_start is None:
                raise ValueError('Table start should not be None')
            elif offset_table == 'eot':
                self._write_eot(
                    table_start,
                    pixels_start,
                    frame_positions,
                    pixels_end
                )
            elif offset_table == 'bot':
                self._write_bot(table_start, pixels_start, frame_positions)
        self.close()

    def _write_preamble(self) -> None:
        """Writes file preamble to file."""
        preamble = b'\x00' * 128
        self._fp.write(preamble)
        self._fp.write(b'DICM')

    def _write_file_meta(self, uid: UID, transfer_syntax: UID) -> None:
        """Writes file meta dataset to file.

        Parameters
        ----------
        uid: UID
            SOP instance uid to include in file.
        transfer_syntax: UID
            Transfer syntax used in file.
        """
        meta_ds = FileMetaDataset()
        meta_ds.TransferSyntaxUID = transfer_syntax
        meta_ds.MediaStorageSOPInstanceUID = uid
        meta_ds.MediaStorageSOPClassUID = WSI_SOP_CLASS_UID
        validate_file_meta(meta_ds)
        write_file_meta_info(self._fp, meta_ds)

    def _write_base(self, dataset: Dataset) -> None:
        """Writes base dataset to file.

        Parameters
        ----------
        dataset: Dataset

        """
        now = datetime.now()
        dataset.ContentDate = datetime.date(now).strftime('%Y%m%d')
        dataset.ContentTime = datetime.time(now).strftime('%H%M%S.%f')
        write_dataset(self._fp, dataset)

    def _write_tag(
        self,
        tag: str,
        value_representation: str,
        length: Optional[int] = None
    ):
        """Write tag, tag VR and length.

        Parameters
        ----------
        tag: str
            Name of tag to write.
        value_representation: str.
            Value representation (VR) of tag to write.
        length: Optional[int] = None
            Length of data after tag. 'Unspecified' (0xFFFFFFFF) if None.

        """
        self._fp.write_tag(Tag(tag))
        self._fp.write(bytes(value_representation, "iso8859"))
        self._fp.write_US(0)
        if length is not None:
            self._fp.write_UL(length)
        else:
            self._fp.write_UL(0xFFFFFFFF)

    def _reserve_eot(
        self,
        number_of_frames: int
    ) -> int:
        """Reserve space in file for extended offset table.

        Parameters
        ----------
        number_of_frames: int
            Number of frames to reserve space for.

        """
        table_start = self._fp.tell()
        BYTES_PER_ITEM = 8
        eot_length = BYTES_PER_ITEM * number_of_frames
        self._write_tag('ExtendedOffsetTable', 'OV', eot_length)
        for index in range(number_of_frames):
            self._write_unsigned_long_long(0)
        self._write_tag('ExtendedOffsetTableLengths', 'OV', eot_length)
        for index in range(number_of_frames):
            self._write_unsigned_long_long(0)
        return table_start

    def _reserve_bot(
        self,
        number_of_frames: int
    ) -> int:
        """Reserve space in file for basic offset table.

        Parameters
        ----------
        number_of_frames: int
            Number of frames to reserve space for.

        """
        table_start = self._fp.tell()
        BYTES_PER_ITEM = 4
        tag_lengths = BYTES_PER_ITEM * number_of_frames
        self._fp.write_tag(ItemTag)
        self._fp.write_UL(tag_lengths)
        for index in range(number_of_frames):

            self._fp.write_UL(0)
        return table_start

    def _write_pixel_data_start(
        self,
        number_of_frames: int,
        offset_table: Optional[str]
    ) -> Tuple[Optional[int], int]:
        """Writes tags starting pixel data and reserves space for BOT or EOT.

        Parameters
        ----------
        number_of_frames: int
            Number of frames to reserve space for in BOT or EOT.
        offset_table: Optional[str] = 'bot'
            Offset table to use, 'bot' basic offset table, 'eot' extended
            offset table, None - no offset table.

        Returns
        ----------
        Tuple[Optional[int], int]
            Start of table (BOT or EOT) and start of pixel data (after BOT).
        """
        table_start: Optional[int] = None
        if offset_table == 'eot':
            table_start = self._reserve_eot(number_of_frames)

        # Write pixel data tag
        self._write_tag('PixelData', 'OB')

        if offset_table == 'bot':
            table_start = self._reserve_bot(number_of_frames)
        else:
            self._fp.write_tag(ItemTag)  # Empty BOT
            self._fp.write_UL(0)

        pixel_data_start = self._fp.tell()

        return table_start, pixel_data_start

    def _write_bot(
        self,
        bot_start: int,
        pixel_data_start: int,
        frame_positions: Sequence[int]
    ) -> None:
        """Writes BOT to file.

        Parameters
        ----------
        bot_start: int
            File position of BOT start
        bot_end: int
            File position of BOT end
        frame_positions: Sequence[int]
            List of file positions for frames, relative to file start

        """
        BYTES_PER_ITEM = 4
        # Check that last BOT entry is not over 2^32 - 1
        last_entry = frame_positions[-1] - pixel_data_start
        if last_entry > 2**32 - 1:
            raise NotImplementedError(
                "Image data exceeds 2^32 - 1 bytes "
                "An extended offset table should be used"
            )

        self._fp.seek(bot_start)  # Go to first BOT entry
        self._check_tag_and_length(
            ItemTag,
            BYTES_PER_ITEM*len(frame_positions),
            False
        )

        for frame_position in frame_positions:  # Write BOT
            self._fp.write_UL(frame_position-pixel_data_start)

    def _write_unsigned_long_long(
        self,
        value: int
    ):
        """Write unsigned long long integer (64 bits) as little endian.

        Parameters
        ----------
        value: int
            Value to write.

        """
        self._fp.write(pack('<Q', value))

    def _write_eot(
        self,
        eot_start: int,
        pixel_data_start: int,
        frame_positions: Sequence[int],
        last_frame_end: int
    ) -> None:
        """Writes EOT to file.

        Parameters
        ----------
        bot_start: int
            File position of EOT start
        pixel_data_start: int
            File position of EOT end
        frame_positions: Sequence[int]
            List of file positions for frames, relative to file start
        last_frame_end: int
            Position of last frame end.

        """
        BYTES_PER_ITEM = 8
        # Check that last BOT entry is not over 2^64 - 1
        last_entry = frame_positions[-1] - pixel_data_start
        if last_entry > 2**64 - 1:
            raise ValueError(
                "Image data exceeds 2^64 - 1 bytes, likely something is wrong"
            )
        self._fp.seek(eot_start)  # Go to EOT table
        self._check_tag_and_length(
            Tag('ExtendedOffsetTable'),
            BYTES_PER_ITEM*len(frame_positions)
        )
        for frame_position in frame_positions:  # Write EOT
            relative_position = frame_position-pixel_data_start
            self._write_unsigned_long_long(relative_position)

        # EOT LENGTHS
        self._check_tag_and_length(
            Tag('ExtendedOffsetTableLengths'),
            BYTES_PER_ITEM*len(frame_positions)
        )
        frame_start = frame_positions[0]
        for frame_end in frame_positions[1:]:  # Write EOT lengths
            frame_length = frame_end - frame_start
            self._write_unsigned_long_long(frame_length)
            frame_start = frame_end

        # Last frame length, end does not include tag and length
        TAG_BYTES = 4
        LENGHT_BYTES = 4
        last_frame_start = frame_start + TAG_BYTES + LENGHT_BYTES
        last_frame_length = last_frame_end - last_frame_start
        self._write_unsigned_long_long(last_frame_length)

    def _write_pixel_data(
        self,
        image_data: ImageData,
        z: float,
        path: str,
        workers: int,
        chunk_size: int,
        scale: int = 1,
        image_format: str = 'jpeg',
        image_options: Dict[str, Any] = {'quality': 95}
    ) -> List[int]:
        """Writes pixel data to file.

        Parameters
        ----------
        image_data: ImageData
            Image data to read pixel tiles from.
        z: float
            Focal plane to write.
        path: str
            Optical path to write.
        workers: int
            Maximum number of thread workers to use.
        chunk_size: int
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.
        scale: int
            Scale factor (1 = No scaling).
        image_format: str = 'jpeg'
            Image format if scaling.
        image_options: Dict[str, Any] = {'quality': 95}
            Image options if scaling.

        Returns
        ----------
        List[int]
            List of frame position (position of ItemTag), relative to start of
            file.
        """
        chunked_tile_points = self._chunk_tile_points(
            image_data,
            chunk_size,
            scale
        )

        if scale == 1:
            def get_tiles_thread(tile_points: Sequence[Point]) -> List[bytes]:
                """Thread function to get tiles as bytes."""
                return image_data.get_encoded_tiles(tile_points, z, path)
            get_tiles = get_tiles_thread
        else:
            def get_scaled_tiles_thread(
                scaled_tile_points: Sequence[Point]
            ) -> List[bytes]:
                """Thread function to get scaled tiles as bytes."""
                return image_data.get_scaled_encoded_tiles(
                    scaled_tile_points,
                    z,
                    path,
                    scale,
                    image_format,
                    image_options
                )
            get_tiles = get_scaled_tiles_thread
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # Each thread result is a list of tiles that is itemized and writen
            frame_positions: List[int] = []
            for thread_result in pool.map(
                get_tiles,
                chunked_tile_points
            ):
                for tile in thread_result:
                    for frame in itemize_frame(tile, 1):
                        frame_positions.append(self._fp.tell())
                        self._fp.write(frame)

        return frame_positions

    def _chunk_tile_points(
        self,
        image_data: ImageData,
        chunk_size: int,
        scale: int = 1
    ) -> Generator[Generator[Point, None, None], None, None]:
        """Divides tile positions in image_data into chunks.

        Parameters
        ----------
        image_data: ImageData
            Image data with tiles to chunk.
        chunk_size: int
            Requested chunk size
        scale: int = 1
            Scaling factor (1 = no scaling).

        Returns
        ----------
        Generator[Generator[Point, None, None], None, None]
            Chunked tile positions
        """
        minimum_chunk_size = getattr(
            image_data,
            'suggested_minimum_chunk_size',
            1
        )
        # If chunk_size is less than minimum_chunk_size, use minimum_chunk_size
        # Otherwise, set chunk_size to highest even multiple of
        # minimum_chunk_size
        chunk_size = max(
            minimum_chunk_size,
            chunk_size//minimum_chunk_size * minimum_chunk_size
        )
        new_tiled_size = image_data.tiled_size.ceil_div(scale)
        # Divide the image tiles up into chunk_size chunks (up to tiled size)
        chunked_tile_points = (
            Region(
                Point(x, y),
                Size(min(chunk_size, new_tiled_size.width - x), 1)
            ).iterate_all()
            for y in range(new_tiled_size.height)
            for x in range(0, new_tiled_size.width, chunk_size)
        )
        return chunked_tile_points

    def _write_pixel_data_end(self) -> None:
        """Writes tags ending pixel data."""
        self._fp.write_tag(SequenceDelimiterTag)
        self._fp.write_UL(0)


class WsiInstance:
    """Represents a level, label, or overview wsi image, containing image data
    and datasets with metadata."""
    def __init__(
        self,
        datasets: Union[WsiDataset, Sequence[WsiDataset]],
        image_data: ImageData
    ):
        """Create a WsiInstance from datasets with metadata and image data.

        Parameters
        ----------
        datasets: Union[WsiDataset, Sequence[WsiDataset]]
            Single dataset or list of datasets.
        image_data: ImageData
            Image data.
        """
        if not isinstance(datasets, Sequence):
            datasets = [datasets]
        self._datasets = datasets
        self._image_data = image_data
        self._identifier, self._uids = self._validate_instance(self.datasets)
        self._wsi_type = self.dataset.wsi_type

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
        depth: Optional[int] = None
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
            ' ImageData ' + self.image_data.pretty_str(indent+1, depth)
        )
        return string

    @property
    def wsi_type(self) -> str:
        """Return wsi type."""
        return self._wsi_type

    @property
    def datasets(self) -> Sequence[WsiDataset]:
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
    def mm_size(self) -> Optional[SizeMm]:
        """Return slide size in mm."""
        return self.dataset.mm_size

    @property
    def mm_depth(self) -> Optional[float]:
        """Return imaged depth in mm."""
        return self.dataset.mm_depth

    @property
    def slice_thickness(self) -> Optional[float]:
        """Return slice thickness."""
        return self.dataset.slice_thickness

    @property
    def slice_spacing(self) -> Optional[float]:
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
    def identifier(self) -> UID:
        """Return identifier (instance uid for single file instance or
        concatenation uid for multiple file instance)."""
        return self._identifier

    @property
    def instance_number(self) -> int:
        return int(self.dataset.instance_number)

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
    def uids(self) -> SlideUids:
        """Return base uids"""
        return self._uids

    @classmethod
    def open(
        cls,
        files: Sequence[WsiDicomFile],
        series_uids: SlideUids,
        series_tile_size: Optional[Size] = None
    ) -> List['WsiInstance']:
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
        List[WsiInstancece]
            List of created instances.
        """
        filtered_files = WsiDicomFile.filter_files(
            files,
            series_uids,
            series_tile_size
        )
        files_grouped_by_instance = WsiDicomFile.group_files(filtered_files)
        return [
            cls(
                [file.dataset for file in instance_files],
                WsiDicomImageData(instance_files)
            )
            for instance_files in files_grouped_by_instance.values()
        ]

    @staticmethod
    def check_duplicate_instance(
        instances: Sequence['WsiInstance'],
        self: object
    ) -> None:
        """Check for duplicates in list of instances. Instances are duplicate
        if instance identifier (file instance uid or concatenation uid) match.
        Stops at first found duplicate and raises WsiDicomUidDuplicateError.

        Parameters
        ----------
        instances: Sequence['WsiInstance']
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
        datasets: Sequence[WsiDataset]
    ) -> Tuple[UID, SlideUids]:
        """Check that no files in instance are duplicate, that all files in
        instance matches (uid, type and size).
        Raises WsiDicomMatchError otherwise.
        Returns the matching file uid.

        Returns
        ----------
        Tuple[UID, SlideUids]
            Instance identifier uid and base uids
        """
        WsiDataset.check_duplicate_dataset(datasets, self)

        base_dataset = datasets[0]
        for dataset in datasets[1:]:
            if not base_dataset.matches_instance(dataset):
                raise WsiDicomError("Datasets in instances does not match")
        return (
            base_dataset.uids.identifier,
            base_dataset.uids.slide,
        )

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
            self.uids.matches(other_instance.uids) and
            self.size == other_instance.size and
            self.tile_size == other_instance.tile_size and
            self.wsi_type == other_instance.wsi_type
        )

    def close(self) -> None:
        self._image_data.close()
