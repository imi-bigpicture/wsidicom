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

import logging
import math
from collections.abc import Sequence
from copy import deepcopy
from enum import Enum
from functools import cached_property
from typing import Any, ClassVar

from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence as DicomSequence
from pydicom.tag import BaseTag
from pydicom.uid import (
    UID,
    VLWholeSlideMicroscopyImageStorage,
    generate_uid,
)
from pydicom.valuerep import MAX_VALUE_LEN, DSfloat

from wsidicom.codec import Encoder
from wsidicom.config import get_settings
from wsidicom.errors import (
    WsiDicomError,
    WsiDicomFileError,
    WsiDicomUidDuplicateError,
)
from wsidicom.geometry import Size, SizeMm
from wsidicom.instance.image_data import ImageData
from wsidicom.metadata.image import ImageType
from wsidicom.tags import (
    LossyImageCompressionMethodTag,
    LossyImageCompressionRatioTag,
    OpticalPathIdentificationSequenceTag,
    OpticalPathIdentifierTag,
    PerFrameFunctionalGroupsSequenceTag,
)
from wsidicom.uid import FileUids, SlideUids


class TileType(Enum):
    FULL = "TILED_FULL"
    SPARSE = "TILED_SPARSE"


class WsiDataset(Dataset):
    """Extend pydicom.dataset.Dataset (containing WSI metadata) with simple
    parsers for attributes specific for WSI. Use snake case to avoid name
    collision with dicom fields (that are handled by pydicom.dataset.Dataset).
    """

    REQUIRED_ATTRIBUTES: ClassVar[tuple[str, ...]] = (
        "SOPInstanceUID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "ImageType",
        "Rows",
        "Columns",
        "TotalPixelMatrixColumns",
        "TotalPixelMatrixRows",
        "SamplesPerPixel",
        "PhotometricInterpretation",
        "BitsStored",
    )
    """DICOM attributes that must be present for the library to be able to read
    an instance. These are the attributes that are dereferenced unconditionally
    while opening a dataset (identity, image and tile geometry, and pixel
    format). Datasets missing any of these are rejected by
    :func:`is_supported_wsi_dicom`."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of dataset {self.uids.instance}"

    @cached_property
    def uids(self) -> FileUids:
        """Return UIDs from dataset.

        Returns
        -------
        FileUids
            Found UIDs from dataset.
        """
        instance_uid = UID(self.SOPInstanceUID)
        concatenation_uid = getattr(self, "SOPInstanceUIDOfConcatenationSource", None)
        frame_of_reference_uid = getattr(self, "FrameOfReferenceUID", None)

        slide_uids = SlideUids(
            self.StudyInstanceUID,
            self.SeriesInstanceUID,
            frame_of_reference_uid,
        )
        file_uids = FileUids(instance_uid, concatenation_uid, slide_uids)
        return file_uids

    @cached_property
    def frame_offset(self) -> int:
        """Return frame offset (offset to first frame in instance if
        concatenated). Is zero if non-concatenated instance or first instance
        in concatenated instance.

        Returns
        -------
        int
            Concatenation offset in number of frames.
        """
        if self.uids.concatenation is None:
            return 0
        try:
            return int(self.ConcatenationFrameOffsetNumber)
        except AttributeError:
            raise WsiDicomError(
                "Concatenated file missing concatenation frame offsetnumber"
            ) from None

    @property
    def frame_count(self) -> int:
        """Return number of frames in instance."""
        return int(getattr(self, "NumberOfFrames", 1))

    @cached_property
    def tile_type(self) -> TileType:
        """Return tiling type of dataset. Raises WsiDicomError if type
        is undetermined.

        Returns
        -------
        TileType
            Tiling type
        """
        tile_type = getattr(self, "DimensionOrganizationType", "TILED_SPARSE")
        if tile_type == "TILED_FULL":
            # By the standard it should be tiled full.
            return TileType.FULL
        if "PerFrameFunctionalGroupsSequence" in self:
            # If no per frame functional sequence we can't make a sparse tile index.
            return TileType.SPARSE
        if self.image_type == ImageType.LABEL:
            # Labels are expected to only have one frame and can be treated as tiled
            # full.
            return TileType.FULL
        number_of_focal_planes = getattr(self, "TotalPixelMatrixFocalPlanes", 1)
        number_of_optical_paths = getattr(self, "NumberOfOpticalPaths", 1)
        if self.frame_count == number_of_focal_planes * number_of_optical_paths:
            # One frame per focal plane and optical path, treat as tiled full.
            return TileType.FULL
        raise WsiDicomError("Undetermined tile type.")

    @cached_property
    def pixel_measure(self) -> Dataset | None:
        """Return Pixel measure dataset from dataset if found.

        Returns
        -------
        Dataset | None
            Found Pixel measure dataset.
        """
        shared_functional_group = getattr(self, "SharedFunctionalGroupsSequence", None)
        if shared_functional_group is None:
            return None
        pixel_measure_sequence = getattr(
            shared_functional_group[0], "PixelMeasuresSequence", None
        )
        if pixel_measure_sequence is None:
            return None
        return pixel_measure_sequence[0]

    @cached_property
    def pixel_spacing(self) -> SizeMm | None:
        """Read pixel spacing from dicom dataset.

        Parameters
        ----------
        ds: Dataset
            Pydicom dataset

        Returns
        -------
        SizeMm
            The pixel spacing in mm/pixel.
        """
        if self.pixel_measure is None:
            return None
        pixel_spacing_values = getattr(self.pixel_measure, "PixelSpacing", None)
        if pixel_spacing_values is not None:
            if any([spacing <= 0 for spacing in pixel_spacing_values]):
                logging.warning(f"Pixel spacing not positive, {pixel_spacing_values}")
                return None
            return SizeMm(pixel_spacing_values[1], pixel_spacing_values[0])
        return None

    @cached_property
    def spacing_between_slices(self) -> float | None:
        """Return spacing between slices."""
        if self.pixel_measure is None:
            return None
        return getattr(self.pixel_measure, "SpacingBetweenSlices", None)

    @cached_property
    def frame_sequence(self) -> DicomSequence:
        """Return per frame functional group sequence if present, otherwise
        shared functional group sequence.

        Returns
        -------
        DicomSequence
            Per frame or shared functional group sequence.
        """
        if "PerFrameFunctionalGroupsSequence" in self and (
            "PlanePositionSlideSequence" in self.PerFrameFunctionalGroupsSequence[0]
        ):
            return self.PerFrameFunctionalGroupsSequence
        elif "SharedFunctionalGroupsSequence" in self:
            return self.SharedFunctionalGroupsSequence
        return DicomSequence([])

    @property
    def ext_depth_of_field(self) -> bool:
        """Return true if instance has extended depth of field
        (several focal planes are combined to one plane)."""
        return self._ext_depth_of_field[0]

    @property
    def ext_depth_of_field_planes(self) -> int | None:
        """Return number of focal planes used for extended depth of
        field."""
        return self._ext_depth_of_field[1]

    @property
    def ext_depth_of_field_plane_distance(self) -> float | None:
        """Return total focal depth used for extended depth of field."""
        return self._ext_depth_of_field[2]

    @cached_property
    def focus_method(self) -> str:
        """Return focus method."""
        return str(getattr(self, "FocusMethod", "AUTO"))

    @cached_property
    def image_size(self) -> Size:
        """Read total pixel size from dataset.

        Returns
        -------
        Size
            The image size
        """
        image_size = Size(self.TotalPixelMatrixColumns, self.TotalPixelMatrixRows)
        if image_size.width <= 0 or image_size.height <= 0:
            raise WsiDicomError("Image size is zero")
        if self.tile_type == TileType.FULL and self.uids.concatenation is None:
            # Check that the number of frames match the image size and tile size.
            # Dont check concatenated instances as the frame count is ambiguous.
            expected_tiled_size = image_size.ceil_div(self.tile_size)
            number_of_focal_planes = getattr(self, "TotalPixelMatrixFocalPlanes", 1)
            number_of_optical_paths = getattr(self, "NumberOfOpticalPaths", 1)
            expected_frame_count = (
                expected_tiled_size.area
                * number_of_focal_planes
                * number_of_optical_paths
            )
            if expected_frame_count != self.frame_count:
                error = (
                    f"Image size {image_size} does not match tile size "
                    f"{self.tile_size} and number of frames {self.frame_count} "
                    f"for tile type {TileType.FULL}."
                )
                if (
                    self.image_type == ImageType.VOLUME
                    and self.frame_count
                    != number_of_focal_planes * number_of_optical_paths
                ):
                    # Be strict on volume images if more than one frame per focal plane
                    # and optical path.
                    raise WsiDicomError(error)
                # Labels and overviews are likely to have only one tile.
                error += " Overriding image size to tile size."
                logging.warning(error)
                image_size = self.tile_size
        return image_size

    @cached_property
    def mm_size(self) -> SizeMm | None:
        """Read mm size from dataset.

        Returns
        -------
        SizeMm
            The size of the image in mm
        """
        mm_width = getattr(self, "ImagedVolumeWidth", None)
        mm_height = getattr(self, "ImagedVolumeHeight", None)
        if mm_width is None or mm_height is None:
            mm_size = None
        else:
            mm_size = SizeMm(mm_width, mm_height)
        return mm_size

    @cached_property
    def mm_depth(self) -> float | None:
        """Return depth of image in mm."""
        return getattr(self, "ImagedVolumeDepth", None)

    @cached_property
    def tile_size(self) -> Size:
        """Read tile size from from dataset.

        Returns
        -------
        Size
            The tile size
        """
        return Size(self.Columns, self.Rows)

    @property
    def samples_per_pixel(self) -> int:
        """Return samples per pixel (3 for RGB)."""
        return self.SamplesPerPixel

    @property
    def bits(self) -> int:
        """Return the number of bits stored for each sample."""
        return self.BitsStored

    @property
    def lossy_compressed(self) -> bool:
        """Return true if image has been lossy compressed."""
        return getattr(self, "LossyImageCompression", None) == "01"

    @property
    def photometric_interpretation(self) -> str:
        """Return photometric interpretation."""
        return self.PhotometricInterpretation

    @cached_property
    def optical_path_sequence(self) -> DicomSequence | None:
        """Return optical path sequence from dataset."""
        return getattr(self, "OpticalPathSequence", None)

    @cached_property
    def z_offset(self) -> float:
        z_offset_getters = (
            lambda: (
                self.TotalPixelMatrixOriginSequence[0].ZOffsetInSlideCoordinateSystem
            ),
            lambda: (
                self.SharedFunctionalGroupsSequence[0]
                .PlanePositionSlideSequence[0]
                .ZOffsetInSlideCoordinateSystem
            ),
        )
        for z_offset_getter in z_offset_getters:
            try:
                return z_offset_getter()
            except AttributeError:
                pass
        return 0.0

    @property
    def number_of_focal_planes(self) -> int:
        """Return number of focal planes in image."""
        return self.get("TotalPixelMatrixFocalPlanes", 1)

    @property
    def slice_thickness(self) -> float | None:
        """Return slice thickness spacing from pixel measure dataset.

        Returns
        -------
        float | None
            Slice thickness or None if unknown.
        """
        if self.pixel_measure is not None:
            slice_thickness = getattr(self.pixel_measure, "SliceThickness", None)
            if slice_thickness is not None:
                return slice_thickness
        if self.mm_depth is not None:
            number_of_focal_planes = getattr(self, "TotalPixelMatrixFocalPlanes", 1)
            return self.mm_depth / number_of_focal_planes
        return None

    @cached_property
    def image_type(self) -> ImageType:
        """Return wsi flavour from wsi type tuple.

        Returns
        -------
        ImageType
            Wsi flavour.
        """
        return self._get_image_type(self.ImageType)

    @classmethod
    def is_supported_wsi_dicom(cls, dataset: Dataset) -> ImageType | None:
        """Check if dataset is a DICOM WSI type that the library can read.

        The dataset is rejected (``None`` returned) if it is not of the WSI SOP
        class, if it is missing any attribute the library dereferences while
        opening an instance (see ``REQUIRED_ATTRIBUTES``), if the image type is
        not supported, or if the pixel representation or planar configuration is
        unsupported.

        Parameters
        ----------
        dataset: Dataset
            Pydicom dataset to check if is a WSI dataset.

        Returns
        -------
        ImageType | None
            WSI image flavor, or None if the dataset is not a supported WSI.
        """

        sop_class_uid: UID | None = getattr(dataset, "SOPClassUID", None)
        if sop_class_uid != VLWholeSlideMicroscopyImageStorage:
            logging.debug(f"Non-wsi image, SOP class {sop_class_uid}.")
            return None

        for name in cls.REQUIRED_ATTRIBUTES:
            if name not in dataset:
                logging.debug(f"Missing required attribute {name}.")
                return None

        try:
            image_type = cls._get_image_type(dataset.ImageType)
        except ValueError:
            logging.debug(f"Non-supported image type {dataset.ImageType}.")
            return None

        pixel_representation = int(getattr(dataset, "PixelRepresentation", 0))
        if pixel_representation != 0:
            logging.debug(f"Unsupported pixel representation {pixel_representation}.")
            return None
        planar_configuration = int(getattr(dataset, "PlanarConfiguration", 0))
        if planar_configuration != 0:
            logging.debug(f"Unsupported planar configuration {planar_configuration}.")
            return None
        return image_type

    @staticmethod
    def check_duplicate_dataset(
        datasets: Sequence["WsiDataset"], caller: object
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
        instance_uids: list[UID] = []

        for dataset in datasets:
            instance_uid = UID(dataset.SOPInstanceUID)
            if instance_uid not in instance_uids:
                instance_uids.append(instance_uid)
            else:
                raise WsiDicomUidDuplicateError(str(dataset), str(caller))

    def matches_instance(self, other_dataset: "WsiDataset") -> bool:
        """Return true if other file is of the same instance as self.

        Parameters
        ----------
        other_dataset: 'WsiDataset'
            Dataset to check.

        Returns
        -------
        bool
            True if same instance.
        """

        return (
            self.uids == other_dataset.uids
            and self.image_size == other_dataset.image_size
            and self.tile_size == other_dataset.tile_size
            and self.tile_type == other_dataset.tile_type
            and (
                getattr(self, "TotalPixelMatrixOriginSequence", None)
                == getattr(other_dataset, "TotalPixelMatrixOriginSequence", None)
            )
        )

    def matches_series(self, uids: SlideUids, tile_size: Size | None = None) -> bool:
        """Check if instance is valid (Uids and tile size match).
        Base uids should match for instances in all types of series,
        tile size should only match for level series.
        """
        if tile_size is not None and tile_size != self.tile_size:
            return False

        return self.uids.slide.matches(uids)

    def read_optical_path_identifier(self, frame: Dataset) -> str:
        """Return optical path identifier from frame, or from self if not
        found."""
        optical_path_sequence = frame.get(
            OpticalPathIdentificationSequenceTag, self.optical_path_sequence
        )
        if optical_path_sequence is None:
            return "0"
        optical_sequence = getattr(
            frame, "OpticalPathIdentificationSequence", self.optical_path_sequence
        )
        if optical_sequence is None:
            return "0"
        optical_path_identifier = optical_sequence[0].get(
            OpticalPathIdentifierTag, None
        )
        if optical_path_identifier is None:
            return "0"
        return optical_path_identifier.value

    def get_multi_value(self, tag: BaseTag) -> list[Any]:
        """Return values for tag as list of values. If tag is not found, return empty
        list. If tag is not multi value, return list with one value.

        Parameters
        ----------
        tag: BaseTag
            Tag to get values for.

        Returns
        -------
        list[Any]
            List of values.
        """
        element = self.get(tag)
        if element is None:
            return []
        vm = getattr(element, "VM", 1)
        if vm > 1 or isinstance(element, MultiValue):
            return [value for value in element]
        return [element.value]

    def as_tiled_full(
        self,
        focal_planes: Sequence[float],
        optical_paths: Sequence[str],
        tiled_size: Size,
        scale: int = 1,
    ) -> "WsiDataset":
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
        -------
        WsiDataset
            Copy of dataset set as tiled full.

        """
        dataset = self._copy_without_per_frame()
        dataset.DimensionOrganizationType = "TILED_FULL"
        # Make a new Shared functional group sequence and Pixel measure
        # sequence if not in dataset, otherwise update the Pixel measure
        # sequence
        shared_functional_group = getattr(
            dataset, "SharedFunctionalGroupsSequence", DicomSequence([Dataset()])
        )

        pixel_measure = getattr(
            shared_functional_group[0],
            "PixelMeasuresSequence",
            DicomSequence([Dataset()]),
        )
        if dataset.pixel_spacing is not None:
            pixel_measure[0].PixelSpacing = [
                DSfloat(dataset.pixel_spacing.height * scale, True),
                DSfloat(dataset.pixel_spacing.width * scale, True),
            ]
        focal_plane_spacing = self._get_spacing_between_slices_for_focal_planes(
            focal_planes
        )
        if focal_plane_spacing is not None:
            pixel_measure[0].SpacingBetweenSlices = DSfloat(focal_plane_spacing, True)
        elif "SpacingBetweenSlices" in pixel_measure[0]:
            # A single focal plane has no spacing; drop any spacing carried over
            # from a multi-plane source that has been split per focal plane.
            del pixel_measure[0].SpacingBetweenSlices

        if self.slice_thickness is not None:
            pixel_measure[0].SliceThickness = DSfloat(dataset.slice_thickness, True)

        shared_functional_group[0].PixelMeasuresSequence = pixel_measure
        dataset.SharedFunctionalGroupsSequence = shared_functional_group

        dataset.TotalPixelMatrixColumns = max(
            math.ceil(dataset.TotalPixelMatrixColumns / scale), 1
        )
        dataset.TotalPixelMatrixRows = max(
            math.ceil(dataset.TotalPixelMatrixRows / scale), 1
        )
        dataset.TotalPixelMatrixFocalPlanes = len(focal_planes)
        dataset.NumberOfOpticalPaths = len(optical_paths)
        dataset.NumberOfFrames = (
            max(tiled_size.ceil_div(scale).area, 1)
            * len(focal_planes)
            * len(optical_paths)
        )

        # Keep only the optical paths written to this instance, so the optical
        # path identity is preserved when the paths have been split across
        # instances (e.g. one instance per optical path).
        optical_path_sequence = getattr(dataset, "OpticalPathSequence", None)
        if optical_path_sequence is not None:
            kept_optical_paths = DicomSequence(
                item
                for item in optical_path_sequence
                if str(item[OpticalPathIdentifierTag].value) in optical_paths
            )
            if len(kept_optical_paths) != 0:
                dataset.OpticalPathSequence = kept_optical_paths

        # Encode the focal plane origin (z) so the planes can be reconstructed
        # on read as ``z_offset + index * spacing``. This is the source z offset
        # for the full set of planes, but differs when the planes have been
        # split across instances (e.g. one instance per focal plane). Preserve
        # the in-plane (x, y) origin when present. Only create a new origin
        # sequence when there is a non-zero z to encode, since x and y are
        # required in the sequence item; a zero z is the default on read.
        focal_plane_origin = getattr(dataset, "TotalPixelMatrixOriginSequence", None)
        if focal_plane_origin is not None:
            focal_plane_origin[0].ZOffsetInSlideCoordinateSystem = DSfloat(
                focal_planes[0], True
            )
        elif focal_planes[0] != 0.0:
            origin_item = Dataset()
            origin_item.XOffsetInSlideCoordinateSystem = DSfloat(0.0, True)
            origin_item.YOffsetInSlideCoordinateSystem = DSfloat(0.0, True)
            origin_item.ZOffsetInSlideCoordinateSystem = DSfloat(focal_planes[0], True)
            dataset.TotalPixelMatrixOriginSequence = DicomSequence([origin_item])

        return dataset

    def update_for_transcoding(self, transcoder: Encoder, scale: int) -> None:
        """Update dataset metadata for transcoding.

        Parameters
        ----------
        transcoder: Encoder
            Encoder being used for transcoding.
        scale: int
            Scale factor applied to the image data.
        """
        self.PhotometricInterpretation = transcoder.photometric_interpretation
        if transcoder.lossy_method:
            self.LossyImageCompression = "01"
            ratios = self.get_multi_value(LossyImageCompressionRatioTag)
            methods = self.get_multi_value(LossyImageCompressionMethodTag)
            if scale != 1:
                ratios.clear()
                methods.clear()
            ratios.append(" " * MAX_VALUE_LEN["DS"])
            methods.append(transcoder.lossy_method.value)
            self.LossyImageCompressionRatio = ratios
            self.LossyImageCompressionMethod = methods

    @classmethod
    def create_instance_dataset(
        cls,
        dataset: Dataset,
        image_type: ImageType,
        image_data: ImageData,
        pyramid_index: int | None = None,
    ) -> "WsiDataset":
        """Return instance dataset for image_data based on base dataset.

        Parameters
        ----------
        base_dataset: Dataset
            Dataset common for all instances.
        image_type:
            Type of instance ('VOLUME', 'LABEL', 'OVERVIEW)
        image_data:
            Image data to create dataset for.
        pyramid_index: int | None = None
            Pyramid index. of image data, if volume image.

        Returns
        -------
        WsiDataset
            Dataset for instance.
        """
        resampled = "NONE"
        if image_type == ImageType.VOLUME:
            if pyramid_index is None:
                raise ValueError("Pyramid index must be set for volume image.")
            if pyramid_index > 0:
                resampled = "RESAMPLED"

        original_or_derived = "ORIGINAL" if resampled == "NONE" else "DERIVED"
        dataset.ImageType = [
            original_or_derived,
            "PRIMARY",
            image_type.value,
            resampled,
        ]
        dataset.SOPInstanceUID = generate_uid(prefix=None)
        shared_functional_group_sequence = Dataset()
        if image_data.pixel_spacing is None:
            if image_type == ImageType.VOLUME:
                raise ValueError(
                    "Image flavor 'VOLUME' requires pixel spacing to be set"
                )
        else:
            pixel_measure_sequence = Dataset()
            pixel_measure_sequence.PixelSpacing = [
                DSfloat(image_data.pixel_spacing.height, True),
                DSfloat(image_data.pixel_spacing.width, True),
            ]
            focal_plane_spacing = cls._get_spacing_between_slices_for_focal_planes(
                image_data.focal_planes
            )
            if focal_plane_spacing is not None:
                pixel_measure_sequence.SpacingBetweenSlices = DSfloat(
                    focal_plane_spacing, True
                )
            # DICOM 2022a part 3 IODs - C.8.12.4.1.2 Imaged Volume Width,
            # Height, Depth. Depth must not be 0. Default to 0.5 microns
            slice_thickness = 0.0005
            pixel_measure_sequence.SliceThickness = DSfloat(slice_thickness, True)
            shared_functional_group_sequence.PixelMeasuresSequence = DicomSequence(
                [pixel_measure_sequence]
            )
            dataset.SharedFunctionalGroupsSequence = DicomSequence(
                [shared_functional_group_sequence]
            )
            if image_data.imaged_size is None:
                dataset.ImagedVolumeWidth = (
                    image_data.image_size.width * image_data.pixel_spacing.width
                )
                dataset.ImagedVolumeHeight = (
                    image_data.image_size.height * image_data.pixel_spacing.height
                )
            else:
                dataset.ImagedVolumeWidth = image_data.imaged_size.width
                dataset.ImagedVolumeHeight = image_data.imaged_size.height
            # SliceThickness is in mm, ImagedVolumeDepth in um
            dataset.ImagedVolumeDepth = DSfloat(slice_thickness * 1000, True)

        # DICOM 2022a part 3 IODs - C.8.12.9 Whole Slide Microscopy Image Frame Type
        # Macro. Analogous to ImageType and shared by all frames so clone
        wsi_frame_type_item = Dataset()
        wsi_frame_type_item.FrameType = dataset.ImageType
        (
            shared_functional_group_sequence.WholeSlideMicroscopyImageFrameTypeSequence
        ) = DicomSequence([wsi_frame_type_item])
        dataset.SharedFunctionalGroupsSequence = DicomSequence(
            [shared_functional_group_sequence]
        )

        if image_data.image_coordinate_system is not None:
            dataset.ImageOrientationSlide = [
                DSfloat(value, True)
                for value in image_data.image_coordinate_system.orientation.values
            ]
            offset_item = Dataset()
            offset_item.XOffsetInSlideCoordinateSystem = DSfloat(
                image_data.image_coordinate_system.origin.x, True
            )
            offset_item.YOffsetInSlideCoordinateSystem = DSfloat(
                image_data.image_coordinate_system.origin.y, True
            )
            if image_data.image_coordinate_system.z_offset is not None:
                offset_item.ZOffsetInSlideCoordinateSystem = DSfloat(
                    image_data.image_coordinate_system.z_offset, True
                )
            dataset.TotalPixelMatrixOriginSequence = DicomSequence([offset_item])

        dataset.DimensionOrganizationType = "TILED_FULL"
        dataset.TotalPixelMatrixColumns = image_data.image_size.width
        dataset.TotalPixelMatrixRows = image_data.image_size.height
        dataset.Columns = image_data.tile_size.width
        dataset.Rows = image_data.tile_size.height
        dataset.NumberOfFrames = (
            image_data.tiled_size.area
            * len(image_data.focal_planes)
            * len(image_data.optical_paths)
        )
        dataset.BitsAllocated = image_data.bits // 8 * 8
        dataset.BitsStored = image_data.bits
        dataset.HighBit = image_data.bits - 1
        dataset.PixelRepresentation = 0
        if image_data.lossy_compression:
            dataset.LossyImageCompression = "01"
            dataset.LossyImageCompressionRatio = [
                DSfloat(item.ratio, auto_format=True)
                for item in image_data.lossy_compression
            ]
            dataset.LossyImageCompressionMethod = [
                item.method.value for item in image_data.lossy_compression
            ]
        else:
            dataset.LossyImageCompression = "00"

        dataset.PhotometricInterpretation = image_data.photometric_interpretation
        dataset.SamplesPerPixel = image_data.samples_per_pixel

        if image_data.samples_per_pixel == 3:
            dataset.PlanarConfiguration = 0

        dataset.FocusMethod = "AUTO"
        dataset.ExtendedDepthOfField = "NO"
        return WsiDataset(dataset)

    def _copy_without_per_frame(self) -> "WsiDataset":
        """Copy dataset excluding PerFrameFunctionalGroupsSequence."""
        dataset = deepcopy(
            {
                tag: elem
                for tag, elem in self.items()
                if tag != PerFrameFunctionalGroupsSequenceTag
            }
        )
        return WsiDataset(dataset)

    @cached_property
    def _ext_depth_of_field(self) -> tuple[bool, int | None, float | None]:
        """Return extended depth of field (enabled, number of focal planes,
        distance between focal planes) from dataset.

        Returns
        -------
        tuple[bool, int | None, float | None]
            If extended depth of field is used, and if used number of focal
            planes and distance between focal planes.
        """
        if getattr(self, "ExtendedDepthOfField", "NO") != "YES":
            return False, None, None

        planes = getattr(self, "NumberOfFocalPlanes", 1)
        distance = getattr(self, "DistanceBetweenFocalPlanes", 0.0)
        if planes is None or distance is None:
            raise WsiDicomFileError(
                self.filepath,
                "Missing NumberOfFocalPlanes or DistanceBetweenFocalPlanes",
            )
        return True, planes, distance

    @staticmethod
    def focal_planes_equally_spaced(focal_planes: Sequence[float]) -> bool:
        """Return whether the focal planes can share one TILED_FULL instance.

        Focal planes can only be encoded in a single TILED_FULL instance if they
        are a single plane or (approximately) equally spaced.

        Parameters
        ----------
        focal_planes: Sequence[float]
            Focal planes to check.

        Returns
        -------
        bool
            True if the focal planes are a single plane or equally spaced.
        """
        try:
            WsiDataset._get_spacing_between_slices_for_focal_planes(focal_planes)
            return True
        except NotImplementedError:
            return False

    @staticmethod
    def _get_spacing_between_slices_for_focal_planes(
        focal_planes: Sequence[float],
    ) -> float | None:
        """Return spacing between slices in mm for focal planes (defined in
        um). Spacing must be the same between all focal planes for TILED_FULL
        arrangement.

        Parameters
        ----------
        focal_planes: Sequence[float]
            Focal planes to calculate spacing for.

        Returns
        -------
        float | None
            Spacing between focal planes, or None if only one focal plane.

        """
        if len(focal_planes) == 1:
            return None
        spacing: float | None = None
        sorted_focal_planes = sorted(focal_planes)
        distance_threshold = get_settings().focal_plane_distance_threshold
        for index in range(len(sorted_focal_planes) - 1):
            this_spacing = sorted_focal_planes[index + 1] - sorted_focal_planes[index]
            if spacing is None:
                spacing = this_spacing
            elif abs(spacing - this_spacing) > distance_threshold:
                raise NotImplementedError(
                    "Image data has non-equal spacing between slices: "
                    f"{spacing, this_spacing}, difference threshold: "
                    f"{distance_threshold}, "
                    "not possible to encode several focal planes in one "
                    "TILED_FULL instance. Split the focal planes into separate "
                    "instances (InstanceSplit.FOCAL_PLANE) to write unequally "
                    "spaced focal planes."
                )
        if spacing is None:
            raise ValueError("Could not calculate spacings.")
        return spacing / 1000.0

    @staticmethod
    def _get_image_type(wsi_type: tuple[str, str, str, str]) -> ImageType:
        """Return wsi flavour from wsi type tuple.

        Returns
        -------
        str
            Wsi flavour.
        """
        IMAGE_TYPE_INDEX_IN_WSI_TYPE = 2
        return ImageType(wsi_type[IMAGE_TYPE_INDEX_IN_WSI_TYPE])
