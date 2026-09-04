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
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
from pydicom.config import RAISE
from pydicom.datadict import dictionary_VR, keyword_for_tag
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence as DicomSequence
from pydicom.tag import BaseTag, Tag
from pydicom.uid import (
    UID,
    VLWholeSlideMicroscopyImageStorage,
    generate_uid,
)
from pydicom.valuerep import MAX_VALUE_LEN, DSfloat

from wsidicom.codec import Encoder
from wsidicom.codec.encoder import LossyCompressionIsoStandard
from wsidicom.config import get_settings
from wsidicom.errors import (
    WsiDicomError,
    WsiDicomFileError,
    WsiDicomUidDuplicateError,
)
from wsidicom.geometry import Size, SizeMm
from wsidicom.instance.image_data import ImageData
from wsidicom.instance.per_frame_group_positions import PerFrameGroupPositions
from wsidicom.metadata import (
    ImageCoordinateSystem,
    ImageType,
    Label,
    LossyCompression,
    Overview,
    Pyramid,
)
from wsidicom.metadata.schema.dicom import (
    BaseWsiMetadata,
    BaseWsiMetadataDicomSchema,
    ImageCoordinateSystemDicomSchema,
    LabelBaseDicomSchema,
    LabelDicomSchema,
    OverviewDicomSchema,
    PyramidDicomSchema,
)
from wsidicom.tags import (
    BitsAllocatedTag,
    BitsStoredTag,
    ColumnPositionInTotalImagePixelMatrixTag,
    ColumnsTag,
    ConcatenationFrameOffsetNumberTag,
    ConcatenationUIDTag,
    DimensionOrganizationTypeTag,
    DistanceBetweenFocalPlanesTag,
    ExtendedDepthOfFieldTag,
    FocusMethodTag,
    FrameOfReferenceUIDTag,
    FrameTypeTag,
    HighBitTag,
    ImagedVolumeDepthTag,
    ImagedVolumeHeightTag,
    ImagedVolumeWidthTag,
    ImageOrientationSlideTag,
    ImageTypeTag,
    InConcatenationNumberTag,
    InConcatenationTotalNumberTag,
    InstanceNumberTag,
    LossyImageCompressionMethodTag,
    LossyImageCompressionRatioTag,
    LossyImageCompressionTag,
    NumberOfFocalPlanesTag,
    NumberOfFramesTag,
    NumberOfOpticalPathsTag,
    OpticalPathIdentificationSequenceTag,
    OpticalPathIdentifierTag,
    OpticalPathSequenceTag,
    PerFrameFunctionalGroupsSequenceTag,
    PhotometricInterpretationTag,
    PixelMeasuresSequenceTag,
    PixelRepresentationTag,
    PixelSpacingTag,
    PlanarConfigurationTag,
    PlanePositionSlideSequenceTag,
    RowPositionInTotalImagePixelMatrixTag,
    RowsTag,
    SamplesPerPixelTag,
    SeriesInstanceUIDTag,
    SharedFunctionalGroupsSequenceTag,
    SliceThicknessTag,
    SOPClassUIDTag,
    SOPInstanceUIDOfConcatenationSourceTag,
    SOPInstanceUIDTag,
    SpacingBetweenSlicesTag,
    StudyInstanceUIDTag,
    TotalPixelMatrixColumnsTag,
    TotalPixelMatrixFocalPlanesTag,
    TotalPixelMatrixOriginSequenceTag,
    TotalPixelMatrixRowsTag,
    WholeSlideMicroscopyImageFrameTypeSequenceTag,
    XOffsetInSlideCoordinateSystemTag,
    YOffsetInSlideCoordinateSystemTag,
    ZOffsetInSlideCoordinateSystemTag,
)
from wsidicom.uid import FileUids, SlideUids

logger = logging.getLogger(__name__)


class TileType(Enum):
    FULL = "TILED_FULL"
    SPARSE = "TILED_SPARSE"


@dataclass(frozen=True)
class ConcatenationPart:
    """Where one instance sits in the concatenation its level is written as.

    Parameters
    ----------
    uid: UID
        Concatenation UID shared by every part of the level.
    source_instance_uid: UID
        SOP Instance UID the parts would have had as a single instance.
    number: int
        Position of this part in the concatenation, counting from one.
    frame_offset: int
        Number of frames the parts before this one hold together.
    total: int | None = None
        Number of parts the concatenation holds, when that is known. Left out
        of the dataset when it is not, as when parts are split by byte size
        while writing.
    """

    uid: UID
    source_instance_uid: UID
    number: int
    frame_offset: int
    total: int | None = None


class WsiDataset:
    """Extend pydicom.dataset.Dataset (containing WSI metadata) with simple
    parsers for attributes specific for WSI. Use snake case to avoid name
    collision with dicom fields (that are handled by pydicom.dataset.Dataset).
    """

    REQUIRED_ATTRIBUTES: ClassVar[tuple[BaseTag, ...]] = (
        SOPInstanceUIDTag,
        StudyInstanceUIDTag,
        SeriesInstanceUIDTag,
        ImageTypeTag,
        RowsTag,
        ColumnsTag,
        TotalPixelMatrixColumnsTag,
        TotalPixelMatrixRowsTag,
        SamplesPerPixelTag,
        PhotometricInterpretationTag,
        BitsStoredTag,
    )
    """DICOM attributes that must be present for the library to be able to read
    an instance. These are the attributes that are dereferenced unconditionally
    while opening a dataset (identity, image and tile geometry, and pixel
    format). Datasets missing any of these are rejected by
    `is_supported`."""

    SUPPORTED_PHOTOMETRIC_INTERPRETATIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "MONOCHROME2",
            "RGB",
            "YBR_FULL",
            "YBR_FULL_422",
            "YBR_PARTIAL_422",
            "YBR_ICT",
            "YBR_RCT",
        }
    )
    """Photometric interpretations the library can read. Covers the values the
    DICOM WSI IOD enumerates (MONOCHROME2, RGB, YBR_ICT, YBR_RCT, YBR_FULL_422);
    see PS3.3 C.8.12.4
    https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_c.8.12.4.html
    Notably excludes MONOCHROME1, which the IOD does not permit. Datasets with any
    other value are rejected by `is_supported`."""

    DEFAULT_Z_OFFSET: ClassVar[float] = 0.0
    """Z offset of a frame that does not state one, which is the focal plane at zero."""

    def __init__(
        self,
        dataset: Dataset,
        frame_positions: PerFrameGroupPositions | None = None,
    ):
        """Create a WSI dataset around a pydicom dataset.

        The dataset is wrapped rather than subclassed: what this adds is reading
        of WSI attributes, and `set` for writing one, which checks the value as
        it is set. The dataset itself is `dataset`.

        Parameters
        ----------
        dataset: Dataset
            Dataset to wrap.
        frame_positions: PerFrameGroupPositions | None = None
            Tile positions the reader took out of the bytes of the Per Frame Functional
            Groups Sequence, given when it did not have to build a dataset for every
            item. The sequence is then not in the dataset, and these stand in for it.
        """
        self._dataset = dataset
        self._frame_positions = frame_positions

    def __deepcopy__(self, memo: dict[int, Any]) -> "WsiDataset":
        """A copy of this holding a copy of the dataset.

        Defined so that the readers cached on this are not carried over to the
        copy by the default copying of the attributes of an object: what was
        worked out from the attributes of this one is worked out again for the
        copy rather than described from the dataset it came from.
        """
        return WsiDataset(deepcopy(self._dataset, memo), self._frame_positions)

    def as_dataset(self) -> Dataset:
        """The wrapped dataset, for giving to something that takes a dataset.

        A method rather than a property so that stepping out of this class is a
        deliberate act and plain to find; what is read from a WSI dataset should
        come from the readers here.
        """
        return self._dataset

    @classmethod
    def is_supported_image_type(cls, dataset: Dataset) -> bool:
        """Whether a dataset is of the WSI SOP class and states a flavour that is read.

        The part of :func:`is_supported` that can be answered from the attributes
        ordered before ``SOPInstanceUID``, so that a reader working through a stream
        can turn an instance away without parsing the rest of it. Answers for a
        dataset holding no more than those attributes, and never accepts what
        :func:`is_supported` rejects.

        Parameters
        ----------
        dataset: Dataset
            Dataset to check, holding at least the attributes ordered before
            ``SOPInstanceUID``.

        Returns
        -------
        bool
            True if the dataset is of the WSI SOP class and states an image type
            this library reads.
        """
        sop_class_uid: UID | None = cls.get_value(dataset, SOPClassUIDTag)
        if sop_class_uid != VLWholeSlideMicroscopyImageStorage:
            logger.debug(f"Non-wsi image, SOP class {sop_class_uid}.")
            return False
        image_type = cls.get_value(dataset, ImageTypeTag)
        if image_type is None:
            logger.debug(f"Missing required attribute {keyword_for_tag(ImageTypeTag)}.")
            return False
        try:
            cls._get_image_type(image_type)
        except (ValueError, IndexError):
            logger.debug(f"Non-supported image type {image_type}.")
            return False
        return True

    @classmethod
    def is_supported(cls, dataset: Dataset) -> bool:
        """Whether a dataset is a WSI instance this library can read.

        False if it is not of the WSI SOP class, if the image type is not one that
        is read, if it is missing any attribute dereferenced while opening an
        instance (see ``REQUIRED_ATTRIBUTES``), if the pixel representation or
        planar configuration is unsupported, if the photometric interpretation is
        not one the codec handles (e.g. MONOCHROME1), or if it is a non-8-bit
        colour image.

        This settles whether an instance is read, wherever it was read from. Every
        attribute it looks at is ordered before the Per Frame Functional Groups
        Sequence, so a reader that has come that far through a stream can ask.

        Parameters
        ----------
        dataset: Dataset
            Dataset to check, holding at least the attributes ordered before the
            Per Frame Functional Groups Sequence.

        Returns
        -------
        bool
            True if the instance can be read.
        """
        if not cls.is_supported_image_type(dataset):
            return False
        for tag in cls.REQUIRED_ATTRIBUTES:
            if tag not in dataset:
                logger.debug(f"Missing required attribute {keyword_for_tag(tag)}.")
                return False

        pixel_representation = int(cls.get_value(dataset, PixelRepresentationTag, 0))
        if pixel_representation != 0:
            logger.debug(f"Unsupported pixel representation {pixel_representation}.")
            return False
        planar_configuration = int(cls.get_value(dataset, PlanarConfigurationTag, 0))
        if planar_configuration != 0:
            logger.debug(f"Unsupported planar configuration {planar_configuration}.")
            return False
        photometric_interpretation = str(dataset.PhotometricInterpretation)
        if photometric_interpretation not in cls.SUPPORTED_PHOTOMETRIC_INTERPRETATIONS:
            logger.debug(
                f"Unsupported photometric interpretation {photometric_interpretation}."
            )
            return False
        bits_stored = int(dataset.BitsStored)
        samples_per_pixel = int(dataset.SamplesPerPixel)
        if bits_stored != 8 and samples_per_pixel != 1:
            # Non-8-bit is only supported for grayscale. 16-bit color is not
            # fundamentally hard (the stitch/downsample pipeline handles it), but
            # no example WSI has been found to test against; would support it if
            # one turned up.
            logger.debug(
                f"Unsupported combination of bits stored {bits_stored} and "
                f"samples per pixel {samples_per_pixel}."
            )
            return False
        return True

    @staticmethod
    def _create_data_element(
        tag: str | BaseTag,
        value: Any,
        value_representation: str | None = None,
    ) -> DataElement:
        """Make an element holding a value, checking the value as it is made.

        Parameters
        ----------
        tag: str | BaseTag
            Tag of the attribute, or the keyword of one that has a keyword.
        value: Any
            Value to set it to, as it would be given to pydicom.
        value_representation: str | None = None
            VR to write the value as, for a tag that does not have one of its
            own, such as a private tag. Taken from the tag when not given.

        The value is always checked, whatever `Settings.dicom_value_validation`
        says. That setting is there for the metadata a caller supplies, which
        wsidicom writes as given; what is written through here is what wsidicom
        worked out for itself, and a value it cannot write conformantly is a
        fault of its own rather than of the metadata it was handed.

        Raises
        ------
        ValueError
            If the tag is not a tag or the keyword of one, if it has no value
            representation of its own and none was given, or if the value does
            not conform to its value representation.
        """
        element_tag = Tag(tag)
        if value_representation is None:
            try:
                value_representation = dictionary_VR(element_tag)
            except KeyError:
                raise ValueError(
                    f"{element_tag} has no value representation of its own, so "
                    "one has to be given to set it."
                ) from None
            if " or " in value_representation:
                raise ValueError(
                    f"{element_tag} is written as {value_representation}, so "
                    "which of them has to be given to set it."
                )
        return DataElement(
            element_tag,
            value_representation,
            value,
            validation_mode=RAISE,
        )

    def replace(self, changes: Mapping[str | BaseTag, Any]) -> "WsiDataset":
        """A copy of this with the given attributes changed.

        A WSI dataset offers no way to change it in place: what wsidicom writes
        is written by making a changed copy, so that the values it writes are
        the ones checked here, while the values read from a file are left as
        they were read.

        The copy holds the elements of this one rather than copies of them, so
        that making one costs the number of attributes rather than the size of
        what they hold: changing an attribute puts a new element in the copy
        and leaves this one as it was. What is reached through `as_dataset` is
        shared, so changing that changes it for both.

        Parameters
        ----------
        changes: Mapping[str | BaseTag, Any]
            Attributes to change, by keyword or by tag.

        Returns
        -------
        WsiDataset
            Copy of this with the attributes changed.

        Raises
        ------
        ValueError
            If a tag is not a tag or the keyword of one, or if a value does not
            conform to its value representation.
        """
        dataset = Dataset()
        for element in self._dataset:
            dataset.add(element)
        file_meta = self._dataset.get("file_meta", None)
        if file_meta is not None:
            dataset.file_meta = file_meta
        self._update_dataset(dataset, changes)
        return WsiDataset(dataset, self._frame_positions)

    @staticmethod
    def get_value(dataset: Dataset, tag: BaseTag, default: Any = None) -> Any:
        """The value of an attribute of a dataset, or `default` if it is not there.

        Asking a dataset for a tag gives the element holding the value, where
        asking it for a keyword gives the value itself. This gives the value,
        so that a tag reads like the keyword it stands for.

        Parameters
        ----------
        dataset: Dataset
            Dataset to read the attribute from.
        tag: BaseTag
            Tag of the attribute.
        default: Any = None
            What to answer with when the dataset does not hold the attribute.
        """
        element = dataset.get(tag, None)
        return default if element is None else element.value

    @staticmethod
    def get_optional_sequence(dataset: Dataset, tag: BaseTag) -> DicomSequence | None:
        """The items of a sequence attribute of a dataset, or None if the dataset
        does not hold the attribute.

        Tells an attribute that is not there from one that is there and holds no
        items. Use `get_sequence` where that difference does not matter.

        Parameters
        ----------
        dataset: Dataset
            Dataset to read the sequence from.
        tag: BaseTag
            Tag of the sequence attribute.
        """
        element = dataset.get(tag, None)
        if element is None:
            return None
        items: DicomSequence = element.value
        return items

    @classmethod
    def get_sequence(cls, dataset: Dataset, tag: BaseTag) -> DicomSequence:
        """The items of a sequence attribute of a dataset.

        Empty when the attribute holds no items and when it is not there at
        all: a sequence that is not there holds nothing, which is what most
        readers of one here make of it.

        Parameters
        ----------
        dataset: Dataset
            Dataset to read the sequence from.
        tag: BaseTag
            Tag of the sequence attribute.
        """
        items = cls.get_optional_sequence(dataset, tag)
        return DicomSequence() if items is None else items

    @classmethod
    def get_sequence_item(cls, dataset: Dataset, tag: BaseTag) -> Dataset | None:
        """The first item of a sequence attribute of a dataset, or None if the
        dataset does not hold the attribute or holds it with no items.

        For the sequences that are defined to hold exactly one item, which is
        every one read through here. How many items a sequence may hold is said
        by the module that defines it and not by the data dictionary, which
        gives every sequence attribute a value multiplicity of one, so this is
        not something that can be derived from the tag.

        A dataset that states more than one item is read for its first, which is
        what it means if it means anything, and logged rather than rejected.

        Parameters
        ----------
        dataset: Dataset
            Dataset to read the sequence from.
        tag: BaseTag
            Tag of the sequence attribute.
        """
        items = cls.get_sequence(dataset, tag)
        if len(items) == 0:
            return None
        if len(items) > 1:
            logger.warning(
                f"{keyword_for_tag(tag)} holds {len(items)} items, where it is "
                "defined to hold one. Reading the first."
            )
        return items[0]

    @classmethod
    def _update_dataset(
        cls, dataset: Dataset, update: Mapping[str | BaseTag, Any]
    ) -> None:
        """Set attributes of a dataset, checking each value as it is set.

        For a dataset being built rather than one being changed, and for the
        datasets held in a sequence: making the element of a sequence checks
        that the value is a sequence and nothing of what the items of it hold,
        so what is put in an item is checked as it is put there.

        Parameters
        ----------
        dataset: Dataset
            Dataset to set the attributes of.
        update: Mapping[str | BaseTag, Any]
            Attributes to set, by keyword or by tag.

        Raises
        ------
        ValueError
            If a tag is not a tag or the keyword of one, or if a value does not
            conform to its value representation.
        """
        for tag, value in update.items():
            if isinstance(value, DataElement):
                if value.tag != Tag(tag):
                    raise ValueError(
                        f"{value.tag} was given for {Tag(tag)}, and an element "
                        "cannot be set under a tag other than its own."
                    )
            else:
                value = cls._create_data_element(tag, value)
            dataset.add(value)

    def overlay(self, dataset: Dataset) -> "WsiDataset":
        """A copy of this with the attributes of `dataset` set on it.

        The elements are carried over as they are, keeping the value
        representation each was made with, which the dictionary cannot always
        give back: an attribute written as one of two is settled by whoever
        made the element and not by the tag.

        Parameters
        ----------
        dataset: Dataset
            Dataset whose attributes to set on the copy.

        Returns
        -------
        WsiDataset
            Copy of this with the attributes of `dataset` set on it.

        Raises
        ------
        ValueError
            If a value does not conform to its value representation.
        """
        return self.replace({element.tag: element for element in dataset})

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WsiDataset):
            return self._dataset == other.as_dataset()
        return self._dataset == other

    @property
    def frame_positions(self) -> PerFrameGroupPositions:
        """Tile position of every frame, in frame order.

        The same either way: the positions the reader took out of the bytes of the per
        frame functional groups sequence, or the sequence when it is in the dataset.
        Only what a frame states is reported, so frames that carry no z offset or
        optical path identifier have none here either.

        Returns
        -------
        PerFrameGroupPositions
            Position of every frame.
        """
        if self._frame_positions is not None:
            return self._frame_positions
        return self._parse_frame_positions()

    def _parse_frame_positions(self) -> PerFrameGroupPositions:
        """Return the position of every frame, parsed from the per frame groups.

        A frame need not carry a z offset or an optical path identifier. If no frame
        carries one, there is no sequence of them and what the instance states applies
        to every frame; if only some do, the file is refused.

        Returns
        -------
        PerFrameGroupPositions
            Position of every frame.

        Raises
        ------
        WsiDicomError
            If the per frame functional groups do not state the tile positions, or if
            only some frames state a z offset or an optical path identifier.
        """
        if not self._has_parsed_per_frame_positions:
            raise WsiDicomError(
                "The per frame functional groups of this instance do not state where "
                "its frames sit, or there are none. A sparse tiled image is required "
                "to give every frame a Plane Position (Slide)."
            )
        columns: list[int] = []
        rows: list[int] = []
        z_offsets: list[float] = []
        identifiers: list[str] = []
        sequence = self.get_sequence(self._dataset, PerFrameFunctionalGroupsSequenceTag)
        for frame in sequence:
            position: Dataset = frame[PlanePositionSlideSequenceTag][0]
            columns.append(
                int(position[ColumnPositionInTotalImagePixelMatrixTag].value)
            )
            rows.append(int(position[RowPositionInTotalImagePixelMatrixTag].value))
            z_offset = position.get(ZOffsetInSlideCoordinateSystemTag, None)
            if z_offset is not None:
                z_offsets.append(float(z_offset.value))
            optical_paths = self.get_sequence(
                frame, OpticalPathIdentificationSequenceTag
            )
            if len(optical_paths) > 1:
                raise WsiDicomError(
                    f"A frame states {len(optical_paths)} optical path identifiers, "
                    "where Optical Path Identification Sequence holds a single item."
                )
            identifier = (
                None
                if len(optical_paths) == 0
                else optical_paths[0].get(OpticalPathIdentifierTag, None)
            )
            if identifier is not None:
                identifiers.append(str(identifier.value))

        frame_count = len(columns)
        for element, values in (
            ("z offset", z_offsets),
            ("optical path identifier", identifiers),
        ):
            if len(values) not in (0, frame_count):
                raise WsiDicomError(
                    f"{len(values)} of {frame_count} frames state a {element}. A per "
                    "frame functional group macro is in every frame or in none of "
                    "them, so there is no reading of this that is not a guess."
                )
        return PerFrameGroupPositions(
            columns=np.asarray(columns, dtype=np.int64),
            rows=np.asarray(rows, dtype=np.int64),
            z_offsets=(
                np.asarray(z_offsets, dtype=np.float64) if len(z_offsets) > 0 else None
            ),
            optical_path_identifiers=(
                np.asarray(identifiers, dtype=np.str_) if len(identifiers) > 0 else None
            ),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of dataset {self.uids.instance}"

    @property
    def instance_uid(self) -> UID:
        """The SOP instance uid of this instance.

        On its own rather than through `uids`, which needs the study and series
        uids as well and so asks more of the dataset than a caller that wants
        only this one.
        """
        return UID(self._dataset.SOPInstanceUID)

    @cached_property
    def uids(self) -> FileUids:
        """Return UIDs from dataset.

        Returns
        -------
        FileUids
            Found UIDs from dataset.
        """
        instance_uid = UID(self._dataset.SOPInstanceUID)
        concatenation_uid = self.get_value(
            self._dataset, SOPInstanceUIDOfConcatenationSourceTag
        )
        frame_of_reference_uid = self.get_value(self._dataset, FrameOfReferenceUIDTag)

        slide_uids = SlideUids(
            self._dataset.StudyInstanceUID,
            self._dataset.SeriesInstanceUID,
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
            return int(self._dataset.ConcatenationFrameOffsetNumber)
        except AttributeError:
            raise WsiDicomError(
                "Concatenated file missing concatenation frame offsetnumber"
            ) from None

    @property
    def frame_count(self) -> int:
        """Return number of frames in instance."""
        return int(self.get_value(self._dataset, NumberOfFramesTag, 1))

    @cached_property
    def tile_type(self) -> TileType:
        """Return tiling type of dataset. Raises WsiDicomError if type
        is undetermined.

        Returns
        -------
        TileType
            Tiling type
        """
        tile_type = self.get_value(
            self._dataset, DimensionOrganizationTypeTag, "TILED_SPARSE"
        )
        if tile_type == "TILED_FULL":
            # By the standard it should be tiled full.
            return TileType.FULL
        if self._has_per_frame_positions:
            # If no per frame functional sequence we can't make a sparse tile index.
            return TileType.SPARSE
        if self.image_type == ImageType.LABEL:
            # Labels are expected to only have one frame and can be treated as tiled
            # full.
            return TileType.FULL
        number_of_focal_planes = self.get_value(
            self._dataset, TotalPixelMatrixFocalPlanesTag, 1
        )
        number_of_optical_paths = self.get_value(
            self._dataset, NumberOfOpticalPathsTag, 1
        )
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
        shared = self.get_sequence_item(
            self._dataset, SharedFunctionalGroupsSequenceTag
        )
        if shared is None:
            return None
        return self.get_sequence_item(shared, PixelMeasuresSequenceTag)

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
        pixel_spacing_values = self.get_value(self.pixel_measure, PixelSpacingTag)
        if pixel_spacing_values is not None:
            if any([spacing <= 0 for spacing in pixel_spacing_values]):
                logger.warning(f"Pixel spacing not positive, {pixel_spacing_values}")
                return None
            return SizeMm(pixel_spacing_values[1], pixel_spacing_values[0])
        return None

    @cached_property
    def spacing_between_slices(self) -> float | None:
        """Return spacing between slices."""
        if self.pixel_measure is None:
            return None
        return self.get_value(self.pixel_measure, SpacingBetweenSlicesTag)

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
        return str(self.get_value(self._dataset, FocusMethodTag, "AUTO"))

    @cached_property
    def image_size(self) -> Size:
        """Read total pixel size from dataset.

        Returns
        -------
        Size
            The image size
        """
        image_size = Size(
            self._dataset.TotalPixelMatrixColumns, self._dataset.TotalPixelMatrixRows
        )
        if image_size.width <= 0 or image_size.height <= 0:
            raise WsiDicomError("Image size is zero")
        if self.tile_type == TileType.FULL and self.uids.concatenation is None:
            # Check that the number of frames match the image size and tile size.
            # Dont check concatenated instances as the frame count is ambiguous.
            expected_tiled_size = image_size.ceil_div(self.tile_size)
            number_of_focal_planes = self.get_value(
                self._dataset, TotalPixelMatrixFocalPlanesTag, 1
            )
            number_of_optical_paths = self.get_value(
                self._dataset, NumberOfOpticalPathsTag, 1
            )
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
                logger.warning(error)
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
        mm_width = self.get_value(self._dataset, ImagedVolumeWidthTag)
        mm_height = self.get_value(self._dataset, ImagedVolumeHeightTag)
        if mm_width is None or mm_height is None:
            mm_size = None
        else:
            mm_size = SizeMm(mm_width, mm_height)
        return mm_size

    @cached_property
    def mm_depth(self) -> float | None:
        """Return depth of image in mm."""
        return self.get_value(self._dataset, ImagedVolumeDepthTag)

    @cached_property
    def tile_size(self) -> Size:
        """Read tile size from from dataset.

        Returns
        -------
        Size
            The tile size
        """
        return Size(self._dataset.Columns, self._dataset.Rows)

    @property
    def samples_per_pixel(self) -> int:
        """Return samples per pixel (3 for RGB)."""
        return self._dataset.SamplesPerPixel

    @property
    def bits(self) -> int:
        """Return the number of bits stored for each sample."""
        return self._dataset.BitsStored

    @property
    def lossy_compressed(self) -> bool:
        """Return true if image has been lossy compressed."""
        lossy = self._dataset.get(LossyImageCompressionTag, None)
        return lossy is not None and lossy.value == "01"

    @cached_property
    def pyramid_metadata(self) -> Pyramid:
        """The pyramid metadata this dataset states.

        Held once worked out, as the other readers here are: deserialising the
        dataset through the metadata schema is more than an attribute lookup,
        and a dataset does not change once made.
        """
        return PyramidDicomSchema().load(self._dataset)

    @cached_property
    def overview_metadata(self) -> Overview:
        """The overview metadata this dataset states."""
        return OverviewDicomSchema().load(self._dataset)

    @cached_property
    def image_coordinate_system(self) -> ImageCoordinateSystem | None:
        """Where on the slide this image sits, or None if it does not say."""
        try:
            return ImageCoordinateSystemDicomSchema().load(self._dataset)
        except TypeError:
            return None

    @cached_property
    def label_metadata(self) -> Label:
        """The label metadata this dataset states."""
        if self.image_type == ImageType.LABEL:
            return LabelDicomSchema().load(self._dataset)
        return LabelBaseDicomSchema().load(self._dataset)

    @cached_property
    def base_metadata(self) -> BaseWsiMetadata:
        """The study, series, patient, equipment and slide metadata this states."""
        return BaseWsiMetadataDicomSchema().load(self._dataset)

    @property
    def lossy_compressions(self) -> list[LossyCompression] | None:
        """The lossy compressions the image has been through, in order.

        None if it has not been lossy compressed. Each step pairs the method it
        was compressed with and the ratio it reached.
        """
        if not self.lossy_compressed:
            return None
        methods = [
            LossyCompressionIsoStandard(value)
            for value in self._get_multi_value(LossyImageCompressionMethodTag)
        ]
        ratios = [
            float(value)
            for value in self._get_multi_value(LossyImageCompressionRatioTag)
        ]
        return [
            LossyCompression(method, ratio)
            for method, ratio in zip(methods, ratios, strict=False)
        ]

    @property
    def lossy_compression_ratios(self) -> list[Any]:
        """The compression ratios as they stand in the dataset, one per step.

        Given as they are written rather than as numbers: the writer reserves a
        blank of the full width for the ratio of the step it is about to write
        and patches the value in once the size is known, so the width of what is
        here matters.
        """
        return self._get_multi_value(LossyImageCompressionRatioTag)

    @property
    def lossy_compression_methods(self) -> list[Any]:
        """The compression methods as they stand in the dataset, one per step.

        One method for each ratio in `lossy_compression_ratios`, in the same
        order: the step that compressed the image data by that ratio.
        """
        return self._get_multi_value(LossyImageCompressionMethodTag)

    @property
    def photometric_interpretation(self) -> str:
        """Return photometric interpretation."""
        return self._dataset.PhotometricInterpretation

    @cached_property
    def optical_path_sequence(self) -> DicomSequence | None:
        """Return optical path sequence from dataset."""
        return self.get_optional_sequence(self._dataset, OpticalPathSequenceTag)

    @cached_property
    def _shared_functional_group(self) -> Dataset | None:
        """The item of the Shared Functional Groups Sequence, if the instance has one.

        By the standard a functional group macro is stated in either the shared groups
        or the per frame groups and not both, so for a sparse tiled image the tile
        positions and the optical path are per frame. A file that instead states a
        value that is the same for every frame once in the shared groups is not
        following that, but what it means is unambiguous, so it is read. Only frames
        that state no value of their own are answered from here, which leaves a file
        that does follow the standard reading exactly as before.
        """
        return self.get_sequence_item(self._dataset, SharedFunctionalGroupsSequenceTag)

    @cached_property
    def z_offset(self) -> float:
        """The z offset in the slide coordinate system that applies to the image.

        Stated by the Total pixel matrix origin sequence, and when that states
        none, by the plane position of the shared functional groups.
        `DEFAULT_Z_OFFSET` when neither states one.
        """
        origin = self.get_sequence_item(
            self._dataset, TotalPixelMatrixOriginSequenceTag
        )
        if origin is not None:
            z_offset = origin.get(ZOffsetInSlideCoordinateSystemTag, None)
            if z_offset is not None:
                return float(z_offset.value)
        return self.read_z_offset()

    @property
    def number_of_focal_planes(self) -> int:
        """Return number of focal planes in image."""
        return self.get_value(self._dataset, TotalPixelMatrixFocalPlanesTag, 1)

    @property
    def slice_thickness(self) -> float | None:
        """Return slice thickness spacing from pixel measure dataset.

        Returns
        -------
        float | None
            Slice thickness or None if unknown.
        """
        if self.pixel_measure is not None:
            slice_thickness: float | None = self.get_value(
                self.pixel_measure, SliceThicknessTag
            )
            if slice_thickness is not None:
                return slice_thickness
        if self.mm_depth is not None:
            number_of_focal_planes = int(
                self.get_value(self._dataset, TotalPixelMatrixFocalPlanesTag, 1)
            )
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
        return self._get_image_type(self._dataset.ImageType)

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
            instance_uid = dataset.instance_uid
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
                self.get_sequence(self._dataset, TotalPixelMatrixOriginSequenceTag)
                == self.get_sequence(
                    other_dataset.as_dataset(), TotalPixelMatrixOriginSequenceTag
                )
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

    def read_optical_path_identifier(self, frame: Dataset | None = None) -> str:
        """Return the optical path identifier that applies to `frame`.

        Parameters
        ----------
        frame: Dataset | None = None
            Per frame functional group item, or None to ask what applies to a frame
            that states no identifier of its own.

        Returns
        -------
        str
            Optical path identifier, "0" if neither the frame, the shared groups nor
            the optical paths of the instance name one.
        """
        for source in (frame, self._shared_functional_group):
            if source is None:
                continue
            identifier = self._optical_path_identifier_of(
                self.get_sequence(source, OpticalPathIdentificationSequenceTag)
            )
            if identifier is not None:
                return identifier
        identifier = self._optical_path_identifier_of(self.optical_path_sequence)
        return "0" if identifier is None else identifier

    def read_z_offset(self, frame: Dataset | None = None) -> float:
        """Return the z offset in the slide coordinate system that applies to `frame`.

        Parameters
        ----------
        frame: Dataset | None = None
            Per frame functional group item, or None to ask what applies to a frame
            that states no offset of its own.

        Returns
        -------
        float
            Z offset, `DEFAULT_Z_OFFSET` if neither the frame nor the shared groups
            state one.
        """
        for source in (frame, self._shared_functional_group):
            if source is None:
                continue
            position = self.get_sequence(source, PlanePositionSlideSequenceTag)
            if len(position) == 0:
                continue
            z_offset = position[0].get(ZOffsetInSlideCoordinateSystemTag, None)
            if z_offset is not None:
                return float(z_offset.value)
        return self.DEFAULT_Z_OFFSET

    @staticmethod
    def _optical_path_identifier_of(sequence: DicomSequence | None) -> str | None:
        """Return the optical path identifier the first item of `sequence` names.

        Parameters
        ----------
        sequence: DicomSequence | None
            Optical Path Identification Sequence or Optical Path Sequence, or None.

        Returns
        -------
        str | None
            The identifier, or None if the sequence is missing, empty, or does not
            name one.
        """
        if sequence is None or len(sequence) == 0:
            return None
        identifier = sequence[0].get(OpticalPathIdentifierTag, None)
        return None if identifier is None else str(identifier.value)

    def _get_multi_value(self, tag: BaseTag) -> list[Any]:
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
        element = self._dataset.get(tag)
        if element is None:
            return []
        vm = getattr(element, "VM", 1)
        if vm > 1 or isinstance(element, MultiValue):
            return [value for value in element]
        return [element.value]

    def as_instance(
        self,
        instance_uid: UID,
        instance_number: int,
        frame_count: int | None = None,
        concatenation: ConcatenationPart | None = None,
    ) -> "WsiDataset":
        """A copy of this identified as one instance being written.

        Parameters
        ----------
        instance_uid: UID
            SOP Instance UID to write the instance as.
        instance_number: int
            Instance Number to write the instance as.
        frame_count: int | None = None
            Number of frames the instance holds, when that differs from the
            number this states.
        concatenation: ConcatenationPart | None = None
            Where the instance sits in the concatenation its level is written
            as, when the level is written as one.

        Returns
        -------
        WsiDataset
            Copy of this identified as the instance described.
        """
        changes: dict[str | BaseTag, Any] = {
            SOPInstanceUIDTag: instance_uid,
            InstanceNumberTag: instance_number,
        }
        if frame_count is not None:
            changes[NumberOfFramesTag] = frame_count
        if concatenation is not None:
            changes[ConcatenationUIDTag] = concatenation.uid
            changes[SOPInstanceUIDOfConcatenationSourceTag] = (
                concatenation.source_instance_uid
            )
            changes[InConcatenationNumberTag] = concatenation.number
            changes[ConcatenationFrameOffsetNumberTag] = concatenation.frame_offset
            if concatenation.total is not None:
                changes[InConcatenationTotalNumberTag] = concatenation.total
        return self.replace(changes)

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
        # The changes are collected and made in one go at the end, so that
        # what is read below is the dataset as it came rather than a dataset
        # part way through being changed.
        changes: dict[str | BaseTag, Any] = {DimensionOrganizationTypeTag: "TILED_FULL"}
        # Make a new Shared functional group sequence and Pixel measure
        # sequence if not in dataset, otherwise update the Pixel measure
        # sequence
        shared_functional_group = self.get_sequence(
            dataset, SharedFunctionalGroupsSequenceTag
        ) or DicomSequence([Dataset()])
        pixel_measure = self.get_sequence(
            shared_functional_group[0], PixelMeasuresSequenceTag
        ) or DicomSequence([Dataset()])
        # What goes in the item of a sequence is checked as it is put there:
        # making the element of the sequence checks that the value is one, and
        # nothing of what the items of it hold.
        measures: dict[str | BaseTag, Any] = {}
        if self.pixel_spacing is not None:
            measures[PixelSpacingTag] = [
                DSfloat(self.pixel_spacing.height * scale, True),
                DSfloat(self.pixel_spacing.width * scale, True),
            ]
        focal_plane_spacing = self._get_spacing_between_slices_for_focal_planes(
            focal_planes
        )
        if focal_plane_spacing is not None:
            measures[SpacingBetweenSlicesTag] = DSfloat(focal_plane_spacing, True)
        elif SpacingBetweenSlicesTag in pixel_measure[0]:
            # A single focal plane has no spacing; drop any spacing carried over
            # from a multi-plane source that has been split per focal plane.
            del pixel_measure[0].SpacingBetweenSlices

        if self.slice_thickness is not None:
            measures[SliceThicknessTag] = DSfloat(self.slice_thickness, True)
        self._update_dataset(pixel_measure[0], measures)

        self._update_dataset(
            shared_functional_group[0], {PixelMeasuresSequenceTag: pixel_measure}
        )
        changes[SharedFunctionalGroupsSequenceTag] = shared_functional_group

        # The raw attributes rather than `image_size`, which checks the frame
        # count against the tile size: the frame count written below is for the
        # tiled full arrangement and does not add up against the size read here.
        changes[TotalPixelMatrixColumnsTag] = max(
            math.ceil(dataset.TotalPixelMatrixColumns / scale), 1
        )
        changes[TotalPixelMatrixRowsTag] = max(
            math.ceil(dataset.TotalPixelMatrixRows / scale), 1
        )
        changes[TotalPixelMatrixFocalPlanesTag] = len(focal_planes)
        changes[NumberOfOpticalPathsTag] = len(optical_paths)
        changes[NumberOfFramesTag] = (
            max(tiled_size.ceil_div(scale).area, 1)
            * len(focal_planes)
            * len(optical_paths)
        )

        # Keep only the optical paths written to this instance, so the optical
        # path identity is preserved when the paths have been split across
        # instances (e.g. one instance per optical path).
        kept_optical_paths = DicomSequence(
            item
            for item in self.get_sequence(self._dataset, OpticalPathSequenceTag)
            if str(item[OpticalPathIdentifierTag].value) in optical_paths
        )
        if len(kept_optical_paths) != 0:
            changes[OpticalPathSequenceTag] = kept_optical_paths

        # Encode the focal plane origin (z) so the planes can be reconstructed
        # on read as ``z_offset + index * spacing``. This is the source z offset
        # for the full set of planes, but differs when the planes have been
        # split across instances (e.g. one instance per focal plane). Preserve
        # the in-plane (x, y) origin when present. Only create a new origin
        # sequence when there is a non-zero z to encode, since x and y are
        # required in the sequence item; a zero z is the default on read.
        focal_plane_origin = self.get_sequence_item(
            dataset, TotalPixelMatrixOriginSequenceTag
        )
        if focal_plane_origin is not None:
            self._update_dataset(
                focal_plane_origin,
                {ZOffsetInSlideCoordinateSystemTag: DSfloat(focal_planes[0], True)},
            )
        elif focal_planes[0] != 0.0:
            origin_item = Dataset()
            self._update_dataset(
                origin_item,
                {
                    XOffsetInSlideCoordinateSystemTag: DSfloat(0.0, True),
                    YOffsetInSlideCoordinateSystemTag: DSfloat(0.0, True),
                    ZOffsetInSlideCoordinateSystemTag: DSfloat(focal_planes[0], True),
                },
            )
            changes[TotalPixelMatrixOriginSequenceTag] = DicomSequence([origin_item])

        return WsiDataset(dataset).replace(changes)

    def update_for_transcoding(self, transcoder: Encoder, scale: int) -> "WsiDataset":
        """A copy of this with the metadata of the transcoded image data.

        Parameters
        ----------
        transcoder: Encoder
            Encoder being used for transcoding.
        scale: int
            Scale factor applied to the image data.

        Returns
        -------
        WsiDataset
            Copy of this describing the transcoded image data.
        """
        changes: dict[str | BaseTag, Any] = {
            PhotometricInterpretationTag: transcoder.photometric_interpretation
        }
        if transcoder.lossy_method:
            changes[LossyImageCompressionTag] = "01"
            ratios = self._get_multi_value(LossyImageCompressionRatioTag)
            methods = self._get_multi_value(LossyImageCompressionMethodTag)
            if scale != 1:
                ratios.clear()
                methods.clear()
            ratios.append(" " * MAX_VALUE_LEN["DS"])
            methods.append(transcoder.lossy_method.value)
            changes[LossyImageCompressionRatioTag] = ratios
            changes[LossyImageCompressionMethodTag] = methods
        return self.replace(changes)

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
        image_type_value = [
            original_or_derived,
            "PRIMARY",
            image_type.value,
            resampled,
        ]
        cls._update_dataset(
            dataset,
            {
                ImageTypeTag: image_type_value,
                SOPInstanceUIDTag: generate_uid(prefix=None),
            },
        )
        shared_functional_group_sequence = Dataset()
        if image_data.pixel_spacing is None:
            if image_type == ImageType.VOLUME:
                raise ValueError(
                    "Image flavor 'VOLUME' requires pixel spacing to be set"
                )
        else:
            # DICOM 2022a part 3 IODs - C.8.12.4.1.2 Imaged Volume Width,
            # Height, Depth. Depth must not be 0. Default to 0.5 microns
            slice_thickness = 0.0005
            measures: dict[str | BaseTag, Any] = {
                PixelSpacingTag: [
                    DSfloat(image_data.pixel_spacing.height, True),
                    DSfloat(image_data.pixel_spacing.width, True),
                ],
                SliceThicknessTag: DSfloat(slice_thickness, True),
            }
            focal_plane_spacing = cls._get_spacing_between_slices_for_focal_planes(
                image_data.focal_planes
            )
            if focal_plane_spacing is not None:
                measures[SpacingBetweenSlicesTag] = DSfloat(focal_plane_spacing, True)
            pixel_measure_sequence = Dataset()
            cls._update_dataset(pixel_measure_sequence, measures)
            cls._update_dataset(
                shared_functional_group_sequence,
                {PixelMeasuresSequenceTag: DicomSequence([pixel_measure_sequence])},
            )
            if image_data.imaged_size is None:
                imaged_width = (
                    image_data.image_size.width * image_data.pixel_spacing.width
                )
                imaged_height = (
                    image_data.image_size.height * image_data.pixel_spacing.height
                )
            else:
                imaged_width = image_data.imaged_size.width
                imaged_height = image_data.imaged_size.height
            cls._update_dataset(
                dataset,
                {
                    SharedFunctionalGroupsSequenceTag: DicomSequence(
                        [shared_functional_group_sequence]
                    ),
                    ImagedVolumeWidthTag: imaged_width,
                    ImagedVolumeHeightTag: imaged_height,
                    # SliceThickness is in mm, ImagedVolumeDepth in um
                    ImagedVolumeDepthTag: DSfloat(slice_thickness * 1000, True),
                },
            )

        # DICOM 2022a part 3 IODs - C.8.12.9 Whole Slide Microscopy Image Frame Type
        # Macro. Analogous to ImageType and shared by all frames so clone
        wsi_frame_type_item = Dataset()
        cls._update_dataset(wsi_frame_type_item, {FrameTypeTag: image_type_value})
        cls._update_dataset(
            shared_functional_group_sequence,
            {
                WholeSlideMicroscopyImageFrameTypeSequenceTag: DicomSequence(
                    [wsi_frame_type_item]
                )
            },
        )
        cls._update_dataset(
            dataset,
            {
                SharedFunctionalGroupsSequenceTag: DicomSequence(
                    [shared_functional_group_sequence]
                )
            },
        )

        if image_data.image_coordinate_system is not None:
            offset: dict[str | BaseTag, Any] = {
                XOffsetInSlideCoordinateSystemTag: DSfloat(
                    image_data.image_coordinate_system.origin.x, True
                ),
                YOffsetInSlideCoordinateSystemTag: DSfloat(
                    image_data.image_coordinate_system.origin.y, True
                ),
            }
            if image_data.image_coordinate_system.z_offset is not None:
                offset[ZOffsetInSlideCoordinateSystemTag] = DSfloat(
                    image_data.image_coordinate_system.z_offset, True
                )
            offset_item = Dataset()
            cls._update_dataset(offset_item, offset)
            cls._update_dataset(
                dataset,
                {
                    ImageOrientationSlideTag: [
                        DSfloat(value, True)
                        for value in (
                            image_data.image_coordinate_system.orientation.values
                        )
                    ],
                    TotalPixelMatrixOriginSequenceTag: DicomSequence([offset_item]),
                },
            )

        written: dict[str | BaseTag, Any] = {
            DimensionOrganizationTypeTag: "TILED_FULL",
            TotalPixelMatrixColumnsTag: image_data.image_size.width,
            TotalPixelMatrixRowsTag: image_data.image_size.height,
            ColumnsTag: image_data.tile_size.width,
            RowsTag: image_data.tile_size.height,
            NumberOfFramesTag: (
                image_data.tiled_size.area
                * len(image_data.focal_planes)
                * len(image_data.optical_paths)
            ),
            BitsAllocatedTag: image_data.bits // 8 * 8,
            BitsStoredTag: image_data.bits,
            HighBitTag: image_data.bits - 1,
            PixelRepresentationTag: 0,
            PhotometricInterpretationTag: image_data.photometric_interpretation,
            SamplesPerPixelTag: image_data.samples_per_pixel,
            FocusMethodTag: "AUTO",
            ExtendedDepthOfFieldTag: "NO",
        }
        if image_data.lossy_compression:
            written[LossyImageCompressionTag] = "01"
            written[LossyImageCompressionRatioTag] = [
                DSfloat(item.ratio, auto_format=True)
                for item in image_data.lossy_compression
            ]
            written[LossyImageCompressionMethodTag] = [
                item.method.value for item in image_data.lossy_compression
            ]
        else:
            written[LossyImageCompressionTag] = "00"
        if image_data.samples_per_pixel == 3:
            written[PlanarConfigurationTag] = 0
        cls._update_dataset(dataset, written)
        return WsiDataset(dataset)

    def _copy_without_per_frame(self) -> Dataset:
        """Copy dataset excluding PerFrameFunctionalGroupsSequence.

        A plain dataset rather than a WSI dataset: the copy is made in order to
        change it, and what is read from it while it is being changed is read
        as DICOM attributes rather than through the readers here, which cache
        what they work out and would go stale as it is changed.
        """
        dataset = Dataset()
        dataset.update(
            deepcopy(
                {
                    tag: elem
                    for tag, elem in self._dataset.items()
                    if tag != PerFrameFunctionalGroupsSequenceTag
                }
            )
        )
        return dataset

    @property
    def _has_per_frame_positions(self) -> bool:
        """Whether the frames carry per-frame tile positions, the marker of a sparse,
        explicitly-positioned image. This is the condition a sparse tile index needs,
        so :func:`tile_type` gates on it.

        Positions from the reader answer the same question: they only exist when it
        found a tile position for every frame, and then the sequence itself is not in
        the dataset.
        """
        return self._frame_positions is not None or self._has_parsed_per_frame_positions

    @property
    def _has_parsed_per_frame_positions(self) -> bool:
        """Whether the per frame functional groups sequence is in the dataset and its
        items carry tile positions (PlanePositionSlideSequence). Checks the first frame
        as representative."""
        frames = self.get_sequence(self._dataset, PerFrameFunctionalGroupsSequenceTag)
        return len(frames) > 0 and PlanePositionSlideSequenceTag in frames[0]

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
        if self.get_value(self._dataset, ExtendedDepthOfFieldTag, "NO") != "YES":
            return False, None, None

        planes = self.get_value(self._dataset, NumberOfFocalPlanesTag, 1)
        distance = self.get_value(self._dataset, DistanceBetweenFocalPlanesTag, 0.0)
        if planes is None or distance is None:
            raise WsiDicomFileError(
                str(self.uids.instance),
                "Missing NumberOfFocalPlanes or DistanceBetweenFocalPlanes",
            )
        return True, planes, distance

    @classmethod
    def focal_planes_equally_spaced(cls, focal_planes: Sequence[float]) -> bool:
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
            cls._get_spacing_between_slices_for_focal_planes(focal_planes)
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
