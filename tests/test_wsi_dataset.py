import logging
from collections.abc import Sequence

import pytest
from pydicom import Dataset
from pydicom.dataelem import DataElement
from pydicom.sequence import Sequence as DicomSequence
from pydicom.tag import BaseTag, Tag
from pydicom.uid import UID, CTImageStorage, generate_uid

from tests.data_gen import create_main_dataset
from wsidicom.config import Settings, use_settings
from wsidicom.errors import WsiDicomError
from wsidicom.geometry import Size, SizeMm
from wsidicom.instance import ImageData, TileType
from wsidicom.instance.dataset import WsiDataset
from wsidicom.metadata import ImageType
from wsidicom.options import DicomValueValidationOption
from wsidicom.tags import (
    LossyImageCompressionRatioTag,
    SharedFunctionalGroupsSequenceTag,
    SOPInstanceUIDTag,
    TotalPixelMatrixOriginSequenceTag,
)


@pytest.fixture
def concatenation():
    yield False


@pytest.fixture
def instance_uid():
    return generate_uid()


@pytest.fixture
def concatenation_uid():
    return generate_uid()


@pytest.fixture
def frame_of_reference_uid():
    return generate_uid()


@pytest.fixture
def study_instance_uid():
    return generate_uid()


@pytest.fixture
def series_instance_uid():
    return generate_uid()


@pytest.fixture
def dataset(
    instance_uid: UID,
    concatenation_uid: UID,
    frame_of_reference_uid: UID,
    study_instance_uid: UID,
    series_instance_uid: UID,
    concatenation: bool,
):
    dataset = Dataset()
    dataset.SOPInstanceUID = instance_uid
    if concatenation:
        dataset.SOPInstanceUIDOfConcatenationSource = concatenation_uid
    dataset.FrameOfReferenceUID = frame_of_reference_uid
    dataset.StudyInstanceUID = study_instance_uid
    dataset.SeriesInstanceUID = series_instance_uid
    dataset.ImageType = ["DERIVED", "PRIMARY", "VOLUME", "RESAMPLED"]
    yield WsiDataset(dataset)


@pytest.fixture
def pixel_spacing():
    yield SizeMm(0.1, 0.2)


@pytest.fixture
def spacing_between_slices():
    yield 0.1


@pytest.fixture
def pixel_measure(pixel_spacing: SizeMm | None, spacing_between_slices: float | None):
    pixel_measure = Dataset()
    if pixel_spacing is not None:
        pixel_measure.PixelSpacing = [pixel_spacing.height, pixel_spacing.width]
    if spacing_between_slices is not None:
        pixel_measure.SpacingBetweenSlices = spacing_between_slices
    yield pixel_measure


@pytest.fixture
def shared_functional_group(pixel_measure: Dataset):
    shared_functional_group = Dataset()
    shared_functional_group.PixelMeasuresSequence = [pixel_measure]
    yield shared_functional_group


def _optical_path_identification(identifier: str) -> DicomSequence:
    """Return an Optical Path Identification Sequence naming `identifier`."""
    item = Dataset()
    item.OpticalPathIdentifier = identifier
    return DicomSequence([item])


def _plane_position(z_offset: float) -> DicomSequence:
    """Return a Plane Position (Slide) Sequence at `z_offset`."""
    item = Dataset()
    item.ZOffsetInSlideCoordinateSystem = z_offset
    return DicomSequence([item])


def _per_frame_groups(
    z_offsets: Sequence[float | None] = (None, None),
    identifiers: Sequence[str | None] = (None, None),
) -> DicomSequence:
    """Return per frame functional groups for frames at the given z and optical path.

    Every frame states a tile position, as a sparse tiled image is required to. A
    frame states a z offset or an optical path identifier only where one is given.
    """
    frames = DicomSequence()
    for z_offset, identifier in zip(z_offsets, identifiers, strict=True):
        position = Dataset()
        position.ColumnPositionInTotalImagePixelMatrix = 1
        position.RowPositionInTotalImagePixelMatrix = 1
        if z_offset is not None:
            position.ZOffsetInSlideCoordinateSystem = z_offset
        frame = Dataset()
        frame.PlanePositionSlideSequence = DicomSequence([position])
        if identifier is not None:
            optical_path = Dataset()
            optical_path.OpticalPathIdentifier = identifier
            frame.OpticalPathIdentificationSequence = DicomSequence([optical_path])
        frames.append(frame)
    return frames


@pytest.mark.unittest
class TestWsiDataset:
    @pytest.mark.parametrize(
        ["values", "expected_values"],
        [(None, []), ("1", ["1"]), (["1", "2"], ["1", "2"])],
    )
    def test_get_multi_value(
        self,
        dataset: WsiDataset,
        values: str | Sequence[str] | None,
        expected_values: Sequence[str],
    ):
        # Arrange
        if values is not None:
            dataset.as_dataset().add(
                DataElement(LossyImageCompressionRatioTag, "CS", values)
            )

        # Act
        read_values = dataset.lossy_compression_ratios

        # Assert
        assert read_values == expected_values

    @pytest.mark.parametrize("concatenation", [True, False])
    def test_uids(
        self,
        dataset: WsiDataset,
        instance_uid: UID,
        concatenation_uid: UID,
        frame_of_reference_uid: UID,
        study_instance_uid: UID,
        series_instance_uid: UID,
        concatenation: bool,
    ):
        # Arrange

        # Act
        uids = dataset.uids

        # Assert
        assert uids.instance == instance_uid
        if concatenation:
            assert uids.concatenation == concatenation_uid
            assert uids.identifier == concatenation_uid
        else:
            assert uids.concatenation is None
            assert uids.identifier == instance_uid
        assert uids.slide.frame_of_reference == frame_of_reference_uid
        assert uids.slide.study_instance == study_instance_uid
        assert uids.slide.series_instance == series_instance_uid

    @pytest.mark.parametrize(
        ["concatenation", "expected_frame_offset"], [(None, 0), (1, 1), (100, 100)]
    )
    def test_frame_offset(
        self,
        dataset: WsiDataset,
        concatenation: int | None,
        expected_frame_offset: int,
    ):
        # Arrange
        if concatenation is not None:
            dataset = dataset.replace({"ConcatenationFrameOffsetNumber": concatenation})

        # Act
        frame_offset = dataset.frame_offset

        # Assert
        assert frame_offset == expected_frame_offset

    @pytest.mark.parametrize(
        ["frame_count", "expected_frame_count"], [(None, 1), (1, 1), (100, 100)]
    )
    def test_frame_count(
        self, dataset: WsiDataset, frame_count: int | None, expected_frame_count: int
    ):
        # Arrange
        if frame_count is not None:
            dataset = dataset.replace({"NumberOfFrames": frame_count})

        # Act
        read_frame_count = dataset.frame_count

        # Assert
        assert read_frame_count == expected_frame_count

    def test_tile_type_tiled_full(self, dataset: WsiDataset):
        # Arrange
        dataset = dataset.replace({"DimensionOrganizationType": "TILED_FULL"})

        # Act
        read_tile_type = dataset.tile_type

        # Assert
        assert read_tile_type == TileType.FULL

    def test_tile_type_tiled_sparse(self, dataset: WsiDataset):
        # Arrange — a real sparse image has per-frame items carrying tile positions
        frame = Dataset()
        frame.PlanePositionSlideSequence = [Dataset()]
        dataset = dataset.replace({"PerFrameFunctionalGroupsSequence": [frame]})

        # Act
        read_tile_type = dataset.tile_type

        # Assert
        assert read_tile_type == TileType.SPARSE

    @pytest.mark.parametrize(
        "per_frame",
        [
            [],  # present but empty
            [Dataset()],  # present, non-empty, but no PlanePositionSlideSequence
        ],
        ids=["empty", "no-plane-position"],
    )
    def test_tile_type_per_frame_without_positions_is_not_sparse(
        self, dataset: WsiDataset, per_frame: list
    ):
        # Arrange — without per-frame tile positions a sparse index can't be built,
        # so it must fall through to the tiled-full heuristics (single frame here)
        # rather than be classed sparse and later fail reading frame positions.
        dataset = dataset.replace({"PerFrameFunctionalGroupsSequence": per_frame})

        # Act
        read_tile_type = dataset.tile_type

        # Assert
        assert read_tile_type == TileType.FULL

    def test_tile_type_label(self, dataset: WsiDataset):
        # Arrange
        dataset = dataset.replace(
            {"ImageType": ["DERIVED", "LABEL", "VOLUME", "RESAMPLED"]}
        )

        # Act
        read_tile_type = dataset.tile_type

        # Assert
        assert read_tile_type == TileType.FULL

    def test_tile_type_single_frame(self, dataset: WsiDataset):
        # Arrange
        dataset = dataset.replace(
            {
                "TotalPixelMatrixFocalPlanes": 1,
                "NumberOfOpticalPaths": 1,
                "NumberOfFrames": 1,
            }
        )

        # Act
        read_tile_type = dataset.tile_type

        # Assert
        assert read_tile_type == TileType.FULL

    def test_pixel_measure(
        self,
        dataset: WsiDataset,
        shared_functional_group: Dataset,
        pixel_measure: Dataset,
    ):
        # Arrange
        dataset = dataset.replace(
            {"SharedFunctionalGroupsSequence": [shared_functional_group]}
        )

        # Act
        read_pixel_measure = dataset.pixel_measure

        # Assert
        assert read_pixel_measure == pixel_measure

    @pytest.mark.parametrize("option", list(DicomValueValidationOption))
    def test_replace_checks_the_value_whatever_the_setting_says(
        self, dataset: WsiDataset, option: DicomValueValidationOption
    ):
        """What is written here is what wsidicom worked out, not what it was given.

        `Settings.dicom_value_validation` is there for the metadata a caller
        supplies. Turning it off must not stop wsidicom checking its own.
        """
        # Arrange
        # Series Description is LO, which allows 64
        settings = Settings(dicom_value_validation=option)

        # Act & Assert
        with use_settings(settings), pytest.raises(ValueError):
            dataset.replace({"SeriesDescription": "X" * 100})

    def test_overlay_keeps_the_value_representation_the_element_was_made_with(
        self, dataset: WsiDataset
    ):
        """The dictionary cannot settle a value representation written as one of two."""
        # Arrange
        # Smallest Image Pixel Value is written as US or SS, and this one as US
        source = Dataset()
        source.add(DataElement(Tag("SmallestImagePixelValue"), "US", 7))

        # Act
        overlaid = dataset.overlay(source).as_dataset()

        # Assert
        assert overlaid[Tag("SmallestImagePixelValue")].VR == "US"

    def test_replace_refuses_an_element_under_another_tag(self, dataset: WsiDataset):
        """An element is set under its own tag, so the two have to agree."""
        # Arrange
        element = DataElement(Tag("PatientName"), "PN", "Smith")

        # Act & Assert
        with pytest.raises(ValueError):
            dataset.replace({Tag("SeriesDescription"): element})

    def test_replace_refuses_a_value_for_an_unsettled_value_representation(
        self, dataset: WsiDataset
    ):
        """A raw value says nothing about which of the two to write it as."""
        # Arrange

        # Act & Assert
        with pytest.raises(ValueError):
            dataset.replace({"SmallestImagePixelValue": 7})

    def test_pixel_measure_reads_the_first_of_several_shared_groups(
        self,
        dataset: WsiDataset,
        shared_functional_group: Dataset,
        pixel_measure: Dataset,
        caplog: pytest.LogCaptureFixture,
    ):
        # Arrange
        # The shared functional groups sequence is defined to hold one item. A
        # dataset that states two is read for the first rather than rejected.
        second_group = Dataset()
        second_group.PixelMeasuresSequence = DicomSequence([Dataset()])
        dataset = dataset.replace(
            {
                SharedFunctionalGroupsSequenceTag: [
                    shared_functional_group,
                    second_group,
                ]
            }
        )

        # Act
        with caplog.at_level(logging.WARNING):
            read_pixel_measure = dataset.pixel_measure

        # Assert
        assert read_pixel_measure == pixel_measure
        assert "SharedFunctionalGroupsSequence holds 2 items" in caplog.text

    def test_z_offset_with_empty_origin_sequence_falls_back(
        self,
        dataset: WsiDataset,
    ):
        # Arrange
        # An origin sequence that is there but holds no item states no z
        # offset, and is read as though it were not there at all.
        dataset = dataset.replace({TotalPixelMatrixOriginSequenceTag: []})

        # Act
        z_offset = dataset.z_offset

        # Assert
        assert z_offset == WsiDataset.DEFAULT_Z_OFFSET

    @pytest.mark.parametrize(
        ["pixel_spacing", "expected_pixel_spacing"],
        [(None, None), (SizeMm(0, 0), None), (SizeMm(0.1, 0.2), SizeMm(0.1, 0.2))],
    )
    def test_pixel_spacing(
        self,
        dataset: WsiDataset,
        shared_functional_group: Dataset,
        expected_pixel_spacing: SizeMm | None,
    ):
        # Arrange
        dataset = dataset.replace(
            {"SharedFunctionalGroupsSequence": [shared_functional_group]}
        )

        # Act
        read_pixel_spacing = dataset.pixel_spacing

        # Assert
        assert read_pixel_spacing == expected_pixel_spacing

    @pytest.mark.parametrize("spacing_between_slices", [None, 0.1])
    def test_spacing_between_slices(
        self,
        dataset: WsiDataset,
        shared_functional_group: Dataset,
        spacing_between_slices: float | None,
    ):
        # Arrange
        dataset = dataset.replace(
            {"SharedFunctionalGroupsSequence": [shared_functional_group]}
        )

        # Act
        read_spacing_between_slices = dataset.spacing_between_slices

        # Assert
        assert read_spacing_between_slices == spacing_between_slices

    @pytest.mark.parametrize(
        ["image_type", "pyramid_index", "expected_image_type"],
        [
            (ImageType.VOLUME, 0, ["ORIGINAL", "PRIMARY", "VOLUME", "NONE"]),
            (ImageType.VOLUME, 1, ["DERIVED", "PRIMARY", "VOLUME", "RESAMPLED"]),
            (ImageType.LABEL, None, ["ORIGINAL", "PRIMARY", "LABEL", "NONE"]),
            (ImageType.OVERVIEW, None, ["ORIGINAL", "PRIMARY", "OVERVIEW", "NONE"]),
        ],
    )
    def test_create_instance_dataset(
        self,
        image_data: ImageData,
        image_type: ImageType,
        pyramid_index: int | None,
        expected_image_type: Sequence[str],
    ):
        # Arrange
        dataset = Dataset()

        # Act
        instance_dataset = WsiDataset.create_instance_dataset(
            dataset, image_type, image_data, pyramid_index
        )

        # Assert
        assert instance_dataset.as_dataset().ImageType == expected_image_type

    def test_is_supported_image_type_supported_volume_returns_true(self):
        # Arrange
        dataset = create_main_dataset()

        # Act
        is_supported_image_type = WsiDataset.is_supported_image_type(dataset)

        # Assert
        assert is_supported_image_type

    def test_is_supported_image_type_answers_from_the_attributes_read_first(self):
        """The check a reader makes before parsing the rest of a stream.

        It is asked of a dataset holding only what is ordered before SOP Instance
        UID, so it must not read an attribute ordered after it.
        """
        # Arrange
        source = create_main_dataset()
        dataset = Dataset()
        for element in source:
            if element.tag < SOPInstanceUIDTag:
                dataset.add(element)

        # Act
        is_supported_image_type = WsiDataset.is_supported_image_type(dataset)

        # Assert
        assert is_supported_image_type

    def test_is_supported_image_type_missing_sop_class_uid_returns_false(self):
        # Arrange
        dataset = create_main_dataset()
        del dataset.SOPClassUID

        # Act
        is_supported_image_type = WsiDataset.is_supported_image_type(dataset)

        # Assert
        assert not is_supported_image_type

    def test_is_supported_image_type_non_wsi_sop_class_returns_false(self):
        # Arrange
        dataset = create_main_dataset()
        dataset.SOPClassUID = CTImageStorage

        # Act
        is_supported_image_type = WsiDataset.is_supported_image_type(dataset)

        # Assert
        assert not is_supported_image_type

    @pytest.mark.parametrize(
        "image_type",
        [
            ["DERIVED", "PRIMARY", "BADFLAVOR", "NONE"],
            ["DERIVED", "PRIMARY", "LOCALIZER", "RESAMPLED"],
            ["DERIVED", "PRIMARY"],  # too few values to state a flavour
        ],
        ids=["unknown", "localizer", "too-few-values"],
    )
    def test_is_supported_image_type_unsupported_image_type_returns_false(
        self, image_type: Sequence[str]
    ):
        # Arrange
        dataset = create_main_dataset()
        dataset.ImageType = image_type

        # Act
        is_supported_image_type = WsiDataset.is_supported_image_type(dataset)

        # Assert
        assert not is_supported_image_type

    def test_is_supported_image_type_missing_image_type_returns_false(self):
        # Arrange — answered rather than raised, as a file is turned away by this
        dataset = create_main_dataset()
        del dataset.ImageType

        # Act
        is_supported_image_type = WsiDataset.is_supported_image_type(dataset)

        # Assert
        assert not is_supported_image_type

    def test_is_supported_supported_volume_returns_true(self):
        # Arrange
        dataset = create_main_dataset()

        # Act
        is_supported = WsiDataset.is_supported(dataset)

        # Assert
        assert is_supported

    @pytest.mark.parametrize("attribute", WsiDataset.REQUIRED_ATTRIBUTES)
    def test_is_supported_missing_required_attribute_returns_false(
        self, attribute: BaseTag
    ):
        # Arrange
        dataset = create_main_dataset()
        del dataset[attribute]

        # Act
        is_supported = WsiDataset.is_supported(dataset)

        # Assert
        assert not is_supported

    def test_is_supported_missing_sop_class_uid_returns_false(self):
        # Arrange
        dataset = create_main_dataset()
        del dataset.SOPClassUID

        # Act
        is_supported = WsiDataset.is_supported(dataset)

        # Assert
        assert not is_supported

    def test_is_supported_non_wsi_sop_class_returns_false(self):
        # Arrange — the only SOP class check a source reading over the web makes
        dataset = create_main_dataset()
        dataset.SOPClassUID = CTImageStorage

        # Act
        is_supported = WsiDataset.is_supported(dataset)

        # Assert
        assert not is_supported

    def test_is_supported_unsupported_image_type_returns_false(self):
        # Arrange — answered rather than raised, as an instance is turned away by this
        dataset = create_main_dataset()
        dataset.ImageType = ["DERIVED", "PRIMARY", "BADFLAVOR", "NONE"]

        # Act
        is_supported = WsiDataset.is_supported(dataset)

        # Assert
        assert not is_supported

    @pytest.mark.parametrize(
        ["attribute", "value"],
        [
            ("PixelRepresentation", 1),  # signed
            ("PlanarConfiguration", 1),  # non-interleaved color
            ("PhotometricInterpretation", "MONOCHROME1"),  # not in the WSI IOD
            ("BitsStored", 16),  # 16-bit color (default dataset is 3 samples)
        ],
    )
    def test_is_supported_unsupported_pixel_format_returns_false(
        self, attribute: str, value: int | str
    ):
        # Arrange
        dataset = create_main_dataset()
        setattr(dataset, attribute, value)

        # Act
        is_supported = WsiDataset.is_supported(dataset)

        # Assert
        assert not is_supported

    def test_is_supported_16bit_grayscale_returns_true(self):
        # Arrange — 16-bit is supported for grayscale, just not for color
        dataset = create_main_dataset()
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16

        # Act
        is_supported = WsiDataset.is_supported(dataset)

        # Assert
        assert is_supported

    @pytest.mark.parametrize(
        ["attribute", "value"],
        [
            ("SOPClassUID", CTImageStorage),
            ("ImageType", ["DERIVED", "PRIMARY", "LOCALIZER", "RESAMPLED"]),
        ],
        ids=["sop-class", "image-type"],
    )
    def test_is_supported_image_type_never_accepts_what_is_supported_rejects(
        self, attribute: str, value: object
    ):
        """The early check is a part of the whole one, so it can stand in for it."""
        # Arrange
        dataset = create_main_dataset()
        setattr(dataset, attribute, value)

        # Act
        is_supported_image_type = WsiDataset.is_supported_image_type(dataset)
        is_supported = WsiDataset.is_supported(dataset)

        # Assert
        assert not is_supported
        assert not is_supported_image_type

    def test_as_tiled_full_preserves_xy_origin_and_sets_z(self):
        # Arrange — create_main_dataset has an origin with x=60, y=10, no z.
        dataset = WsiDataset(create_main_dataset())

        # Act
        result = dataset.as_tiled_full([5.0], ["0"], Size(2, 2), 1)

        # Assert
        origin = result.as_dataset().TotalPixelMatrixOriginSequence[0]
        assert float(origin.XOffsetInSlideCoordinateSystem) == 60.0
        assert float(origin.YOffsetInSlideCoordinateSystem) == 10.0
        assert float(origin.ZOffsetInSlideCoordinateSystem) == 5.0

    def test_as_tiled_full_no_spurious_origin_when_absent_and_z_zero(self):
        # Arrange — a single plane at z=0 with no origin sequence.
        source = create_main_dataset()
        del source.TotalPixelMatrixOriginSequence
        dataset = WsiDataset(source)

        # Act
        result = dataset.as_tiled_full([0.0], ["0"], Size(2, 2), 1)

        # Assert — no invalid (x/y-less) origin item is created.
        assert "TotalPixelMatrixOriginSequence" not in result.as_dataset()

    def test_as_tiled_full_creates_valid_origin_when_absent_and_z_nonzero(self):
        # Arrange — a non-zero focal plane with no origin sequence.
        source = create_main_dataset()
        del source.TotalPixelMatrixOriginSequence
        dataset = WsiDataset(source)

        # Act
        result = dataset.as_tiled_full([5.0], ["0"], Size(2, 2), 1)

        # Assert — required x/y are present (defaulted) alongside z.
        origin = result.as_dataset().TotalPixelMatrixOriginSequence[0]
        assert float(origin.XOffsetInSlideCoordinateSystem) == 0.0
        assert float(origin.YOffsetInSlideCoordinateSystem) == 0.0
        assert float(origin.ZOffsetInSlideCoordinateSystem) == 5.0

    def test_as_tiled_full_filters_optical_path_sequence_to_written_paths(self):
        # Arrange — add a second optical path "1" to the source.
        source = create_main_dataset()
        second_path = Dataset()
        second_path.OpticalPathIdentifier = "1"
        source.OpticalPathSequence.append(second_path)
        dataset = WsiDataset(source)

        # Act — write only optical path "1".
        result = dataset.as_tiled_full([0.0], ["1"], Size(2, 2), 1)

        # Assert
        identifiers = [
            str(item.OpticalPathIdentifier)
            for item in result.as_dataset().OpticalPathSequence
        ]
        assert identifiers == ["1"]
        assert result.as_dataset().NumberOfOpticalPaths == 1

    def test_as_tiled_full_drops_slice_spacing_for_single_focal_plane(self):
        # Arrange — a source with multi-plane slice spacing set.
        source = create_main_dataset()
        source.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[
            0
        ].SpacingBetweenSlices = "2.0"
        dataset = WsiDataset(source)

        # Act — split to a single focal plane.
        result = dataset.as_tiled_full([5.0], ["0"], Size(2, 2), 1)

        # Assert — stale spacing is removed.
        pixel_measure = (
            result.as_dataset()
            .SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
        )
        assert "SpacingBetweenSlices" not in pixel_measure

    def test_optical_path_identifier_from_the_frame_wins_over_the_shared_groups(self):
        # Arrange - the standard place for it in a sparse image is the frame.
        source = create_main_dataset()
        source.SharedFunctionalGroupsSequence[
            0
        ].OpticalPathIdentificationSequence = _optical_path_identification("shared")
        frame = Dataset()
        frame.OpticalPathIdentificationSequence = _optical_path_identification("frame")
        dataset = WsiDataset(source)

        # Act
        identifier = dataset.read_optical_path_identifier(frame)

        # Assert
        assert identifier == "frame"

    def test_optical_path_identifier_falls_back_to_the_shared_groups(self):
        # Arrange - stating it once for all frames is not where the standard puts it,
        # but it is unambiguous, so a frame that states none is answered from there.
        source = create_main_dataset()
        source.SharedFunctionalGroupsSequence[
            0
        ].OpticalPathIdentificationSequence = _optical_path_identification("shared")
        dataset = WsiDataset(source)

        # Act
        identifier = dataset.read_optical_path_identifier()

        # Assert - and not "0", the first optical path of the instance.
        assert identifier == "shared"

    def test_optical_path_identifier_falls_back_to_the_optical_paths(self):
        # Arrange - nothing states it per frame or once for all frames.
        dataset = WsiDataset(create_main_dataset())

        # Act
        identifier = dataset.read_optical_path_identifier()

        # Assert
        assert identifier == "0"

    def test_z_offset_from_the_frame_wins_over_the_shared_groups(self):
        # Arrange - the standard place for it in a sparse image is the frame.
        source = create_main_dataset()
        source.SharedFunctionalGroupsSequence[
            0
        ].PlanePositionSlideSequence = _plane_position(2.0)
        frame = Dataset()
        frame.PlanePositionSlideSequence = _plane_position(1.0)
        dataset = WsiDataset(source)

        # Act
        z_offset = dataset.read_z_offset(frame)

        # Assert
        assert z_offset == 1.0

    def test_z_offset_falls_back_to_the_shared_groups(self):
        # Arrange - one focal plane stated once for all frames rather than per frame.
        source = create_main_dataset()
        source.SharedFunctionalGroupsSequence[
            0
        ].PlanePositionSlideSequence = _plane_position(2.0)
        dataset = WsiDataset(source)

        # Act
        z_offset = dataset.read_z_offset()

        # Assert - and not the default of zero.
        assert z_offset == 2.0

    def test_z_offset_falls_back_to_zero(self):
        # Arrange - nothing states it per frame or once for all frames.
        dataset = WsiDataset(create_main_dataset())

        # Act
        z_offset = dataset.read_z_offset()

        # Assert
        assert z_offset == 0.0

    @pytest.mark.parametrize(
        ["z_offsets", "identifiers"],
        [
            ([1.0, None], [None, None]),
            ([None, None], ["1", None]),
        ],
    )
    def test_frame_positions_refuse_an_element_only_some_frames_state(
        self,
        z_offsets: Sequence[float | None],
        identifiers: Sequence[str | None],
    ):
        # Arrange - a macro in the per frame groups is in every frame or in none, so
        # there is no reading of this file that is better than a guess.
        source = create_main_dataset()
        source.PerFrameFunctionalGroupsSequence = _per_frame_groups(
            z_offsets, identifiers
        )
        dataset = WsiDataset(source)

        # Act & Assert
        with pytest.raises(WsiDicomError):
            _ = dataset.frame_positions

    @pytest.mark.parametrize(
        ["z_offsets", "identifiers", "expected_z", "expected_identifiers"],
        [
            ([None, None], [None, None], None, None),
            ([1.0, 2.0], [None, None], [1.0, 2.0], None),
            ([None, None], ["0", "1"], None, ["0", "1"]),
        ],
    )
    def test_frame_positions_of_an_element_every_frame_states_or_none_do(
        self,
        z_offsets: Sequence[float | None],
        identifiers: Sequence[str | None],
        expected_z: list[float] | None,
        expected_identifiers: list[str] | None,
    ):
        # Arrange
        source = create_main_dataset()
        source.PerFrameFunctionalGroupsSequence = _per_frame_groups(
            z_offsets, identifiers
        )
        dataset = WsiDataset(source)

        # Act
        positions = dataset.frame_positions

        # Assert - None where no frame states it, so what the instance says applies.
        if expected_z is None:
            assert positions.z_offsets is None
        else:
            assert positions.z_offsets is not None
            assert list(positions.z_offsets) == expected_z
        if expected_identifiers is None:
            assert positions.optical_path_identifiers is None
        else:
            assert positions.optical_path_identifiers is not None
            assert list(positions.optical_path_identifiers) == expected_identifiers

    def test_frame_positions_refuse_more_than_one_optical_path_identifier(self):
        # Arrange - Optical Path Identification Sequence holds a single item, so which
        # path a frame stating several is on cannot be told.
        source = create_main_dataset()
        frames = DicomSequence()
        for identifiers in (["0", "1"], ["2", "3"]):
            position = Dataset()
            position.ColumnPositionInTotalImagePixelMatrix = 1
            position.RowPositionInTotalImagePixelMatrix = 1
            frame = Dataset()
            frame.PlanePositionSlideSequence = DicomSequence([position])
            frame.OpticalPathIdentificationSequence = DicomSequence(
                [
                    _optical_path_identification(identifier)[0]
                    for identifier in identifiers
                ]
            )
            frames.append(frame)
        source.PerFrameFunctionalGroupsSequence = frames
        dataset = WsiDataset(source)

        # Act & Assert
        with pytest.raises(WsiDicomError):
            _ = dataset.frame_positions
