from typing import Optional, Sequence, Union
from pydicom import Dataset

import pytest

from wsidicom.instance.dataset import WsiDataset
from pydicom.dataelem import DataElement
from wsidicom.tags import LossyImageCompressionRatioTag
from pydicom.uid import generate_uid, UID
from wsidicom.instance import TileType
from wsidicom.geometry import SizeMm


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
    dataset = WsiDataset()
    dataset.SOPInstanceUID = instance_uid
    if concatenation:
        dataset.SOPInstanceUIDOfConcatenationSource = concatenation_uid
    dataset.FrameOfReferenceUID = frame_of_reference_uid
    dataset.StudyInstanceUID = study_instance_uid
    dataset.SeriesInstanceUID = series_instance_uid
    dataset.ImageType = ["DERIVED", "PRIMARY", "VOLUME", "RESAMPLED"]
    yield dataset


@pytest.fixture
def pixel_spacing():
    yield SizeMm(0.1, 0.2)


@pytest.fixture
def spacing_between_slices():
    yield 0.1


@pytest.fixture
def pixel_measure(
    pixel_spacing: Optional[SizeMm], spacing_between_slices: Optional[float]
):
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


class TestWsiDataset:
    @pytest.mark.parametrize(
        ["values", "expected_values"],
        [(None, []), ("1", ["1"]), (["1", "2"], ["1", "2"])],
    )
    def test_get_multi_value(
        self,
        dataset: WsiDataset,
        values: Optional[Union[str, Sequence[str]]],
        expected_values: Sequence[str],
    ):
        # Arrange
        if values is not None:
            dataset.add(DataElement(LossyImageCompressionRatioTag, "CS", values))

        # Act
        read_values = dataset.get_multi_value(LossyImageCompressionRatioTag)

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
        concatenation: Optional[int],
        expected_frame_offset: int,
    ):
        # Arrange
        if concatenation is not None:
            dataset.ConcatenationFrameOffsetNumber = concatenation

        # Act
        frame_offset = dataset.frame_offset

        # Assert
        assert frame_offset == expected_frame_offset

    @pytest.mark.parametrize(
        ["frame_count", "expected_frame_count"], [(None, 1), (1, 1), (100, 100)]
    )
    def test_frame_count(
        self, dataset: WsiDataset, frame_count: Optional[int], expected_frame_count: int
    ):
        # Arrange
        if frame_count is not None:
            dataset.NumberOfFrames = frame_count

        # Act
        read_frame_count = dataset.frame_count

        # Assert
        assert read_frame_count == expected_frame_count

    def test_tile_type_tiled_full(self, dataset: WsiDataset):
        # Arrange
        dataset.DimensionOrganizationType = "TILED_FULL"

        # Act
        read_tile_type = dataset.tile_type

        # Assert
        assert read_tile_type == TileType.FULL

    def test_tile_type_tiled_sparse(self, dataset: WsiDataset):
        # Arrange
        dataset.PerFrameFunctionalGroupsSequence = []

        # Act
        read_tile_type = dataset.tile_type

        # Assert
        assert read_tile_type == TileType.SPARSE

    def test_tile_type_label(self, dataset: WsiDataset):
        # Arrange
        dataset.ImageType = ["DERIVED", "LABEL", "VOLUME", "RESAMPLED"]

        # Act
        read_tile_type = dataset.tile_type

        # Assert
        assert read_tile_type == TileType.FULL

    def test_tile_type_single_frame(self, dataset: WsiDataset):
        # Arrange
        dataset.TotalPixelMatrixFocalPlanes = 1
        dataset.NumberOfOpticalPaths = 1
        dataset.NumberOfFrames = 1

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
        dataset.SharedFunctionalGroupsSequence = [shared_functional_group]

        # Act
        read_pixel_measure = dataset.pixel_measure

        # Assert
        assert read_pixel_measure == pixel_measure

    @pytest.mark.parametrize(
        ["pixel_spacing", "expected_pixel_spacing"],
        [(None, None), (SizeMm(0, 0), None), (SizeMm(0.1, 0.2), SizeMm(0.1, 0.2))],
    )
    def test_pixel_spacing(
        self,
        dataset: WsiDataset,
        shared_functional_group: Dataset,
        expected_pixel_spacing: Optional[SizeMm],
    ):
        # Arrange
        dataset.SharedFunctionalGroupsSequence = [shared_functional_group]

        # Act
        read_pixel_spacing = dataset.pixel_spacing

        # Assert
        assert read_pixel_spacing == expected_pixel_spacing

    @pytest.mark.parametrize("spacing_between_slices", [None, 0.1])
    def test_spacing_between_slices(
        self,
        dataset: WsiDataset,
        shared_functional_group: Dataset,
        spacing_between_slices: Optional[float],
    ):
        # Arrange
        dataset.SharedFunctionalGroupsSequence = [shared_functional_group]

        # Act
        read_spacing_between_slices = dataset.spacing_between_slices

        # Assert
        assert read_spacing_between_slices == spacing_between_slices
