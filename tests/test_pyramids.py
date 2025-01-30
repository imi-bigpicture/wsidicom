from typing import List, Optional, Tuple

import pytest
from pydicom import Dataset
from pydicom.uid import UID, generate_uid

from wsidicom.geometry import Orientation, PointMm, Size, SizeMm
from wsidicom.instance.dataset import ImageType, WsiDataset
from wsidicom.instance.instance import WsiInstance
from wsidicom.metadata.image import ImageCoordinateSystem
from wsidicom.series.pyramids import Pyramids
from wsidicom.uid import SlideUids


class WsiTestInstance(WsiInstance):
    def __init__(
        self,
        size: Size,
        tile_size: Size,
        pixel_spacing: SizeMm,
        study_instance_uid: UID,
        series_instance_uid: UID,
        frame_of_reference_uid: UID,
        image_coordinate_system: ImageCoordinateSystem,
        ext_depth_of_field: Optional[Tuple[int, float]],
    ):
        self._size = size
        self._tile_size = tile_size
        self._pixel_spacing = pixel_spacing
        self._image_coordinate_system = image_coordinate_system
        if ext_depth_of_field is not None:
            self._ext_depth_of_field_planes = ext_depth_of_field[0]
            self._ext_depth_of_field_plane_distance = ext_depth_of_field[1]
            self._ext_depth_of_field = True
        else:
            self._ext_depth_of_field_planes = None
            self._ext_depth_of_field_plane_distance = None
            self._ext_depth_of_field = False
        dataset = Dataset()
        dataset.SOPInstanceUID = generate_uid()
        dataset.ImagedVolumeWidth = 10
        dataset.ImagedVolumeHeight = 10
        self._datasets = [WsiDataset(dataset)]
        self._image_data = None
        self._identifier = generate_uid()
        self._uids = SlideUids(
            study_instance_uid,
            series_instance_uid,
            frame_of_reference_uid,
        )
        self._image_type = ImageType.VOLUME

    @property
    def size(self) -> Size:
        return self._size

    @property
    def tile_size(self) -> Size:
        return self._tile_size

    @property
    def pixel_spacing(self) -> SizeMm:
        return self._pixel_spacing

    @property
    def image_coordinate_system(self) -> ImageCoordinateSystem:
        return self._image_coordinate_system

    @property
    def ext_depth_of_field(self) -> bool:
        return self._ext_depth_of_field

    @property
    def ext_depth_of_field_planes(self) -> Optional[int]:
        return self._ext_depth_of_field_planes

    @property
    def ext_depth_of_field_plane_distance(self) -> Optional[float]:
        return self._ext_depth_of_field_plane_distance


def create_pyramid_instance(
    image_coordinate_system: ImageCoordinateSystem,
    ext_depth_of_field: Optional[Tuple[int, float]],
    study_instance_uid: UID,
    series_instance_uid: UID,
    frame_of_reference_uid: UID,
):
    size = Size(100, 100)
    tile_size = Size(10, 10)
    pixel_spacing = SizeMm(0.5, 0.5)
    return WsiTestInstance(
        size,
        tile_size,
        pixel_spacing,
        study_instance_uid,
        series_instance_uid,
        frame_of_reference_uid,
        image_coordinate_system,
        ext_depth_of_field,
    )


@pytest.fixture()
def study_instance_uid():
    return generate_uid()


@pytest.fixture()
def series_instance_uid():
    return generate_uid()


@pytest.fixture()
def frame_of_reference_uid():
    return generate_uid()


class TestPyramids:
    @pytest.mark.parametrize(
        [
            "instance_definitions",
            "expected_pyramid_count",
        ],
        [
            [
                [
                    (PointMm(0, 0), Orientation((0, -1, 0, 1, 0, 0)), None),
                    (PointMm(0, 0), Orientation((0, -1, 0, 1, 0, 0)), None),
                ],
                1,
            ],
            [
                [
                    (PointMm(0, 0), Orientation((0, -1, 0, 1, 0, 0)), (5, 0.5)),
                    (PointMm(0, 0), Orientation((0, -1, 0, 1, 0, 0)), (5, 0.5)),
                ],
                1,
            ],
            [
                [
                    (PointMm(0, 0), Orientation((0, -1, 0, 1, 0, 0)), None),
                    (PointMm(0, 0), Orientation((0, -1, 0, 1, 0, 0)), (5, 0.5)),
                ],
                2,
            ],
            [
                [
                    (PointMm(0, 0), Orientation((0, -1, 0, 1, 0, 0)), None),
                    (PointMm(2, 2), Orientation((0, -1, 0, 1, 0, 0)), None),
                ],
                2,
            ],
            [
                [
                    (PointMm(0, 0), Orientation((0, -1, 0, 1, 0, 0)), None),
                    (PointMm(0.001, 0.001), Orientation((0, -1, 0, 1, 0, 0)), None),
                ],
                1,
            ],
        ],
    )
    def test_open_number_of_created_pyramids(
        self,
        instance_definitions: List[
            Tuple[PointMm, Orientation, Optional[Tuple[int, float]]]
        ],
        study_instance_uid: UID,
        series_instance_uid: UID,
        frame_of_reference_uid: UID,
        expected_pyramid_count: int,
    ):
        # Arrange
        instances = [
            create_pyramid_instance(
                ImageCoordinateSystem(
                    instance_definition[0], instance_definition[1].rotation
                ),
                instance_definition[2],
                study_instance_uid,
                series_instance_uid,
                frame_of_reference_uid,
            )
            for instance_definition in instance_definitions
        ]

        # Act
        pyramids = Pyramids.open(instances, [])

        # Assert
        assert len(pyramids) == expected_pyramid_count
