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

from typing import Optional

import pytest

from wsidicom import WsiDicom
from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.file.wsidicom_file_image_data import WsiDicomFileImageData
from wsidicom.geometry import Point, PointMm, Region, RegionMm, Size, SizeMm


@pytest.mark.unittest
class TestWsiDicomLevels:
    def test_mm_to_pixel(self, wsi: WsiDicom):
        # Arrange
        wsi_level = wsi.levels.get_level(0)
        mm_region = RegionMm(position=PointMm(0, 0), size=SizeMm(1, 1))
        new_size = int(1 / 0.1242353)

        # Act
        pixel_region = wsi_level.mm_to_pixel(mm_region)

        # Assert
        assert pixel_region.position == Point(0, 0)
        assert pixel_region.size == Size(new_size, new_size)

    def test_find_closest_level(self, wsi: WsiDicom):
        # Arrange
        # Act
        closest_level = wsi.levels.get_closest_by_level(2)

        # Assert
        assert closest_level.level == 0

    def test_find_closest_pixel_spacing(self, wsi: WsiDicom):
        # Arrange
        # Act
        closest_level = wsi.levels.get_closest_by_pixel_spacing(SizeMm(0.5, 0.5))

        # Assert
        assert closest_level.level == 0

    def test_find_closest_size(self, wsi: WsiDicom):
        # Arrange
        # Act
        closest_level = wsi.levels.get_closest_by_size(Size(100, 100))

        # Assert
        assert closest_level.level == 0

    def test_calculate_scale(self, wsi: WsiDicom):
        # Arrange
        wsi_level = wsi.levels.get_level(0)

        # Act
        scale = wsi_level.calculate_scale(5)

        # Assert
        assert scale == 32

    def test_get_frame_number(self, wsi: WsiDicom):
        # Arrange
        base_level = wsi.levels.get_level(0)
        instance = base_level.get_instance()
        image_data = instance._image_data
        assert isinstance(image_data, WsiDicomFileImageData)

        # Act
        number = image_data.tiles.get_frame_index(Point(0, 0), 0, "0")

        # Assert
        assert number == 0

    def test_get_blank_color(self, wsi: WsiDicom):
        # Arrange
        base_level = wsi.levels.get_level(0)
        instance = base_level.get_instance()
        image_data = instance.image_data

        # Act
        color = image_data._get_blank_color(image_data.photometric_interpretation)

        # Assert
        assert color == (255, 255, 255)

    def test_get_file_frame_in_range(self, wsi: WsiDicom):
        # Arrange
        base_level = wsi.levels.get_level(0)
        instance = base_level.get_instance()
        image_data = instance._image_data
        assert isinstance(image_data, WsiDicomFileImageData)

        # Act
        file = image_data._get_file(0)

        # Assert
        assert file == (image_data._files[0])

    def test_get_file_frame_out_of_range_throws(self, wsi: WsiDicom):
        # Arrange
        base_level = wsi.levels.get_level(0)
        instance = base_level.get_instance()
        image_data = instance._image_data
        assert isinstance(image_data, WsiDicomFileImageData)

        # Act & Assert
        with pytest.raises(WsiDicomNotFoundError):
            image_data._get_file(10)

    @pytest.mark.parametrize(
        ["region", "z", "path", "expected_result"],
        [
            (Region(Point(0, 0), Size(0, 0)), 0, "0", True),
            (Region(Point(0, 0), Size(0, 2)), 0, "0", False),
            (Region(Point(0, 0), Size(0, 0)), 1, "0", False),
            (Region(Point(0, 0), Size(0, 0)), 0, "1", False),
        ],
    )
    def test_valid_tiles(
        self, wsi: WsiDicom, region: Region, z: float, path: str, expected_result: bool
    ):
        # Arrange
        base_level = wsi.levels.get_level(0)
        instance = base_level.get_instance()
        image_data = instance._image_data

        # Act
        test = image_data.valid_tiles(region, z, path)

        # Assert
        assert test == expected_result

    @pytest.mark.parametrize(
        ["region", "expected_result"],
        [
            (
                Region(position=Point(0, 0), size=Size(100, 100)),
                Region(Point(0, 0), Size(1, 1)),
            ),
            (
                Region(position=Point(0, 0), size=Size(1024, 1024)),
                Region(Point(0, 0), Size(1, 1)),
            ),
            (
                Region(position=Point(300, 400), size=Size(500, 500)),
                Region(Point(0, 0), Size(1, 1)),
            ),
        ],
    )
    def test_get_tile_range(
        self, wsi: WsiDicom, region: Region, expected_result: Region
    ):
        # Arrange
        base_level = wsi.levels.get_level(0)
        instance = base_level.get_instance()

        # Act
        tile_range = instance.image_data._get_tile_range(region, 0, "0")

        # Assert
        assert tile_range == expected_result

    @pytest.mark.parametrize(
        ["region", "expected_size"],
        [
            (Region(position=Point(0, 0), size=Size(100, 100)), Size(100, 100)),
            (Region(position=Point(0, 0), size=Size(2000, 2000)), Size(154, 290)),
            (Region(position=Point(200, 300), size=Size(100, 100)), Size(0, 0)),
        ],
    )
    def test_crop_region_to_level_size(
        self, wsi: WsiDicom, region: Region, expected_size: Size
    ):
        # Arrange
        base_level = wsi.levels.get_level(0)
        image_size = base_level.size

        # Act
        cropped_region = region.crop(image_size)

        # Assert
        assert cropped_region.size == expected_size

    @pytest.mark.parametrize(
        ["region", "expected_result"],
        [
            (Region(position=Point(0, 0), size=Size(100, 100)), True),
            (Region(position=Point(150, 0), size=Size(10, 100)), False),
        ],
    )
    def test_valid_pixel(self, wsi: WsiDicom, region: Region, expected_result: bool):
        # Arrange
        # 154x290
        wsi_level = wsi.levels.get_level(0)

        # Act
        valid = wsi_level.valid_pixels(region)

        # Assert
        assert valid == expected_result

    @pytest.mark.parametrize(["level", "expected_result"], [(1, True), (20, False)])
    def test_valid_level(self, wsi: WsiDicom, level: int, expected_result: bool):
        # Arrange
        # Act
        valid = wsi.levels.valid_level(level)

        # Assert
        assert valid == expected_result

    @pytest.mark.parametrize(
        ["z", "path"], [(None, None), (0, None), (None, "0"), (0, "0")]
    )
    def test_get_instance_defaulting(
        self, wsi: WsiDicom, z: Optional[float], path: Optional[str]
    ):
        # Arrange
        wsi_level = wsi.levels.get_level(0)

        # Act
        instance = wsi_level.get_instance(z, path)

        # Assert
        assert instance == wsi_level.default_instance

    def test_lowest_single_tile_level(self, wsi: WsiDicom):
        # Arrange
        tile_size = wsi.tile_size

        # Act
        lowest_single_tile_level = wsi.levels.lowest_single_tile_level

        # Assert
        assert tile_size.all_greater_than(wsi.size // (2**lowest_single_tile_level))
        assert tile_size.any_less_than(
            wsi.size // (2 ** (lowest_single_tile_level - 1))
        )
