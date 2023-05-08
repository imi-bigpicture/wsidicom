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

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import pytest
from parameterized import parameterized

from tests.data_gen import create_layer_file
from wsidicom import WsiDicom
from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.file.wsidicom_file_image_data import WsiDicomFileImageData
from wsidicom.geometry import Point, PointMm, Region, RegionMm, Size, SizeMm


@pytest.mark.unittest
class WsiDicomLevelsTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tempdir: TemporaryDirectory
        self.slide: WsiDicom

    @classmethod
    def setUpClass(cls):
        cls.tempdir = TemporaryDirectory()
        dirpath = Path(cls.tempdir.name)
        test_file_path = dirpath.joinpath("test_im.dcm")
        create_layer_file(test_file_path)
        cls.slide = WsiDicom.open(cls.tempdir.name)

    @classmethod
    def tearDownClass(cls):
        cls.slide.close()
        cls.tempdir.cleanup()

    def test_mm_to_pixel(self):
        # Arrange
        wsi_level = self.slide.levels.get_level(0)
        mm_region = RegionMm(position=PointMm(0, 0), size=SizeMm(1, 1))
        new_size = int(1 / 0.1242353)

        # Act
        pixel_region = wsi_level.mm_to_pixel(mm_region)

        # Assert
        self.assertEqual(pixel_region.position, Point(0, 0))
        self.assertEqual(pixel_region.size, Size(new_size, new_size))

    def test_find_closest_level(self):
        # Arrange
        # Act
        closest_level = self.slide.levels.get_closest_by_level(2)

        # Assert
        self.assertEqual(closest_level.level, 0)

    def test_find_closest_pixel_spacing(self):
        # Arrange
        # Act
        closest_level = self.slide.levels.get_closest_by_pixel_spacing(SizeMm(0.5, 0.5))

        # Assert
        self.assertEqual(closest_level.level, 0)

    def test_find_closest_size(self):
        # Arrange
        # Act
        closest_level = self.slide.levels.get_closest_by_size(Size(100, 100))

        # Assert
        self.assertEqual(closest_level.level, 0)

    def test_calculate_scale(self):
        # Arrange
        wsi_level = self.slide.levels.get_level(0)

        # Act
        scale = wsi_level.calculate_scale(5)

        # Assert
        self.assertEqual(scale, 32)

    def test_get_frame_number(self):
        # Arrange
        base_level = self.slide.levels.get_level(0)
        instance = base_level.get_instance()
        image_data = instance._image_data
        assert isinstance(image_data, WsiDicomFileImageData)

        # Act
        number = image_data.tiles.get_frame_index(Point(0, 0), 0, "0")

        # Assert
        self.assertEqual(number, 0)

    def test_get_blank_color(self):
        # Arrange
        base_level = self.slide.levels.get_level(0)
        instance = base_level.get_instance()
        image_data = instance.image_data

        # Act
        color = image_data._get_blank_color(image_data.photometric_interpretation)

        # Assert
        self.assertEqual(color, (255, 255, 255))

    def test_get_file_frame_in_range(self):
        # Arrange
        base_level = self.slide.levels.get_level(0)
        instance = base_level.get_instance()
        image_data = instance._image_data
        assert isinstance(image_data, WsiDicomFileImageData)

        # Act
        file = image_data._get_file(0)

        # Assert
        self.assertEqual(file, (image_data._files[0]))

    def test_get_file_frame_out_of_range_throws(self):
        # Arrange
        base_level = self.slide.levels.get_level(0)
        instance = base_level.get_instance()
        image_data = instance._image_data
        assert isinstance(image_data, WsiDicomFileImageData)

        # Act & Assert
        self.assertRaises(WsiDicomNotFoundError, image_data._get_file, 10)

    @parameterized.expand(
        [
            (Region(Point(0, 0), Size(0, 0)), 0, "0", True),
            (Region(Point(0, 0), Size(0, 2)), 0, "0", False),
            (Region(Point(0, 0), Size(0, 0)), 1, "0", False),
            (Region(Point(0, 0), Size(0, 0)), 0, "1", False),
        ]
    )
    def test_valid_tiles(
        self, region: Region, z: float, path: str, expected_result: bool
    ):
        # Arrange
        base_level = self.slide.levels.get_level(0)
        instance = base_level.get_instance()
        image_data = instance._image_data

        # Act
        test = image_data.valid_tiles(region, z, path)

        # Assert
        self.assertEqual(test, expected_result)

    @parameterized.expand(
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
        ]
    )
    def test_get_tile_range(self, region: Region, expected: Region):
        # Arrange
        base_level = self.slide.levels.get_level(0)
        instance = base_level.get_instance()

        # Act
        tile_range = instance.image_data._get_tile_range(region, 0, "0")

        # Assert
        self.assertEqual(tile_range, expected)

    @parameterized.expand(
        [
            (Region(position=Point(0, 0), size=Size(100, 100)), Size(100, 100)),
            (Region(position=Point(0, 0), size=Size(2000, 2000)), Size(154, 290)),
            (Region(position=Point(200, 300), size=Size(100, 100)), Size(0, 0)),
        ]
    )
    def test_crop_region_to_level_size(self, region: Region, expected_size: Size):
        # Arrange
        base_level = self.slide.levels.get_level(0)
        image_size = base_level.size

        # Act
        cropped_region = region.crop(image_size)

        # Assert
        self.assertEqual(cropped_region.size, expected_size)

    @parameterized.expand(
        [
            (Region(position=Point(0, 0), size=Size(100, 100)), True),
            (Region(position=Point(150, 0), size=Size(10, 100)), False),
        ]
    )
    def test_valid_pixel(self, region: Region, expected_result: bool):
        # Arrange
        # 154x290
        wsi_level = self.slide.levels.get_level(0)

        # Act
        valid = wsi_level.valid_pixels(region)

        # Assert
        self.assertEqual(valid, expected_result)

    @parameterized.expand([(1, True), (20, False)])
    def test_valid_level(self, level: int, expected_result: bool):
        # Arrange
        # Act
        valid = self.slide.levels.valid_level(level)

        # Assert
        self.assertEqual(valid, expected_result)

    @parameterized.expand([(None, None), (0, None), (None, "0"), (0, "0")])
    def test_get_instance_defaulting(self, z: Optional[float], path: Optional[str]):
        # Arrange
        wsi_level = self.slide.levels.get_level(0)

        # Act
        instance = wsi_level.get_instance(z, path)

        # Assert
        self.assertEqual(instance, wsi_level.default_instance)

    def test_lowest_single_tile_level(self):
        # Arrange
        tile_size = self.slide.tile_size

        # Act
        lowest_single_tile_level = self.slide.levels.lowest_single_tile_level

        # Assert
        self.assertTrue(
            tile_size.all_greater_than(
                self.slide.size // (2**lowest_single_tile_level)
            )
        )
        self.assertTrue(
            tile_size.any_less_than(
                self.slide.size // (2 ** (lowest_single_tile_level - 1))
            )
        )
