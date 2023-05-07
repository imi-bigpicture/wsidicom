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
from typing import List, Optional, Type

import numpy as np
import pytest
from parameterized import parameterized
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence

from tests.data_gen import create_layer_file, create_main_dataset
from wsidicom import WsiDicom
from wsidicom.conceptcode import (
    CidConceptCode,
    Code,
    IlluminationCode,
    IlluminationColorCode,
)
from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.file.wsidicom_file_image_data import WsiDicomFileImageData
from wsidicom.geometry import Point, PointMm, Region, RegionMm, Size, SizeMm
from wsidicom.optical import Illumination, Lut, OpticalManager, OpticalPath


@pytest.mark.unittest
class WsiDicomInterfaceTests(unittest.TestCase):
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
                Region(position=Point(x=0, y=0), size=Size(width=100, height=100)),
                Point(0, 0),
                Size(1024, 1024),
                Region(position=Point(0, 0), size=Size(100, 100)),
            ),
            (
                Region(position=Point(x=0, y=0), size=Size(width=1500, height=1500)),
                Point(0, 0),
                Size(1024, 1024),
                Region(position=Point(0, 0), size=Size(1024, 1024)),
            ),
            (
                Region(
                    position=Point(x=1200, y=1200), size=Size(width=300, height=300)
                ),
                Point(1, 1),
                Size(1024, 1024),
                Region(position=Point(176, 176), size=Size(300, 300)),
            ),
        ]
    )
    def test_crop_tile(
        self, region: Region, point: Point, size: Size, expected_result: Region
    ):
        # Arrange

        # Act
        cropped_region = region.inside_crop(point, size)

        # Assert
        self.assertEqual(cropped_region, expected_result)

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

    @parameterized.expand(
        [
            (
                [256, 0, 8],
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\x00\x00",
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\x00\x00",
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\xff\x00",
                2,
                np.linspace(0, 255, 256, dtype=np.uint16),
            ),
            (
                [256, 0, 16],
                b"\x01\x00\x00\x01\xff\xff",
                b"\x01\x00\x00\x01\x00\x00",
                b"\x01\x00\x00\x01\x00\x00",
                0,
                np.linspace(0, 65535, 256, dtype=np.uint16),
            ),
        ]
    )
    def test_parse_lut(
        self,
        red_lut_descriptor: List[int],
        red_lut: bytes,
        green_lut: bytes,
        blue_lut: bytes,
        channel: int,
        expected: np.ndarray,
    ):
        # Arrange
        ds = Dataset()
        ds.RedPaletteColorLookupTableDescriptor = red_lut_descriptor
        ds.SegmentedRedPaletteColorLookupTableData = red_lut
        ds.SegmentedGreenPaletteColorLookupTableData = green_lut
        ds.SegmentedBluePaletteColorLookupTableData = blue_lut
        expected_lut = np.zeros((3, 256), dtype=np.uint16)
        expected_lut[channel, :] = expected

        # Act
        lut = Lut(DicomSequence([ds]))

        # Assert
        self.assertTrue(np.array_equal(lut.get(), expected_lut))

    def test_recreate_optical_module(self):
        # Arrange
        ds = create_main_dataset()
        original_optical = Dataset()
        original_optical.OpticalPathSequence = ds.OpticalPathSequence
        original_optical.NumberOfOpticalPaths = ds.NumberOfOpticalPaths

        # Act
        restored_optical_ds = self.slide.optical.insert_into_ds(Dataset())

        # Assert
        self.assertEqual(original_optical, restored_optical_ds)

    def test_make_optical(self):
        # Arrange
        illumination_method = IlluminationCode("Transmission illumination")
        illumination_color = IlluminationColorCode("Full Spectrum")
        illumination = Illumination(
            illumination_method=[illumination_method],
            illumination_color=illumination_color,
        )
        path = OpticalPath(
            identifier="1",
            illumination=illumination,
            photometric_interpretation="YBR_FULL_422",
            icc_profile=bytes(0),
        )

        # Act
        optical = OpticalManager([path])

        # Assert
        self.assertEqual(optical.get("1"), path)

    @parameterized.expand(
        (
            code_class,
            code,
        )
        for code_class in CidConceptCode.__subclasses__()
        for code in code_class.cid.values()
    )
    def test_create_code_from_meaning(
        self, code_class: Type[CidConceptCode], code: Code
    ):
        # Arrange

        # Act
        created_code = code_class(code.meaning)

        # Assert
        self.assertEqual(code, created_code)
