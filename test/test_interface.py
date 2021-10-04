import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from wsidicom.conceptcode import (CidConceptCode, IlluminationCode,
                                  IlluminationColorCode)
from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.interface import (DicomImageData, Point, PointMm, Region,
                                RegionMm, Size, SizeMm, WsiDicom)
from wsidicom.optical import Illumination, Lut, OpticalManager, OpticalPath

from .data_gen import create_layer_file, create_main_dataset


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
        wsi_level = self.slide.levels.get_level(0)
        mm_region = RegionMm(
            position=PointMm(0, 0),
            size=SizeMm(1, 1)
        )
        pixel_region = wsi_level.mm_to_pixel(mm_region)
        self.assertEqual(pixel_region.position, Point(0, 0))
        new_size = int(1 / 0.1242353)
        self.assertEqual(pixel_region.size, Size(new_size, new_size))

    def test_find_closest_level(self):
        closest_level = self.slide.levels.get_closest_by_level(2)
        self.assertEqual(closest_level.level, 0)

    def test_find_closest_pixel_spacing(self):
        closest_level = self.slide.levels.get_closest_by_pixel_spacing(
            SizeMm(0.5, 0.5)
        )
        self.assertEqual(closest_level.level, 0)

    def test_find_closest_size(self):
        closest_level = self.slide.levels.get_closest_by_size(Size(100, 100))
        self.assertEqual(closest_level.level, 0)

    def test_calculate_scale(self):
        wsi_level = self.slide.levels.get_level(0)
        scale = wsi_level.calculate_scale(5)
        self.assertEqual(scale, 2 ** (5-0))

    def test_get_frame_number(self):
        base_level = self.slide.levels.get_level(0)
        instance = base_level.get_instance()
        image_data: DicomImageData = instance._image_data
        number = image_data.tiles.get_frame_index(Point(0, 0), 0, '0')
        self.assertEqual(number, 0)

    def test_get_blank_color(self):
        base_level = self.slide.levels.get_level(0)
        instance = base_level.get_instance()
        image_data: DicomImageData = instance.image_data
        color = image_data._get_blank_color(
            image_data.photometric_interpretation)
        self.assertEqual(color, (255, 255, 255))

    def test_get_frame_file(self):
        base_level = self.slide.levels.get_level(0)
        instance = base_level.get_instance()
        image_data: DicomImageData = instance._image_data
        file = image_data._get_file(0)
        self.assertEqual(file, (image_data._files[0]))

        self.assertRaises(
            WsiDicomNotFoundError,
            image_data._get_file,
            10
        )

    def test_valid_tiles(self):
        base_level = self.slide.levels.get_level(0)
        instance = base_level.get_instance()
        image_data: DicomImageData = instance._image_data
        test = image_data.valid_tiles(

            Region(Point(0, 0), Size(0, 0)), 0, '0'
        )
        self.assertTrue(test)

        test = image_data.valid_tiles(
            Region(Point(0, 0), Size(0, 2)), 0, '0'
        )
        self.assertFalse(test)

        test = image_data.valid_tiles(
            Region(Point(0, 0), Size(0, 0)), 1, '0'
        )
        self.assertFalse(test)

        test = image_data.valid_tiles(
            Region(Point(0, 0), Size(0, 0)), 0, '1'
        )
        self.assertFalse(test)

    def test_crop_tile(self):
        base_level = self.slide.levels.get_level(0)
        instance = base_level.get_instance()
        region = Region(
            position=Point(x=0, y=0),
            size=Size(width=100, height=100)
        )
        cropped_region = instance.crop_tile(Point(x=0, y=0), region)
        expected = Region(
            position=Point(0, 0),
            size=Size(100, 100)
        )
        self.assertEqual(cropped_region, expected)

        region = Region(
            position=Point(x=0, y=0),
            size=Size(width=1500, height=1500)
        )
        cropped_region = instance.crop_tile(Point(x=0, y=0), region)
        expected = Region(
            position=Point(0, 0),
            size=Size(1024, 1024)
        )
        self.assertEqual(cropped_region, expected)

        region = Region(
            position=Point(x=1200, y=1200),
            size=Size(width=300, height=300)
        )
        cropped_region = instance.crop_tile(Point(x=1, y=1), region)
        expected = Region(
            position=Point(176, 176),
            size=Size(300, 300)
        )
        self.assertEqual(cropped_region, expected)

    def test_get_tiles(self):
        base_level = self.slide.levels.get_level(0)
        instance = base_level.get_instance()
        region = Region(
            position=Point(0, 0),
            size=Size(100, 100)
        )
        get_tiles = instance.get_tile_range(region, 0, '0')
        expected = Region(Point(0, 0), Size(0, 0))
        self.assertEqual(get_tiles, expected)

        region = Region(
            position=Point(0, 0),
            size=Size(1024, 1024)
        )
        get_tiles = instance.get_tile_range(region, 0, '0')
        expected = Region(Point(0, 0), Size(0, 0))
        self.assertEqual(get_tiles, expected)

        region = Region(
            position=Point(300, 400),
            size=Size(500, 500)
        )
        get_tiles = instance.get_tile_range(region, 0, '0')
        expected = Region(Point(0, 0), Size(0, 0))
        self.assertEqual(get_tiles, expected)

    def test_crop_region_to_level_size(self):
        base_level = self.slide.levels.get_level(0)
        instance = base_level.get_instance()
        image_size = base_level.size
        region = Region(
            position=Point(0, 0),
            size=Size(100, 100)
        )
        cropped_region = instance.crop_to_level_size(region)
        self.assertEqual(region.size, cropped_region.size)
        region = Region(
            position=Point(0, 0),
            size=Size(2000, 2000)
        )
        cropped_region = instance.crop_to_level_size(region)
        self.assertEqual(image_size - region.position, cropped_region.size)
        region = Region(
            position=Point(200, 300),
            size=Size(100, 100)
        )
        cropped_region = instance.crop_to_level_size(region)
        self.assertEqual(Size(0, 0), cropped_region.size)

    def test_size_class(self):
        size0 = Size(10, 10)
        size1 = Size(1, 1)
        self.assertEqual(size0 - size1, Size(9, 9))

        self.assertEqual(size0 * 2, Size(20, 20))

        self.assertEqual(size0 // 3, Size(3, 3))

        self.assertEqual(size0.to_tuple(), (10, 10))

    def test_point_class(self):
        point0 = Point(10, 10)
        point1 = Point(2, 2)
        point2 = Point(3, 3)
        size0 = Size(2, 2)

        self.assertEqual(point1 * point0, Point(20, 20))
        self.assertEqual(point0 * size0, Point(20, 20))
        self.assertEqual(point0 * 2, Point(20, 20))
        self.assertEqual(point0 // 3, Point(3, 3))
        self.assertEqual(point0 % point1, Point(0, 0))
        self.assertEqual(point0 % point2, Point(1, 1))
        self.assertEqual(point0 % size0, Point(0, 0))
        self.assertEqual(point0 + point1, Point(12, 12))
        self.assertEqual(point0 + 2, Point(12, 12))
        self.assertEqual(point0 + size0, Point(12, 12))
        self.assertEqual(point0 - point1, Point(8, 8))
        self.assertEqual(point0 - 2, Point(8, 8))
        self.assertEqual(point0 - size0, Point(8, 8))
        self.assertEqual(Point.max(point0, point1), point0)
        self.assertEqual(Point.min(point0, point1), point1)
        self.assertEqual(point0.to_tuple(), (10, 10))

    def test_valid_pixel(self):
        wsi_level = self.slide.levels.get_level(0)
        # 154x290
        region = Region(
                position=Point(0, 0),
                size=Size(100, 100)
            )
        self.assertTrue(wsi_level.valid_pixels(region))
        region = Region(
            position=Point(150, 0),
            size=Size(10, 100)
        )
        self.assertFalse(wsi_level.valid_pixels(region))

    def test_write_indexer(self):
        wsi_level = self.slide.levels.get_level(0)
        instance = wsi_level.get_instance()

        write_index = Point(0, 0)
        tile = Point(0, 0)
        region = Region(
            position=Point(0, 0),
            size=Size(2048, 2048)
        )
        tile_crop = instance.crop_tile(tile, region)
        write_index = instance._write_indexer(
            write_index,
            tile_crop.size,
            region.size,
        )
        self.assertEqual(write_index, Point(1024, 0))

        tile = Point(1, 0)
        tile_crop = instance.crop_tile(tile, region)
        write_index = instance._write_indexer(
            write_index,
            tile_crop.size,
            region.size,
            )
        self.assertEqual(write_index, Point(0, 1024))

        tile = Point(0, 1)
        tile_crop = instance.crop_tile(tile, region)
        write_index = instance._write_indexer(
            write_index,
            tile_crop.size,
            region.size,
            )
        self.assertEqual(write_index, Point(1024, 1024))

        write_index = Point(0, 0)
        tile = Point(0, 0)
        region = Region(
            position=Point(512, 512),
            size=Size(1024, 1024)
        )
        tile_crop = instance.crop_tile(tile, region)
        write_index = instance._write_indexer(
            write_index,
            tile_crop.size,
            region.size,
        )
        self.assertEqual(write_index, Point(512, 0))

        tile = Point(1, 0)
        tile_crop = instance.crop_tile(tile, region)
        write_index = instance._write_indexer(
            write_index,
            tile_crop.size,
            region.size,
            )
        self.assertEqual(write_index, Point(0, 512))

        tile = Point(0, 1)
        tile_crop = instance.crop_tile(tile, region)
        write_index = instance._write_indexer(
            write_index,
            tile_crop.size,
            region.size,
            )
        self.assertEqual(write_index, Point(512, 512))

    def test_valid_level(self):
        self.assertTrue(self.slide.levels.valid_level(1))
        self.assertFalse(self.slide.levels.valid_level(20))

    def test_get_instance(self):
        wsi_level = self.slide.levels.get_level(0)
        instance = wsi_level.get_instance()
        self.assertEqual(instance, wsi_level.default_instance)
        instance = wsi_level.get_instance(path='0')
        self.assertEqual(instance, wsi_level.default_instance)
        instance = wsi_level.get_instance(z=0)
        self.assertEqual(instance, wsi_level.default_instance)

    def test_parse_lut(self):
        ds = pydicom.dataset.Dataset()
        ds.RedPaletteColorLookupTableDescriptor = [256, 0, 8]
        ds.SegmentedRedPaletteColorLookupTableData = (
            b'\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\x00\x00'
        )
        ds.SegmentedGreenPaletteColorLookupTableData = (
            b'\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\x00\x00'
        )
        ds.SegmentedBluePaletteColorLookupTableData = (
            b'\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\xff\x00'
        )
        lut = Lut(DicomSequence([ds]))
        test = np.zeros((3, 256), dtype=np.uint16)
        test[2, :] = np.linspace(0, 255, 256, dtype=np.uint16)
        print(lut.get())
        print(test)
        self.assertTrue(np.array_equal(lut.get(), test))

        ds = pydicom.dataset.Dataset()
        ds.RedPaletteColorLookupTableDescriptor = [256, 0, 16]
        ds.SegmentedRedPaletteColorLookupTableData = (
            b'\x01\x00\x00\x01\xff\xff'
        )
        ds.SegmentedGreenPaletteColorLookupTableData = (
            b'\x01\x00\x00\x01\x00\x00'
        )
        ds.SegmentedBluePaletteColorLookupTableData = (
            b'\x01\x00\x00\x01\x00\x00'
        )
        lut = Lut(DicomSequence([ds]))
        test = np.zeros((3, 256), dtype=np.uint16)
        test[0, :] = np.linspace(0, 65535, 256, dtype=np.uint16)
        self.assertTrue(np.array_equal(lut.get(), test))

    def test_recreate_optical_module(self):
        ds = create_main_dataset()
        original_optical = Dataset()
        original_optical.OpticalPathSequence = ds.OpticalPathSequence
        original_optical.NumberOfOpticalPaths = ds.NumberOfOpticalPaths
        restored_optical_ds = self.slide.optical.insert_into_ds(Dataset())
        self.assertEqual(original_optical, restored_optical_ds)

    def test_make_optical(self):
        illumination_method = IlluminationCode('Transmission illumination')
        illumination_color = IlluminationColorCode('Full Spectrum')
        illumination = Illumination(
            illumination_method=[illumination_method],
            illumination_color=illumination_color
        )
        path = OpticalPath(
            identifier='1',
            illumination=illumination,
            photometric_interpretation='YBR_FULL_422',
            icc_profile=bytes(0)
        )
        optical = OpticalManager([path])

        self.assertEqual(optical.get('1'), path)

    def test_concept_codes(self):
        for code_class in CidConceptCode.__subclasses__():
            print(code_class)
            available_codes = code_class.cid.values()
            for available_code in available_codes:
                code = code_class(available_code.meaning)
                self.assertEqual(
                    available_code,
                    code
                )
