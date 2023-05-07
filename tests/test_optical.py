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
from typing import List

import numpy as np
import pytest
from parameterized import parameterized
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence

from tests.data_gen import create_layer_file, create_main_dataset
from wsidicom import WsiDicom
from wsidicom.conceptcode import (
    IlluminationCode,
    IlluminationColorCode,
)
from wsidicom.optical import Illumination, Lut, OpticalManager, OpticalPath


@pytest.mark.unittest
class WsiDicomOpticalTests(unittest.TestCase):
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
