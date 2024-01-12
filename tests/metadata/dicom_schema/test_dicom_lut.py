#    Copyright 2023 SECTRA AB
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

import io
from typing import Sequence, Tuple

import numpy as np
import pytest
from pydicom import Dataset

from wsidicom.metadata.optical_path import (
    ConstantLutSegment,
    LinearLutSegment,
    Lut,
    LutDataType,
    LutSegment,
)
from wsidicom.metadata.schema.dicom.optical_path import (
    LutDicomFormatter,
    LutDicomParser,
)


class TestDicomLut:
    @pytest.mark.parametrize(
        [
            "descriptor",
            "red_lut",
            "green_lut",
            "blue_lut",
            "expected",
        ],
        [
            (
                (256, 0, 8),
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\x00\x00",
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\x00\x00",
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\xff\x00",
                Lut(
                    [ConstantLutSegment(0, 256)],
                    [ConstantLutSegment(0, 256)],
                    [LinearLutSegment(0, 255, 256)],
                    np.uint8,
                ),
            ),
            (
                (256, 0, 16),
                b"\x01\x00\x00\x01\xff\xff",
                b"\x01\x00\x00\x01\x00\x00",
                b"\x01\x00\x00\x01\x00\x00",
                Lut(
                    [LinearLutSegment(0, 65535, 256)],
                    [ConstantLutSegment(0, 256)],
                    [ConstantLutSegment(0, 256)],
                    np.uint16,
                ),
            ),
            (
                (256, 0, 16),
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\x00\x00",
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\x00\x00",
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\xff\x00",
                Lut(
                    [ConstantLutSegment(0, 256)],
                    [ConstantLutSegment(0, 256)],
                    [LinearLutSegment(0, 255, 256)],
                    np.uint16,
                ),
            ),
        ],
    )
    def test_from_dataset(
        self,
        descriptor: Tuple[int, int, int],
        red_lut: bytes,
        green_lut: bytes,
        blue_lut: bytes,
        expected: Lut,
    ):
        # Arrange
        dataset = Dataset()
        dataset.RedPaletteColorLookupTableDescriptor = descriptor
        dataset.GreenPaletteColorLookupTableDescriptor = descriptor
        dataset.BluePaletteColorLookupTableDescriptor = descriptor
        dataset.SegmentedRedPaletteColorLookupTableData = red_lut
        dataset.SegmentedGreenPaletteColorLookupTableData = green_lut
        dataset.SegmentedBluePaletteColorLookupTableData = blue_lut

        # Act
        lut = LutDicomParser.from_dataset([dataset])

        # Assert
        assert lut == expected

    @pytest.mark.parametrize(
        ["segment_data", "expected_segments"],
        [
            (
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\x00\x00",
                [ConstantLutSegment(0, 256)],
            ),
            (
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\xff\x00",
                [LinearLutSegment(0, 255, 256)],
            ),
        ],
    )
    def test_parse_segments(
        self, segment_data: bytes, expected_segments: Sequence[LutSegment]
    ):
        # Arrange

        # Act
        parsed = LutDicomParser._parse_segments(segment_data, np.uint16)

        # Assert
        assert list(parsed) == expected_segments

    @pytest.mark.parametrize(
        [
            "lut",
            "expected_descriptor",
            "expected_red_lut",
            "expected_green_lut",
            "expected_blue_lut",
        ],
        [
            (
                Lut(
                    [ConstantLutSegment(0, 256)],
                    [ConstantLutSegment(0, 256)],
                    [LinearLutSegment(0, 255, 256)],
                    np.uint8,
                ),
                (256, 0, 8),
                b"\x00\x01\x00\x01\xff\x00",
                b"\x00\x01\x00\x01\xff\x00",
                b"\x00\x01\x00\x01\xff\xff",
            ),
            (
                Lut(
                    [LinearLutSegment(0, 65535, 256)],
                    [ConstantLutSegment(0, 256)],
                    [ConstantLutSegment(0, 256)],
                    np.uint16,
                ),
                (256, 0, 16),
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\xff\xff",
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\x00\x00",
                b"\x00\x00\x01\x00\x00\x00\x01\x00\xff\x00\x00\x00",
            ),
        ],
    )
    def test_to_dataset(
        self,
        lut: Lut,
        expected_descriptor: Tuple[int, int, int],
        expected_red_lut: bytes,
        expected_green_lut: bytes,
        expected_blue_lut: bytes,
    ):
        # Arrange

        # Act
        dataset_sequence = LutDicomFormatter.to_dataset(lut)

        # Assert
        assert len(dataset_sequence) == 1
        dataset = dataset_sequence[0]
        assert isinstance(dataset, Dataset)
        assert dataset.RedPaletteColorLookupTableDescriptor == expected_descriptor
        assert dataset.GreenPaletteColorLookupTableDescriptor == expected_descriptor
        assert dataset.BluePaletteColorLookupTableDescriptor == expected_descriptor
        assert dataset.SegmentedRedPaletteColorLookupTableData == expected_red_lut
        assert dataset.SegmentedGreenPaletteColorLookupTableData == expected_green_lut
        assert dataset.SegmentedBluePaletteColorLookupTableData == expected_blue_lut

    @pytest.mark.parametrize(
        ["given_data_type", "data", "expected_data_type"],
        [
            (np.uint8, b"\x00\x01", np.uint8),
            (np.uint16, b"\x00\x01", np.uint8),
            (np.uint8, b"\x00\x00\x01\x00", np.uint16),
            (np.uint16, b"\x00\x00\x01\x00", np.uint16),
        ],
    )
    def test_determine_correct_data_type(
        self,
        given_data_type: LutDataType,
        data: bytes,
        expected_data_type: LutDataType,
    ):
        # Arrange

        # Act
        with io.BytesIO(data) as buffer:
            determined_data_type = LutDicomParser._determine_correct_data_type(
                buffer, given_data_type
            )

        # Assert
        assert determined_data_type == expected_data_type
