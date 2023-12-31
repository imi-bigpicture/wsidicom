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

import struct
import numpy as np
import pytest

from wsidicom.codec.rle import RleCodec


@pytest.mark.skipif(not RleCodec.is_available(), reason="Image codecs not available")
class TestRle:
    @pytest.mark.parametrize(
        ["data", "expected_segment_count"],
        [
            (np.array([[121, 121], [27, 63]], dtype=np.uint8), 1),
            (np.array([[121, 121], [27, 63]], dtype=np.uint16), 2),
            (
                np.array(
                    [
                        [[121, 122, 123], [121, 122, 123]],
                        [[27, 28, 29], [63, 64, 65]],
                    ],
                    dtype=np.uint8,
                ),
                3,
            ),
            (
                np.array(
                    [
                        [[121, 122, 123], [121, 122, 123]],
                        [[27, 28, 29], [63, 64, 65]],
                    ],
                    dtype=np.uint16,
                ),
                6,
            ),
        ],
    )
    def test_encode(self, data: np.ndarray, expected_segment_count: int):
        # Arrange
        # Act
        encoded = RleCodec.encode(data)

        # Assert
        header = encoded[0:64]
        segment_count = struct.unpack("<L", header[0:4])[0]
        assert segment_count == expected_segment_count
        segment_positions = struct.unpack("<15L", header[4:64])
        assert segment_positions[0] == 64
        assert all(segment % 2 == 0 for segment in segment_positions[0:segment_count])
        assert all(segment == 0 for segment in segment_positions[segment_count:])

    @pytest.mark.parametrize(
        ["data", "expected_data"],
        [
            (
                b"\x01\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                + b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                + b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                + b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xffy"
                + b"\x01\x1b?\x00",
                np.array([[121, 121], [27, 63]], dtype=np.uint8),
            ),
            (
                b"\x02\x00\x00\x00@\x00\x00\x00D\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                + b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                + b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                + b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x00"
                + b"\xff\x00\xffy\x01\x1b?\x00",
                np.array([[121, 121], [27, 63]], dtype=np.uint16),
            ),
            (
                b"\x03\x00\x00\x00@\x00\x00\x00F\x00\x00\x00L\x00\x00\x00\x00\x00\x00"
                + b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                + b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                + b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xffy\x01\x1b?"
                + b"\x00\xffz\x01\x1c@\x00\xff{\x01\x1dA\x00",
                np.array(
                    [
                        [[121, 122, 123], [121, 122, 123]],
                        [[27, 28, 29], [63, 64, 65]],
                    ],
                    dtype=np.uint8,
                ),
            ),
            (
                b"\x06\x00\x00\x00@\x00\x00\x00D\x00\x00\x00J\x00\x00\x00N\x00\x00"
                + b"\x00T\x00\x00\x00X\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                + b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
                + b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x00\xff\x00\xffy"
                + b"\x01\x1b?\x00\xff\x00\xff\x00\xffz\x01\x1c@\x00\xff\x00\xff\x00"
                + b"\xff{\x01\x1dA\x00",
                np.array(
                    [
                        [[121, 122, 123], [121, 122, 123]],
                        [[27, 28, 29], [63, 64, 65]],
                    ],
                    dtype=np.uint16,
                ),
            ),
        ],
    )
    def test_decode(self, data: bytes, expected_data: np.ndarray):
        # Arrange
        rows = expected_data.shape[0]
        columns = expected_data.shape[1]
        if expected_data.dtype == np.uint8:
            bits = 8
        else:
            bits = 16

        # Act
        decoded = RleCodec.decode(data, rows, columns, bits)

        # Assert
        assert np.array_equal(decoded, expected_data)
