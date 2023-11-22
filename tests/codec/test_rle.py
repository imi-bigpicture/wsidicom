import struct
import numpy as np
import pytest

from wsidicom.codec.rle import RleCodec


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
