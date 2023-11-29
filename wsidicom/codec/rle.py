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

"""Module for handling DICOM RLE data with imagecodecs PackBits codec."""

import io
from struct import pack, unpack
from typing import List, Sequence, Tuple

import numpy as np
from wsidicom.codec.optionals import (
    packbits_decode,
    packbits_encode,
    IMAGE_CODECS_AVAILABLE,
)


class RleCodec:
    """Codec for encoding and decoding DICOM RLE data with imagecodecs PackBits codec."""

    _bits_to_compnents_per_channel = {
        8: 1,
        16: 2,
    }
    _data_type_to_components_per_channel = {
        np.dtype(np.uint8): 1,
        np.dtype(np.uint16): 2,
    }

    @classmethod
    def is_available(cls) -> bool:
        """Return true if codec is available.

        Returns
        ----------
        bool
            True if codec is available.
        """
        return IMAGE_CODECS_AVAILABLE

    @classmethod
    def encode(cls, data: np.ndarray) -> bytes:
        """Encode data to DICOM RLE.

        Parameters
        ----------
        data : np.ndarray
            Data to encode. Must be 2D or 3D array with uint8 or uint16 dtype.

        Returns
        ----------
        bytes
            Encoded data. First 64 bytes is the RLE header, followed by RLE segments."""
        if not cls.is_available():
            raise RuntimeError("Image codecs not available.")
        components_per_channel = cls._data_type_to_components_per_channel[data.dtype]
        rows = data.shape[0]
        cols = data.shape[1]
        channel_shape = (rows, cols, components_per_channel)
        # Reshape so that there is channels to iterate over
        if data.ndim == 2:
            data = data.reshape((data.shape[0], data.shape[1], 1))
        channel_count = data.shape[2]
        segment_positions: List[int] = []
        with io.BytesIO() as buffer:
            # For each channel
            for channel_index in range(channel_count):
                channel = data[:, :, channel_index]
                if channel.dtype != np.uint8:
                    # Split into high and low byte components
                    channel = channel.copy().view(np.uint8)
                # Reshape so that there is byte component to iterate over
                channel = channel.reshape(channel_shape)
                # Encode each byte component in reverse (high first)
                for component_index in reversed(range(components_per_channel)):
                    segment_positions.append(64 + buffer.tell())
                    component = channel[:, :, component_index]
                    encoded_component = packbits_encode(component)
                    buffer.write(encoded_component)
                    if len(encoded_component) % 2 == 1:
                        # Segment must be even length
                        buffer.write(b"\x00")

            encoded = buffer.getvalue()
        image_codec_header = cls._create_header(segment_positions)
        return image_codec_header + encoded

    @classmethod
    def decode(cls, data: bytes, rows: int, cols: int, bits: int) -> np.ndarray:
        """Decode DICOM RLE data.

        Parameters
        ----------
        data : bytes
            RLE data to decode. Should have a RLE header followed by RLE segments.
        rows : int
            Number of rows in decoded data.
        cols : int
            Number of columns in decoded data.
        bits : int
            Number of bits per pixel. Must be 8 or 16.
        """
        if not cls.is_available():
            raise RuntimeError("Image codecs not available.")
        components_per_channel = cls._bits_to_compnents_per_channel[bits]
        segments = cls._parse_header(data)
        channels = len(segments) // components_per_channel
        with io.BytesIO() as buffer:
            for segment_start, segment_end in segments:
                segment_data = packbits_decode(data[segment_start:segment_end])
                buffer.write(segment_data)
            # Read data is uint8
            array = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
            # Data is ordered by colunm, row, (high-low) byte, and channel
            array = array.reshape(channels, components_per_channel, rows, cols)
            # Transpose to row, col, channel, (high-low) byte
            array = array.transpose((2, 3, 0, 1))
            if bits == 16:
                # Combine high and low bytes
                array = array.copy().view(np.uint16).byteswap()
            return array.squeeze()

    @staticmethod
    def _parse_header(data: bytes) -> Sequence[Tuple[int, int]]:
        """Parse RLE header into segment start and end positions."""
        header_items = unpack("<16L", data[0:64])
        segments = header_items[0]
        segment_positions = header_items[1 : segments + 1] + (len(data),)
        return [
            (segment_positions[index], segment_positions[index + 1])
            for index in range(len(segment_positions) - 1)
        ]

    @staticmethod
    def _create_header(segment_positions: Sequence[int]):
        """Create RLE header from segment positions."""
        if len(segment_positions) < 1:
            raise ValueError("Too few segments.")
        if len(segment_positions) > 15:
            raise ValueError("Too many segments.")
        if any(segment % 2 == 1 for segment in segment_positions):
            raise ValueError("Segment positions must be even.")
        segment_positions = list(segment_positions)
        segment_count = len(segment_positions)
        empty_segment_count = 15 - segment_count
        segment_positions.extend([0 for _ in range(empty_segment_count)])
        return pack(
            "<16L",
            segment_count,
            *segment_positions,
        )
