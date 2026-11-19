#    Copyright 2026 SECTRA AB
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

"""Lossless optimization of encoded JPEG images."""

from pydicom.uid import UID, JPEGBaseline8Bit

from wsidicom.codec.optionals import IMAGE_CODECS_AVAILABLE, jpeg_transcode
from wsidicom.codec.recoder import Recoder


class JpegOptimizer(Recoder):
    """Recodes encoded JPEG images with Huffman tables optimal for each image.

    The DCT coefficients are entropy-coded again and not decoded to pixels, so the
    pixels of an optimized image are identical to those of the original. The
    optimized image is always a sequential JPEG. A progressive image, which a JPEG
    Baseline image should not be, therefore comes out conforming to JPEG Baseline,
    and can be larger than the original.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Return True if images can be optimized, which needs imagecodecs."""
        return IMAGE_CODECS_AVAILABLE

    @classmethod
    def supports_transfer_syntax(cls, transfer_syntax: UID) -> bool:
        """Return True if images of a transfer syntax can be optimized.

        Parameters
        ----------
        transfer_syntax: UID
            Transfer syntax of the encoded images.

        Returns
        -------
        bool
            True for JPEG Baseline (8-bit).
        """
        return transfer_syntax == JPEGBaseline8Bit

    def recode(self, encoded: bytes) -> bytes:
        """Return an encoded JPEG image recoded with optimal Huffman tables.

        Parameters
        ----------
        encoded: bytes
            Encoded JPEG image.

        Returns
        -------
        bytes
            The image entropy-coded with Huffman tables optimal for it, with
            unchanged pixels.
        """
        if jpeg_transcode is None:
            raise RuntimeError("Optimizing JPEG images requires imagecodecs.")
        return bytes(jpeg_transcode(encoded, optimize=True))
