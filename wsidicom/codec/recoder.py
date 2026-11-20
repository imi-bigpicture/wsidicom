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

"""Recoding of encoded images."""

from abc import ABCMeta, abstractmethod
from typing import Optional

from pydicom.uid import UID, JPEGBaseline8Bit

from wsidicom.codec.encoder import Encoder
from wsidicom.codec.optionals import IMAGE_CODECS_AVAILABLE, jpeg_transcode
from wsidicom.codec.settings import JpegSettings, Settings


class Recoder(metaclass=ABCMeta):
    """Recodes an encoded image into another encoded image, keeping its pixels.

    A recoder is created for encoding settings, and recodes encoded images into
    images that meet them without decoding the images. The recoded images can be
    of another transfer syntax than the images recoded.
    """

    def __init__(self, settings: Settings) -> None:
        """Create a recoder for settings.

        Parameters
        ----------
        settings: Settings
            Settings that the recoded images are to meet.
        """
        self._settings = settings

    @property
    def settings(self) -> Settings:
        """Settings that the recoded images meet."""
        return self._settings

    @property
    def transfer_syntax(self) -> UID:
        """Transfer syntax of the recoded images."""
        return self._settings.transfer_syntax

    @classmethod
    def create_for_settings(cls, settings: Settings) -> Optional["Recoder"]:
        """Create the recoder to recode images with to meet settings.

        Parameters
        ----------
        settings: Settings
            Settings that images are to meet.

        Returns
        -------
        Recoder | None
            Recoder for the settings, or None if no recoder applies to them.

        Raises
        ------
        RuntimeError
            If a recoder applies to the settings but is not available.
        """
        recoder = next(
            (
                recoder
                for recoder in cls._recoders()
                if recoder.supports_settings(settings)
            ),
            None,
        )
        if recoder is None:
            return None
        if not recoder.is_available():
            raise RuntimeError(
                f"{recoder.__name__} is needed for {settings} but is not available."
            )
        return recoder(settings)

    @staticmethod
    def _recoders() -> "tuple[type[Recoder], ...]":
        """Return the recoders, in order of preference."""
        return (JpegOptimizer,)

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if the recoder can be used."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def supports_settings(cls, settings: Settings) -> bool:
        """Return True if the recoder applies to encoding settings.

        Parameters
        ----------
        settings: Settings
            Encoding settings.

        Returns
        -------
        bool
            True if images recoded with the recoder meet the settings.
        """
        raise NotImplementedError()

    @abstractmethod
    def supports_transfer_syntax(self, transfer_syntax: UID) -> bool:
        """Return True if images of a transfer syntax can be recoded.

        Parameters
        ----------
        transfer_syntax: UID
            Transfer syntax of the encoded images.

        Returns
        -------
        bool
            True if images of the transfer syntax can be recoded.
        """
        raise NotImplementedError()

    @abstractmethod
    def recode(self, encoded: bytes) -> bytes:
        """Return an encoded image recoded.

        Parameters
        ----------
        encoded: bytes
            Encoded image.

        Returns
        -------
        bytes
            The recoded image, with the pixels of the encoded image.
        """
        raise NotImplementedError()

    @abstractmethod
    def matching_encoder(self, encoder: Encoder) -> Encoder:
        """Return an encoder that encodes images as this recoder recodes them.

        Parameters
        ----------
        encoder: Encoder
            Encoder to base the returned encoder on.

        Returns
        -------
        Encoder
            An encoder with the settings of the given encoder, changed so that the
            images it encodes are as if recoded. The given encoder if it already
            encodes them so, or if the recoder does not apply to its images.
        """
        raise NotImplementedError()


class JpegOptimizer(Recoder):
    """Recodes encoded JPEG images with Huffman tables optimal for each image.

    Applies to JPEG Baseline settings that ask for optimized Huffman tables. The
    DCT coefficients are entropy-coded again and not decoded to pixels, so the
    pixels of an optimized image are identical to those of the original. The
    optimized image is always a sequential JPEG. A progressive image, which a JPEG
    Baseline image should not be, therefore comes out conforming to JPEG Baseline,
    and can be larger than the original.
    """

    _TRANSFER_SYNTAX = JPEGBaseline8Bit
    """JPEG Baseline, the transfer syntax of the images optimized and of the result."""

    @classmethod
    def is_available(cls) -> bool:
        """Return True if images can be optimized, which needs imagecodecs."""
        return IMAGE_CODECS_AVAILABLE

    @classmethod
    def supports_settings(cls, settings: Settings) -> bool:
        """Return True for JPEG Baseline settings with optimized Huffman tables."""
        return (
            isinstance(settings, JpegSettings)
            and settings.optimize
            and settings.transfer_syntax == cls._TRANSFER_SYNTAX
        )

    def supports_transfer_syntax(self, transfer_syntax: UID) -> bool:
        """Return True for JPEG Baseline (8-bit), the only images optimized."""
        return transfer_syntax == self._TRANSFER_SYNTAX

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

    def matching_encoder(self, encoder: Encoder) -> Encoder:
        """Return the encoder, optimizing its Huffman tables if JPEG Baseline.

        A JPEG Baseline encoder that does not optimize its Huffman tables is
        replaced by one created from the same settings that does.
        """
        settings = encoder.settings
        if (
            not isinstance(settings, JpegSettings)
            or not self.supports_transfer_syntax(settings.transfer_syntax)
            or settings.optimize
        ):
            return encoder
        return Encoder.create_for_settings(settings.with_optimize(True))
