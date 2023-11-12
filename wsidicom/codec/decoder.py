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

"""Module with decoders for image data."""

import io
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, Optional, Type

import numpy as np
from highdicom.frame import decode_frame
from imagecodecs import (
    JPEG2K,
    JPEG8,
    JPEGLS,
    jpeg2k_decode,
    jpeg8_decode,
    jpegls_decode,
)
from PIL import Image
from PIL.Image import Image as PILImage
from pydicom import Dataset
from pydicom import config as pydicom_config
from pydicom.uid import (
    JPEG2000,
    UID,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGExtended12Bit,
    JPEGLosslessP14,
    JPEGLosslessSV1,
    JPEGLSLossless,
    JPEGLSNearLossless,
)

from wsidicom import config
from wsidicom.geometry import Size


class Decoder(metaclass=ABCMeta):
    @abstractmethod
    def decode(self, frame: bytes) -> PILImage:
        """Decode frame into Pillow Image.

        Parameters
        ----------
        frame : bytes
            Encoded frame.

        Returns
        -------
        PIL.Image
            Pillow Image.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def is_supported(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int
    ) -> bool:
        """Return true if decoder supports transfer syntax.

        Parameters
        ----------
        transfer_syntax : pydicom.uid.UID
            Transfer syntax.
        samples_per_pixel : int
            Number of samples per pixel.
        bits : int
            Number of bits per sample.

        Returns
        -------
        bool
            True if decoder supports transfer syntax.

        """
        raise NotImplementedError()

    @classmethod
    def has_decoder(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int
    ) -> bool:
        """Return true if decoder supports transfer syntax.

        Parameters
        ----------
        transfer_syntax : pydicom.uid.UID
            Transfer syntax.
        samples_per_pixel : int
            Number of samples per pixel.
        bits : int
            Number of bits per sample.

        Returns
        -------
        bool
            True there is a decoder that supports transfer syntax.

        """
        if bits != 8 and samples_per_pixel != 1:
            # Pillow only supports 8 bit color images
            return False

        return cls._select_decoder(transfer_syntax, samples_per_pixel, bits) is not None

    @classmethod
    def create(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int, dataset: Dataset
    ) -> "Decoder":
        """Create a decoder that supports the transfer syntax.

        Parameters
        ----------
        transfer_syntax : UID
            Transfer syntax.
        samples_per_pixel : int
            Number of samples per pixel.
        bits : int
            Number of bits per sample.
        dataset : Dataset
            DICOM dataset.

        Returns
        -------
        Decoder
            Decoder for transfer syntax.

        """
        if bits != 8 and samples_per_pixel != 1:
            # Pillow only supports 8 bit color images
            raise ValueError(
                f"Non-supported combination of bits {bits} and  "
                f"samples per pixel {samples_per_pixel}. "
                "Non-8 bit images are only supported for grayscale images."
            )
        decoder = cls._select_decoder(transfer_syntax, samples_per_pixel, bits)
        if decoder is None:
            raise ValueError(f"Unsupported transfer syntax: {transfer_syntax}.")
        if decoder == PillowDecoder:
            return PillowDecoder()
        elif decoder == ImageCodecsDecoder:
            return ImageCodecsDecoder(transfer_syntax)
        elif decoder == PydicomDecoder:
            return PydicomDecoder(
                transfer_syntax=transfer_syntax,
                size=Size(dataset.Rows, dataset.Columns),
                samples_per_pixel=samples_per_pixel,
                bits_allocated=bits,
                bits_stored=dataset.BitsStored,
                photometric_interpretation=dataset.PhotometricInterpretation,
                pixel_representation=dataset.PixelRepresentation,
                planar_configuration=dataset.PlanarConfiguration,
            )
        raise ValueError(f"Unsupported transfer syntax: {transfer_syntax}")

    @classmethod
    def _select_decoder(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int
    ) -> Optional[Type["Decoder"]]:
        """Select decoder based on transfer syntax.

        Parameters
        ----------
        transfer_syntax : UID
            Transfer syntax.
        samples_per_pixel : int
            Number of samples per pixel.
        bits : int
            Number of bits per sample.

        Returns
        -------
        Optional[Type[Decoder]]
            Decoder class. None if no decoder supports transfer syntax.

        """
        decoders: Dict[str, Type[Decoder]] = {
            "pillow": PillowDecoder,
            "image_codecs": ImageCodecsDecoder,
            "pydicom": PydicomDecoder,
        }
        if config.settings.prefered_decoder is not None:
            if config.settings.prefered_decoder not in decoders:
                raise ValueError(
                    f"Unknown prefered decoder: {config.settings.prefered_decoder}."
                )
            decoder = decoders[config.settings.prefered_decoder]
            if not decoder.is_supported(transfer_syntax, samples_per_pixel, bits):
                return None
            return decoder
        return next(
            (
                decoder
                for decoder in decoders.values()
                if decoder.is_supported(transfer_syntax, samples_per_pixel, bits)
            ),
            None,
        )


class PillowDecoder(Decoder):
    """Decoder that uses Pillow to decode images."""

    _supported_transfer_syntaxes = [JPEGBaseline8Bit, JPEG2000, JPEG2000Lossless]

    def decode(self, frame: bytes) -> PILImage:
        return Image.open(io.BytesIO(frame))

    @classmethod
    def is_supported(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int
    ) -> bool:
        if bits not in [8, 16]:
            # Pillow only supports 8 and 16 bit images
            return False
        if bits != 8 and samples_per_pixel != 1:
            # Pillow only supports 8 bit color images
            return False
        return transfer_syntax in cls._supported_transfer_syntaxes


class PydicomDecoder(Decoder):
    """Decoder that uses pydicom to decode images."""

    def __init__(
        self,
        transfer_syntax: UID,
        size: Size,
        samples_per_pixel: int,
        bits_allocated: int,
        bits_stored: int,
        photometric_interpretation: str,
        pixel_representation: int,
        planar_configuration: int,
    ):
        """Initialize decoder.

        Parameters
        ----------
        transfer_syntax : UID
            Transfer syntax.
        size : Size
            Size of image.
        samples_per_pixel : int
            Number of samples per pixel.
        bits_allocated : int
            Number of bits allocated.
        bits_stored : int
            Number of bits stored.
        photometric_interpretation : str
            Photometric interpretation.
        pixel_representation : int
            Pixel representation.
        planar_configuration : int
            Planar configuration.
        """
        self._transfer_syntax = transfer_syntax
        self._size = size
        self._samples_per_pixel = samples_per_pixel
        self._bits_allocated = bits_allocated
        self._bits_stored = bits_stored
        self._photometric_interpretation = photometric_interpretation
        self._pixel_representation = pixel_representation
        self._planar_configuration = planar_configuration

    def decode(self, frame: bytes) -> PILImage:
        array = decode_frame(
            value=frame,
            transfer_syntax_uid=self._transfer_syntax,
            rows=self._size.height,
            columns=self._size.width,
            samples_per_pixel=self._samples_per_pixel,
            bits_allocated=self._bits_allocated,
            bits_stored=self._bits_stored,
            photometric_interpretation=self._photometric_interpretation,
            pixel_representation=self._pixel_representation,
            planar_configuration=self._planar_configuration,
        )
        return Image.fromarray(array)

    @classmethod
    def is_supported(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int
    ) -> bool:
        available_handlers = (
            handler
            for handler in pydicom_config.pixel_data_handlers
            if handler.is_available()
        )
        return any(
            available_handler.supports_transfer_syntax(transfer_syntax)
            for available_handler in available_handlers
        )


class ImageCodecsDecoder(Decoder):
    """Decoder that uses imagecodecs to decode images."""

    _supported_transfer_syntaxes = {
        JPEGBaseline8Bit: (jpeg8_decode, JPEG8),
        JPEGExtended12Bit: (jpeg8_decode, JPEG8),
        JPEGLosslessP14: (jpeg8_decode, JPEG8),
        JPEGLosslessSV1: (jpeg8_decode, JPEG8),
        JPEGLSLossless: (jpegls_decode, JPEGLS),
        JPEGLSNearLossless: (jpegls_decode, JPEGLS),
        JPEG2000Lossless: (jpeg2k_decode, JPEG2K),
        JPEG2000: (jpeg2k_decode, JPEG2K),
    }

    def __init__(self, transfer_syntax: UID) -> None:
        """Initialize decoder.

        Parameters
        ----------
        transfer_syntax : UID
            Transfer syntax.
        """
        decoder = self._get_decoder(transfer_syntax)
        if decoder is None:
            raise ValueError(f"Unsupported transfer syntax: {transfer_syntax}.")
        self._decoder = decoder

    def decode(self, frame: bytes) -> PILImage:
        decoded = self._decoder(frame)
        return Image.fromarray(decoded)

    @classmethod
    def is_supported(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int
    ) -> bool:
        decoder = cls._get_decoder(transfer_syntax)
        return decoder is not None

    @classmethod
    def _get_decoder(
        cls, transfer_syntax: UID
    ) -> Optional[Callable[[bytes], np.ndarray]]:
        """Get imagecodes decoder for transfer syntax.

        Parameters
        ----------
        transfer_syntax : UID
            Transfer syntax.

        Returns
        -------
        Optional[Callable[[bytes], np.ndarray]]
            Decoder. None if no decoder supports transfer syntax.
        """
        if transfer_syntax not in cls._supported_transfer_syntaxes:
            return None
        decoder, codec = cls._supported_transfer_syntaxes[transfer_syntax]
        assert decoder is not None and codec is not None
        if not codec.available:
            return None
        return decoder
