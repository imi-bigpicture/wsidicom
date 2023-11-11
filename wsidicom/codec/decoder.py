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
        """Decode frame into Pillow Image."""
        raise NotImplementedError()

    @classmethod
    def is_supported(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int
    ) -> bool:
        """Return true if decoder supports transfer syntax."""
        if bits != 8 and samples_per_pixel != 1:
            # Pillow only supports 8 bit color images
            return False

        return cls._select_decoder(transfer_syntax, samples_per_pixel, bits) is not None

    @classmethod
    def create(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int, dataset: Dataset
    ) -> "Decoder":
        """Create a decoder that supports the transfer syntax."""
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
        if transfer_syntax not in cls._supported_transfer_syntaxes:
            return None
        decoder, codec = cls._supported_transfer_syntaxes[transfer_syntax]
        assert decoder is not None and codec is not None
        if not codec.available:
            return None
        return decoder
