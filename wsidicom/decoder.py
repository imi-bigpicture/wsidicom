import io
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, Optional, Type

from highdicom.frame import decode_frame
from imagecodecs import jpeg2k_decode, jpeg_decode, jpegls_decode, JPEG2K, JPEG8, JPEGLS
from PIL import Image
from PIL.Image import Image as PILImage
import numpy as np
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
    def is_supported(cls, transfer_syntax: UID) -> bool:
        """Return true if decoder supports transfer syntax."""
        return cls._select_decoder(transfer_syntax) is not None

    @classmethod
    def create(cls, transfer_syntax: UID, dataset: Dataset) -> "Decoder":
        """Create a decoder that supports the transfer syntax."""
        decoder = cls._select_decoder(transfer_syntax)
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
                samples_per_pixel=dataset.SamplesPerPixel,
                bits_allocated=dataset.BitsAllocated,
                bits_stored=dataset.BitsStored,
                photometric_interpretation=dataset.PhotometricInterpretation,
                pixel_representation=dataset.PixelRepresentation,
                planar_configuration=dataset.PlanarConfiguration,
            )
        else:
            raise ValueError(f"Unsupported transfer syntax: {transfer_syntax}")

    @classmethod
    def _select_decoder(cls, transfer_syntax: UID) -> Optional[Type["Decoder"]]:
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
            if not decoder.is_supported(transfer_syntax):
                return None
            return decoder
        return next(
            (
                decoder
                for decoder in decoders.values()
                if decoder.is_supported(transfer_syntax)
            ),
            None,
        )


class PillowDecoder(Decoder):
    _supported_transfer_syntaxes = [JPEGBaseline8Bit, JPEG2000, JPEG2000Lossless]

    def decode(self, frame: bytes) -> PILImage:
        return Image.open(io.BytesIO(frame))

    @classmethod
    def is_supported(cls, transfer_syntax: UID) -> bool:
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
    def is_supported(cls, transfer_syntax: UID) -> bool:
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
        "jpeg8": [
            JPEGBaseline8Bit,
            JPEGExtended12Bit,
            JPEGLosslessP14,
            JPEGLosslessSV1,
        ],
        "jpegls": [
            JPEGLSLossless,
            JPEGLSNearLossless,
        ],
        "jpeg2k": [
            JPEG2000Lossless,
            JPEG2000,
        ],
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
    def is_supported(cls, transfer_syntax: UID) -> bool:
        decoder = cls._get_decoder(transfer_syntax)
        return decoder is not None

    @classmethod
    def _get_decoder(
        cls, transfer_syntax: UID
    ) -> Optional[Callable[[bytes], np.ndarray]]:
        decoder_name = next(
            (
                decoder_name
                for decoder_name, transfer_syntaxes in cls._supported_transfer_syntaxes.items()
                if transfer_syntax in transfer_syntaxes
            ),
            None,
        )
        if decoder_name == "jpeg8" and JPEG8.available:
            return jpeg_decode
        if decoder_name == "jpegls" and JPEGLS.available:
            return jpegls_decode
        if decoder_name == "jpeg2k" and JPEG2K.available:
            return jpeg2k_decode
