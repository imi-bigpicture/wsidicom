import io
from abc import ABCMeta, abstractmethod
from typing import List, Type

from highdicom.frame import decode_frame
from imagecodecs.imagecodecs import jpeg2k_decode, jpeg_decode, jpegls_decode
from PIL import Image
from PIL.Image import Image as PILImage
from pydicom import Dataset
from pydicom.pixel_data_handlers import (
    gdcm_handler,
    jpeg_ls_handler,
    numpy_handler,
    pillow_handler,
    pylibjpeg_handler,
    rle_handler,
)
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

from wsidicom.geometry import Size


class Decoder(metaclass=ABCMeta):
    @abstractmethod
    def decode(self, frame: bytes) -> PILImage:
        """Decode frame into Pillow Image."""
        raise NotImplementedError()

    @classmethod
    def is_supported(cls, transfer_syntax: UID) -> bool:
        """Return true if decoder supports transfer syntax."""
        decoders: List[Type[Decoder]] = [
            PillowDecoder,
            ImageCodecsDecoder,
            ImageCodecsJpegLsDecoder,
            ImageCodecsJpeg2000Decoder,
            PydicomDecoder,
        ]
        return any(decoder.is_supported(transfer_syntax) for decoder in decoders)

    @classmethod
    def create(cls, transfer_syntax: UID, dataset: Dataset) -> "Decoder":
        """Create a decoder that supports the transfer syntax."""
        if PillowDecoder.is_supported(transfer_syntax):
            return PillowDecoder()
        elif ImageCodecsDecoder.is_supported(transfer_syntax):
            return ImageCodecsDecoder(transfer_syntax)
        elif PydicomDecoder.is_supported(transfer_syntax):
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
            for handler in [
                pillow_handler,
                numpy_handler,
                gdcm_handler,
                jpeg_ls_handler,
                pylibjpeg_handler,
                rle_handler,
            ]
            if handler.is_available()
        )
        return any(
            available_handler.supports_transfer_syntax(transfer_syntax)
            for available_handler in available_handlers
        )


class ImageCodecsDecoder(Decoder):
    _supported_transfer_syntaxes = [
        JPEGBaseline8Bit,
        JPEGExtended12Bit,
        JPEGLosslessP14,
        JPEGLosslessSV1,
        JPEGLSLossless,
        JPEGLSNearLossless,
        JPEG2000Lossless,
        JPEG2000,
    ]
    _jpeg_transfer_syntaxes = [
        JPEGBaseline8Bit,
        JPEGExtended12Bit,
        JPEGLosslessP14,
        JPEGLosslessSV1,
    ]
    _jpeg_ls_transfer_syntaxes = [JPEGLSLossless, JPEGLSNearLossless]
    _jpeg2000_transfer_syntaxes = [JPEG2000Lossless, JPEG2000]

    def __init__(self, transfer_syntax: UID) -> None:
        if transfer_syntax in self._jpeg_transfer_syntaxes:
            self._decoder = jpeg_decode
        elif transfer_syntax in self._jpeg_ls_transfer_syntaxes:
            self._decoder = jpegls_decode
        elif transfer_syntax in self._jpeg2000_transfer_syntaxes:
            self._decoder = jpeg2k_decode
        else:
            raise ValueError(f"Unsupported transfer syntax: {transfer_syntax}.")

    def decode(self, frame: bytes) -> PILImage:
        decoded = self._decoder(frame)
        return Image.fromarray(decoded)

    @classmethod
    def is_supported(cls, transfer_syntax: UID) -> bool:
        return transfer_syntax in cls._supported_transfer_syntaxes
