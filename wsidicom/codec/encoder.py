from abc import ABCMeta, abstractmethod
from typing import Union
from PIL.Image import Image as PILImage
from pydicom import Dataset
from pydicom.uid import (
    JPEG2000,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGExtended12Bit,
    JPEGLosslessP14,
    JPEGLosslessSV1,
    JPEGLSLossless,
    JPEGLSNearLossless,
)
from imagecodecs import (
    JPEG8,
    jpeg2k_encode,
    jpeg8_encode,
    jpegls_encode,
)
from pydicom.pixel_data_handlers.util import pixel_dtype
import numpy as np
from pydicom.pixel_data_handlers.rle_handler import rle_encode_frame
from wsidicom.codec.settings import (
    Channels,
    JpegSettings,
    JpegLsNearLosslessSettings,
    JpegLosslessSettings,
    Jpeg2kLosslessSettings,
    Jpeg2kSettings,
    JpegLsLosslessSettings,
    NumpySettings,
    RleSettings,
    Settings,
)


class Encoder(metaclass=ABCMeta):
    @abstractmethod
    def encode(self, image: PILImage) -> bytes:
        """Encode image into bytes."""
        raise NotImplementedError()

    @classmethod
    def create(cls, settings: Settings) -> "Encoder":
        """Create an encoder that supports the transfer syntax."""
        if isinstance(settings, (JpegSettings, JpegLosslessSettings)):
            return JpegEncoder(settings)
        if isinstance(settings, (JpegLsNearLosslessSettings, JpegLsLosslessSettings)):
            return JpegLsEncoder(settings)
        if isinstance(settings, (Jpeg2kSettings, Jpeg2kLosslessSettings)):
            return Jpeg2kEncoder(settings)
        if isinstance(settings, NumpySettings):
            return NumpyEncoder(settings)
        if isinstance(settings, RleSettings):
            return RleEncoder()
        if isinstance(settings, JpegLsLosslessSettings):
            return JpegLsEncoder(settings)
        raise ValueError(f"Unsupported encoder settings: {settings}")


class JpegEncoder(Encoder):
    _supported_transfer_syntaxes = [
        JPEGBaseline8Bit,
        JPEGExtended12Bit,
        JPEGLosslessP14,
        JPEGLosslessSV1,
    ]

    def __init__(
        self,
        settings: Union[JpegSettings, JpegLosslessSettings],
    ) -> None:
        self._bits = settings.bits
        if self._bits == 8:
            self._dtype = np.uint8
        else:
            self._dtype = np.uint16
        if settings.channels == Channels.GRAYSCALE:
            self._output_colorspace = JPEG8.CS.GRAYSCALE
        elif settings.channels == Channels.YBR:
            self._output_colorspace = JPEG8.CS.YCbCr
        elif settings.channels == Channels.RGB:
            self._output_colorspace = JPEG8.CS.RGB
        else:
            raise ValueError(f"Unsupported channels: {settings.channels}.")
        if isinstance(settings, JpegSettings):
            self._lossless = False
            self._predictor = None
            self._level = settings.quality
            self._subsampling = (
                settings.subsampling.value if settings.subsampling else None
            )
        elif isinstance(settings, JpegLosslessSettings):
            self._lossless = True
            self._predictor = settings.predictor
            self._level = None
            self._subsampling = None
        else:
            raise ValueError(f"Unsupported encoder settings: {type(settings)}.")

    def encode(self, image: PILImage) -> bytes:
        return jpeg8_encode(
            np.array(image).astype(self._dtype),
            level=self._level,
            lossless=self._lossless,
            bitspersample=self._bits,
            subsampling=self._subsampling,
            outcolorspace=self._output_colorspace,
        )


class JpegLsEncoder(Encoder):
    """Encoder that uses jpegls to encode image."""

    _supported_transfer_syntaxes = [
        JPEGLSLossless,
        JPEGLSNearLossless,
    ]

    def __init__(
        self,
        settings: Union[JpegLsLosslessSettings, JpegLsNearLosslessSettings],
    ) -> None:
        self._bits = settings.bits
        if isinstance(settings, JpegLsLosslessSettings):
            self._level = 0
        elif isinstance(settings, JpegLsNearLosslessSettings):
            self._level = settings.level
        else:
            raise ValueError(f"Unsupported encoder settings: {type(settings)}.")

    def encode(self, image: PILImage) -> bytes:
        """Encode image into bytes."""
        return jpegls_encode(np.array(image), level=self._level)


class Jpeg2kEncoder(Encoder):
    """Encoder that uses jpeg2k to encode image."""

    _supported_transfer_syntaxes = [
        JPEG2000Lossless,
        JPEG2000,
    ]

    def __init__(self, settings: Union[Jpeg2kSettings, Jpeg2kLosslessSettings]) -> None:
        self._bits = settings.bits
        if settings.channels == Channels.YBR:
            self._multiple_component_transform = True
        else:
            self._multiple_component_transform = False
        if isinstance(settings, Jpeg2kSettings):
            self._level = settings.level
            self._reversible = False
        elif isinstance(settings, Jpeg2kLosslessSettings):
            self._level = 0
            self._reversible = True
        else:
            raise ValueError(f"Unsupported encoder settings: {type(settings)}.")

    def encode(self, image: PILImage) -> bytes:
        """Encode image into bytes."""
        return jpeg2k_encode(
            np.array(image),
            level=self._level,
            reversible=self._reversible,
            bitspersample=self._bits,
            codecformat="J2K",
            mct=self._multiple_component_transform,
        )


class NumpyEncoder(Encoder):
    """Encoder that uses numpy to encode image."""

    def __init__(self, settings: NumpySettings) -> None:
        dataset = Dataset()
        dataset.BitsAllocated = settings.allocated_bits
        dataset.PixelRepresentation = settings.pixel_representation
        dataset.is_little_endian = settings.little_endian
        self._dtype = pixel_dtype(dataset)

    def encode(self, image: PILImage) -> bytes:
        """Encode image into bytes."""
        return np.array(image).astype(self._dtype).tobytes()


class RleEncoder(Encoder):
    """Encoder that uses rle encoder to encode image."""

    def encode(self, image: PILImage) -> bytes:
        """Encode image into bytes."""
        return rle_encode_frame(np.array(image))
