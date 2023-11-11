from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Optional, Type, Union
from PIL.Image import Image as PILImage
from pydicom import Dataset
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
    ExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    RLELossless,
)
from imagecodecs import (
    JPEG2K,
    JPEG8,
    JPEGLS,
    jpeg2k_encode,
    jpeg8_encode,
    jpegls_encode,
)
from pydicom.pixel_data_handlers.util import pixel_dtype
import numpy as np
from pydicom.pixel_data_handlers.rle_handler import rle_encode_frame


class Channels(Enum):
    GRAYSCALE = "grayscale"
    YBR = "ybr"
    RGB = "rgb"

    @classmethod
    def from_photometric_interpretation(
        cls, photometric_interpretation: str
    ) -> "Channels":
        ybr_photometric_interpretations = [
            "YBR_FULL",
            "YBR_FULL_422",
            "YBR_PARTIAL_422",
            "YBR_ICT",
            "YBR_RCT",
        ]
        if photometric_interpretation == "MONOCHROME2":
            return cls.GRAYSCALE
        if photometric_interpretation in ybr_photometric_interpretations:
            return cls.YBR
        if photometric_interpretation == "RGB":
            return cls.RGB
        raise ValueError(
            f"Unsupported photometric interpretation: {photometric_interpretation}."
        )


class Subsampling(Enum):
    R444 = "444"
    R422 = "422"
    R420 = "420"
    R411 = "411"
    R440 = "440"


@dataclass
class Settings(metaclass=ABCMeta):
    bits: int = 8
    channels: Channels = Channels.YBR

    @property
    @abstractmethod
    def transfer_syntax(self) -> UID:
        """Return transfer syntax."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def extension(self) -> str:
        """Return file extension."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def photometric_interpretation(self) -> str:
        """Return photometric interpretation."""
        raise NotImplementedError()

    @property
    def allocated_bits(self) -> int:
        """Return allocated bits."""
        return self.bits // 8 * 8

    @property
    def high_bit(self) -> int:
        """Return high bit."""
        return self.bits - 1

    @classmethod
    def create(cls, dataset: Dataset, transfer_syntax: UID) -> "Settings":
        jpeg_transfer_syntaxes = [
            JPEGBaseline8Bit,
            JPEGExtended12Bit,
        ]
        jpeg_lossless_transfer_syntaxes = [
            JPEGLosslessP14,
            JPEGLosslessSV1,
        ]
        jpeg_ls_transfer_syntaxes = [
            JPEGLSLossless,
            JPEGLSNearLossless,
        ]
        jpeg_2000_transfer_syntaxes = [
            JPEG2000Lossless,
            JPEG2000,
        ]
        uncompressed_transfer_syntaxes = [
            ExplicitVRLittleEndian,
            ExplicitVRBigEndian,
        ]
        channels = Channels.from_photometric_interpretation(
            dataset.PhotometricInterpretation
        )
        bits = dataset.BitsStored
        if transfer_syntax in jpeg_transfer_syntaxes:
            return JpegSettings(bits, channels)
        if transfer_syntax in jpeg_lossless_transfer_syntaxes:
            return JpegLosslessSettings(bits, channels)
        if transfer_syntax in jpeg_ls_transfer_syntaxes:
            if channels != Channels.GRAYSCALE:
                raise ValueError("JPEG-LS encoder only supports grayscale images.")
            if transfer_syntax == JPEGLSNearLossless:
                return JpegLsNearLosslessSettings(bits)
            return JpegLsLosslessSettings(bits)
        if transfer_syntax in jpeg_2000_transfer_syntaxes:
            if transfer_syntax == JPEG2000:
                return Jpeg2kSettings(bits, channels)
            return Jpeg2kLosslessSettings(bits, channels)
        if transfer_syntax == RLELossless:
            return RleSettings(bits, channels)
        if transfer_syntax in uncompressed_transfer_syntaxes:
            if channels == Channels.YBR:
                raise ValueError("Numpy encoder does not support ybr images.")
            return NumpySettings(
                bits,
                channels,
                dataset.PixelRepresentation,
                dataset.is_little_endian
                if dataset.is_little_endian is not None
                else True,
            )
        raise ValueError(f"Unsupported transfer syntax: {transfer_syntax}.")


@dataclass
class JpegSettings(Settings):
    quality: int = 80
    subsampling: Optional[Subsampling] = Subsampling.R420

    @property
    def transfer_syntax(self) -> UID:
        if self.bits == 8:
            return JPEGBaseline8Bit
        return JPEGExtended12Bit

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        if self.channels == Channels.YBR:
            return "YBR_FULL_422"
        return "RGB"

    @property
    def extension(self) -> str:
        return "jpg"


@dataclass
class JpegLosslessSettings(Settings):
    predictor: Optional[int] = None

    @property
    def transfer_syntax(self) -> UID:
        if self.predictor is None:
            return JPEGLosslessSV1
        return JPEGLosslessP14

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        if self.channels == Channels.YBR:
            return "YBR_FULL_422"
        return "RGB"

    @property
    def extension(self) -> str:
        return "jpg"


@dataclass
class JpegLsNearLosslessSettings(Settings):
    level: int = 1
    channels: Literal[Channels.GRAYSCALE] = Channels.GRAYSCALE

    @property
    def transfer_syntax(self) -> UID:
        return JPEGLSNearLossless

    @property
    def photometric_interpretation(self) -> str:
        return "MONOCHROME2"

    @property
    def extension(self) -> str:
        return "jls"


@dataclass
class JpegLsLosslessSettings(Settings):
    @property
    def transfer_syntax(self) -> UID:
        return JPEGLSLossless

    @property
    def photometric_interpretation(self) -> str:
        return "MONOCHROME2"

    @property
    def extension(self) -> str:
        return "jls"


@dataclass
class Jpeg2kSettings(Settings):
    level: float = 80

    @property
    def transfer_syntax(self) -> UID:
        return JPEG2000

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        if self.channels == Channels.YBR:
            return "YBR_ICT"
        return "RGB"

    @property
    def extension(self) -> str:
        return "jp2"


@dataclass
class Jpeg2kLosslessSettings(Settings):
    @property
    def transfer_syntax(self) -> UID:
        return JPEG2000Lossless

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        if self.channels == Channels.YBR:
            return "YBR_RCT"
        return "RGB"

    @property
    def extension(self) -> str:
        return "jp2"


@dataclass
class NumpySettings(Settings):
    channels: Union[Literal[Channels.GRAYSCALE], Literal[Channels.RGB]] = Channels.RGB
    pixel_representation: int = 0
    little_endian: bool = True

    @property
    def transfer_syntax(self) -> UID:
        if self.little_endian:
            return ExplicitVRLittleEndian
        return ExplicitVRBigEndian

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        return "RGB"

    @property
    def extension(self) -> str:
        return "raw"


@dataclass
class RleSettings(Settings):
    @property
    def transfer_syntax(self) -> UID:
        return RLELossless

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        if self.channels == Channels.YBR:
            return "YBR_FULL"
        return "RGB"

    @property
    def extension(self) -> str:
        return "rle"


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
