from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union
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
    ImplicitVRLittleEndian,
    RLELossless,
)


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

    @property
    def samples_per_pixel(self) -> int:
        """Return samples per pixel."""
        return 1 if self.channels == Channels.GRAYSCALE else 3

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
                return JpegLsLosslessSettings(bits, level=1)
            return JpegLsLosslessSettings(bits)
        if transfer_syntax in jpeg_2000_transfer_syntaxes:
            if transfer_syntax == JPEG2000:
                return Jpeg2kSettings(bits, channels)
            return Jpeg2kSettings(bits, channels, level=None)
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
                dataset.is_implicit_VR if dataset.is_implicit_VR is not None else True,
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
        return ".jpg"


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
        return ".jpg"


@dataclass
class JpegLsLosslessSettings(Settings):
    level: Optional[int] = None
    channels: Literal[Channels.GRAYSCALE] = Channels.GRAYSCALE

    @property
    def transfer_syntax(self) -> UID:
        if self.level is None:
            return JPEGLSLossless
        return JPEGLSNearLossless

    @property
    def photometric_interpretation(self) -> str:
        return "MONOCHROME2"

    @property
    def extension(self) -> str:
        return ".jls"


@dataclass
class Jpeg2kSettings(Settings):
    level: Optional[int] = 80

    @property
    def transfer_syntax(self) -> UID:
        if self.level is None:
            return JPEG2000Lossless
        return JPEG2000

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        if self.channels == Channels.YBR:
            if self.level is None:
                return "YBR_RCT"
            return "YBR_ICT"
        return "RGB"

    @property
    def extension(self) -> str:
        return ".jp2"


@dataclass
class NumpySettings(Settings):
    channels: Union[Literal[Channels.GRAYSCALE], Literal[Channels.RGB]] = Channels.RGB
    little_endian: bool = True
    explicit_vr: bool = True
    pixel_representation: int = 0

    @property
    def transfer_syntax(self) -> UID:
        if self.little_endian:
            if self.explicit_vr:
                return ExplicitVRLittleEndian
            return ImplicitVRLittleEndian
        return ExplicitVRBigEndian

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        return "RGB"

    @property
    def extension(self) -> str:
        return ".raw"


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
        return ".rle"
