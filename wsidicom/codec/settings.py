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

"""Module with settings for encoding image data."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union
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
    """Color channels in encoded image."""

    GRAYSCALE = "grayscale"
    YBR = "ybr"
    RGB = "rgb"

    @classmethod
    def from_photometric_interpretation(
        cls, photometric_interpretation: str
    ) -> "Channels":
        """Return channels matching photometric interpretation."""
        ybr_photometric_interpretations = [
            "YBR_FULL",
            "YBR_FULL_422",
            "YBR_PARTIAL_422",
            "YBR_ICT",
            "YBR_RCT",
            "YBR",
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
    """JPEG chroma subsampling settings."""

    R444 = "Full horizontal, full vertical (4:4:4)"
    R422 = "Half horisontal, full vertical (4:2:2)"
    R420 = "Half horisontal, half vertical (4:2:0)"
    R411 = "Quarter horisontal, full vertical (4:1:1)"
    R440 = "Full horizontal, half vertical (4:4:0)"

    @classmethod
    def from_string(cls, string: Optional[str]) -> "Subsampling":
        """Return subsampling matching string."""
        if string is None or string == "444":
            return Subsampling.R444
        if string == "422":
            return Subsampling.R422
        if string == "420":
            return Subsampling.R420
        if string == "411":
            return Subsampling.R411
        if string == "440":
            return Subsampling.R440
        raise ValueError(f"Unsupported subsampling: {string}.")


class Settings(metaclass=ABCMeta):
    def __init__(self, bits: int, channels: Channels):
        """
        Initialize encoding settings.

        Parameters:
        ----------
            bits: int
                Number of bits per pixel.
            channels: Channels
                Color channels in encoded image."""
        self._bits = bits
        self._channels = channels

    @property
    def bits(self) -> int:
        """Return bits."""
        return self._bits

    @property
    def channels(self) -> Channels:
        """Return channels."""
        return self._channels

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
    def create(
        cls,
        transfer_syntax: UID,
        bits: int,
        photometric_interpretation: str,
    ) -> "Settings":
        """Create settings based on properties dataset and transfer syntax.

        Parameters:
        ----------
            transfer_syntax: UID
                Transfer syntax to create settings for.
            bits: int
                Number of bits per pixel.
            photometric_interpretation: str
                Photometric interpretation of image.

        Returns:
        ----------
            Settings
                Settings matching dataset and transfer syntax.
        """
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
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
            ExplicitVRBigEndian,
        ]
        channels = Channels.from_photometric_interpretation(photometric_interpretation)
        if transfer_syntax in jpeg_transfer_syntaxes:
            return JpegSettings(bits=bits, channels=channels)
        if transfer_syntax in jpeg_lossless_transfer_syntaxes:
            return JpegLosslessSettings(bits=bits, channels=channels)
        if transfer_syntax in jpeg_ls_transfer_syntaxes:
            if channels == Channels.YBR:
                raise ValueError("JPEG-LS does not support conversion to ybr.")
            if transfer_syntax == JPEGLSNearLossless:
                return JpegLsSettings(level=1, bits=bits, channels=channels)
            return JpegLsSettings(level=0, bits=bits, channels=channels)
        if transfer_syntax in jpeg_2000_transfer_syntaxes:
            if transfer_syntax == JPEG2000:
                return Jpeg2kSettings(bits=bits, channels=channels)
            return Jpeg2kSettings(level=0, bits=bits, channels=channels)
        if transfer_syntax == RLELossless:
            return RleSettings(bits=bits, channels=channels)
        if transfer_syntax in uncompressed_transfer_syntaxes:
            if channels == Channels.YBR:
                raise ValueError("Numpy encoder does not support ybr images.")
            return NumpySettings(
                bits=bits,
                channels=channels,
                little_endian=transfer_syntax.is_little_endian,
                implicit_vr=transfer_syntax.is_implicit_VR,
            )
        raise ValueError(f"Unsupported transfer syntax: {transfer_syntax}.")


class JpegSettings(Settings):
    """JPEG encoding settings."""

    def __init__(
        self,
        quality: int = 80,
        bits: int = 8,
        channels: Channels = Channels.YBR,
        subsampling: Subsampling = Subsampling.R422,
    ):
        """
        Initialize JPEG encoding settings.

        Parameters:
        ----------
            quality: int = 80
                JPEG quality factor. 0-100, recommended to not use higher than 95.
            bits: int
                Number of bits per pixel.
            channels: Channels = Channels.YBR
                Color channels in encoded image.
            subsampling: Subsampling = Subsampling.R420
                Chroma subsampling settings.
        """
        self._quality = quality
        self._subsampling = subsampling
        super().__init__(bits, channels)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(quality={self.quality}, bits={self.bits}, "
            f"channels={self.channels}, subsampling={self.subsampling})"
        )

    @property
    def quality(self) -> int:
        """Return quality."""
        return self._quality

    @property
    def subsampling(self) -> Subsampling:
        """Return subsampling."""
        return self._subsampling

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


class JpegLosslessSettings(Settings):
    def __init__(
        self,
        predictor: int = 1,
        bits: int = 8,
        channels: Channels = Channels.YBR,
    ):
        """
        Initialize JPEG lossless encoding settings.

        Parameters:
        ----------
            predictor: int = 1
                JPEG predictor setting. 1-7. Use 1 for SV1.
            bits: int = 8
                Number of bits per pixel.
            channels: Channels = Channels.YBR
                Color channels in encoded image.
        """
        self._predictor = predictor
        super().__init__(bits, channels)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(predictor={self.predictor}, bits={self.bits}, "
            f"channels={self.channels})"
        )

    @property
    def predictor(self) -> int:
        """Return predictor."""
        return self._predictor

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
            return "YBR_FULL"
        return "RGB"

    @property
    def extension(self) -> str:
        return ".jpg"


@dataclass
class JpegLsSettings(Settings):
    def __init__(
        self,
        level: int = 0,
        bits: int = 8,
        channels: Channels = Channels.RGB,
    ):
        """
        Initialize JPEG-LS lossless encoding settings. Only supports grayscale images.

        Parameters:
        ----------
            level: int = 0
                JPEG-LS near-lossless level. 0-9. 0 is lossless.
            bits: int = 8
                Number of bits per pixel.
        """
        self._level = level
        super().__init__(bits, channels)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(level={self.level}, bits={self.bits})"

    @property
    def level(self) -> int:
        """Return level."""
        return self._level

    @property
    def transfer_syntax(self) -> UID:
        if self.level == 0:
            return JPEGLSLossless
        return JPEGLSNearLossless

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        return "RGB"

    @property
    def extension(self) -> str:
        return ".jls"


class Jpeg2kSettings(Settings):
    def __init__(
        self, level: int = 80, bits: int = 8, channels: Channels = Channels.YBR
    ):
        """
        Initialize JPEG 2000 encoding settings.

        Parameters:
        ----------
            level: int = 80
                JPEG 2000 compression level in dB. Set to < 1 or > 1000 for lossless.
            bits: int = 8
                Number of bits per pixel.
            channels: Channels = Channels.YBR
                Color channels in encoded image.
        """
        self._level = level
        super().__init__(bits, channels)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(level={self.level}, bits={self.bits}, "
            f"channels={self.channels})"
        )

    @property
    def level(self) -> int:
        """Return level."""
        return self._level

    @property
    def lossless(self) -> bool:
        """Return True if lossless, else False."""
        return self.level < 1 or self.level > 1000

    @property
    def transfer_syntax(self) -> UID:
        if self.lossless:
            return JPEG2000Lossless
        return JPEG2000

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        if self.channels == Channels.YBR:
            if self.lossless:
                return "YBR_RCT"
            return "YBR_ICT"
        return "RGB"

    @property
    def extension(self) -> str:
        return ".jp2"


class NumpySettings(Settings):
    def __init__(
        self,
        bits: int = 8,
        channels: Union[
            Literal[Channels.GRAYSCALE], Literal[Channels.RGB]
        ] = Channels.RGB,
        little_endian: bool = True,
        implicit_vr: bool = False,
    ):
        """
        Initialize numpy encoding settings.

        Parameters:
        ----------
            bits: int = 8
                Number of bits per pixel.
            channels: Union[
                Literal[Channels.GRAYSCALE], Literal[Channels.RGB]
            ] = Channels.RGB,
                Color channels in encoded image.
            little_endian: bool = True
                Endianness of encoded image.
            implicit_vr: bool = False
                VR encoding of transfer syntax.
        """
        self._little_endian = little_endian
        self._implicit_vr = implicit_vr
        super().__init__(bits, channels)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(bits={self.bits}, channels={self.channels}, "
            f"little_endian={self.little_endian}, implicit_vr={self.implicit_vr})"
        )

    @property
    def little_endian(self) -> bool:
        """Return little endian."""
        return self._little_endian

    @property
    def implicit_vr(self) -> bool:
        """Return implicit vr."""
        return self._implicit_vr

    @property
    def transfer_syntax(self) -> UID:
        if self.little_endian:
            if self.implicit_vr:
                return ImplicitVRLittleEndian
            return ExplicitVRLittleEndian
        return ExplicitVRBigEndian

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        return "RGB"

    @property
    def extension(self) -> str:
        return ".raw"


class RleSettings(Settings):
    def __init__(self, bits: int = 8, channels: Channels = Channels.RGB):
        super().__init__(bits, channels)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bits={self.bits}, channels={self.channels})"

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
