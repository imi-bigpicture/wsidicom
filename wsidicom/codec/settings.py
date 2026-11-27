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

import copy
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import ClassVar, Literal, Self

from pydicom.uid import (
    HTJ2K,
    JPEG2000,
    UID,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    HTJ2KLossless,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGExtended12Bit,
    JPEGLossless,
    JPEGLosslessSV1,
    JPEGLSLossless,
    JPEGLSNearLossless,
    RLELossless,
)

from wsidicom.uid import JPEGXL, JPEGXLLossless


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
    R422 = "Half horizontal, full vertical (4:2:2)"
    R420 = "Half horizontal, half vertical (4:2:0)"
    R411 = "Quarter horizontal, full vertical (4:1:1)"
    R440 = "Full horizontal, half vertical (4:4:0)"

    @classmethod
    def from_string(cls, string: str | None) -> "Subsampling":
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
    _DEFAULT_BITS: ClassVar[int] = 8
    """Bits per sample if not given."""
    _DEFAULT_COLOR_CHANNELS: ClassVar[Channels] = Channels.YBR
    """Channels if not given, and the channels of an adapted color image."""
    _SUPPORTED_CHANNELS: ClassVar[tuple[Channels, ...]] = tuple(Channels)
    """Channels that can be given."""

    def __init__(
        self, bits: int | None = None, channels: Channels | None = None
    ) -> None:
        """
        Initialize encoding settings.

        Parameters
        ----------
            bits: int | None = None
                Number of bits per pixel. If None, 8, or the bits of the image
                the settings are adapted to.
            channels: Channels | None = None
                Color channels in encoded image. If None, the default color
                channels of the format, or the channels matching the image the
                settings are adapted to.

        Raises
        ------
        ValueError
            If the format does not support the channels.
        """
        if channels is not None and channels not in self._SUPPORTED_CHANNELS:
            raise ValueError(
                f"{type(self).__name__} does not support {channels.value} channels."
            )
        self._given_bits = bits
        self._given_channels = channels

    @property
    def bits(self) -> int:
        """Return bits."""
        if self._given_bits is None:
            return self._DEFAULT_BITS
        return self._given_bits

    @property
    def channels(self) -> Channels:
        """Return channels."""
        if self._given_channels is None:
            return self._DEFAULT_COLOR_CHANNELS
        return self._given_channels

    def adapted_to(self, bits: int, samples_per_pixel: int) -> Self:
        """Return settings adapted to the given bits and samples per pixel.

        Bits and channels not specified when the settings were created follow the
        given values, while specified ones are kept. Bits the format can not
        encode, and more than 8 bits with more than one sample per pixel, are not
        used, so the settings then do not match the given values.

        Parameters
        ----------
        bits: int
            Number of bits per sample.
        samples_per_pixel: int
            Number of samples per pixel.

        Returns
        -------
        Self
            Adapted settings. These settings if nothing changes.
        """
        adapted_bits = self.bits
        if (
            self._given_bits is None
            and self._can_encode_bits(bits)
            and (samples_per_pixel == 1 or bits <= 8)
        ):
            adapted_bits = bits
        adapted_channels = self.channels
        if self._given_channels is None and samples_per_pixel == 1:
            adapted_channels = Channels.GRAYSCALE
        if adapted_bits == self.bits and adapted_channels == self.channels:
            return self
        adapted = copy.copy(self)
        adapted._given_bits = adapted_bits
        adapted._given_channels = adapted_channels
        return adapted

    def _can_encode_bits(self, bits: int) -> bool:
        """Return True if the format can encode samples of this many bits."""
        return 1 < bits <= 16

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
        """Create settings based on properties in dataset and transfer syntax.

        Parameters
        ----------
            transfer_syntax: UID
                Transfer syntax to create settings for.
            bits: int
                Number of bits per pixel.
            photometric_interpretation: str
                Photometric interpretation of image.

        Returns
        -------
            Settings
                Settings matching dataset and transfer syntax.
        """
        jpeg_transfer_syntaxes = [
            JPEGBaseline8Bit,
            JPEGExtended12Bit,
        ]
        jpeg_lossless_transfer_syntaxes = [
            JPEGLossless,
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
        ht_jpeg_2000_transfer_syntaxes = [
            HTJ2K,
            HTJ2KLossless,
            # HTJ2KLosslessRPCL # Not supported in imagecodec jpeg2k encoder
        ]
        jpeg_xl_transfer_syntaxes = [JPEGXL, JPEGXLLossless]
        uncompressed_transfer_syntaxes = [
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
            ExplicitVRBigEndian,
        ]
        channels = Channels.from_photometric_interpretation(photometric_interpretation)
        if transfer_syntax in jpeg_transfer_syntaxes:
            return JpegSettings(bits=bits, channels=channels)
        if transfer_syntax in jpeg_lossless_transfer_syntaxes:
            # Predictor 1 gives SV1. Any other gives the general transfer syntax,
            # where 4 compresses tissue well.
            predictor = 1 if transfer_syntax == JPEGLosslessSV1 else 4
            # There is no color transform in lossless jpeg, so color images are
            # encoded as RGB also if stated as something else.
            if channels == Channels.YBR:
                channels = Channels.RGB
            return JpegLosslessSettings(predictor, bits=bits, channels=channels)
        if transfer_syntax in jpeg_2000_transfer_syntaxes:
            if transfer_syntax == JPEG2000:
                return Jpeg2kSettings(bits=bits, channels=channels)
            return Jpeg2kSettings(levels=0, bits=bits, channels=channels)
        if transfer_syntax in ht_jpeg_2000_transfer_syntaxes:
            if transfer_syntax == HTJ2K:
                return HTJpeg2000Settings(bits=bits, channels=channels)
            return HTJpeg2000Settings(levels=0, bits=bits, channels=channels)
        if transfer_syntax in jpeg_xl_transfer_syntaxes:
            # The color space of the decoded pixels is given by the codestream,
            # and color images are stated as RGB when encoded.
            if channels == Channels.YBR:
                channels = Channels.RGB
            if transfer_syntax == JPEGXL:
                return JpegXlSettings(level=None, bits=bits, channels=channels)
            return JpegXlSettings(level=100, bits=bits, channels=channels)
        # The remaining formats can not store the samples as ybr.
        if channels == Channels.YBR:
            raise ValueError(
                f"Unsupported transfer syntax for ybr channels: {transfer_syntax}."
            )
        if transfer_syntax in jpeg_ls_transfer_syntaxes:
            if transfer_syntax == JPEGLSNearLossless:
                return JpegLsSettings(level=1, bits=bits, channels=channels)
            return JpegLsSettings(level=0, bits=bits, channels=channels)
        if transfer_syntax == RLELossless:
            return RleSettings(bits=bits, channels=channels)
        if transfer_syntax in uncompressed_transfer_syntaxes:
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
        bits: int | None = None,
        channels: Channels | None = None,
        subsampling: Subsampling = Subsampling.R422,
        optimize: bool = False,
    ) -> None:
        """
        Initialize JPEG encoding settings.

        Parameters
        ----------
            quality: int = 80
                JPEG quality factor. 0-100, recommended to not use higher than 95.
            bits: int | None = None
                Number of bits per pixel, see `Settings`.
            channels: Channels | None = None
                Color channels in encoded image, see `Settings`.
            subsampling: Subsampling = Subsampling.R422
                Chroma subsampling settings. Ignored for RGB, which has no chroma
                components.
            optimize: bool = False
                If to build Huffman tables optimal for each encoded image instead of
                using the default tables. This makes the encoded images smaller
                without changing their pixels, at the cost of slower encoding.
        """
        self._quality = quality
        self._subsampling = subsampling
        self._optimize = optimize
        super().__init__(bits, channels)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(quality={self.quality}, bits={self.bits}, "
            f"channels={self.channels}, subsampling={self.subsampling}, "
            f"optimize={self.optimize})"
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
    def optimize(self) -> bool:
        """Return if Huffman tables are optimized for each encoded image."""
        return self._optimize

    def with_optimize(self, optimize: bool) -> "JpegSettings":
        """Return a copy of the settings with optimized Huffman tables set or not.

        Parameters
        ----------
        optimize: bool
            If to build Huffman tables optimal for each encoded image.

        Returns
        -------
        JpegSettings
            Settings equal to these except for `optimize`.
        """
        changed = copy.copy(self)
        changed._optimize = optimize
        return changed

    def _can_encode_bits(self, bits: int) -> bool:
        return bits in (8, 12)

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
    """JPEG lossless encoding settings.

    Color images are encoded as RGB, as there is no color transform in lossless
    JPEG. `Channels.YBR` is not supported.
    """

    _DEFAULT_COLOR_CHANNELS = Channels.RGB
    _SUPPORTED_CHANNELS = (Channels.GRAYSCALE, Channels.RGB)

    def __init__(
        self,
        predictor: int = 1,
        bits: int | None = None,
        channels: Literal[Channels.GRAYSCALE, Channels.RGB] | None = None,
    ) -> None:
        """
        Initialize JPEG lossless encoding settings.

        Parameters
        ----------
            predictor: int = 1
                JPEG predictor (selection value), 1-7. 1 gives the transfer syntax
                JPEG Lossless SV1, and 2-7 give JPEG Lossless.
            bits: int | None = None
                Number of bits per pixel, see `Settings`.
            channels: Literal[Channels.GRAYSCALE, Channels.RGB] | None = None
                Color channels in encoded image, see `Settings`.
        """
        if predictor not in range(1, 8):
            raise ValueError(f"Unsupported predictor: {predictor}. Use 1-7.")
        self._predictor = predictor
        super().__init__(bits, channels)

    def __repr__(self) -> str:
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
        if self.predictor == 1:
            return JPEGLosslessSV1
        return JPEGLossless

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        return "RGB"

    @property
    def extension(self) -> str:
        return ".jpg"


class JpegLsSettings(Settings):
    _DEFAULT_COLOR_CHANNELS = Channels.RGB
    _SUPPORTED_CHANNELS = (Channels.GRAYSCALE, Channels.RGB)

    def __init__(
        self,
        level: int = 0,
        bits: int | None = None,
        channels: Literal[Channels.GRAYSCALE, Channels.RGB] | None = None,
    ) -> None:
        """
        Initialize JPEG-LS lossless encoding settings. Only supports grayscale images.

        Parameters
        ----------
            level: int = 0
                JPEG-LS near-lossless level. 0-9. 0 is lossless.
            bits: int | None = None
                Number of bits per pixel, see `Settings`.
            channels: Literal[Channels.GRAYSCALE, Channels.RGB] | None = None
                Color channels in encoded image, see `Settings`.
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
        self,
        levels: float | Sequence[float] = 80,
        bits: int | None = None,
        channels: Channels | None = None,
    ) -> None:
        """
        Initialize JPEG 2000 encoding settings.

        Parameters
        ----------
            levels: float | Sequence[float] = 80
                JPEG 2000 compression levels in dB. Set to 0 for lossless.
            bits: int | None = None
                Number of bits per pixel, see `Settings`.
            channels: Channels | None = None
                Color channels in encoded image, see `Settings`.
        """
        if isinstance(levels, (int, float)):
            levels = [levels]
        self._levels = levels
        super().__init__(bits, channels)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(levels={self.levels}, bits={self.bits}, "
            f"channels={self.channels})"
        )

    @property
    def levels(self) -> Sequence[float]:
        """Return level."""
        return self._levels

    @property
    def lossless(self) -> bool:
        """Return True if lossless, else False."""
        return self.levels[-1] == 0

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


class HTJpeg2000Settings(Jpeg2kSettings):
    @property
    def transfer_syntax(self) -> UID:
        if self.lossless:
            return HTJ2KLossless
        return HTJ2K

    @property
    def extension(self) -> str:
        return ".jph"


class JpegXlSettings(Settings):
    """JPEG XL encoding settings.

    Color images are stated as RGB. Lossless JPEG XL stores the RGB samples, while
    lossy stores them in the XYB color space, which the codestream states, so that
    they are decoded as RGB. `Channels.YBR` is not supported.
    """

    _DEFAULT_COLOR_CHANNELS = Channels.RGB
    _SUPPORTED_CHANNELS = (Channels.GRAYSCALE, Channels.RGB)

    def __init__(
        self,
        level: int | None = None,
        bits: int | None = None,
        channels: Literal[Channels.GRAYSCALE, Channels.RGB] | None = None,
    ) -> None:
        """
        Initialize JPEG XL encoding settings.

        Parameters
        ----------
            level: int | None = None
                JPEG XL compression level. -100-100. Set to 100 for lossless.
            bits: int | None = None
                Number of bits per pixel, see `Settings`.
            channels: Literal[Channels.GRAYSCALE, Channels.RGB] | None = None
                Color channels in encoded image, see `Settings`.
        """
        self._level = level
        self._lossless = level is not None and level >= 100
        super().__init__(bits, channels)

    @property
    def level(self) -> int | None:
        """Return level."""
        return self._level

    @property
    def lossless(self) -> bool:
        """Return True if lossless, else False."""
        return self._lossless

    @property
    def transfer_syntax(self) -> UID:
        if self._lossless:
            return JPEGXLLossless
        return JPEGXL

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        return "RGB"

    @property
    def extension(self) -> str:
        return ".jxl"


class NumpySettings(Settings):
    _DEFAULT_COLOR_CHANNELS = Channels.RGB
    _SUPPORTED_CHANNELS = (Channels.GRAYSCALE, Channels.RGB)

    def __init__(
        self,
        bits: int | None = None,
        channels: Literal[Channels.GRAYSCALE, Channels.RGB] | None = None,
        little_endian: bool = True,
        implicit_vr: bool = False,
    ) -> None:
        """
        Initialize numpy encoding settings.

        Parameters
        ----------
            bits: int | None = None
                Number of bits per pixel, see `Settings`.
            channels: Literal[Channels.GRAYSCALE, Channels.RGB] | None = None
                Color channels in encoded image, see `Settings`.
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

    def _can_encode_bits(self, bits: int) -> bool:
        return bits in (8, 16)

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
    _DEFAULT_COLOR_CHANNELS = Channels.RGB
    _SUPPORTED_CHANNELS = (Channels.GRAYSCALE, Channels.RGB)

    def __init__(
        self,
        bits: int | None = None,
        channels: Literal[Channels.GRAYSCALE, Channels.RGB] | None = None,
    ) -> None:
        super().__init__(bits, channels)

    def _can_encode_bits(self, bits: int) -> bool:
        return bits in (8, 16)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bits={self.bits}, channels={self.channels})"

    @property
    def transfer_syntax(self) -> UID:
        return RLELossless

    @property
    def photometric_interpretation(self) -> str:
        if self.channels == Channels.GRAYSCALE:
            return "MONOCHROME2"
        return "RGB"

    @property
    def extension(self) -> str:
        return ".rle"
