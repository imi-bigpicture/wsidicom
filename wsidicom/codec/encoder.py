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

"""Module with encoders for image data."""

from abc import ABCMeta, abstractmethod
from enum import Enum
import io
from typing import Dict, Generic, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from PIL import Image as Pillow
from PIL.Image import Image
from pydicom import Dataset
from pydicom.pixel_data_handlers.util import pixel_dtype
from pydicom.uid import (
    UID,
    JPEGBaseline8Bit,
    JPEGExtended12Bit,
    JPEG2000,
    JPEGLSNearLossless,
)
from wsidicom.codec.optionals import (
    IMAGE_CODECS_AVAILABLE,
    PYLIBJPEGRLE_AVAILABLE,
    rle_encode_frame,
    JPEG8,
    jpeg2k_encode,
    jpeg8_encode,
    jpegls_encode,
)
from wsidicom.codec.rle import RleCodec

from wsidicom.codec.settings import (
    Channels,
    Jpeg2kSettings,
    JpegLosslessSettings,
    JpegLsSettings,
    JpegSettings,
    NumpySettings,
    RleSettings,
    Settings,
    Subsampling,
)

SettingsType = TypeVar("SettingsType", bound=Settings)


class LossyCompressionIsoStandard(Enum):
    JPEG_LOSSY = "ISO_10918_1"
    JPEG_LS_NEAR_LOSSLESS = "ISO_14495_1"
    JPEG_2000_IRREVERSIBLE = "ISO_15444_1"

    @classmethod
    def transfer_syntax_to_iso(
        cls, transfer_syntax: UID
    ) -> Optional["LossyCompressionIsoStandard"]:
        if transfer_syntax in [JPEGBaseline8Bit, JPEGExtended12Bit]:
            return cls.JPEG_LOSSY
        elif transfer_syntax == JPEGLSNearLossless:
            return cls.JPEG_LS_NEAR_LOSSLESS
        elif transfer_syntax == JPEG2000:
            return cls.JPEG_2000_IRREVERSIBLE
        return None


class Encoder(Generic[SettingsType], metaclass=ABCMeta):
    """Abstract base class for encoders."""

    def __init__(self, settings: SettingsType):
        """Initialize encoder.

        Parameters
        ----------
        settings: Settings
            Settings for the encoder.
        """
        self._settings = settings

    @abstractmethod
    def encode(self, image: Union[Image, np.ndarray]) -> bytes:
        """Encode image into bytes.

        Parameters
        ----------
        image: Union[Image, np.ndarray]
            Image to encode.

        Returns
        ----------
        bytes
            Encoded image.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def lossy(self) -> bool:
        """Return True if encoder is lossy."""
        raise NotImplementedError()

    @property
    def lossy_method(self) -> Optional[LossyCompressionIsoStandard]:
        """Return ISO standard name of compression if encoder is lossy."""
        return LossyCompressionIsoStandard.transfer_syntax_to_iso(self.transfer_syntax)

    @classmethod
    @abstractmethod
    def supports_settings(cls, settings: Settings) -> bool:
        """Return True if encoder supports settings."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if encoder is available."""
        raise NotImplementedError()

    @property
    def settings(self) -> SettingsType:
        """Return the settings."""
        return self._settings

    @property
    def transfer_syntax(self) -> UID:
        """Return the transfer syntax UID."""
        return self._settings.transfer_syntax

    @property
    def bits(self) -> int:
        """Return the number of bits per sample."""
        return self._settings.bits

    @property
    def allocated_bits(self) -> int:
        """Return the number of allocated bits."""
        return self._settings.allocated_bits

    @property
    def high_bit(self) -> int:
        """Return the high bit."""
        return self._settings.high_bit

    @property
    def channels(self) -> Channels:
        """Return the number of channels."""
        return self._settings.channels

    @property
    def samples_per_pixel(self) -> int:
        """Return the number of samples per pixel."""
        return self._settings.samples_per_pixel

    @property
    def photometric_interpretation(self) -> str:
        """Return the photometric interpretation."""
        return self._settings.photometric_interpretation

    @classmethod
    def create(
        cls, transfer_syntax: UID, bits: int, photometric_interpretation: str
    ) -> "Encoder":
        """Create an encoder using settings.

        Parameters
        ----------
        settings: SettingsType
            Settings for the encoder.

        Returns
        ----------
        Encoder[SettingsType]
            Encoder for settings.
        """
        settings = Settings.create(
            transfer_syntax,
            bits,
            photometric_interpretation,
        )
        return cls.create_for_settings(settings)

    @classmethod
    def create_for_settings(cls, settings: SettingsType) -> "Encoder[SettingsType]":
        """Create an encoder using settings.

        Parameters
        ----------
        settings: SettingsType
            Settings for the encoder.

        Returns
        ----------
        Encoder[SettingsType]
            Encoder for settings.
        """
        encoder = cls._select_encoder(settings)
        if encoder is None:
            raise ValueError(f"Unsupported encoder settings: {settings}")
        return encoder(settings)

    @staticmethod
    def _select_encoder(
        settings: SettingsType,
    ) -> Optional["Type[Encoder[SettingsType]]"]:
        """Select encoder for settings.

        Parameters
        ----------
        settings: SettingsType
            Settings for the encoder.

        Returns
        ----------
        Optional[Type[Encoder[SettingsType]]]
            Encoder for settings, or None if no encoder is available.
        """
        # Sort encoders by preference
        encoders_by_settings: Dict[Type[Settings], Tuple[Type[Encoder], ...]] = {
            JpegSettings: (JpegEncoder, PillowEncoder),
            JpegLosslessSettings: (JpegEncoder,),
            JpegLsSettings: (JpegLsEncoder,),
            Jpeg2kSettings: (Jpeg2kEncoder, PillowEncoder),
            NumpySettings: (NumpyEncoder,),
            RleSettings: (PylibjpegRleEncoder, ImageCodecsRleEncoder),
        }
        return next(
            (
                encoder
                for encoder in encoders_by_settings[type(settings)]
                if encoder.is_available() and encoder.supports_settings(settings)
            ),
            None,
        )


class PillowEncoder(Encoder[Union[JpegSettings, Jpeg2kSettings]]):
    """Encoder that uses pillow to encode image."""

    _supported_subsamplings = {
        Subsampling.R444: 0,
        Subsampling.R422: 1,
        Subsampling.R420: 2,
    }

    def __init__(self, settings: Union[JpegSettings, Jpeg2kSettings]):
        if isinstance(settings, JpegSettings):
            if settings.subsampling not in self._supported_subsamplings:
                raise ValueError(f"Unsupported subsampling: {settings.subsampling}.")
            subsampling = self._supported_subsamplings[settings.subsampling]
            self._pillow_settings = {
                "quality": settings.quality,
                "subsampling": subsampling,
            }
            self._format = "jpeg"
        elif isinstance(settings, Jpeg2kSettings):
            self._pillow_settings = {
                "quality_mode": "dB",
                "quality_layers": [settings.level],
                "irreversible": not settings.lossless,
                "mct": settings.channels == Channels.YBR,
                "no_jp2": True,
            }
            self._format = "jpeg2000"
        else:
            raise ValueError(f"Unsupported encoder settings: {type(settings)}.")
        super().__init__(settings)

    @property
    def lossy(self) -> bool:
        return isinstance(self.settings, JpegSettings) or (
            isinstance(self.settings, Jpeg2kSettings) and not self.settings.lossless
        )

    def encode(self, image: Union[Image, np.ndarray]) -> bytes:
        if not isinstance(image, Image):
            image = Pillow.fromarray(image)
        with io.BytesIO() as buffer:
            image.save(buffer, format=self._format, **self._pillow_settings)  # type: ignore
            return buffer.getvalue()

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def supports_settings(cls, settings: Settings) -> bool:
        if isinstance(settings, JpegSettings):
            return (
                settings.bits == 8
                and settings.subsampling in cls._supported_subsamplings
            )
        return isinstance(settings, Jpeg2kSettings) and settings.bits == 8


class JpegEncoder(Encoder[Union[JpegSettings, JpegLosslessSettings]]):
    """JPEG encoder using image codecs."""

    def __init__(
        self,
        settings: Union[JpegSettings, JpegLosslessSettings],
    ) -> None:
        """Initialize JPEG encoder.

        Parameters
        ----------
        settings: Union[JpegSettings, JpegLosslessSettings]
            Settings for the encoder.

        """
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
                settings.subsampling.name.lstrip("R") if settings.subsampling else None
            )
        elif isinstance(settings, JpegLosslessSettings):
            self._lossless = True
            self._predictor = settings.predictor
            self._level = None
            self._subsampling = None
        else:
            raise ValueError(f"Unsupported encoder settings: {type(settings)}.")
        super().__init__(settings)

    @property
    def lossy(self) -> bool:
        return not self._lossless

    def encode(self, image: Union[Image, np.ndarray]) -> bytes:
        if not self.is_available():
            raise RuntimeError("Image codecs not available.")
        return jpeg8_encode(
            np.array(image).astype(self._dtype),
            level=self._level,
            lossless=self._lossless,
            bitspersample=self._bits,
            subsampling=self._subsampling,
            outcolorspace=self._output_colorspace,
            predictor=self._predictor,
        )

    @classmethod
    def is_available(cls) -> bool:
        return IMAGE_CODECS_AVAILABLE

    @classmethod
    def supports_settings(cls, settings: Settings) -> bool:
        return isinstance(settings, (JpegSettings, JpegLosslessSettings))


class JpegLsEncoder(Encoder[JpegLsSettings]):
    """Encoder that uses jpegls to encode image."""

    def __init__(
        self,
        settings: JpegLsSettings,
    ) -> None:
        """Initialize JPEG-LS encoder.

        Parameters
        ----------
        settings: JpegLsSettings
            Settings for the encoder.
        """
        self._bits = settings.bits
        if settings.level is None or settings.level == 0:
            self._level = 0
        else:
            self._level = settings.level
        super().__init__(settings)

    @property
    def lossy(self) -> bool:
        return self._level != 0

    def encode(self, image: Union[Image, np.ndarray]) -> bytes:
        """Encode image into bytes."""
        if not self.is_available():
            raise RuntimeError("Image codecs not available.")
        if isinstance(image, Image) and image.mode not in ("L", "RGB", "I;16", "I;16L"):
            raise ValueError(f"Unsupported mode: {image.mode}.")
        return jpegls_encode(np.array(image), level=self._level)

    @classmethod
    def is_available(cls) -> bool:
        return IMAGE_CODECS_AVAILABLE

    @classmethod
    def supports_settings(cls, settings: Settings) -> bool:
        return isinstance(settings, JpegLsSettings)


class Jpeg2kEncoder(Encoder[Jpeg2kSettings]):
    """Encoder that uses jpeg2k to encode image."""

    def __init__(self, settings: Jpeg2kSettings) -> None:
        """Initialize JPEG 2000 encoder.

        Parameters
        ----------
        settings: Jpeg2kSettings
            Settings for the encoder.
        """
        self._bits = settings.bits
        if settings.channels == Channels.YBR:
            self._multiple_component_transform = True
        else:
            self._multiple_component_transform = False
        if settings.lossless:
            self._level = 0
            self._reversible = True
        else:
            self._level = settings.level
            self._reversible = False
        super().__init__(settings)

    @property
    def lossy(self) -> bool:
        return not self.settings.lossless

    def encode(self, image: Union[Image, np.ndarray]) -> bytes:
        if not self.is_available():
            raise RuntimeError("Image codecs not available.")
        if isinstance(image, Image) and image.mode not in ("L", "RGB", "I;16", "I;16L"):
            raise ValueError(f"Unsupported mode: {image.mode}.")
        return jpeg2k_encode(
            np.array(image),
            level=self._level,
            reversible=self._reversible,
            bitspersample=self._bits,
            codecformat="J2K",
            mct=self._multiple_component_transform,
        )

    @classmethod
    def is_available(cls) -> bool:
        return IMAGE_CODECS_AVAILABLE

    @classmethod
    def supports_settings(cls, settings: Settings) -> bool:
        if not isinstance(settings, Jpeg2kSettings):
            return False
        if (
            settings.level < 1 or settings.level > 1000
        ) and settings.channels == Channels.YBR:
            # jpeg2k_encode does not enable mct for lossless compression.
            # Remove this when https://github.com/cgohlke/imagecodecs/issues/88 is fixed.
            return False
        return True


class NumpyEncoder(Encoder[NumpySettings]):
    """Encoder that uses numpy to encode image."""

    def __init__(self, settings: NumpySettings) -> None:
        """Initialize numpy encoder.

        Parameters
        ----------
        settings: NumpySettings
            Settings for the encoder.
        """
        dataset = Dataset()
        dataset.BitsAllocated = settings.allocated_bits
        dataset.PixelRepresentation = 0
        dataset.is_little_endian = settings.little_endian
        self._dtype = pixel_dtype(dataset)
        super().__init__(settings)

    @property
    def lossy(self) -> bool:
        return False

    def encode(self, image: Union[Image, np.ndarray]) -> bytes:
        if isinstance(image, Image) and image.mode not in ("L", "RGB", "I;16", "I;16L"):
            raise ValueError(f"Unsupported mode: {image.mode}.")
        return np.array(image).astype(self._dtype).tobytes()

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def supports_settings(cls, settings: Settings) -> bool:
        return isinstance(settings, NumpySettings)


class RleEncoder(Encoder[RleSettings]):
    def __init__(self, settings: RleSettings) -> None:
        """Initialize RLE encoder.

        Parameters
        ----------
        settings: RleSettings
            Settings for the encoder.
        """
        self._bits = settings.bits
        if settings.bits == 8:
            self._dtype = np.uint8
        else:
            self._dtype = np.uint16
        if settings.channels == Channels.GRAYSCALE:
            self._samples_per_pixel = 1
        else:
            self._samples_per_pixel = 3
        super().__init__(settings)

    def encode(self, image: Union[Image, np.ndarray]) -> bytes:
        if isinstance(image, Image) and image.mode not in ("L", "RGB", "I;16", "I;16L"):
            raise ValueError(f"Unsupported mode: {image.mode}.")
        return self._encode(np.array(image).astype(self._dtype))

    @property
    def lossy(self) -> bool:
        return False

    @classmethod
    def supports_settings(cls, settings: Settings) -> bool:
        return isinstance(settings, RleSettings)

    @abstractmethod
    def _encode(self, image: np.ndarray) -> bytes:
        raise NotImplementedError()


class ImageCodecsRleEncoder(RleEncoder):
    """Encoder that uses image codecs PackBits to encode image."""

    def _encode(self, image: np.ndarray):
        if not self.is_available():
            raise RuntimeError("Image codecs not available.")
        return RleCodec.encode(image)

    @classmethod
    def supports_settings(cls, settings: Settings) -> bool:
        if settings.bits != 8:
            return False
        return isinstance(settings, RleSettings)

    @classmethod
    def is_available(cls) -> bool:
        return IMAGE_CODECS_AVAILABLE


class PylibjpegRleEncoder(RleEncoder):
    """Encoder that uses pylibjpeg-rle to encode image."""

    def _encode(self, image: np.ndarray) -> bytes:
        if not self.is_available():
            raise RuntimeError("Pylibjpeg-rle not available.")
        rows, cols = image.shape[0:2]
        return rle_encode_frame(
            np.array(image).astype(self._dtype).tobytes(),
            rows=rows,
            cols=cols,
            bpp=self._bits,
            spp=self._samples_per_pixel,
            byteorder="<",
        )

    @classmethod
    def is_available(cls) -> bool:
        return PYLIBJPEGRLE_AVAILABLE
