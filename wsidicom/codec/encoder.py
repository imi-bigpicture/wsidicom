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
from typing import Union

import numpy as np
from imagecodecs import (
    JPEG8,
    jpeg2k_encode,
    jpeg8_encode,
    jpegls_encode,
)
from PIL.Image import Image as PILImage
from pydicom import Dataset
from pydicom.pixel_data_handlers.util import pixel_dtype
from pydicom.uid import (
    UID,
    JPEG2000,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGExtended12Bit,
    JPEGLosslessP14,
    JPEGLosslessSV1,
    JPEGLSLossless,
    JPEGLSNearLossless,
)
from rle.utils import encode_frame as rle_encode_frame

from wsidicom.codec.settings import (
    Channels,
    Jpeg2kSettings,
    JpegLosslessSettings,
    JpegLsLosslessSettings,
    JpegSettings,
    NumpySettings,
    RleSettings,
    Settings,
)


class Encoder(metaclass=ABCMeta):
    """Abstract base class for encoders."""

    def __init__(self, settings: Settings):
        """Initialize encoder.

        Parameters
        ----------
        settings: Settings
            Settings for the encoder.
        """
        self._settings = settings

    @abstractmethod
    def encode(self, image: Union[PILImage, np.ndarray]) -> bytes:
        """Encode image into bytes.

        Parameters
        ----------
        image: Union[PILImage, np.ndarray]
            Image to encode.

        Returns
        -------
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
    def create(cls, settings: Settings) -> "Encoder":
        """Create an encoder using settings.

        Parameters
        ----------
        settings: Settings
            Settings for the encoder.

        Returns
        -------
        Encoder
            Encoder for settings.
        """
        if isinstance(settings, (JpegSettings, JpegLosslessSettings)):
            return JpegEncoder(settings)
        if isinstance(settings, JpegLsLosslessSettings):
            return JpegLsEncoder(settings)
        if isinstance(settings, JpegLsLosslessSettings):
            return JpegLsEncoder(settings)
        if isinstance(settings, Jpeg2kSettings):
            return Jpeg2kEncoder(settings)
        if isinstance(settings, NumpySettings):
            return NumpyEncoder(settings)
        if isinstance(settings, RleSettings):
            return RleEncoder(settings)

        raise ValueError(f"Unsupported encoder settings: {settings}")


class JpegEncoder(Encoder):
    """JPEG encoder."""

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

    def encode(self, image: Union[PILImage, np.ndarray]) -> bytes:
        return jpeg8_encode(
            np.array(image).astype(self._dtype),
            level=self._level,
            lossless=self._lossless,
            bitspersample=self._bits,
            subsampling=self._subsampling,
            outcolorspace=self._output_colorspace,
            predictor=self._predictor,
        )

    @property
    def lossy(self) -> bool:
        return not self._lossless


class JpegLsEncoder(Encoder):
    """Encoder that uses jpegls to encode image."""

    _supported_transfer_syntaxes = [
        JPEGLSLossless,
        JPEGLSNearLossless,
    ]

    def __init__(
        self,
        settings: JpegLsLosslessSettings,
    ) -> None:
        """Initialize JPEG-LS encoder.

        Parameters
        ----------
        settings: JpegLsLosslessSettings
            Settings for the encoder.
        """
        self._bits = settings.bits
        if settings.level is None or settings.level == 0:
            self._level = 0
        else:
            self._level = settings.level
        super().__init__(settings)

    def encode(self, image: Union[PILImage, np.ndarray]) -> bytes:
        """Encode image into bytes."""
        return jpegls_encode(np.array(image), level=self._level)

    @property
    def lossy(self) -> bool:
        return self._level != 0


class Jpeg2kEncoder(Encoder):
    """Encoder that uses jpeg2k to encode image."""

    _supported_transfer_syntaxes = [
        JPEG2000Lossless,
        JPEG2000,
    ]

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
        if settings.level < 1 or settings.level > 1000:
            self._level = 0
            self._reversible = True
        else:
            self._level = settings.level
            self._reversible = False
        super().__init__(settings)

    def encode(self, image: Union[PILImage, np.ndarray]) -> bytes:
        return jpeg2k_encode(
            np.array(image),
            level=self._level,
            reversible=self._reversible,
            bitspersample=self._bits,
            codecformat="J2K",
            mct=self._multiple_component_transform,
        )

    @property
    def lossy(self) -> bool:
        return not self._reversible


class NumpyEncoder(Encoder):
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
        dataset.PixelRepresentation = settings.pixel_representation
        dataset.is_little_endian = settings.little_endian
        self._dtype = pixel_dtype(dataset)
        super().__init__(settings)

    def encode(self, image: Union[PILImage, np.ndarray]) -> bytes:
        return np.array(image).astype(self._dtype).tobytes()

    @property
    def lossy(self) -> bool:
        return False


class RleEncoder(Encoder):
    """Encoder that uses rle encoder to encode image."""

    def __init__(self, settings: RleSettings) -> None:
        """Initialize rle encoder.

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

    def encode(self, image: Union[PILImage, np.ndarray]) -> bytes:
        if isinstance(image, PILImage):
            rows, cols = image.size
        else:
            rows, cols = image.shape[0:2]
        return rle_encode_frame(
            np.array(image).astype(self._dtype).tobytes(),
            rows=rows,
            cols=cols,
            bpp=self._bits,
            spp=self._samples_per_pixel,
            byteorder="<",
        )

    @property
    def lossy(self) -> bool:
        return False
