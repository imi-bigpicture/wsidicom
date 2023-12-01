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

"""Module with decoders for image data."""

import io
from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, Optional, Type

import numpy as np
from PIL import Image as Pillow
from PIL.Image import Image
from pydicom import config as pydicom_config
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
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
    RLELossless,
)

from wsidicom import config
from wsidicom.codec.rle import RleCodec
from wsidicom.geometry import Size

from wsidicom.codec.optionals import (
    JPEG2K,
    JPEG8,
    JPEGLS,
    jpeg2k_decode,
    jpeg8_decode,
    jpegls_decode,
    IMAGE_CODECS_AVAILABLE,
    PYLIBJPEGRLE_AVAILABLE,
    rle_decode_frame,
)


class Decoder(metaclass=ABCMeta):
    @abstractmethod
    def decode(self, frame: bytes) -> Image:
        """Decode frame into Pillow Image.

        Parameters
        ----------
        frame : bytes
            Encoded frame.

        Returns
        ----------
        PIL.Image
            Pillow Image.
        """
        raise NotImplementedError()

    @classmethod
    def is_available(cls) -> bool:
        """Return true if decoder is available.

        Returns
        ----------
        bool
            True if decoder is available.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def is_supported(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int
    ) -> bool:
        """Return true if decoder supports transfer syntax.

        Parameters
        ----------
        transfer_syntax : pydicom.uid.UID
            Transfer syntax.
        samples_per_pixel : int
            Number of samples per pixel.
        bits : int
            Number of bits per sample.

        Returns
        ----------
        bool
            True if decoder supports transfer syntax.

        """
        raise NotImplementedError()

    @classmethod
    def create(
        cls,
        transfer_syntax: UID,
        samples_per_pixel: int,
        bits: int,
        size: Size,
        photometric_interpretation: str,
    ) -> "Decoder":
        """Create a decoder that supports the transfer syntax.

        Parameters
        ----------
        transfer_syntax : UID
            Transfer syntax.
        samples_per_pixel : int
            Number of samples per pixel.
        bits : int
            Number of bits per sample.
        size : Size
            Size of image.
        photometric_interpretation : str
            Photometric interpretation.

        Returns
        ----------
        Decoder
            Decoder for transfer syntax.

        """
        if bits != 8 and samples_per_pixel != 1:
            # Pillow only supports 8 bit color images
            raise ValueError(
                f"Non-supported combination of bits {bits} and  "
                f"samples per pixel {samples_per_pixel}. "
                "Non-8 bit images are only supported for grayscale images."
            )
        decoder = cls.select_decoder(transfer_syntax, samples_per_pixel, bits)
        if decoder is None:
            raise ValueError(f"Unsupported transfer syntax: {transfer_syntax}.")
        if decoder == PillowDecoder:
            return PillowDecoder()
        elif decoder == ImageCodecsDecoder:
            return ImageCodecsDecoder(transfer_syntax)
        elif decoder == PylibjpegRleDecoder:
            return PylibjpegRleDecoder(size, samples_per_pixel, bits)
        # Does not support padded data
        # elif decoder == ImageCodecsRleDecoder:
        #     return ImageCodecsRleDecoder(size, samples_per_pixel, bits)
        elif decoder == PydicomDecoder:
            return PydicomDecoder(
                transfer_syntax=transfer_syntax,
                size=size,
                samples_per_pixel=samples_per_pixel,
                bits_allocated=(bits // 8) * 8,
                bits_stored=bits,
                photometric_interpretation=photometric_interpretation,
            )
        raise ValueError(f"Unsupported transfer syntax: {transfer_syntax}")

    @classmethod
    def select_decoder(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int
    ) -> Optional[Type["Decoder"]]:
        """Select decoder based on transfer syntax.

        Parameters
        ----------
        transfer_syntax : UID
            Transfer syntax.
        samples_per_pixel : int
            Number of samples per pixel.
        bits : int
            Number of bits per sample.

        Returns
        ----------
        Optional[Type[Decoder]]
            Decoder class. None if no decoder supports transfer syntax.

        """
        if bits != 8 and samples_per_pixel != 1:
            # Pillow only supports 8 bit color images
            return None
        decoders: Dict[str, Type[Decoder]] = {
            "pillow": PillowDecoder,
            "imagecodecs": ImageCodecsDecoder,
            "pylibjpeg_rle": PylibjpegRleDecoder,
            # imagecodes packbit does not allow padded data
            # Uncomment when https://github.com/cgohlke/imagecodecs/issues/86 is fixed
            # "imagecodecs_rle": ImageCodecsRleDecoder,
            "pydicom": PydicomDecoder,
        }
        if config.settings.prefered_decoder is not None:
            if config.settings.prefered_decoder not in decoders:
                raise ValueError(
                    f"Unknown preferred decoder: {config.settings.prefered_decoder}."
                )
            decoder = decoders[config.settings.prefered_decoder]
            if not decoder.is_available() or not decoder.is_supported(
                transfer_syntax, samples_per_pixel, bits
            ):
                return None
            return decoder
        return next(
            (
                decoder
                for decoder in decoders.values()
                if decoder.is_available()
                and decoder.is_supported(transfer_syntax, samples_per_pixel, bits)
            ),
            None,
        )

    @staticmethod
    def _set_mode(image: Image) -> Image:
        """Convert image to mode that is fully supported by Pillow."""
        if image.mode.startswith("I;16"):
            # Pillow does not support scaling of 16 bit images
            return image.convert("I")
        return image


class PillowDecoder(Decoder):
    """Decoder that uses Pillow to decode images."""

    _supported_transfer_syntaxes = [JPEGBaseline8Bit, JPEG2000, JPEG2000Lossless]

    def decode(self, frame: bytes) -> Image:
        image = Pillow.open(io.BytesIO(frame))
        return self._set_mode(image)

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

    @classmethod
    def is_available(cls) -> bool:
        return True


class NumpyBasedDecoder(Decoder):
    """Base for decoder that returns numpy array."""

    def decode(self, frame: bytes) -> Image:
        decoded = self._decode(frame)
        image = Pillow.fromarray(decoded)
        return self._set_mode(image)

    @abstractmethod
    def _decode(self, frame: bytes) -> np.ndarray:
        """Decode frame into numpy array.

        Parameters
        ----------
        frame : bytes
            Encoded frame.

        Returns
        ----------
        np.ndarray
            Numpy array of decoded image."""
        raise NotImplementedError()


class PydicomDecoder(NumpyBasedDecoder):
    """Decoder that uses pydicom to decode images."""

    def __init__(
        self,
        transfer_syntax: UID,
        size: Size,
        samples_per_pixel: int,
        bits_allocated: int,
        bits_stored: int,
        photometric_interpretation: str,
    ):
        """Initialize decoder.

        Parameters
        ----------
        transfer_syntax : UID
            Transfer syntax.
        size : Size
            Size of image.
        samples_per_pixel : int
            Number of samples per pixel.
        bits_allocated : int
            Number of bits allocated.
        bits_stored : int
            Number of bits stored.
        photometric_interpretation : str
            Photometric interpretation.

        """
        handler = self._get_handler(transfer_syntax)
        if handler is None:
            raise ValueError(f"Unsupported transfer syntax: {transfer_syntax}.")
        self._handler = handler
        self._transfer_syntax = transfer_syntax
        dataset = Dataset()
        dataset.file_meta = FileMetaDataset()
        dataset.file_meta.TransferSyntaxUID = transfer_syntax
        dataset.Rows = size.height
        dataset.Columns = size.width
        dataset.SamplesPerPixel = samples_per_pixel
        dataset.BitsAllocated = bits_allocated
        dataset.BitsStored = bits_stored
        dataset.HighBit = bits_stored - 1
        dataset.PhotometricInterpretation = photometric_interpretation
        dataset.PixelRepresentation = 0
        if samples_per_pixel == 3:
            self._reshape_size = (size.height, size.width, samples_per_pixel)
            dataset.PlanarConfiguration = 0
        else:
            self._reshape_size = (size.height, size.width)
        self._dataset = dataset

    def _decode(self, frame: bytes) -> np.ndarray:
        if self._transfer_syntax.is_encapsulated:
            self._dataset.PixelData = encapsulate(frames=[frame])
        else:
            self._dataset.PixelData = frame
        return self._handler(self._dataset).reshape(self._reshape_size)

    @classmethod
    def is_supported(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int
    ) -> bool:
        handler = cls._get_handler(transfer_syntax)
        return handler is not None

    @classmethod
    def _get_handler(
        cls, transfer_syntax: UID
    ) -> Optional[Callable[[Dataset], np.ndarray]]:
        """Return pydicom pixel data handler that supports transfer syntax.

        A pydicom pixel data handler should implement methods `Is_available()`,
        `supports_transfer_syntax()` that takes a transfer syntax and `get_pixeldata()`
        that takes a dataset.
        """
        available_handlers = (
            handler
            for handler in pydicom_config.pixel_data_handlers
            if handler.is_available()
        )
        return next(
            (
                handler.get_pixeldata
                for handler in available_handlers
                if handler.supports_transfer_syntax(transfer_syntax)
            ),
            None,
        )

    @classmethod
    def is_available(cls) -> bool:
        return True


class ImageCodecsDecoder(NumpyBasedDecoder):
    """Decoder that uses imagecodecs to decode images."""

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
        """Initialize decoder.

        Parameters
        ----------
        transfer_syntax : UID
            Transfer syntax.
        """
        decoder = self._get_decoder(transfer_syntax)
        if decoder is None:
            raise ValueError(f"Unsupported transfer syntax: {transfer_syntax}.")
        self._decoder = decoder

    def _decode(self, frame: bytes) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("Image codecs not available.")
        return self._decoder(frame)

    @classmethod
    def is_supported(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int
    ) -> bool:
        if not cls.is_available():
            return False
        decoder = cls._get_decoder(transfer_syntax)
        return decoder is not None

    @classmethod
    def is_available(cls) -> bool:
        return IMAGE_CODECS_AVAILABLE

    @classmethod
    def _get_decoder(
        cls, transfer_syntax: UID
    ) -> Optional[Callable[[bytes], np.ndarray]]:
        """Get imagecodes decoder for transfer syntax.

        Parameters
        ----------
        transfer_syntax : UID
            Transfer syntax.

        Returns
        ----------
        Optional[Callable[[bytes], np.ndarray]]
            Decoder. None if no decoder supports transfer syntax.
        """
        if transfer_syntax not in cls._supported_transfer_syntaxes:
            return None
        decoder, codec = cls._supported_transfer_syntaxes[transfer_syntax]
        assert decoder is not None and codec is not None
        if not codec.available:
            return None
        return decoder


class RleDecoder(NumpyBasedDecoder):
    def __init__(self, size: Size, samples_per_pixel: int, bits: int):
        if not self.is_supported(RLELossless, samples_per_pixel, bits):
            raise ValueError(
                f"Unsupported combination of samples per pixel {samples_per_pixel} "
                f"and bits {bits}."
            )
        if bits == 8:
            self._dtype = np.uint8
        else:
            self._dtype = np.uint16
        self._size = size
        self._bits = bits
        self._samples_per_pixel = samples_per_pixel
        if samples_per_pixel == 3:
            self._reshape_size = (samples_per_pixel, size.height, size.width)
        else:
            self._reshape_size = (size.height, size.width)

    @classmethod
    def is_supported(
        cls, transfer_syntax: UID, samples_per_pixel: int, bits: int
    ) -> bool:
        if transfer_syntax != RLELossless:
            return False
        if samples_per_pixel == 3:
            return bits == 8
        if samples_per_pixel == 1:
            return bits in [8, 16]
        return False


class ImageCodecsRleDecoder(RleDecoder):
    @classmethod
    def is_available(cls):
        return IMAGE_CODECS_AVAILABLE

    def _decode(self, frame: bytes) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("Image codecs not available.")
        return RleCodec.decode(frame, self._size.height, self._size.width, self._bits)


class PylibjpegRleDecoder(RleDecoder):
    """Decoder that uses pylibjpeg-rle to decode images."""

    @classmethod
    def is_available(cls) -> bool:
        return PYLIBJPEGRLE_AVAILABLE

    def _decode(self, frame: bytes) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("Pylibjpeg-rle not available.")
        decoded_data: bytes = rle_decode_frame(frame, self._size.area, self._bits, "<")
        decoded = np.frombuffer(decoded_data, dtype=self._dtype).reshape(
            self._reshape_size
        )
        if self._samples_per_pixel == 3:
            decoded = decoded.transpose((1, 2, 0))
        return decoded
