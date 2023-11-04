import io
from abc import ABCMeta, abstractmethod

from highdicom.frame import decode_frame
from PIL import Image
from PIL.Image import Image as PILImage
from pydicom.pixel_data_handlers import (
    gdcm_handler,
    jpeg_ls_handler,
    numpy_handler,
    pillow_handler,
    pylibjpeg_handler,
    rle_handler,
)
from pydicom.uid import JPEG2000, UID, JPEG2000Lossless, JPEGBaseline8Bit

from wsidicom.geometry import Size
from wsidicom.instance.dataset import WsiDataset


class Decoder(metaclass=ABCMeta):
    @abstractmethod
    def decode(self, frame: bytes) -> PILImage:
        """Decode frame into Pillow Image."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def is_supported(cls, transfer_syntax: UID) -> bool:
        """Return true if decode supports transfer syntax."""
        raise NotImplementedError()


class PillowDecoder(Decoder):
    def decode(self, frame: bytes) -> PILImage:
        return Image.open(io.BytesIO(frame))

    @classmethod
    def is_supported(cls, transfer_syntax: UID) -> bool:
        return transfer_syntax in [JPEG2000, JPEG2000Lossless, JPEGBaseline8Bit]


class PydicomDecoder(Decoder):
    def __init__(
        self,
        transfer_syntax: UID,
        size: Size,
        samples_per_pixel: int,
        bits_allocated: int,
        bits_stored: int,
        photometric_interpretation: str,
    ):
        self._transfer_syntax = transfer_syntax
        self._size = size
        self._samples_per_pixel = samples_per_pixel
        self._bits_allocated = bits_allocated
        self._bits_stored = bits_stored
        self._photometric_interpretation = photometric_interpretation

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


class DecoderFactory:
    @classmethod
    def create(cls, transfer_syntax: UID, dataset: WsiDataset) -> Decoder:
        """Create a decoder that supports the transfer syntax."""
        if PillowDecoder.is_supported(transfer_syntax):
            return PillowDecoder()
        elif PydicomDecoder.is_supported(transfer_syntax):
            return PydicomDecoder(
                transfer_syntax=transfer_syntax,
                size=dataset.tile_size,
                samples_per_pixel=dataset.samples_per_pixel,
                bits_allocated=dataset.BitsAllocated,
                bits_stored=dataset.BitsStored,
                photometric_interpretation=dataset.photometric_interpretation,
            )
        else:
            raise ValueError(f"Unsupported transfer syntax: {transfer_syntax}")
