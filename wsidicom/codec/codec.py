from typing import Iterable, Optional, Union

import numpy as np
from wsidicom.codec.decoder import Decoder
from wsidicom.codec.encoder import Encoder
from pydicom.uid import (
    UID,
    AllTransferSyntaxes,
)
from PIL.Image import Image as PILImage

from wsidicom.codec.settings import Settings
from wsidicom.geometry import Size


class Codec:
    def __init__(self, decoder: Decoder, encoder: Encoder):
        """Create codec with the given decoder and encoder."""
        self._decoder = decoder
        self._encoder = encoder

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @property
    def decoder(self) -> Decoder:
        return self._decoder

    @classmethod
    def supported_transfer_syntaxes(
        cls,
        samples_per_pixel: int,
        bits: int,
        photometric_interpretation: str,
        pixel_representation: int,
        possible_transfer_syntaxes: Optional[Iterable[UID]] = None,
    ) -> Iterable[UID]:
        """Return supported transfer syntaxes for the given parameters.

        Parameters
        ----------
        samples_per_pixel: int
            Samples per pixel of the image.
        bits: int
            Bits per sample of the image.
        photometric_interpretation: str
            Photometric interpretation of the image.
        pixel_representation: int
            Pixel representation of the image.
        possible_transfer_syntaxes: Optional[Iterable[UID]]
            Possible transfer syntaxes to check for support.

        Returns
        ----------
        Iterable[UID]
            Supported transfer syntaxes for the given parameters.
        """
        if possible_transfer_syntaxes is None:
            possible_transfer_syntaxes = AllTransferSyntaxes
        return (
            transfer_syntax
            for transfer_syntax in AllTransferSyntaxes
            if cls.is_supported(
                transfer_syntax,
                samples_per_pixel,
                bits,
                photometric_interpretation,
                pixel_representation,
            )
        )

    def decode(self, data: bytes) -> PILImage:
        return self.decoder.decode(data)

    def encode(self, image: Union[PILImage, np.ndarray]) -> bytes:
        return self.encoder.encode(image)

    @classmethod
    def create(
        cls,
        transfer_syntax: UID,
        samples_per_pixel: int,
        bits: int,
        size: Size,
        photometric_interpretation: str,
        pixel_representation: int,
        planar_configuration: int,
    ) -> "Codec":
        """Create codec for the given parameters.

        Parameters
        ----------
        transfer_syntax: UID
            Transfer syntax of the image.
        samples_per_pixel: int
            Samples per pixel of the image.
        bits: int
            Bits per sample of the image.
        size: Size
            Size of the image.
        photometric_interpretation: str
            Photometric interpretation of the image.
        pixel_representation: int
            Pixel representation of the image.
        planar_configuration: int
            Planar configuration of the image.

        Returns
        ----------
        Codec
            Codec for the given parameters.
        """
        decoder = Decoder.create(
            transfer_syntax,
            samples_per_pixel,
            bits,
            size,
            photometric_interpretation,
            pixel_representation,
            planar_configuration,
        )
        encoder = cls._create_encoder(
            transfer_syntax,
            samples_per_pixel,
            bits,
            photometric_interpretation,
            pixel_representation,
        )
        return cls(decoder, encoder)

    @classmethod
    def is_supported(
        cls,
        transfer_syntax: UID,
        samples_per_pixel: int,
        bits: int,
        photometric_interpretation: str,
        pixel_representation: int,
    ) -> bool:
        """Return True if codec supports the given parameters.

        Parameters
        ----------
        transfer_syntax: UID
            Transfer syntax of the image.
        samples_per_pixel: int
            Samples per pixel of the image.
        bits: int
            Bits per sample of the image.
        photometric_interpretation: str
            Photometric interpretation of the image.
        pixel_representation: int
            Pixel representation of the image.

        Returns
        ----------
        bool
            True if codec supports the given parameters.
        """
        decoder = Decoder.select_decoder(transfer_syntax, samples_per_pixel, bits)
        if decoder is None:
            return False
        try:
            cls._create_encoder(
                transfer_syntax,
                samples_per_pixel,
                bits,
                photometric_interpretation,
                pixel_representation,
            )
        except Exception:
            return False
        return True

    @classmethod
    def _create_encoder(
        cls,
        transfer_syntax: UID,
        samples_per_pixel: int,
        bits: int,
        photometric_interpretation: str,
        pixel_representation: int,
    ) -> Encoder:
        encoder_settings = Settings.create(
            transfer_syntax,
            bits,
            photometric_interpretation,
            pixel_representation,
        )
        return Encoder.create(encoder_settings)
