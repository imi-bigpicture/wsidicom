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

"""Module with codec supporting different formats."""

from typing import Iterable, Optional, Union

import numpy as np
from PIL.Image import Image
from pydicom.uid import (
    UID,
    AllTransferSyntaxes,
)

from wsidicom.codec.decoder import Decoder
from wsidicom.codec.encoder import Encoder
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
            )
        )

    def decode(self, data: bytes) -> Image:
        return self.decoder.decode(data)

    def encode(self, image: Union[Image, np.ndarray]) -> bytes:
        return self.encoder.encode(image)

    @classmethod
    def create(
        cls,
        transfer_syntax: UID,
        samples_per_pixel: int,
        bits: int,
        size: Size,
        photometric_interpretation: str,
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
        )
        encoder = Encoder.create(
            transfer_syntax,
            bits,
            photometric_interpretation,
        )
        return cls(decoder, encoder)

    @classmethod
    def is_supported(
        cls,
        transfer_syntax: UID,
        samples_per_pixel: int,
        bits: int,
        photometric_interpretation: str,
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

        Returns
        ----------
        bool
            True if codec supports the given parameters.
        """
        decoder = Decoder.select_decoder(transfer_syntax, samples_per_pixel, bits)
        if decoder is None:
            return False
        try:
            Encoder.create(
                transfer_syntax,
                bits,
                photometric_interpretation,
            )
        except Exception:
            return False
        return True
