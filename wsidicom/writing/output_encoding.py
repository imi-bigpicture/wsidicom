#    Copyright 2026 SECTRA AB
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

"""How the tiles of each source end up encoded in the output."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from pydicom.uid import UID

from wsidicom.codec import Encoder, Recoder, Settings
from wsidicom.instance import ImageData
from wsidicom.options import TranscodingMode


@dataclass(frozen=True)
class OutputEncoding(metaclass=ABCMeta):
    """How the tiles of a source end up encoded in the output."""

    generated_tiles_encoder: Encoder
    """Encoder for the tiles of the levels generated from the source."""
    transfer_syntax: UID
    """Transfer syntax of the tiles written from the source."""

    @property
    @abstractmethod
    def transcoder(self) -> Encoder | None:
        """Encoder the tiles of the source are encoded again with.

        None if their pixels keep the encoding of the source.
        """
        raise NotImplementedError()


@dataclass(frozen=True)
class CopiedEncoding(OutputEncoding):
    """The encoded tiles of the source are written as they are."""

    @property
    def transcoder(self) -> None:
        """None, as the tiles are written as they are."""
        return None


@dataclass(frozen=True)
class RecodedEncoding(OutputEncoding):
    """The encoded tiles of the source are recoded and then written."""

    recoder: Recoder
    """Recoder the tiles are recoded with."""

    @property
    def transcoder(self) -> None:
        """None, as recoding does not change the pixels of the tiles."""
        return None


@dataclass(frozen=True)
class TranscodedEncoding(OutputEncoding):
    """The tiles of the source are decoded and encoded again."""

    @property
    def transcoder(self) -> Encoder:
        """The encoder of the generated tiles, as the tiles are encoded with it."""
        return self.generated_tiles_encoder


class EncodingResolver:
    """Resolves how the tiles of each source are encoded when written."""

    def __init__(
        self, transcoding: Settings | Encoder | None, mode: TranscodingMode
    ) -> None:
        """Create a resolver for transcoding settings or an encoder.

        Parameters
        ----------
        transcoding: Settings | Encoder | None
            Settings or encoder for transcoding. Settings are adapted to the
            pixel format of each source and encoded with the encoder created for
            them, while an encoder is used as it is. If None, the encoding of
            each source is kept. If a recoder applies to the settings, the tiles it
            can recode are recoded rather than copied or transcoded.
        mode: TranscodingMode
            Which sources are transcoded with the transcoder.

        Raises
        ------
        RuntimeError
            If a recoder applies to the settings of the transcoder but is not
            available.
        """
        if isinstance(transcoding, Settings):
            self._settings = transcoding
            transcoder = Encoder.create_for_settings(transcoding)
        else:
            self._settings = None
            transcoder = transcoding
        self._transcoder = transcoder
        self._adapted_transcoders: dict[tuple[int, int], Encoder] = {}
        self._mode = mode
        self._recoder = (
            Recoder.create_for_settings(transcoder.settings)
            if transcoder is not None
            else None
        )

    def resolve(self, image_data: ImageData) -> OutputEncoding:
        """Return how the tiles of image data are encoded when written.

        The tiles are transcoded if the mode is to always transcode or if the
        image data can only give decoded tiles. Otherwise they are recoded if
        the recoder can recode them, transcoded if the mode is to match the
        transcoder and they are in another transfer syntax than it, and else
        copied.

        Parameters
        ----------
        image_data: ImageData
            Image data of the source.

        Returns
        -------
        OutputEncoding
            How the tiles of the image data are encoded.

        Raises
        ------
        ValueError
            If the tiles are transcoded with an encoder whose bits or samples
            per pixel do not match those of the image data.
        """
        transcoder = self._transcoder_for(image_data)
        if transcoder is not None and self._mode == TranscodingMode.ALWAYS:
            # The user wants every tile decoded and encoded again with the transcoder.
            return self._transcoded(transcoder, image_data)
        if image_data.transcoder is not None:
            # The image data can only give decoded tiles, so they must be encoded.
            return self._transcoded(
                # The transcoder of the user, if it asks for another transfer
                # syntax, and else the own transcoder of the image data.
                transcoder
                if transcoder is not None
                and image_data.transfer_syntax != transcoder.transfer_syntax
                else image_data.transcoder,
                image_data,
            )
        if self._recoder is not None and self._recoder.supports_transfer_syntax(
            image_data.transfer_syntax
        ):
            # The tiles can be changed without decoding them, so their pixels are kept.
            encoder = self._generated_tiles_encoder(image_data)
            return RecodedEncoding(
                encoder,
                self._recoder.transfer_syntax,
                self._recoder,
            )
        if (
            self._mode == TranscodingMode.MATCH
            and transcoder is not None
            and image_data.transfer_syntax != transcoder.transfer_syntax
        ):
            # The tiles are in another transfer syntax than the user asks for.
            return self._transcoded(transcoder, image_data)
        # The tiles are already in the transfer syntax asked for, none was asked
        # for, or they are kept in their own.
        encoder = self._generated_tiles_encoder(image_data)
        return CopiedEncoding(encoder, image_data.transfer_syntax)

    def _transcoder_for(self, image_data: ImageData) -> Encoder | None:
        """Return the transcoder of the user for image data.

        An encoder given by the user is returned as it is. For settings, an
        encoder is created for the settings adapted to the pixel format of the
        image data, once for each pixel format.
        """
        if self._settings is None or self._transcoder is None:
            return self._transcoder
        pixel_format = (image_data.bits, image_data.samples_per_pixel)
        encoder = self._adapted_transcoders.get(pixel_format)
        if encoder is None:
            adapted = self._settings.adapted_to(*pixel_format)
            encoder = (
                self._transcoder
                if adapted is self._settings
                else Encoder.create_for_settings(adapted)
            )
            self._adapted_transcoders[pixel_format] = encoder
        return encoder

    def _transcoded(
        self, transcoder: Encoder, image_data: ImageData
    ) -> TranscodedEncoding:
        """Return the encoding of image data transcoded with an encoder.

        Raises
        ------
        ValueError
            If the bits or samples per pixel of the encoder do not match those of
            the image data.
        """
        if not self._can_encode(transcoder, image_data):
            raise ValueError(
                "Transcode settings must match image data bits and "
                "photometric interpretation."
            )
        transcoder = self._matching_encoder(transcoder)
        return TranscodedEncoding(transcoder, transcoder.transfer_syntax)

    def _generated_tiles_encoder(self, image_data: ImageData) -> Encoder:
        """Return the encoder for levels generated from image data kept encoded.

        The tiles of the generated levels are new, so they are encoded with the
        transcoder if it can encode the pixels of the image data, and else with
        the encoder of the image data. Either is matched to the recoder.
        """
        transcoder = self._transcoder_for(image_data)
        if transcoder is not None and self._can_encode(transcoder, image_data):
            return self._matching_encoder(transcoder)
        return self._matching_encoder(image_data.encoder)

    @staticmethod
    def _can_encode(encoder: Encoder, image_data: ImageData) -> bool:
        """Return True if an encoder can encode the decoded pixels of image data.

        That is, if its bits and samples per pixel are those of the image data.
        """
        return (
            encoder.bits == image_data.bits
            and encoder.samples_per_pixel == image_data.samples_per_pixel
        )

    def _matching_encoder(self, encoder: Encoder) -> Encoder:
        """Return the encoder to encode new tiles with in place of an encoder.

        New tiles are encoded as the recoder recodes tiles, so that they match
        the tiles that are recoded.
        """
        if self._recoder is None:
            return encoder
        return self._recoder.matching_encoder(encoder)
