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

"""Tests of encoder settings."""

import numpy as np
import pytest
from pydicom.uid import UID, JPEGLossless, JPEGLosslessSV1

from wsidicom.codec import Channels, Decoder, Encoder
from wsidicom.codec.settings import (
    HTJpeg2000Settings,
    Jpeg2kSettings,
    JpegLosslessSettings,
    JpegLsSettings,
    JpegSettings,
    JpegXlSettings,
    NumpySettings,
    RleSettings,
    Settings,
    Subsampling,
)
from wsidicom.geometry import Size

ALL_SETTINGS = [
    JpegSettings,
    JpegLosslessSettings,
    JpegLsSettings,
    Jpeg2kSettings,
    HTJpeg2000Settings,
    JpegXlSettings,
    NumpySettings,
    RleSettings,
]
SETTINGS_SUPPORTING_16_BITS = [
    JpegLosslessSettings,
    JpegLsSettings,
    Jpeg2kSettings,
    HTJpeg2000Settings,
    JpegXlSettings,
    NumpySettings,
    RleSettings,
]


@pytest.mark.unittest
class TestSettingsAdaptedTo:
    @pytest.mark.parametrize(
        ["settings_type", "expected_channels"],
        [
            (JpegSettings, Channels.YBR),
            (JpegLosslessSettings, Channels.YBR),
            (JpegLsSettings, Channels.RGB),
            (Jpeg2kSettings, Channels.YBR),
            (HTJpeg2000Settings, Channels.YBR),
            (JpegXlSettings, Channels.YBR),
            (NumpySettings, Channels.RGB),
            (RleSettings, Channels.RGB),
        ],
    )
    def test_not_given_bits_and_channels_report_defaults(
        self, settings_type: type[Settings], expected_channels: Channels
    ) -> None:
        # Arrange

        # Act
        settings = settings_type()

        # Assert
        assert settings.bits == 8
        assert settings.channels == expected_channels

    @pytest.mark.parametrize("settings_type", ALL_SETTINGS)
    def test_8_bit_color_image_keeps_the_settings(
        self, settings_type: type[Settings]
    ) -> None:
        # Arrange
        settings = settings_type()

        # Act
        adapted = settings.adapted_to(8, 3)

        # Assert
        assert adapted is settings

    @pytest.mark.parametrize("settings_type", ALL_SETTINGS)
    def test_grayscale_image_gives_grayscale(
        self, settings_type: type[Settings]
    ) -> None:
        # Arrange
        settings = settings_type()

        # Act
        adapted = settings.adapted_to(8, 1)

        # Assert
        assert type(adapted) is settings_type
        assert adapted.channels == Channels.GRAYSCALE
        assert adapted.samples_per_pixel == 1
        assert settings.channels != Channels.GRAYSCALE

    @pytest.mark.parametrize("settings_type", SETTINGS_SUPPORTING_16_BITS)
    def test_16_bit_grayscale_image_gives_16_bits(
        self, settings_type: type[Settings]
    ) -> None:
        # Arrange
        settings = settings_type()

        # Act
        adapted = settings.adapted_to(16, 1)

        # Assert
        assert adapted.bits == 16
        assert adapted.channels == Channels.GRAYSCALE
        assert settings.bits == 8

    @pytest.mark.parametrize(["bits", "expected_bits"], [(12, 12), (16, 8)])
    def test_jpeg_only_takes_bits_it_can_encode(
        self, bits: int, expected_bits: int
    ) -> None:
        # Arrange
        settings = JpegSettings()

        # Act
        adapted = settings.adapted_to(bits, 1)

        # Assert
        assert adapted.bits == expected_bits

    @pytest.mark.parametrize("settings_type", ALL_SETTINGS)
    def test_16_bit_color_image_gives_8_bits(
        self, settings_type: type[Settings]
    ) -> None:
        # Arrange
        settings = settings_type()

        # Act
        adapted = settings.adapted_to(16, 3)

        # Assert
        assert adapted.bits == 8

    def test_given_bits_and_channels_are_kept(self) -> None:
        # Arrange
        settings = Jpeg2kSettings(bits=8, channels=Channels.RGB)

        # Act
        adapted = settings.adapted_to(16, 1)

        # Assert
        assert adapted is settings

    def test_given_bits_are_kept_while_channels_are_adapted(self) -> None:
        # Arrange
        settings = Jpeg2kSettings(bits=12)

        # Act
        adapted = settings.adapted_to(16, 1)

        # Assert
        assert adapted.bits == 12
        assert adapted.channels == Channels.GRAYSCALE

    def test_other_settings_are_kept(self) -> None:
        # Arrange
        settings = JpegSettings(quality=55, subsampling=Subsampling.R444, optimize=True)

        # Act
        adapted = settings.adapted_to(8, 1)

        # Assert
        assert isinstance(adapted, JpegSettings)
        assert adapted.quality == 55
        assert adapted.subsampling == Subsampling.R444
        assert adapted.optimize

    def test_with_optimize_keeps_bits_and_channels_not_given(self) -> None:
        # Arrange
        settings = JpegSettings()

        # Act
        adapted = settings.with_optimize(True).adapted_to(8, 1)

        # Assert
        assert adapted.channels == Channels.GRAYSCALE
        assert isinstance(adapted, JpegSettings)
        assert adapted.optimize
        assert not settings.optimize

    @pytest.mark.parametrize(
        ["settings", "bits"],
        [
            (JpegSettings(), 8),
            (JpegSettings(), 12),
            (Jpeg2kSettings(levels=0), 8),
            (Jpeg2kSettings(levels=0), 16),
            (HTJpeg2000Settings(levels=0), 16),
            (JpegXlSettings(level=100), 16),
            (JpegLsSettings(), 16),
        ],
    )
    def test_adapted_settings_encode_grayscale_image(
        self, settings: Settings, bits: int
    ) -> None:
        # Arrange
        dtype = np.uint8 if bits == 8 else np.uint16
        pixels = np.tile(np.linspace(0, 2**bits - 1, 64).astype(dtype), (64, 1))
        encoder = Encoder.create_for_settings(settings.adapted_to(bits, 1))

        # Act
        encoded = encoder.encode(pixels)

        # Assert
        decoder = Decoder.create(
            encoder.transfer_syntax,
            samples_per_pixel=1,
            bits=bits,
            size=Size(64, 64),
            photometric_interpretation=encoder.photometric_interpretation,
        )
        decoded = np.asarray(decoder.decode(encoded))
        assert decoded.shape[:2] == pixels.shape
        error = np.abs(decoded.astype(int) - pixels.astype(int)).max()
        assert error <= (2**bits) * 0.02


@pytest.mark.unittest
class TestJpegLosslessSettings:
    @pytest.mark.parametrize(
        ["predictor", "expected_transfer_syntax"],
        [
            (1, JPEGLosslessSV1),
            *((predictor, JPEGLossless) for predictor in range(2, 8)),
        ],
    )
    def test_predictor_gives_transfer_syntax(
        self, predictor: int, expected_transfer_syntax: UID
    ) -> None:
        # Arrange

        # Act
        settings = JpegLosslessSettings(predictor)

        # Assert
        assert settings.transfer_syntax == expected_transfer_syntax

    def test_default_gives_sv1(self) -> None:
        # Arrange

        # Act
        settings = JpegLosslessSettings()

        # Assert
        assert settings.predictor == 1
        assert settings.transfer_syntax == JPEGLosslessSV1

    @pytest.mark.parametrize("predictor", [0, 8, None])
    def test_unsupported_predictor_raises(self, predictor: int | None) -> None:
        # Arrange

        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported predictor"):
            JpegLosslessSettings(predictor)  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize("transfer_syntax", [JPEGLossless, JPEGLosslessSV1])
    def test_created_settings_keep_transfer_syntax(self, transfer_syntax: UID) -> None:
        # Arrange

        # Act
        settings = Settings.create(transfer_syntax, 8, "RGB")

        # Assert
        assert settings.transfer_syntax == transfer_syntax
