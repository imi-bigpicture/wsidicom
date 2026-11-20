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

"""Integration tests for how the levels generated when saving are encoded.

A small synthetic pyramid is saved with missing levels added, with its stored
levels either all in one transfer syntax or in different ones, and the generated
levels of the saved files are inspected.
"""

from pathlib import Path

import numpy as np
import pytest
from pydicom import Dataset
from pydicom.uid import JPEG2000, UID, JPEGBaseline8Bit

from wsidicom import WsiDicom
from wsidicom.codec import Channels, Encoder, Jpeg2kSettings, JpegSettings
from wsidicom.file import OffsetTableType, WsiDicomFileTarget
from wsidicom.geometry import Point, Size, SizeMm
from wsidicom.group import Level
from wsidicom.instance import ImageData, TileType, WsiInstance
from wsidicom.metadata import ImageCoordinateSystem, ImageType, LossyCompression
from wsidicom.metadata.uid_generator import CallableUidGenerator
from wsidicom.series import Pyramid, Pyramids

IMAGE_SIZE = 1024
"""Width and height of the base level, giving levels 0 to 3."""
TILE_SIZE = 128
"""Width and height of the tiles."""
BASE_PIXEL_SPACING = 0.0005
"""Pixel spacing of the base level, in mm."""


class SyntheticImageData(ImageData):
    """Tiled image data of pixels held in memory, encoded with an encoder."""

    def __init__(self, pixels: np.ndarray, encoder: Encoder, scale: int) -> None:
        super().__init__(encoder)
        self._pixels = pixels
        self._scale = scale

    @property
    def transfer_syntax(self) -> UID:
        return self.encoder.transfer_syntax

    @property
    def image_size(self) -> Size:
        return Size(self._pixels.shape[1], self._pixels.shape[0])

    @property
    def tile_size(self) -> Size:
        return Size(TILE_SIZE, TILE_SIZE)

    @property
    def pixel_spacing(self) -> SizeMm:
        spacing = BASE_PIXEL_SPACING * self._scale
        return SizeMm(spacing, spacing)

    @property
    def imaged_size(self) -> SizeMm | None:
        return None

    @property
    def samples_per_pixel(self) -> int:
        return 3

    @property
    def photometric_interpretation(self) -> str:
        return self.encoder.photometric_interpretation

    @property
    def bits(self) -> int:
        return 8

    @property
    def image_coordinate_system(self) -> ImageCoordinateSystem | None:
        return None

    @property
    def thread_safe(self) -> bool:
        return True

    @property
    def lossy_compression(self) -> list[LossyCompression] | None:
        if self.encoder.lossy_method is None:
            return None
        return [LossyCompression(self.encoder.lossy_method, 10.0)]

    @property
    def transcoder(self) -> Encoder | None:
        return None

    def get_decoded_tile(
        self, tile_point: Point, z: float, path: str, cache: bool = True
    ) -> np.ndarray:
        y = tile_point.y * TILE_SIZE
        x = tile_point.x * TILE_SIZE
        return self._pixels[y : y + TILE_SIZE, x : x + TILE_SIZE]

    def get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        return self.encoder.encode(self.get_decoded_tile(tile, z, path))


@pytest.mark.integration
@pytest.mark.parametrize("tile_type", [TileType.FULL])
class TestGeneratedLevelEncodingIntegration:
    JPEG = Encoder.create_for_settings(JpegSettings())
    """Encoder of stored JPEG Baseline levels."""
    JPEG_2000 = Encoder.create_for_settings(Jpeg2kSettings())
    """Encoder of stored lossy JPEG 2000 levels."""

    @staticmethod
    def base_pixels() -> np.ndarray:
        """Return smooth base level pixels whose channels all differ."""
        ramp = np.linspace(0, 1, IMAGE_SIZE)
        horizontal, vertical = np.meshgrid(ramp, ramp)
        channels = [
            40 + 160 * horizontal,
            200 - 160 * vertical,
            40 + 80 * (horizontal + vertical),
        ]
        return np.ascontiguousarray(np.stack(channels, axis=-1).astype(np.uint8))

    @classmethod
    def level_pixels(cls, level: int) -> np.ndarray:
        """Return the base level pixels downsampled to a level."""
        scale = 2**level
        size = IMAGE_SIZE // scale
        blocks = cls.base_pixels().reshape(size, scale, size, scale, 3)
        return np.ascontiguousarray(blocks.mean(axis=(1, 3)).round().astype(np.uint8))

    @classmethod
    def create_pyramids(
        cls, base_dataset: Dataset, stored_levels: dict[int, Encoder]
    ) -> Pyramids:
        """Return a pyramid of the levels given, each with its encoder.

        The instances are described with the metadata of the base dataset.
        """
        base_pixel_spacing = SizeMm(BASE_PIXEL_SPACING, BASE_PIXEL_SPACING)
        levels = [
            Level(
                [
                    WsiInstance.create_instance(
                        SyntheticImageData(cls.level_pixels(level), encoder, 2**level),
                        base_dataset,
                        ImageType.VOLUME,
                        level,
                    )
                ],
                base_pixel_spacing,
            )
            for level, encoder in stored_levels.items()
        ]
        return Pyramids([Pyramid(levels, [])])

    @staticmethod
    def save(
        pyramids: Pyramids,
        output_path: Path,
        uid_generator: CallableUidGenerator,
        transcoding: JpegSettings | None,
    ) -> None:
        """Save the pyramids with missing levels added."""
        with WsiDicomFileTarget(
            output_path,
            uid_generator,
            1,
            None,
            OffsetTableType.BASIC,
            None,
            None,
            True,
            False,
            transcoding,
        ) as target:
            target.save(pyramids, None, None, True)

    @staticmethod
    def compression_ratio(level: Level) -> float:
        """Return the compression ratio of the tiles of a saved level."""
        image_data = level.default_instance.image_data
        compressed_size = sum(
            len(
                image_data.get_encoded_tile(
                    Point(x, y), image_data.default_z, image_data.default_path
                )
            )
            for y in range(image_data.tiled_size.height)
            for x in range(image_data.tiled_size.width)
        )
        uncompressed_size = (
            image_data.tiled_size.area
            * image_data.tile_size.area
            * image_data.samples_per_pixel
            * (image_data.bits // 8)
        )
        return uncompressed_size / compressed_size

    @classmethod
    def assert_generated_level(
        cls, level: Level, transfer_syntax: UID, encoder: Encoder
    ) -> None:
        """Assert that a generated level is encoded with and described as given.

        Its pixels must also be close to those of the base level downsampled,
        which they are not if the tiles are described with another photometric
        interpretation than they are encoded in.
        """
        instance = level.default_instance
        dataset = instance.dataset
        assert encoder.lossy_method is not None
        assert instance.image_data.transfer_syntax == transfer_syntax
        assert dataset.photometric_interpretation == encoder.photometric_interpretation
        assert dataset.lossy_compressed
        assert dataset.lossy_compression_methods == [encoder.lossy_method.value]
        assert dataset.lossy_compression_ratios == [
            pytest.approx(cls.compression_ratio(level), rel=0.01)
        ]
        tile = np.asarray(
            level.get_tile(Point(0, 0), instance.image_data.default_z, "0")
        ).astype(int)
        expected = cls.level_pixels(level.level)[:TILE_SIZE, :TILE_SIZE].astype(int)
        assert np.abs(tile - expected).mean() < 5

    def test_generated_levels_keep_encoding_of_level_below_them(
        self,
        main_dataset: Dataset,
        tmp_path: Path,
        uid_generator: CallableUidGenerator,
    ) -> None:
        # Arrange
        pyramids = self.create_pyramids(main_dataset, {0: self.JPEG})

        # Act
        self.save(pyramids, tmp_path, uid_generator, None)

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            levels = saved_wsi.pyramids[0].levels
            assert [level.level for level in levels] == [0, 1, 2, 3]
            for level in levels[1:]:
                self.assert_generated_level(level, JPEGBaseline8Bit, self.JPEG)

    def test_generated_levels_of_mixed_levels_keep_encoding_of_level_below_them(
        self,
        main_dataset: Dataset,
        tmp_path: Path,
        uid_generator: CallableUidGenerator,
    ) -> None:
        # Arrange
        pyramids = self.create_pyramids(main_dataset, {0: self.JPEG, 2: self.JPEG_2000})

        # Act
        self.save(pyramids, tmp_path, uid_generator, None)

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            pyramid = saved_wsi.pyramids[0]
            self.assert_generated_level(pyramid.get(1), JPEGBaseline8Bit, self.JPEG)
            self.assert_generated_level(pyramid.get(3), JPEG2000, self.JPEG_2000)

    def test_generated_levels_are_encoded_with_transcoding_settings(
        self,
        main_dataset: Dataset,
        tmp_path: Path,
        uid_generator: CallableUidGenerator,
    ) -> None:
        # Arrange
        # The stored level is copied, as it is in the transfer syntax of the
        # settings, while the generated levels are encoded with the settings.
        pyramids = self.create_pyramids(main_dataset, {0: self.JPEG})
        settings = JpegSettings(channels=Channels.RGB)

        # Act
        self.save(pyramids, tmp_path, uid_generator, settings)

        # Assert
        transcoder = Encoder.create_for_settings(settings)
        with WsiDicom.open(tmp_path) as saved_wsi:
            levels = saved_wsi.pyramids[0].levels
            base_dataset = levels[0].default_instance.dataset
            assert base_dataset.photometric_interpretation == "YBR_FULL_422"
            for level in levels[1:]:
                self.assert_generated_level(level, JPEGBaseline8Bit, transcoder)

    def test_generated_levels_of_mixed_levels_are_encoded_with_transcoding_settings(
        self,
        main_dataset: Dataset,
        tmp_path: Path,
        uid_generator: CallableUidGenerator,
    ) -> None:
        # Arrange
        # Level 0 is copied and level 2 transcoded, and the levels generated from
        # either are encoded with the settings.
        pyramids = self.create_pyramids(main_dataset, {0: self.JPEG, 2: self.JPEG_2000})
        settings = JpegSettings(channels=Channels.RGB)

        # Act
        self.save(pyramids, tmp_path, uid_generator, settings)

        # Assert
        transcoder = Encoder.create_for_settings(settings)
        with WsiDicom.open(tmp_path) as saved_wsi:
            pyramid = saved_wsi.pyramids[0]
            transcoded_dataset = pyramid.get(2).default_instance.dataset
            assert transcoded_dataset.photometric_interpretation == "RGB"
            assert transcoded_dataset.lossy_compression_methods == [
                "ISO_15444_1",
                "ISO_10918_1",
            ]
            self.assert_generated_level(pyramid.get(1), JPEGBaseline8Bit, transcoder)
            self.assert_generated_level(pyramid.get(3), JPEGBaseline8Bit, transcoder)
