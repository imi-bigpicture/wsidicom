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

from collections.abc import Sequence

import pytest
from pydicom.uid import UID

from wsidicom import WsiDicom
from wsidicom.cache import DecodedFrameCache, EncodedFrameCache
from wsidicom.codec import Codec
from wsidicom.errors import WsiDicomOutOfBoundsError
from wsidicom.geometry import Point
from wsidicom.instance.dataset import WsiDataset
from wsidicom.instance.wsidicom_image_data import WsiDicomImageData

CACHE_SIZE = 100 * 1024 * 1024


class WsiDicomTestImageData(WsiDicomImageData):
    """Image data serving one frame of its own, given at construction, through an
    encoded frame cache it may share with other image data (as DICOMweb image data
    do)."""

    def __init__(
        self,
        datasets: Sequence[WsiDataset],
        codec: Codec,
        decoded_frame_cache: DecodedFrameCache,
        encoded_frame_cache: EncodedFrameCache,
        transfer_syntax: UID,
        frame: bytes,
    ):
        self._transfer_syntax = transfer_syntax
        self._frame = frame
        self._encoded_frame_cache = encoded_frame_cache
        super().__init__(datasets, codec, decoded_frame_cache)

    @property
    def transfer_syntax(self) -> UID:
        return self._transfer_syntax

    def _get_tile_frame(self, frame_index: int) -> bytes:
        return self._frame

    def _get_encoded_tile_frame(self, frame_index: int) -> bytes:
        return self._encoded_frame_cache.get_tile_frame(
            self._cache_id, frame_index, self._get_tile_frame
        )


@pytest.mark.unittest
class TestWsiDicomImageData:
    def test_serves_own_frame_when_sharing_a_cache_with_collected_image_data(
        self, wsi: WsiDicom
    ):
        """An image data serves its own frames, never those of a collected one.

        The frame caches are held by the source and shared by every image data it
        produces, so an image data created after another has been collected must
        not be served the frames cached for that other one.
        """
        # Arrange
        instance = next(iter(wsi.pyramid.levels[0].instances.values()))
        datasets = instance.datasets
        dataset = datasets[0]
        transfer_syntax = instance.image_data.transfer_syntax
        codec = Codec.create(
            transfer_syntax,
            dataset.samples_per_pixel,
            dataset.bits,
            dataset.tile_size,
            dataset.photometric_interpretation,
        )
        decoded_frame_cache = DecodedFrameCache(CACHE_SIZE)
        encoded_frame_cache = EncodedFrameCache(CACHE_SIZE)
        image_data_count = 100

        # Act
        served_frames: list[tuple[bytes, bytes]] = []
        for index in range(image_data_count):
            frame = index.to_bytes(4, "little")
            image_data = WsiDicomTestImageData(
                datasets,
                codec,
                decoded_frame_cache,
                encoded_frame_cache,
                transfer_syntax,
                frame,
            )
            served_frames.append(
                (
                    frame,
                    image_data.get_encoded_tile(
                        Point(0, 0), instance.default_z, instance.default_path
                    ),
                )
            )
            del image_data

        # Assert
        mismatches = [
            (expected, served)
            for expected, served in served_frames
            if served != expected
        ]
        assert not mismatches, (
            f"{len(mismatches)} of {image_data_count} image data were served "
            f"another image data's frame, first {mismatches[0]}"
        )

    @pytest.mark.parametrize("past", ["column", "row"])
    def test_get_encoded_tile_one_past_last_column_or_row_raises(
        self, wsi: WsiDicom, past: str
    ):
        """A tile one past the last column is not served the first tile of the next
        row (nor one past the last row the first tile of the next plane)."""
        # Arrange
        image_data = next(iter(wsi.pyramid.levels[0].instances.values())).image_data
        tiled_size = image_data.tiled_size
        tile = (
            Point(tiled_size.width, 0)
            if past == "column"
            else Point(0, tiled_size.height)
        )

        # Act & Assert
        with pytest.raises(WsiDicomOutOfBoundsError):
            image_data.get_encoded_tile(
                tile, image_data.default_z, image_data.default_path
            )
