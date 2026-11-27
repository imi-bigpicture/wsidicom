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

import numpy as np
import pytest
from pydicom.uid import UID, ExplicitVRLittleEndian

from wsidicom.codec import Encoder
from wsidicom.geometry import Point, Size, SizeMm
from wsidicom.instance.image_data import ImageData
from wsidicom.metadata import ImageCoordinateSystem, LossyCompression


class _FocalPlanesImageData(ImageData):
    """Minimal ImageData subclass for testing focal-plane selection."""

    def __init__(self, focal_planes: list[float]) -> None:
        super().__init__(Encoder.create(ExplicitVRLittleEndian, 8, "RGB"))
        self._focal_planes = focal_planes

    @property
    def focal_planes(self) -> list[float]:
        return self._focal_planes

    @property
    def transfer_syntax(self) -> UID:
        return ExplicitVRLittleEndian

    @property
    def image_size(self) -> Size:
        return Size(1, 1)

    @property
    def tile_size(self) -> Size:
        return Size(1, 1)

    @property
    def pixel_spacing(self) -> SizeMm | None:
        return None

    @property
    def imaged_size(self) -> SizeMm | None:
        return None

    @property
    def samples_per_pixel(self) -> int:
        return 3

    @property
    def photometric_interpretation(self) -> str:
        return "RGB"

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
        return None

    @property
    def transcoder(self) -> Encoder | None:
        return None

    def get_decoded_tile(
        self,
        tile_point: Point,
        z: float,
        path: str,
        cache: bool = True,
    ) -> np.ndarray:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        return b""


@pytest.mark.unittest
class TestImageDataDefaultZ:
    @pytest.mark.parametrize(
        ["focal_planes", "expected"],
        [
            ([5.0], 5.0),
            ([0.0, 1.0, 2.0], 1.0),
            ([-2.0, -1.0, 0.0, 1.0, 2.0], 0.0),
            ([10.0, 20.0, 30.0], 20.0),
            ([-30.0, -20.0, -10.0], -20.0),
            ([30.0, 10.0, 20.0], 20.0),
        ],
    )
    def test_default_z_returns_middle_focal_plane(
        self, focal_planes: list[float], expected: float
    ) -> None:
        image_data = _FocalPlanesImageData(focal_planes)

        assert image_data.default_z == expected
