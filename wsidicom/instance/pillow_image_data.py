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

from pathlib import Path
from typing import Optional, Union

from PIL import Image as Pillow
from PIL.Image import Image
from pydicom.uid import UID, JPEGBaseline8Bit

from wsidicom.codec import Encoder
from wsidicom.geometry import (
    Point,
    Size,
    SizeMm,
)
from wsidicom.instance.image_coordinate_system import ImageCoordinateSystem
from wsidicom.instance.image_data import ImageData


class PillowImageData(ImageData):
    def __init__(self, image: Image):
        self._image = image.convert("RGB")
        encoder = Encoder.create(
            self.transfer_syntax,
            self.bits,
            self.photometric_interpretation,
        )
        super().__init__(encoder)

    @classmethod
    def from_file(cls, file: Union[str, Path]) -> "PillowImageData":
        image = Pillow.open(file)
        return cls(image)

    @property
    def transfer_syntax(self) -> UID:
        return JPEGBaseline8Bit

    @property
    def image_size(self) -> Size:
        return Size.from_tuple(self._image.size)

    @property
    def tile_size(self) -> Size:
        return self.image_size

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        return None

    @property
    def samples_per_pixel(self) -> int:
        return 3

    @property
    def bits(self) -> int:
        return 8

    @property
    def lossy_compressed(self) -> bool:
        return True

    @property
    def photometric_interpretation(self) -> str:
        return "YBR_FULL_422"

    @property
    def image_coordinate_system(self) -> Optional[ImageCoordinateSystem]:
        return None

    def _get_decoded_tile(self, tile_point: Point, z: float, path: str) -> Image:
        if tile_point != Point(0, 0):
            raise ValueError("Can only get Point(0, 0) from non-tiled image.")
        return self._image

    def _get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        if tile != Point(0, 0):
            raise ValueError("Can only get Point(0, 0) from non-tiled image.")
        return self.encoder.encode(self._image)
