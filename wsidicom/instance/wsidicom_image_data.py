#    Copyright 2021, 2022, 2023 SECTRA AB
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

import io
from abc import ABCMeta, abstractmethod
from functools import cached_property
from typing import List, Optional, Sequence

from PIL import Image
from PIL.Image import Image as PILImage

from wsidicom.errors import WsiDicomOutOfBoundsError
from wsidicom.geometry import Point, Region, Size, SizeMm
from wsidicom.instance.dataset import TileType, WsiDataset
from wsidicom.instance.image_data import ImageData
from wsidicom.instance.image_origin import ImageOrigin
from wsidicom.instance.tile_index.full_tile_index import FullTileIndex
from wsidicom.instance.tile_index.sparse_tile_index import SparseTileIndex
from wsidicom.instance.tile_index.tile_index import TileIndex


class WsiDicomImageData(ImageData, metaclass=ABCMeta):
    def __init__(self, datasets: Sequence[WsiDataset]):
        self._datasets = datasets

    @abstractmethod
    def _get_tile_frame(self, frame_index: int) -> bytes:
        """Return tile frame for frame index.

        Parameters
        ----------
        frame_index: int
             Frame index to get

        Returns
        ----------
        bytes
            The frame in bytes
        """
        raise NotImplementedError()

    @cached_property
    def tiles(self) -> TileIndex:
        if self._datasets[0].tile_type == TileType.FULL:
            return FullTileIndex(self._datasets)
        else:
            return SparseTileIndex(self._datasets)

    @property
    def image_size(self) -> Size:
        """The pixel size of the image."""
        return self.tiles.image_size

    @property
    def tile_size(self) -> Size:
        """The pixel tile size of the image."""
        return self.tiles.tile_size

    @property
    def focal_planes(self) -> List[float]:
        """Focal planes avaiable in the image defined in um."""
        return self.tiles.focal_planes

    @property
    def optical_paths(self) -> List[str]:
        """Optical paths avaiable in the image."""
        return self.tiles.optical_paths

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        """Size of the pixels in mm/pixel."""
        return self._datasets[0].pixel_spacing

    @property
    def photometric_interpretation(self) -> str:
        """Return photometric interpretation."""
        return self._datasets[0].photometric_interpretation

    @property
    def samples_per_pixel(self) -> int:
        """Return samples per pixel (1 or 3)."""
        return self._datasets[0].samples_per_pixel

    @cached_property
    def image_origin(self) -> ImageOrigin:
        """Return the image origin of the image data."""
        return ImageOrigin.from_dataset(self._datasets[0])

    def _get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        """Return image bytes for tile defined by tile (x, y), z,
        and optical path."""
        frame_index = self._get_frame_index(tile, z, path)
        if frame_index == -1:
            return self.blank_encoded_tile
        return self._get_tile_frame(frame_index)

    def _get_decoded_tile(self, tile_point: Point, z: float, path: str) -> PILImage:
        """Return Pillow image for tile defined by tile (x, y), z,
        and optical path."""
        frame_index = self._get_frame_index(tile_point, z, path)
        if frame_index == -1:
            return self.blank_tile
        frame = self._get_tile_frame(frame_index)
        return Image.open(io.BytesIO(frame))

    def _get_frame_index(self, tile: Point, z: float, path: str) -> int:
        """Return frame index for tile. Raises WsiDicomOutOfBoundsError if
        tile, z, or path is not valid.

        Parameters
        ----------
        tile: Point
             Tile coordinate
        z: float
            Z coordinate
        path: str
            Optical identifier

        Returns
        ----------
        int
            Tile frame index
        """
        tile_region = Region(position=tile, size=Size(0, 0))
        if not self.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBoundsError(
                f"Tile region {tile_region}", f"plane {self.tiles.tiled_size}"
            )
        frame_index = self.tiles.get_frame_index(tile, z, path)
        return frame_index

    def is_sparse(self, tile: Point, z: float, path: str) -> bool:
        return self.tiles.get_frame_index(tile, z, path) == -1
