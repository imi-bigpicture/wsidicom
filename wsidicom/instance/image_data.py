#    Copyright 2021, 2022, 2023, 2026 SECTRA AB
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


from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator
from functools import cached_property

import numpy as np
from pydicom.uid import UID

from wsidicom.codec import Encoder
from wsidicom.geometry import Point, Region, Size, SizeMm
from wsidicom.metadata import ImageCoordinateSystem, LossyCompression


class ImageData(metaclass=ABCMeta):
    """Raw, format-specific access to a single image's tiles and metadata.

    Implement this to add support for a new image or file format. Subclasses
    must provide the geometry/metadata properties and the single-tile
    primitives ``get_decoded_tile`` and ``get_encoded_tile``. Optional
    overrides: the batch variants (``get_decoded_tiles``, ``get_encoded_tiles``,
    ``get_encoded_and_decoded_tile``, ``get_encoded_and_decoded_tiles``) for
    more efficient bulk fetching; ``focal_planes`` / ``optical_paths`` when
    more than one is present.
    """

    _default_z: float | None = None

    def __init__(self, encoder: Encoder):
        self._encoder = encoder

    @property
    def encoder(self) -> Encoder:
        """Encoder for this image's tiles."""
        return self._encoder

    @property
    @abstractmethod
    def transfer_syntax(self) -> UID:
        """Return the uid of the transfer syntax of the image."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def image_size(self) -> Size:
        """Return the pixel size of the image."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def tile_size(self) -> Size:
        """Return the pixel size of a tile in the image, or image size if not tiled."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def pixel_spacing(self) -> SizeMm | None:
        """Return the size of the pixels in mm/pixel."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def imaged_size(self) -> SizeMm | None:
        """Return the imaged width and height of the slide in mm."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def samples_per_pixel(self) -> int:
        """Return number of samples per pixel (e.g. 3 for RGB)."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def photometric_interpretation(self) -> str:
        """Return the photometric interpretation of the image data."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def bits(self) -> int:
        """Should return the number of bits stored for each sample."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def image_coordinate_system(self) -> ImageCoordinateSystem | None:
        """Should return the image origin of the image data."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def thread_safe(self) -> bool:
        """Should return True if the image data can be accessed by multiple threads."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def lossy_compression(self) -> list[LossyCompression] | None:
        """Should return None if the image has never been lossy compressed, otherwise a
        list of the lossy compression method and ratio (30.0 means 30:1 compression)
        or an empty list if compression method is not in DICOM standard or ratio is
        unknown."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def transcoder(self) -> Encoder | None:
        """Return transcoder used for image data if image data can't read encoded data
        directly. Return None if no transcoder is needed."""
        raise NotImplementedError()

    @abstractmethod
    def get_decoded_tile(
        self,
        tile_point: Point,
        z: float,
        path: str,
        cache: bool = True,
    ) -> np.ndarray:
        """Return the pixels of a tile.

        The tile-read primitive. Format adapters decode to numpy; Pillow, when
        needed, is derived at the read boundary.

        Parameters
        ----------
        tile_point: Point
            Tile to get.
        z: float
            Z coordinate.
        path: str
            Optical path.
        cache: bool = True
            Whether to cache the decoded tile.

        Returns
        -------
        np.ndarray
            Tile as ``(rows, columns)`` or ``(rows, columns, samples)``.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        """Return bytes for tile.

        Parameters
        ----------
        tile: Point
            Tile to get.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        -------
        bytes
            Tile as bytes.
        """
        raise NotImplementedError()

    def get_decoded_tiles(
        self,
        tiles: Iterable[Point],
        z: float,
        path: str,
        cache: bool = True,
    ) -> Iterator[np.ndarray]:
        """Return the pixels for multiple tiles.

        Implementations can override this with a more efficient batch method.

        Parameters
        ----------
        tiles: Iterable[Point]
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.
        cache: bool = True
            Whether to cache the decoded tiles.

        Returns
        -------
        Iterator[np.ndarray]
            Tiles as pixels, in the order of ``tiles``.
        """
        return (self.get_decoded_tile(tile, z, path, cache) for tile in tiles)

    def get_encoded_tiles(
        self, tiles: Iterable[Point], z: float, path: str
    ) -> Iterator[bytes]:
        """Return bytes for multiple tiles.

        Implementations can override this with a more efficient method for
        getting multiple tiles.

        Parameters
        ----------
        tiles: Iterable[Point]
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        -------
        Iterator[bytes]
            Tiles as bytes, in the order of ``tiles``.
        """
        return (self.get_encoded_tile(tile, z, path) for tile in tiles)

    def get_encoded_and_decoded_tile(
        self, tile: Point, z: float, path: str
    ) -> tuple[bytes, np.ndarray]:
        """Return both the encoded bytes and the pixels for a tile.

        Default implementation calls ``get_encoded_tile`` and
        ``get_decoded_tile`` separately. Subclasses can override to read from
        the source only once.

        Parameters
        ----------
        tile: Point
            Tile to get.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        -------
        tuple[bytes, np.ndarray]
            The encoded bytes and the pixels of the tile.
        """
        return (
            self.get_encoded_tile(tile, z, path),
            self.get_decoded_tile(tile, z, path),
        )

    def get_encoded_and_decoded_tiles(
        self, tiles: Iterable[Point], z: float, path: str
    ) -> Iterator[tuple[bytes, np.ndarray]]:
        """Return both the encoded bytes and the pixels for multiple tiles.

        Default implementation calls ``get_encoded_and_decoded_tile`` per tile.
        Subclasses can override for batch-efficient reading.

        Parameters
        ----------
        tiles: Iterable[Point]
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        -------
        Iterator[tuple[bytes, np.ndarray]]
            For each tile, the encoded bytes and pixels, in the order of
            ``tiles``.
        """
        return (self.get_encoded_and_decoded_tile(tile, z, path) for tile in tiles)

    @property
    def focal_planes(self) -> list[float]:
        """Focal planes available in the image defined in um."""
        return [0.0]

    @property
    def optical_paths(self) -> list[str]:
        """Optical paths available in the image."""
        return ["0"]

    @property
    def suggested_minimum_chunk_size(self) -> int:
        """Suggested minimum number of tiles to read in a batch.

        Subclasses may override this to indicate that reading tiles in larger
        batches is more efficient.
        """
        return 1

    @property
    def tiled_size(self) -> Size:
        """The size (columns and rows of tiles) when divided into tiles.

        Equals (1, 1) if image is not tiled.
        """
        return self.image_size.ceil_div(self.tile_size)

    @property
    def image_region(self) -> Region:
        """The pixel region the image covers."""
        return Region(Point(0, 0), self.image_size)

    @property
    def plane_region(self) -> Region:
        """The tile region the image covers."""
        return Region(position=Point(0, 0), size=self.tiled_size)

    @property
    def blank_color(self) -> int | tuple[int, int, int]:
        """Return grey level or RGB background color."""
        return self._get_blank_color(self.photometric_interpretation)

    @property
    def dtype(self) -> np.dtype:
        """Return the numpy dtype of the image data: uint8 for up to 8 bits per
        sample, uint16 for up to 16."""
        if self.bits <= 8:
            return np.dtype(np.uint8)
        if self.bits <= 16:
            return np.dtype(np.uint16)
        raise ValueError(f"Unsupported bits per sample: {self.bits}.")

    @cached_property
    def blank_tile(self) -> np.ndarray:
        """Return the background tile pixels."""
        shape: tuple[int, ...] = (self.tile_size.height, self.tile_size.width)
        if self.samples_per_pixel != 1:
            shape = shape + (self.samples_per_pixel,)
        return np.full(shape, self.blank_color, self.dtype)

    @cached_property
    def blank_encoded_tile(self) -> bytes:
        """Return encoded background tile."""
        return self.encoder.encode(self.blank_tile)

    @property
    def default_z(self) -> float:
        """The focal plane to use as default.

        Return single defined focal plane (in um) if only one focal plane is
        defined. Return the middle focal plane if several are defined.
        """
        if self._default_z is None:
            default = 0
            if len(self.focal_planes) > 1:
                smallest = min(self.focal_planes)
                largest = max(self.focal_planes)
                middle = (largest - smallest) / 2
                default = min(
                    range(len(self.focal_planes)),
                    key=lambda i: abs(self.focal_planes[i] - middle),
                )
            self._default_z = self.focal_planes[default]
        return self._default_z

    @property
    def default_path(self) -> str:
        """Return the first defined optical path as default optical path identifier."""
        return self.optical_paths[0]

    def valid_tiles(self, region: Region, z: float, path: str) -> bool:
        """Check if tile region is inside image and z coordinate and optical path
        exists.

        Parameters
        ----------
        region: Region
            Tile region.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        -------
        bool
            True if tile region is inside image and z coordinate and optical path
            exists.
        """
        return (
            region.is_inside(self.plane_region)
            and (z in self.focal_planes)
            and (path in self.optical_paths)
        )

    @staticmethod
    def _get_blank_color(
        photometric_interpretation: str,
    ) -> int | tuple[int, int, int]:
        """Return color to use for blank tiles.

        Parameters
        ----------
        photometric_interpretation: str
            The photometric interpretation of the dataset.

        Returns
        -------
        int | tuple[int, int, int]
            Grey level or RGB color.
        """
        BLACK = 0
        WHITE = 255
        if photometric_interpretation == "MONOCHROME2":
            return BLACK
        return (WHITE, WHITE, WHITE)

    def pretty_str(self, indent: int = 0, depth: int | None = None) -> str:
        """Return pretty string of object."""
        return str(self)
