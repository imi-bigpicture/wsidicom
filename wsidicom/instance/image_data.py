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

"""Module implementing generic ImageData for accessing image data."""

from abc import ABCMeta, abstractmethod
from typing import (
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from PIL import Image as Pillow
from PIL.Image import Image
from pydicom.uid import UID

from wsidicom.codec import Encoder
from wsidicom.errors import WsiDicomOutOfBoundsError
from wsidicom.geometry import Point, Region, Size, SizeMm
from wsidicom.instance.image_coordinate_system import ImageCoordinateSystem
from wsidicom.thread import ConditionalThreadPoolExecutor
from wsidicom.config import settings


class ImageData(metaclass=ABCMeta):
    """
    Generic class for image data.

    Can be inherited to implement support for other image/file formats. Subclasses
    should implement properties to get transfer_syntax, image_size, tile_size,
    pixel_spacing,  samples_per_pixel, and photometric_interpretation and methods
    _get_decoded_tile(), _get_encoded_tile(), and close(). Optionally the methods
    _get_decoded_tiles() and _get_encoded_tiles() can be overridden for more efficient
    fetching of multiple tiles.

    Additionally properties focal_planes and/or optical_paths should be
    overridden if multiple focal planes or optical paths are implemented.
    """

    _default_z: Optional[float] = None
    _blank_tile: Optional[Image] = None
    _encoded_blank_tile: Optional[bytes] = None

    def __init__(self, encoder: Encoder):
        self._encoder = encoder

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
    def pixel_spacing(self) -> Optional[SizeMm]:
        """Return the size of the pixels in mm/pixel."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def samples_per_pixel(self) -> int:
        """Return number of samples per pixel (e.g. 3 for RGB)."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def photometric_interpretation(self) -> str:
        """Return the photophotometric interpretation of the image data."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def bits(self) -> int:
        """Should return the number of bits stored for each sample."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def lossy_compressed(self) -> bool:
        """Should return True if the image has been lossy compressed."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def image_coordinate_system(self) -> Optional[ImageCoordinateSystem]:
        """Should return the image origin of the image data."""
        raise NotImplementedError()

    @abstractmethod
    def _get_decoded_tile(self, tile_point: Point, z: float, path: str) -> Image:
        """
        Return Pillow image for tile.

        Parameters
        ----------
        tile: Point
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        ----------
        Image
            Tile as Pillow Image.
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        """
        Return bytes for tile.

        Parameters
        ----------
        tile: Point
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        ----------
        bytes
            Tile as bytes.
        """
        raise NotImplementedError()

    @property
    def tiled_size(self) -> Size:
        """
        The size (columns and rows of tiles) of the image when divided into tiles.

        Equals (1, 1) if image is not tiled.
        """
        return self.image_size.ceil_div(self.tile_size)

    @property
    def image_region(self) -> Region:
        """The pixel region the image covers."""
        return Region(Point(0, 0), self.image_size)

    @property
    def focal_planes(self) -> List[float]:
        """Focal planes available in the image defined in um."""
        return [0.0]

    @property
    def optical_paths(self) -> List[str]:
        """Optical paths available in the image."""
        return ["0"]

    @property
    def image_mode(self) -> str:
        """Return Pillow image mode (e.g. RGB) for image data."""
        if self.samples_per_pixel == 1:
            if self.bits == 8:
                return "L"
            elif self.bits == 16:
                return "I"
        elif self.samples_per_pixel == 3:
            return "RGB"
        raise NotImplementedError()

    @property
    def blank_color(self) -> Union[int, Tuple[int, int, int]]:
        """Return grey level or RGB background color."""
        return self._get_blank_color(self.photometric_interpretation)

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
        """Return pretty string of object."""
        return str(self)

    @property
    def default_z(self) -> float:
        """
        The focal plane to use as default.

        Return single defined focal plane (in um) if only one focal plane defined.
        Return the middle focal plane if several focal planes are defined.
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

    @property
    def plane_region(self) -> Region:
        return Region(position=Point(0, 0), size=self.tiled_size)

    @property
    def blank_tile(self) -> Image:
        """Return background tile."""
        if self._blank_tile is None:
            self._blank_tile = self._create_blank_tile()
        return self._blank_tile

    @property
    def blank_encoded_tile(self) -> bytes:
        """Return encoded background tile."""
        if self._encoded_blank_tile is None:
            self._encoded_blank_tile = self.encoder.encode(self.blank_tile)
        return self._encoded_blank_tile

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    def get_decoded_tiles(
        self, tiles: Iterable[Point], z: float, path: str
    ) -> Iterator[Image]:
        """
        Return Pillow images for tiles.

        Parameters
        ----------
        tiles: Iterable[Point]
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        ----------
        Iterator[Image]
            Tiles as Images.
        """
        return self._get_decoded_tiles(tiles, z, path)

    def get_encoded_tiles(
        self,
        tiles: Iterable[Point],
        z: float,
        path: str,
        scale: int = 1,
        reencoder: Optional[Encoder] = None,
    ) -> Iterator[bytes]:
        """
        Return bytes for tiles.

        Parameters
        ----------
        tiles: Iterable[Point]
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.
        scale: int = 1
            Scale to use for downscaling.
        reencoder: Optional[Encoder] = None
            Encoder to use for re-encoding. If None do not re-encode tiles.

        Returns
        ----------
        Iterator[Image]
            Tiles as Images.
        """
        if reencoder is not None:
            return self._get_reencoded_tiles(tiles, z, path, scale, reencoder)
        if scale == 1:
            return self._get_encoded_tiles(tiles, z, path)
        return (self.get_scaled_encoded_tile(tile, z, path, scale) for tile in tiles)

    def get_scaled_tile(
        self,
        scaled_tile_point: Point,
        z: float,
        path: str,
        scale: int,
        workers: int = 1,
    ) -> Image:
        """
        Return scaled tile as Pillow image.

        Parameters
        ----------
        scaled_tile_point: Point,
            Scaled position of tile to get.
        z: float
            Z coordinate.
        path: str
            Optical path.
        scale: int
            Scale to use for downscaling.
        workers: int = 1
            Workers to use for threading. Default to not use threading as method is
            likely already used in a threading context.

        Returns
        ----------
        Image
            Scaled tiled as Pillow image.
        """
        image = Pillow.new(
            mode=self.image_mode,
            size=(self.tile_size * scale).to_tuple(),
            color=self.blank_color,
        )
        # Get decoded tiles for the region covering the scaled tile
        # in the image data
        tile_region = Region(scaled_tile_point * scale, Size(1, 1) * scale)
        tile_region = tile_region.crop(self.tiled_size)

        def paste(image: Image, tile_point: Point, tile: Image):
            image_coordinate = (tile_point - tile_region.start) * self.tile_size
            image.paste(tile, image_coordinate.to_tuple())

        self._paste_tiles(
            image,
            tile_region,
            z,
            path,
            paste,
            workers,
        )

        return image.resize(
            self.tile_size.to_tuple(), resample=settings.pillow_resampling_filter
        )

    def get_tile(self, tile_point: Point, z: float, path: str) -> Image:
        """
        Get tile as Pillow image.

        If frame is inside tile geometry but no tile exists in frame data (sparse)
        returns blank image. Crops tile to be inside image boundary.

        Parameters
        ----------
        tile_point: Point
            Tile x, y coordinate.
        z: float
            Z coordinate.
        path: str
            Optical path.


        Returns
        ----------
        Image
            Tile image.
        """
        tile = self._get_decoded_tile(tile_point, z, path)
        return self._crop_tile(tile_point, tile)

    def get_tiles(self, tiles: Iterable[Point], z: float, path: str) -> Iterator[Image]:
        return (
            self._crop_tile(tile_point, tile)
            for tile_point, tile in zip(tiles, self.get_decoded_tiles(tiles, z, path))
        )

    def get_scaled_encoded_tile(
        self, scaled_tile_point: Point, z: float, path: str, scale: int
    ) -> bytes:
        """
        Return scaled tile as bytes.

        Parameters
        ----------
        scaled_tile_point: Point,
            Scaled position of tile to get.
        z: float
            Z coordinate.
        path: str
            Optical path.
        Scale: int
            Scale to use for downscaling.

        Returns
        ----------
        bytes
            Scaled tile as bytes.
        """
        image = self.get_scaled_tile(scaled_tile_point, z, path, scale)
        return self.encoder.encode(image)

    def get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        """
        Get tile as bytes.

        If frame is inside tile geometry but no tile exists in frame data (sparse)
        returns encoded blank image. Crops tile to be inside image boundary.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        ----------
        bytes
            Tile image as bytes.
        """
        # Check if tile is an edge tile that should be cropped
        cropped_tile_region = self.image_region.inside_crop(tile, self.tile_size)
        if cropped_tile_region.size == self.tile_size:
            return self._get_encoded_tile(tile, z, path)
        image = self._get_decoded_tile(tile, z, path)
        image.crop(box=cropped_tile_region.box_from_origin)
        return self.encoder.encode(image)

    def stitch_tiles(self, region: Region, path: str, z: float, threads: int) -> Image:
        """Stitches tiles together to form requested image.

        Parameters
        ----------
        region: Region
             Pixel region to stitch to image
        path: str
            Optical path
        z: float
            Z coordinate
        threads: int
            Number of threads to use for read.

        Returns
        ----------
        Image
            Stitched image
        """

        tile_region = self._get_tile_range(region, z, path)
        if tile_region.size.area == 1:
            tile = self._get_decoded_tile(tile_region.start, z, path)
            tile_crop = region.inside_crop(tile_region.start, self.tile_size)
            if tile_crop.size != self.tile_size:
                tile = tile.crop(box=tile_crop.box)
            return tile
        image = Pillow.new(mode=self.image_mode, size=region.size.to_tuple())
        # The tiles are cropped prior to pasting. This offset is the equal to the first
        # (upper left) tiles size, and is added to the image coordinate for tiles not
        # in the first row or column.
        offset = (tile_region.start * self.tile_size) - region.start

        def paste(image: Image, tile_point: Point, tile: Image):
            image_coordinate = Point(
                offset.x * (tile_point.x != tile_region.start.x),
                offset.y * (tile_point.y != tile_region.start.y),
            )
            image_coordinate += (tile_point - tile_region.start) * self.tile_size
            tile_crop = region.inside_crop(tile_point, self.tile_size)
            if tile_crop.size != self.tile_size:
                tile = tile.crop(box=tile_crop.box)
            image.paste(tile, image_coordinate.to_tuple())

        self._paste_tiles(
            image,
            tile_region,
            z,
            path,
            paste,
            threads,
        )
        return image

    def valid_tiles(self, region: Region, z: float, path: str) -> bool:
        """
        Check if tile region is inside image and z coordinate and optical path exists.

        Parameters
        ----------
        region: Region
            Tile region.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        ----------
        bool
            True if tile region is inside image and z coordinate and optical path
            exists.
        """
        return (
            region.is_inside(self.plane_region)
            and (z in self.focal_planes)
            and (path in self.optical_paths)
        )

    def _get_decoded_tiles(
        self, tiles: Iterable[Point], z: float, path: str
    ) -> Iterator[Image]:
        """
        Return Pillow images for tiles. Implementations can override this with a more
        efficent method for getting multiple tiles.

        Parameters
        ----------
        tiles: Iterable[Point]
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        ----------
        Iterator[Image]
            Iterator of tiles as Images.
        """
        return (self._get_decoded_tile(tile, z, path) for tile in tiles)

    def _get_encoded_tiles(
        self, tiles: Iterable[Point], z: float, path: str
    ) -> Iterator[bytes]:
        """
        Return bytes for tiles. Implementations can override this with a more efficent
        method for getting multiple tiles.

        Parameters
        ----------
        tiles: Iterable[Point]
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.

        Returns
        ----------
        Iterator[bytes]
            Iterator of tiles as bytes.
        """
        return (self._get_encoded_tile(tile, z, path) for tile in tiles)

    def _crop_tile(self, tile_point: Point, tile: Image) -> Image:
        tile_crop = self.image_region.inside_crop(tile_point, self.tile_size)
        # Check if tile is an edge tile that should be cropped
        if tile_crop.size != self.tile_size:
            return tile.crop(box=tile_crop.box)
        return tile

    def _paste_tiles(
        self,
        image: Image,
        tile_region: Region,
        z: float,
        path: str,
        paste_method: Callable[[Image, Point, Image], None],
        threads: int,
    ):
        """
        Paste tiles in region using method.

        Use threading if number of tiles to paste
        is larger than one and requested worker count is more than one.

        Parameters
        ----------
        image: Image
            Image to paste into.
        tile_region: Region
            Tile region of tiles to paste.
        z: float
            Z coordinate.
        path: str
            Optical path.
        paste_method: Callable[[Image, Point, Image], None]
            Method that accepts a image, a tile point and tile to paste and returns None.
        threads: int
            Number of workers to use.
        """

        def thread_paste(tile_points: Iterable[Point]) -> None:
            tile_points = list(tile_points)
            for tile_point, tile in zip(
                tile_points, self.get_decoded_tiles(tile_points, z, path)
            ):
                paste_method(image, tile_point, tile)

        with ConditionalThreadPoolExecutor(
            max_workers=threads, force_iteration=True
        ) as pool:
            pool.map(thread_paste, tile_region.chunked_iterate_all(threads))

    def _get_reencoded_tiles(
        self,
        tiles: Iterable[Point],
        z: float,
        path: str,
        scale: int,
        reencoder: Encoder,
    ) -> Iterator[bytes]:
        """
        Return re-encoded bytes for tiles.

        Parameters
        ----------
        tiles: Iterable[Point]
            Tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.
        scale: int
            Scale to use for downscaling.
        reencoder: Encoder
            Encoder to use for re-encoding.

        """
        if scale == 1:
            decoded_tiles = self._get_decoded_tiles(tiles, z, path)
        else:
            decoded_tiles = (
                self.get_scaled_tile(tile, z, path, scale) for tile in tiles
            )
        return (reencoder.encode(tile) for tile in decoded_tiles)

    @staticmethod
    def _get_blank_color(
        photometric_interpretation: str,
    ) -> Union[int, Tuple[int, int, int]]:
        """Return color to use blank tiles.

        Parameters
        ----------
        photometric_interpretation: str
            The photomoetric interpretation of the dataset

        Returns
        ----------
        Union[int, Tuple[int, int, int]]
            Grey level or RGB color,

        """
        BLACK = 0
        WHITE = 255
        if photometric_interpretation == "MONOCHROME2":
            return BLACK
        return (WHITE, WHITE, WHITE)

    def _create_blank_tile(self) -> Image:
        """Create blank tile for instance.

        Returns
        ----------
        Image
            Blank tile image
        """
        return Pillow.new(
            mode=self.image_mode,
            size=self.tile_size.to_tuple(),
            color=self.blank_color,
        )

    def _get_tile_range(self, pixel_region: Region, z: float, path: str) -> Region:
        """Return range of tiles to cover pixel region.

        Parameters
        ----------
        pixel_region: Region
            Pixel region of tiles to get
        z: float
            Z coordinate of tiles to get
        path: str
            Optical path identifier of tiles to get

        Returns
        ----------
        Region
            Region of tiles for stitching image
        """

        tile_region = Region.from_points(
            pixel_region.start // self.tile_size,
            (pixel_region.end - 1) // self.tile_size + 1,
        )
        if not self.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBoundsError(
                f"Tile region {tile_region}", f"tiled size {self.tiled_size}"
            )
        return tile_region
