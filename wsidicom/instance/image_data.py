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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from PIL import Image
from PIL.Image import Image as PILImage
from pydicom.uid import JPEG2000, UID, JPEG2000Lossless, JPEGBaseline8Bit

from wsidicom.errors import WsiDicomOutOfBoundsError
from wsidicom.geometry import (
    Point,
    Region,
    Size,
    SizeMm,
)
from wsidicom.instance.image_origin import ImageOrigin


class ImageData(metaclass=ABCMeta):
    """Generic class for image data that can be inherited to implement support
    for other image/file formats. Subclasses should implement properties to get
    transfer_syntax, image_size, tile_size, pixel_spacing,  samples_per_pixel,
    and photometric_interpretation and methods get_tile() and close().
    Additionally properties focal_planes and/or optical_paths should be
    overridden if multiple focal planes or optical paths are implemented."""

    _default_z: Optional[float] = None
    _blank_tile: Optional[PILImage] = None
    _encoded_blank_tile: Optional[bytes] = None

    @property
    @abstractmethod
    def transfer_syntax(self) -> UID:
        """Should return the uid of the transfer syntax of the image."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def image_size(self) -> Size:
        """Should return the pixel size of the image."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def tile_size(self) -> Size:
        """Should return the pixel tile size of the image, or pixel size of
        the image if not tiled."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def pixel_spacing(self) -> Optional[SizeMm]:
        """Should return the size of the pixels in mm/pixel."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def samples_per_pixel(self) -> int:
        """Should return number of samples per pixel (e.g. 3 for RGB)."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def photometric_interpretation(self) -> str:
        """Should return the photophotometric interpretation of the image
        data."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def image_origin(self) -> ImageOrigin:
        """Should return the image origin of the image data."""
        raise NotImplementedError()

    @abstractmethod
    def _get_decoded_tile(self, tile_point: Point, z: float, path: str) -> PILImage:
        """Should return Image for tile defined by tile (x, y), z,
        and optical path."""
        raise NotImplementedError()

    @abstractmethod
    def _get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        """Should return image bytes for tile defined by tile (x, y), z,
        and optical path."""
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Should close any open files."""
        raise NotImplementedError()

    @property
    def tiled_size(self) -> Size:
        """The size of the image when divided into tiles, e.g. number of
        columns and rows of tiles. Equals (1, 1) if image is not tiled."""
        return self.image_size.ceil_div(self.tile_size)

    @property
    def image_region(self) -> Region:
        return Region(Point(0, 0), self.image_size)

    @property
    def focal_planes(self) -> List[float]:
        """Focal planes avaiable in the image defined in um."""
        return [0.0]

    @property
    def optical_paths(self) -> List[str]:
        """Optical paths avaiable in the image."""
        return ["0"]

    @property
    def image_mode(self) -> str:
        """Return Pillow image mode (e.g. RGB) for image data"""
        if self.samples_per_pixel == 1:
            return "L"
        elif self.samples_per_pixel == 3:
            return "RGB"
        raise NotImplementedError()

    @property
    def blank_color(self) -> Tuple[int, int, int]:
        """Return RGB background color."""
        return self._get_blank_color(self.photometric_interpretation)

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
        return str(self)

    @property
    def default_z(self) -> float:
        """Return single defined focal plane (in um) if only one focal plane
        defined. Return the middle focal plane if several focal planes are
        defined."""
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
        """Return the first defined optical path as default optical path
        identifier."""
        return self.optical_paths[0]

    @property
    def plane_region(self) -> Region:
        return Region(position=Point(0, 0), size=self.tiled_size - 1)

    @property
    def blank_tile(self) -> PILImage:
        """Return background tile."""
        if self._blank_tile is None:
            self._blank_tile = self._create_blank_tile()
        return self._blank_tile

    @property
    def blank_encoded_tile(self) -> bytes:
        """Return encoded background tile."""
        if self._encoded_blank_tile is None:
            self._encoded_blank_tile = self.encode(self.blank_tile)
        return self._encoded_blank_tile

    def get_decoded_tiles(
        self, tiles: Iterable[Point], z: float, path: str
    ) -> List[PILImage]:
        """Return tiles for tile defined by tile (x, y), z, and optical
        path.

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
        List[PILImage]
            Tiles as Images.
        """
        return [self._get_decoded_tile(tile, z, path) for tile in tiles]

    def get_encoded_tiles(
        self, tiles: Iterable[Point], z: float, path: str
    ) -> List[bytes]:
        """Return tiles for tile defined by tile (x, y), z, and optical
        path.

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
        List[bytes]
            Tiles in bytes.
        """
        return [self._get_encoded_tile(tile, z, path) for tile in tiles]

    def get_scaled_tile(
        self, scaled_tile_point: Point, z: float, path: str, scale: int
    ) -> PILImage:
        """Return scaled tile defined by tile (x, y), z, optical
        path and scale.

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
        PILImage
            Scaled tiled as Image.
        """
        image = Image.new(
            mode=self.image_mode,  # type: ignore
            size=(self.tile_size * scale).to_tuple(),
            color=self.blank_color[: self.samples_per_pixel],
        )
        # Get decoded tiles for the region covering the scaled tile
        # in the image data
        tile_points = Region(scaled_tile_point * scale, Size(1, 1) * scale)
        origin = tile_points.start
        for tile_point in tile_points.iterate_all():
            if (tile_point.x < self.tiled_size.width) and (
                tile_point.y < self.tiled_size.height
            ):
                tile = self._get_decoded_tile(tile_point, z, path)
                image_coordinate = (tile_point - origin) * self.tile_size
                image.paste(tile, image_coordinate.to_tuple())

        return image.resize(
            self.tile_size.to_tuple(), resample=Image.Resampling.BILINEAR
        )

    def get_scaled_encoded_tile(
        self,
        scaled_tile_point: Point,
        z: float,
        path: str,
        scale: int,
        image_format: str,
        image_options: Dict[str, Any],
    ) -> bytes:
        """Return scaled encoded tile defined by tile (x, y), z, optical
        path and scale.

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
        image_format: str
            Image format, e.g. 'JPEG', for encoding.
        image_options: Dict[str, Any].
            Dictionary of options for encoding.

        Returns
        ----------
        bytes
            Scaled tile as bytes.
        """
        image = self.get_scaled_tile(scaled_tile_point, z, path, scale)
        with io.BytesIO() as buffer:
            image.save(buffer, format=image_format, **image_options)
            return buffer.getvalue()

    def get_scaled_encoded_tiles(
        self,
        scaled_tile_points: Iterable[Point],
        z: float,
        path: str,
        scale: int,
        image_format: str,
        image_options: Dict[str, Any],
    ) -> List[bytes]:
        """Return scaled encoded tiles defined by tile (x, y) positions, z,
        optical path and scale.

        Parameters
        ----------
        scaled_tile_points: Iterable[Point],
            Scaled position of tiles to get.
        z: float
            Z coordinate.
        path: str
            Optical path.
        Scale: int
            Scale to use for downscaling.
        image_format: str
            Image format, e.g. 'JPEG', for encoding.
        image_options: Dict[str, Any].
            Dictionary of options for encoding.

        Returns
        ----------
        List[bytes]
            Scaled tiles as bytes.
        """
        return [
            self.get_scaled_encoded_tile(
                scaled_tile_point, z, path, scale, image_format, image_options
            )
            for scaled_tile_point in scaled_tile_points
        ]

    def valid_tiles(self, region: Region, z: float, path: str) -> bool:
        """Check if tile region is inside tile geometry and z coordinate and
        optical path exists.

        Parameters
        ----------
        region: Region
            Tile region.
        z: float
            Z coordinate.
        path: str
            Optical path.
        """
        return (
            region.is_inside(self.plane_region)
            and (z in self.focal_planes)
            and (path in self.optical_paths)
        )

    def encode(self, image: PILImage) -> bytes:
        """Encode image using transfer syntax.

        Parameters
        ----------
        image: PILImage
            Image to encode

        Returns
        ----------
        bytes
            Encoded image as bytes

        """
        image_format, image_options = self._image_settings(self.transfer_syntax)
        with io.BytesIO() as buffer:
            image.save(buffer, format=image_format, **image_options)
            return buffer.getvalue()

    @staticmethod
    def _image_settings(transfer_syntax: UID) -> Tuple[str, Dict[str, Any]]:
        """Return image format and options for creating encoded tiles as in the
        used transfer syntax.

        Parameters
        ----------
        transfer_syntax: pydicom.uid
            Transfer syntax to match image format and options to

        Returns
        ----------
        tuple[str, dict[str, int]]
            image format and image options

        """
        if transfer_syntax == JPEGBaseline8Bit:
            image_format = "jpeg"
            image_options = {"quality": 95}
        elif transfer_syntax == JPEG2000:
            image_format = "jpeg2000"
            image_options = {"irreversible": True}
        elif transfer_syntax == JPEG2000Lossless:
            image_format = "jpeg2000"
            image_options = {"irreversible": False}
        else:
            raise NotImplementedError("Only supports jpeg and jpeg2000")
        return (image_format, image_options)

    @staticmethod
    def _get_blank_color(photometric_interpretation: str) -> Tuple[int, int, int]:
        """Return color to use blank tiles.

        Parameters
        ----------
        photometric_interpretation: str
            The photomoetric interpretation of the dataset

        Returns
        ----------
        Tuple[int, int, int]
            RGB color,

        """
        BLACK = 0
        WHITE = 255
        if photometric_interpretation == "MONOCHROME2":
            return (BLACK, BLACK, BLACK)  # Monocrhome2 is black
        return (WHITE, WHITE, WHITE)

    def _create_blank_tile(self) -> PILImage:
        """Create blank tile for instance.

        Returns
        ----------
        PILImage
            Blank tile image
        """
        return Image.new(
            mode=self.image_mode,  # type: ignore
            size=self.tile_size.to_tuple(),
            color=self.blank_color[: self.samples_per_pixel],
        )

    def stitch_tiles(self, region: Region, path: str, z: float) -> PILImage:
        """Stitches tiles together to form requested image.

        Parameters
        ----------
        region: Region
             Pixel region to stitch to image
        path: str
            Optical path
        z: float
            Z coordinate

        Returns
        ----------
        PILImage
            Stitched image
        """

        image = Image.new(
            mode=self.image_mode, size=region.size.to_tuple()  # type: ignore
        )
        stitching_tiles = self._get_tile_range(region, z, path)

        write_index = Point(x=0, y=0)
        tile = stitching_tiles.position
        for tile in stitching_tiles.iterate_all(include_end=True):
            tile_image = self.get_tile(tile, z, path, region)
            image.paste(tile_image, write_index.to_tuple())
            write_index = self._write_indexer(
                write_index, Size.from_tuple(tile_image.size), region.size
            )
        return image

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
        start = pixel_region.start // self.tile_size
        end = pixel_region.end.ceil_div(self.tile_size) - 1
        tile_region = Region.from_points(start, end)
        if not self.valid_tiles(tile_region, z, path):
            raise WsiDicomOutOfBoundsError(
                f"Tile region {tile_region}", f"tiled size {self.tiled_size}"
            )
        return tile_region

    @staticmethod
    def _write_indexer(index: Point, previous_size: Size, image_size: Size) -> Point:
        """Increment index in x by previous width until index x exceds image
        size. Then resets index x to 0 and increments index y by previous
        height. Requires that tiles are scanned row by row.

        Parameters
        ----------
        index: Point
            The last write index position
        previouis_size: Size
            The size of the last written last tile
        image_size: Size
            The size of the image to be written

        Returns
        ----------
        Point
            The position (upper right) in image to insert the next tile into
        """
        index.x += previous_size.width
        if index.x >= image_size.width:
            index.x = 0
            index.y += previous_size.height
        return index

    def get_tile(
        self, tile: Point, z: float, path: str, crop: Union[bool, Region] = True
    ) -> PILImage:
        """Get tile image at tile coordinate x, y. If frame is inside tile
        geometry but no tile exists in frame data (sparse) returns blank image.
        Optional crop tile to crop_region.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate.
        z: float
            Z coordinate.
        path: str
            Optical path.
        crop: Union[bool, Region] = True
            If to crop tile to image size (True, default) or to region.

        Returns
        ----------
        PILImage
            Tile image.
        """
        image = self._get_decoded_tile(tile, z, path)
        if crop is False:
            return image

        if isinstance(crop, bool):
            crop = self.image_region
        tile_crop = crop.inside_crop(tile, self.tile_size)
        if tile_crop.size == self.tile_size:
            return image

        return image.crop(box=tile_crop.box)

    def get_encoded_tile(
        self, tile: Point, z: float, path: str, crop: Union[bool, Region] = True
    ) -> bytes:
        """Get tile bytes at tile coordinate x, y
        If frame is inside tile geometry but no tile exists in
        frame data (sparse) returns encoded blank image.

        Parameters
        ----------
        tile: Point
            Tile x, y coordinate.
        z: float
            Z coordinate.
        path: str
            Optical path.
        crop: Union[bool, Region] = True
            If to crop tile to image size (True, default) or to region.

        Returns
        ----------
        bytes
            Tile image as bytes.
        """
        tile_frame = self._get_encoded_tile(tile, z, path)
        if crop is False:
            return tile_frame

        if isinstance(crop, bool):
            crop = self.image_region
        # Check if tile is an edge tile that should be croped
        cropped_tile_region = crop.inside_crop(tile, self.tile_size)
        if cropped_tile_region.size != self.tile_size:
            image = Image.open(io.BytesIO(tile_frame))
            image.crop(box=cropped_tile_region.box_from_origin)
            tile_frame = self.encode(image)
        return tile_frame
