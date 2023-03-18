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
from pathlib import Path
from typing import List, Optional, OrderedDict, Sequence, Union

from PIL import Image
from PIL.Image import Image as PILImage
from pydicom.uid import UID

from wsidicom.dataset import TileType
from wsidicom.errors import WsiDicomNotFoundError, WsiDicomOutOfBoundsError
from wsidicom.wsidicom_file.wsidicom_file import WsiDicomFile
from wsidicom.geometry import Point, Region, Size, SizeMm
from wsidicom.image_data import ImageData, ImageOrigin
from wsidicom.tile_index import FullTileIndex, SparseTileIndex


class WsiDicomFileImageData(ImageData):
    """Represents image data read from dicom file(s). Image data can
    be sparsly or fully tiled and/or concatenated."""

    def __init__(self, files: Union[WsiDicomFile, Sequence[WsiDicomFile]]) -> None:
        """Create WsiDicomFileImageData from frame data in files.

        Parameters
        ----------
        files: Union[WsiDicomFile, Sequence[WsiDicomFile]]
            Single or list of WsiDicomFiles containing frame data.
        """
        if not isinstance(files, Sequence):
            files = [files]

        # Key is frame offset
        self._files = OrderedDict(
            (file.frame_offset, file)
            for file in sorted(files, key=lambda file: file.frame_offset)
        )

        base_file = files[0]
        datasets = [file.dataset for file in self._files.values()]
        if base_file.dataset.tile_type == TileType.FULL:
            self.tiles = FullTileIndex(datasets)
        else:
            self.tiles = SparseTileIndex(datasets)

        self._pixel_spacing = datasets[0].pixel_spacing
        self._transfer_syntax = base_file.transfer_syntax
        self._default_z: Optional[float] = None
        self._photometric_interpretation = datasets[0].photometric_interpretation
        self._samples_per_pixel = datasets[0].samples_per_pixel
        self._image_origin = ImageOrigin.from_dataset(datasets[0])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._files.values()})"

    def __str__(self) -> str:
        return f"{type(self).__name__} of files {self._files.values()}"

    @property
    def files(self) -> List[Path]:
        return [file.filepath for file in self._files.values()]

    @property
    def transfer_syntax(self) -> UID:
        """The uid of the transfer syntax of the image."""
        return self._transfer_syntax

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
        return self._pixel_spacing

    @property
    def photometric_interpretation(self) -> str:
        """Return photometric interpretation."""
        return self._photometric_interpretation

    @property
    def samples_per_pixel(self) -> int:
        """Return samples per pixel (1 or 3)."""
        return self._samples_per_pixel

    @property
    def image_origin(self) -> ImageOrigin:
        return self._image_origin

    def _get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        frame_index = self._get_frame_index(tile, z, path)
        if frame_index == -1:
            return self.blank_encoded_tile
        return self._get_tile_frame(frame_index)

    def _get_decoded_tile(self, tile_point: Point, z: float, path: str) -> PILImage:
        frame_index = self._get_frame_index(tile_point, z, path)
        if frame_index == -1:
            return self.blank_tile
        frame = self._get_tile_frame(frame_index)
        return Image.open(io.BytesIO(frame))

    def _get_file(self, frame_index: int) -> WsiDicomFile:
        """Return file contaning frame index. Raises WsiDicomNotFoundError if
        frame is not found.

        Parameters
        ----------
        frame_index: int
             Frame index to get

        Returns
        ----------
        WsiDicomFile
            File containing the frame
        """
        for frame_offset, file in self._files.items():
            if (
                frame_index < frame_offset + file.frame_count
                and frame_index >= frame_offset
            ):
                return file

        raise WsiDicomNotFoundError(f"Frame index {frame_index}", "instance")

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
        file = self._get_file(frame_index)
        tile_frame = file.read_frame(frame_index)
        return tile_frame

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

    def close(self) -> None:
        for file in self._files.values():
            file.close()
