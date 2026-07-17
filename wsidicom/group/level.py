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

import math
from collections.abc import Iterable

import numpy as np

from wsidicom.config import settings
from wsidicom.errors import (
    WsiDicomNoResolutionError,
    WsiDicomOutOfBoundsError,
)
from wsidicom.geometry import Point, Region, Size, SizeMm
from wsidicom.group.group import Instances
from wsidicom.instance import WsiInstance
from wsidicom.stringprinting import dict_pretty_str
from wsidicom.thread import ReadExecutor


class Level(Instances):
    """Represents a level in the pyramid and contains one or more instances
    having the same pyramid level index, pixel spacing, and size but possibly
    different focal planes and/or optical paths.
    """

    def __init__(self, instances: Iterable[WsiInstance], base_pixel_spacing: SizeMm):
        """Create a level from list of WsiInstances. Assign the pyramid level
        index from pixel spacing of base level.

        Parameters
        ----------
        instances: Sequence[WsiInstance]
            Instances to build the level.
        base_pixel_spacing: SizeMm
            Pixel spacing of base level.
        """
        super().__init__(instances)
        self._base_pixel_spacing = base_pixel_spacing
        self._level = self._assign_level(self._base_pixel_spacing)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.instances}, {self._base_pixel_spacing})"

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(self, indent: int = 0, depth: int | None = None) -> str:
        string = f"Level: {self.level}, size: {self.size} px, mpp: {self.mpp} um/px"
        if depth is not None:
            depth -= 1
            if depth < 0:
                return string
        string += " Instances: " + dict_pretty_str(self.instances, indent, depth)
        return string

    @property
    def pyramid(self) -> str:
        """Return string representation of the level"""
        return (
            f"Level [{self.level}]"
            f" tiles: {self.default_instance.tiled_size},"
            f" size: {self.size}, mpp: {self.mpp} um/px"
        )

    @property
    def tile_size(self) -> Size:
        return self.default_instance.tile_size

    @property
    def level(self) -> int:
        """Return pyramid level"""
        return self._level

    @property
    def mpp(self) -> SizeMm:
        if self.pixel_spacing is None:
            raise WsiDicomNoResolutionError()
        return self.pixel_spacing * 1000.0

    @property
    def pixel_spacing(self) -> SizeMm:
        if self._pixel_spacing is None:
            raise WsiDicomNoResolutionError()
        return self._pixel_spacing

    def matches(self, other_group: "Instances") -> bool:
        """Check if level matches other level. If strict common Uids should
        match. Wsi type and tile size should always match.

        Parameters
        ----------
        other_group: Instances
            Other level to match against.

        Returns
        -------
        bool
            True if other level matches.
        """
        return super().matches(other_group) and (
            not settings.strict_tile_size_check
            or (
                isinstance(other_group, Level)
                and other_group.tile_size == self.tile_size
            )
        )

    def get_highest_level(self) -> int:
        """Return lowest level that produces a single pixel sized image.

        Returns
        -------
        int
            Relative level where the pixel size becomes 1x1.
        """
        return math.ceil(math.log2(max(self.size.width, self.size.height)))

    def get_lowest_single_tile_level(self) -> int:
        """Return lowest level that produces a single tile sized image.

        Returns
        -------
        int
            Relative level where the pixel size becomes a single tile size.
        """
        return math.ceil(
            math.log2(
                max(
                    self.size.width / self.tile_size.width,
                    self.size.height / self.tile_size.height,
                )
            )
        )

    def get_scaled_tile(
        self,
        tile: Point,
        level: int,
        z: float | None = None,
        path: str | None = None,
        crop_to_image_boundary: bool = True,
        *,
        executor: ReadExecutor,
    ) -> np.ndarray:
        """Return tile in another level by scaling a region.

        Parameters
        ----------
        tile: Point
            Non scaled tile coordinate
        level: int
            Level to scale from
        z: float | None = None
            Z coordinate
        path: str | None = None
            Optical path
        crop_to_image_boundary: bool = True
            If True, edge tiles are cropped to remove the part outside the image
            (as defined by the level size). If False, the cropped tile is padded
            back to the full tile size with the background color.
        executor: ReadExecutor
            Executor that splits image data reads across worker threads.

        Returns
        -------
        np.ndarray
            Tile pixels.
        """
        scale = self.calculate_scale(level)
        instance = self.get_instance(z, path)
        scaled_region = Region.from_tile(tile, instance.tile_size) * scale
        cropped_region = scaled_region.crop(instance.image_data.image_size)
        if not self.valid_pixels(cropped_region):
            raise WsiDicomOutOfBoundsError(
                f"Region {cropped_region}", f"level size {self.size}"
            )
        array = self.get_region(
            cropped_region,
            z,
            path,
            output_size=cropped_region.size.ceil_div(scale),
            executor=executor,
        )
        if crop_to_image_boundary:
            return array
        # Pad the cropped edge tile back to full tile size with the background.
        # Blank tile is cached and shared, so copy before writing.
        image_data = instance.image_data
        canvas = image_data.blank_tile.copy()
        canvas[: array.shape[0], : array.shape[1]] = array
        return canvas

    def get_scaled_encoded_tile(
        self,
        tile: Point,
        level: int,
        z: float | None = None,
        path: str | None = None,
        crop_to_image_boundary: bool = True,
        *,
        executor: ReadExecutor,
    ) -> bytes:
        """Return encoded tile in another level by scaling a region.

        Parameters
        ----------
        tile: Point
            Non scaled tile coordinate
        level: int
           Level to scale from
        z: float | None = None
            Z coordinate
        path: str | None = None
            Optical path
        crop_to_image_boundary: bool = True
            If True, edge tiles are cropped to remove the part outside the image.
            If False, the cropped tile is padded back to the full tile size with
            the background color.
        executor: ReadExecutor
            Executor that splits image data reads across worker threads.

        Returns
        -------
        bytes
            A transfer syntax encoded tile
        """
        image = self.get_scaled_tile(
            tile, level, z, path, crop_to_image_boundary, executor=executor
        )
        instance = self.get_instance(z, path)
        return instance.image_data.encoder.encode(image)

    def calculate_scale(self, level_to: int) -> int:
        """Return scaling factor to given level.

        Parameters
        ----------
        level_to -- index of level to scale to

        Returns
        -------
        int
            Scaling factor between this level and given level
        """
        return int(2 ** (level_to - self.level))

    def _assign_level(self, base_pixel_spacing: SizeMm) -> int:
        """Return (2^level scale factor) based on pixel spacing.
        Will round to closest integer. Raises NotImplementedError if level is
        to far from integer.

        Parameters
        ----------
        base_pixel_spacing: SizeMm
            The pixel spacing of the base level

        Returns
        -------
        int
            The pyramid order of the level
        """
        float_level = math.log2(self.pixel_spacing.width / base_pixel_spacing.width)
        level = int(round(float_level))
        tolerance = settings.level_scale_tolerance
        if not math.isclose(float_level, level, abs_tol=tolerance):
            raise NotImplementedError(
                f"Levels needs to be integer. Got {float_level} that is more than set"
                f"tolerance {tolerance} from the closest integer {level}. "
                f"Base spacing is {base_pixel_spacing}, this level has spacing "
                f"{self.pixel_spacing}.",
            )
        return level


class BaseLevel(Level):
    def __init__(self, instances: Iterable[WsiInstance]):
        """Create a base level from list of WsiInstances.

        Parameters
        ----------
        instances: Sequence[WsiInstance]
            Instances to build the level.
        """
        pixel_spacing = next(iter(instances)).pixel_spacing
        if pixel_spacing is None:
            raise WsiDicomNoResolutionError()
        super().__init__(instances, pixel_spacing)
