#    Copyright 2021, 2022 SECTRA AB
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

import logging
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Tuple, Union, Sequence


@dataclass
class SizeMm:
    width: float
    height: float

    def __mul__(self, factor: Union[int, float, "Size", "Point"]) -> "SizeMm":
        if isinstance(factor, (int, float)):
            return SizeMm(factor * self.width, factor * self.height)
        elif isinstance(factor, Size):
            return SizeMm(factor.width * self.width, factor.height * self.height)
        elif isinstance(factor, Point):
            return SizeMm(factor.x * self.width, factor.y * self.height)
        return NotImplemented

    def __truediv__(self, divider: Union[int, float, "SizeMm"]) -> "SizeMm":
        if isinstance(divider, (int, float)):
            return SizeMm(self.width / divider, self.height / divider)
        elif isinstance(divider, SizeMm):
            return SizeMm(self.width / divider.width, self.height / divider.height)
        return NotImplemented

    def __floordiv__(self, divider: Union[int, float, "SizeMm"]) -> "Size":
        if isinstance(divider, (int, float)):
            return Size(int(self.width / divider), int(self.height / divider))
        elif isinstance(divider, SizeMm):
            return Size(
                int(self.width / divider.width), int(self.height / divider.height)
            )
        return NotImplemented

    def to_int_tuple(self) -> Tuple[int, int]:
        return int(self.width), int(self.height)

    def to_tuple(self) -> Tuple[float, float]:
        return self.width, self.height

    @classmethod
    def from_tuple(cls, input: Union[Tuple[float, float], Sequence[float]]) -> "SizeMm":
        try:
            return cls(input[0], input[1])
        except IndexError:
            raise ValueError("input did not contain two values")


@dataclass
class PointMm:
    x: float
    y: float

    def __mul__(self, factor: Union[int, float]) -> "PointMm":
        if isinstance(factor, (int, float)):
            return PointMm(factor * self.x, factor * self.y)
        return NotImplemented

    def __truediv__(self, divider: Union[int, float, "PointMm", "SizeMm"]) -> "PointMm":
        if isinstance(divider, (int, float)):
            return PointMm(self.x / divider, self.y / divider)
        elif isinstance(divider, PointMm):
            return PointMm(self.x / divider.x, self.y / divider.y)
        elif isinstance(divider, SizeMm):
            return PointMm(self.x / divider.width, self.y / divider.height)
        return NotImplemented

    def __floordiv__(self, divider: Union[int, float, "PointMm", "SizeMm"]) -> "Point":
        if isinstance(divider, (int, float)):
            return Point(int(self.x // divider), int(self.y // divider))
        elif isinstance(divider, PointMm):
            return Point(int(self.x // divider.x), int(self.y // divider.y))
        elif isinstance(divider, SizeMm):
            return Point(int(self.x // divider.width), int(self.y // divider.height))
        return NotImplemented

    def __add__(self, value: Union[int, float, "PointMm", "SizeMm"]) -> "PointMm":
        if isinstance(value, (int, float)):
            return PointMm(self.x + value, self.y + value)
        elif isinstance(value, SizeMm):
            return PointMm(self.x + value.width, self.y + value.height)
        elif isinstance(value, PointMm):
            return PointMm(self.x + value.x, self.y + value.y)
        return NotImplemented

    def __sub__(self, value: Union[int, float, "PointMm", "SizeMm"]) -> "PointMm":
        if isinstance(value, (int, float)):
            return PointMm(self.x - value, self.y - value)
        elif isinstance(value, SizeMm):
            return PointMm(self.x - value.width, self.y - value.height)
        elif isinstance(value, PointMm):
            return PointMm(self.x - value.x, self.y - value.y)
        return NotImplemented

    def __neg__(self) -> "PointMm":
        return PointMm(-self.x, -self.y)

    @classmethod
    def from_tuple(
        cls, input: Union[Tuple[float, float], Sequence[float]]
    ) -> "PointMm":
        try:
            return cls(input[0], input[1])
        except IndexError:
            raise ValueError("input did not contain two values")

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Size:
    width: int
    height: int

    def __neg__(self) -> "Size":
        return Size(-self.width, -self.height)

    def __sub__(self, value: Union[int, "Size", "Point"]) -> "Size":
        if isinstance(value, int):
            return Size(self.width - value, self.height - value)
        elif isinstance(value, Size):
            return Size(self.width - value.width, self.height - value.height)
        elif isinstance(value, Point):
            return Size(self.width - value.x, self.height - value.y)
        return NotImplemented

    def __add__(self, value: Union[int, "Size", "Point"]) -> "Size":
        if isinstance(value, int):
            return Size(self.width + value, self.height + value)
        elif isinstance(value, Size):
            return Size(self.width + value.width, self.height + value.height)
        elif isinstance(value, Point):
            return Size(self.width + value.x, self.height + value.y)
        return NotImplemented

    def __mul__(self, factor: Union[int, float, "Size", "Point"]) -> "Size":
        if isinstance(factor, (int, float)):
            return Size(int(factor * self.width), int(factor * self.height))
        elif isinstance(factor, Size):
            return Size(factor.width * self.width, factor.height * self.height)
        elif isinstance(factor, Point):
            return Size(factor.x * self.width, factor.y * self.height)
        return NotImplemented

    def __floordiv__(self, divider: Union[int, "Size", SizeMm]) -> "Size":
        if isinstance(divider, (int, float)):
            return Size(int(self.width / divider), int(self.height / divider))
        elif isinstance(divider, (Size, SizeMm)):
            return Size(
                int(self.width / divider.width), int(self.height / divider.height)
            )
        return NotImplemented

    def ceil_div(self, divider: Union[int, "Size", SizeMm]) -> "Size":
        if isinstance(divider, (int, float)):
            return Size(
                math.ceil(self.width / divider), math.ceil(self.height / divider)
            )
        elif isinstance(divider, (Size, SizeMm)):
            return Size(
                math.ceil(self.width / divider.width),
                math.ceil(self.height / divider.height),
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.width, self.height))

    def all_less_than(self, item: "Size") -> bool:
        """If all dimensions is smaller than corresponding dimension in item."""
        return all(self._comparision_operation(item, int.__lt__))

    def all_less_than_or_equal(self, item: "Size") -> bool:
        """If all dimensions is smaller or equal to corresponding dimension in item."""
        return all(self._comparision_operation(item, int.__le__))

    def all_greater_than(self, item: "Size") -> bool:
        """If all dimensions is greater than to corresponding dimension in item."""
        return all(self._comparision_operation(item, int.__gt__))

    def all_greater_than_or_equal(self, item: "Size") -> bool:
        """If all dimensions is greater or equal to corresponding dimension in item."""
        return all(self._comparision_operation(item, int.__ge__))

    def any_less_than(self, item: "Size") -> bool:
        """If any dimension is smaller than corresponding dimension in item."""
        return any(self._comparision_operation(item, int.__lt__))

    def any_less_than_or_equal(self, item: "Size") -> bool:
        """If any dimension is smaller or equal to corresponding dimension in item."""
        return any(self._comparision_operation(item, int.__le__))

    def any_greater_than(self, item: "Size") -> bool:
        """If any dimension is greater than to corresponding dimension in item."""
        return any(self._comparision_operation(item, int.__gt__))

    def any_greater_than_or_equal(self, item: "Size") -> bool:
        """If any dimension is greater or equal to corresponding dimension in item."""
        return any(self._comparision_operation(item, int.__ge__))

    def _comparision_operation(
        self, item: "Size", operation: Callable[[int, int], bool]
    ) -> Iterable[bool]:
        return (
            operation(dimension, other_dimension)
            for (dimension, other_dimension) in zip(
                [self.width, self.height], [item.width, item.height]
            )
        )

    @classmethod
    def from_points(cls, point_1: "Point", point_2: "Point") -> "Size":
        return cls(point_2.x - point_1.x, point_2.y - point_1.y)

    def to_tuple(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @classmethod
    def from_tuple(cls, input: Union[Tuple[int, int], Sequence[int]]) -> "Size":
        try:
            return cls(input[0], input[1])
        except IndexError:
            raise ValueError("input did not contain two values")

    @classmethod
    def max(cls, size_1: "Size", size_2: "Size") -> "Size":
        return cls(
            width=max(size_1.width, size_2.width),
            height=max(size_1.height, size_2.height),
        )

    def ceil(self) -> "Size":
        return Size(
            width=int(math.ceil(self.width)), height=int(math.ceil(self.height))
        )

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Point:
    x: int
    y: int

    def __str__(self) -> str:
        return f"{self.x},{self.y}"

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __mul__(self, factor: Union[int, float, Size, "Point"]) -> "Point":
        if isinstance(factor, (int, float)):
            return Point(int(factor * self.x), int(factor * self.y))
        elif isinstance(factor, Size):
            return Point(factor.width * self.x, factor.height * self.y)
        elif isinstance(factor, Point):
            return Point(factor.x * self.x, factor.y * self.y)
        return NotImplemented

    def __floordiv__(
        self, divider: Union[int, float, "Point", Size, SizeMm]
    ) -> "Point":
        if isinstance(divider, (int, float)):
            return Point(int(self.x / divider), int(self.y / divider))
        elif isinstance(divider, Point):
            return Point(int(self.x / divider.x), int(self.y / divider.y))
        elif isinstance(divider, (Size, SizeMm)):
            return Point(int(self.x / divider.width), int(self.y / divider.height))
        return NotImplemented

    def ceil_div(self, divider: Union[int, float, Size, SizeMm]) -> "Point":
        if isinstance(divider, (int, float)):
            return Point(math.ceil(self.x / divider), math.ceil(self.y / divider))
        elif isinstance(divider, (Size, SizeMm)):
            return Point(
                math.ceil(self.x / divider.width), math.ceil(self.y / divider.height)
            )
        return NotImplemented

    def __mod__(self, divider: Union[Size, "Point"]) -> "Point":
        if isinstance(divider, Size):
            return Point(self.x % divider.width, self.y % divider.height)
        elif isinstance(divider, Point):
            return Point(self.x % divider.x, self.y % divider.y)
        return NotImplemented

    def __add__(self, value: Union[int, float, Size, "Point"]) -> "Point":
        if isinstance(value, (int, float)):
            return Point(int(self.x + value), int(self.y + value))
        elif isinstance(value, Size):
            return Point(self.x + value.width, self.y + value.height)
        elif isinstance(value, Point):
            return Point(self.x + value.x, self.y + value.y)
        return NotImplemented

    def __sub__(self, value: Union[int, float, Size, "Point"]) -> "Point":
        if isinstance(value, (int, float)):
            return Point(int(self.x - value), int(self.y - value))
        elif isinstance(value, Size):
            return Point(self.x - value.width, self.y - value.height)
        elif isinstance(value, Point):
            return Point(self.x - value.x, self.y - value.y)
        return NotImplemented

    @classmethod
    def max(cls, point_1: "Point", point_2: "Point") -> "Point":
        return cls(x=max(point_1.x, point_2.x), y=max(point_1.y, point_2.y))

    @classmethod
    def min(cls, point_1: "Point", point_2: "Point") -> "Point":
        return cls(x=min(point_1.x, point_2.x), y=min(point_1.y, point_2.y))

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, input: Union[Tuple[int, int], Sequence[int]]) -> "Point":
        try:
            return cls(input[0], input[1])
        except IndexError:
            raise ValueError("input did not contain two values")


@dataclass
class Region:
    position: Point
    size: Size

    @property
    def start(self) -> Point:
        return self.position

    @property
    def end(self) -> Point:
        end: Point = self.position + self.size
        return end

    def __mul__(self, value: int) -> "Region":
        if isinstance(value, int):
            return Region(position=self.position * value, size=self.size * value)
        return NotImplemented

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.start.x, self.start.y, self.end.x, self.end.y

    @property
    def box_from_origin(self) -> Tuple[int, int, int, int]:
        return 0, 0, self.size.width, self.size.height

    def iterate_all(self) -> Iterator[Point]:
        return (
            Point(x, y)
            for y in range(self.start.y, self.end.y)
            for x in range(self.start.x, self.end.x)
        )

    def chunked_iterate_all(self, chunks: int) -> Iterator[Iterator[Point]]:
        points = list(self.iterate_all())
        points_count = len(points)
        chunk_size = math.ceil(points_count / chunks)
        for chunk_index in range(chunks):
            chunk_start = chunk_index * chunk_size
            chunk_end = min((chunk_index + 1) * chunk_size, points_count)
            yield (point for point in points[chunk_start:chunk_end])

    @classmethod
    def from_points(cls, point_1: "Point", point_2: "Point") -> "Region":
        return cls(
            position=point_1,
            size=Size(width=point_2.x - point_1.x, height=point_2.y - point_1.y),
        )

    @classmethod
    def from_tile(cls, tile: "Point", size: "Size") -> "Region":
        return cls(position=tile * size, size=size)

    def is_inside(self, test_region: "Region") -> bool:
        return bool(
            (self.start.x >= test_region.start.x)
            and (self.start.y >= test_region.start.y)
            and (self.end.x <= test_region.end.x)
            and (self.end.y <= test_region.end.y)
        )

    def crop(self, other_region: Union[Size, "Region"]) -> "Region":
        """Crop other_region to be inside this region. If the other_region is
        'Size', a region starting from Point(0, 0) is created.

        Parameters
        ----------
        other_region: Union[Size, 'Region']
            Region to be cropped.

        Returns
        ----------
        Region
            Cropped region.
        """
        if isinstance(other_region, Size):
            other_region = Region(Point(0, 0), other_region)
        start = Point.min(Point.max(other_region.position, self.position), self.end)
        end = Point.max(Point.min(other_region.end, self.end), self.position)
        size = Size.from_points(start, end)
        return Region(position=start, size=size)

    def inside_crop(self, point: Point, tile_size: Size) -> "Region":
        """Crop point to be inside and with the same origin as this region.

        Parameters
        ----------
        point: Point
            Point to be cropped.

        Returns
        ----------
        Region
            Cropped region.
        """
        tile_region = Region(position=point * tile_size, size=tile_size)
        cropped_tile_region = self.crop(tile_region)
        cropped_tile_region.position = cropped_tile_region.position % tile_size
        return cropped_tile_region

    def zoom(self, zoom: float) -> "Region":
        """Return center-zoomed region.

        Parameters
        ----------
        zoom: float
            The zoom level.

        Returns
        ----------
        Region
            Zoomed region.
        """
        center = self.start + self.size // 2
        new_center = center * zoom
        new_start = new_center - self.size // 2
        return Region(new_start, self.size)


@dataclass
class RegionMm:
    position: PointMm
    size: SizeMm

    def __init__(self, position: PointMm, size: SizeMm):
        if size.width < 0:
            size.width = -size.width
            position.x -= size.width
        if size.height < 0:
            size.height = -size.height
            position.y -= size.height
        self.position = position
        self.size = size

    @property
    def start(self) -> PointMm:
        return self.position

    @property
    def end(self) -> PointMm:
        end = self.position + self.size
        return end

    def __add__(self, value: PointMm) -> "RegionMm":
        if isinstance(value, PointMm):
            return RegionMm(self.position + value, self.size)
        return NotImplemented

    def __sub__(self, value: PointMm) -> "RegionMm":
        if isinstance(value, PointMm):
            return RegionMm(self.position - value, self.size)
        return NotImplemented

    def zoom(self, zoom: float) -> "RegionMm":
        """Return center-zoomed region.

        Parameters
        ----------
        zoom: float
            The zoom level.

        Returns
        ----------
        RegionMm
            Zoomed region.
        """
        center = self.start + self.size / 2
        new_center = center * zoom
        new_start = new_center - self.size / 2
        return RegionMm(new_start, self.size)


class Orientation:
    def __init__(self, orientation: Tuple[float, float, float, float, float, float]):
        if orientation[0] != -orientation[4] or orientation[1] != orientation[3]:
            logging.warning(
                f"Orientation {orientation} is not "
                "orthogonal with equal lengths with column rotated 90 deg from row"
            )
        self._orientation = orientation
        self._transform = self._create_transform(orientation)
        self._reverse = self._create_reverse(self._transform)

    @property
    def rotation(self) -> float:
        return (
            (math.atan2(-self._orientation[0], self._orientation[3]) / math.pi)
            * 180
            % 360
        )

    @property
    def values(self) -> Tuple[float, float, float, float, float, float]:
        return self._orientation

    def apply_transform(self, point: PointMm):
        return PointMm(
            self._transform[0][0] * point.x + self._transform[0][1] * point.y,
            self._transform[1][0] * point.x + self._transform[1][1] * point.y,
        )

    def apply_reverse_transform(self, point: PointMm):
        return PointMm(
            self._reverse[0][0] * point.x + self._reverse[0][1] * point.y,
            self._reverse[1][0] * point.x + self._reverse[1][1] * point.y,
        )

    @staticmethod
    def _create_transform(
        orientation: Tuple[float, float, float, float, float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float],]:
        return (
            (orientation[0], orientation[3]),
            (orientation[1], orientation[4]),
        )

    @staticmethod
    def _create_reverse(
        transform: Tuple[
            Tuple[float, float],
            Tuple[float, float],
        ]
    ) -> Tuple[Tuple[float, float], Tuple[float, float],]:
        determinant = 1 / (
            transform[0][0] * transform[1][1] - transform[0][1] * transform[1][0]
        )
        return (
            (determinant * transform[1][1], -determinant * transform[0][1]),
            (-determinant * transform[1][0], determinant * transform[0][0]),
        )
