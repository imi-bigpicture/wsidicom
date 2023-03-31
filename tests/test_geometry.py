#    Copyright 2022 SECTRA AB
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

import unittest

import pytest
from parameterized import parameterized

from wsidicom.geometry import (
    Orientation,
    Point,
    PointMm,
    Region,
    RegionMm,
    Size,
    SizeMm,
)
from wsidicom.instance import ImageOrigin


@pytest.mark.unittest
class WsiDicomGeomtryTests(unittest.TestCase):
    def test_size_class(self):
        size0 = Size(10, 10)
        size1 = Size(1, 1)
        self.assertEqual(size0 - size1, Size(9, 9))

        self.assertEqual(size0 * 2, Size(20, 20))

        self.assertEqual(size0 // 3, Size(3, 3))

        self.assertEqual(size0.to_tuple(), (10, 10))

    def test_point_class(self):
        point0 = Point(10, 10)
        point1 = Point(2, 2)
        point2 = Point(3, 3)
        size0 = Size(2, 2)

        self.assertEqual(point1 * point0, Point(20, 20))
        self.assertEqual(point0 * size0, Point(20, 20))
        self.assertEqual(point0 * 2, Point(20, 20))
        self.assertEqual(point0 // 3, Point(3, 3))
        self.assertEqual(point0 % point1, Point(0, 0))
        self.assertEqual(point0 % point2, Point(1, 1))
        self.assertEqual(point0 % size0, Point(0, 0))
        self.assertEqual(point0 + point1, Point(12, 12))
        self.assertEqual(point0 + 2, Point(12, 12))
        self.assertEqual(point0 + size0, Point(12, 12))
        self.assertEqual(point0 - point1, Point(8, 8))
        self.assertEqual(point0 - 2, Point(8, 8))
        self.assertEqual(point0 - size0, Point(8, 8))
        self.assertEqual(Point.max(point0, point1), point0)
        self.assertEqual(Point.min(point0, point1), point1)
        self.assertEqual(point0.to_tuple(), (10, 10))

    def test_region_mm(self):
        region = RegionMm(PointMm(1.0, 2.0), SizeMm(3.0, 4.0))
        self.assertEqual(region.start, PointMm(1.0, 2.0))
        self.assertEqual(region.end, PointMm(4.0, 6.0))

    def test_region_mm_subtract(self):
        region = RegionMm(PointMm(1.0, 2.0), SizeMm(3.0, 4.0))
        region = region - PointMm(1.0, 2.0)
        self.assertEqual(region.start, PointMm(0.0, 0.0))
        self.assertEqual(region.end, PointMm(3.0, 4.0))

    def test_region_mm_add(self):
        region = RegionMm(PointMm(1.0, 2.0), SizeMm(3.0, 4.0))
        region = region + PointMm(1.0, 2.0)
        self.assertEqual(region.start, PointMm(2.0, 4.0))
        self.assertEqual(region.end, PointMm(5.0, 8.0))

    @parameterized.expand(
        [
            (  # Image x along slide y, Image y along slide x
                RegionMm(PointMm(2.0, 4.0), SizeMm(1.0, 2.0)),
                ImageOrigin(PointMm(1.0, 2.0), Orientation([0, 1, 0, 1, 0, 0])),
                PointMm(2.0, 1.0),
                PointMm(4.0, 2.0),
            ),
            (  # Image x reversed to slide y, Image y reversed to slide x
                RegionMm(PointMm(1.0, 4.0), SizeMm(2.0, 1.0)),
                ImageOrigin(PointMm(4.0, 8.0), Orientation([0, -1, 0, -1, 0, 0])),
                PointMm(3.0, 1.0),
                PointMm(4.0, 3.0),
            ),
            (  # Image x along slide x, Image y reversed to slide y
                RegionMm(PointMm(2.0, 5.0), SizeMm(2.0, 1.0)),
                ImageOrigin(PointMm(1.0, 8.0), Orientation([1, 0, 0, 0, -1, 0])),
                PointMm(1.0, 2.0),
                PointMm(3.0, 3.0),
            ),
            (  # Image x reversed to slide x, Image y along slide y
                RegionMm(PointMm(2.0, 3.0), SizeMm(2.0, 3.0)),
                ImageOrigin(PointMm(5.0, 2.0), Orientation([-1, 0, 0, 0, 1, 0])),
                PointMm(1.0, 1.0),
                PointMm(3.0, 4.0),
            ),
        ]
    )
    def test_region_mm_to_other_origin(
        self,
        region: RegionMm,
        origin: ImageOrigin,
        expected_start: PointMm,
        expected_end: PointMm,
    ):
        transformed_region = origin.transform_region(region)
        self.assertEqual(transformed_region.start, expected_start)
        self.assertEqual(transformed_region.end, expected_end)

    @parameterized.expand(
        [
            (Region(Point(3, 4), Size(6, 4)), Point(9, 10), 2.0),
            (Region(Point(9, 10), Size(6, 4)), Point(3, 4), 0.5),
            (Region(Point(3, 4), Size(6, 4)), Point(15, 16), 3.0),
            (Region(Point(4, 7), Size(2, 6)), Point(9, 17), 2.0),
            (Region(Point(9, 17), Size(2, 6)), Point(4, 7), 0.5),
            (Region(Point(4, 7), Size(2, 6)), Point(14, 27), 3.0),
        ]
    )
    def test_region_zoom(self, region: Region, expected_start: Point, zoom: float):
        zoomed_region = region.zoom(zoom)

        self.assertEqual(zoomed_region.start, expected_start)
        self.assertEqual(zoomed_region.size, region.size)

    @parameterized.expand(
        [
            (RegionMm(PointMm(3, 4), SizeMm(6, 4)), PointMm(9, 10), 2.0),
            (RegionMm(PointMm(9, 10), SizeMm(6, 4)), PointMm(3, 4), 0.5),
            (RegionMm(PointMm(3, 4), SizeMm(6, 4)), PointMm(15, 16), 3.0),
            (RegionMm(PointMm(4, 7), SizeMm(2, 6)), PointMm(9, 17), 2.0),
            (RegionMm(PointMm(9, 17), SizeMm(2, 6)), PointMm(4, 7), 0.5),
            (RegionMm(PointMm(4, 7), SizeMm(2, 6)), PointMm(14, 27), 3.0),
        ]
    )
    def test_region_mm_zoom(
        self, region: RegionMm, expected_start: PointMm, zoom: float
    ):
        zoomed_region = region.zoom(zoom)

        self.assertEqual(zoomed_region.start, expected_start)
        self.assertEqual(zoomed_region.size, region.size)
