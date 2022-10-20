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

from wsidicom.geometry import Orientation, Point, PointMm, Region, RegionMm, Size, SizeMm
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
        self.assertEqual(
            region.start,
            PointMm(1.0, 2.0)
        )
        self.assertEqual(
            region.end,
            PointMm(4.0, 6.0)
        )

    def test_region_mm_subtract(self):
        region = RegionMm(PointMm(1.0, 2.0), SizeMm(3.0, 4.0))
        region = region - PointMm(1.0, 2.0)
        self.assertEqual(
            region.start,
            PointMm(0.0, 0.0)
        )
        self.assertEqual(
            region.end,
            PointMm(3.0, 4.0)
        )

    def test_region_mm_add(self):
        region = RegionMm(PointMm(1.0, 2.0), SizeMm(3.0, 4.0))
        region = region + PointMm(1.0, 2.0)
        self.assertEqual(
            region.start,
            PointMm(2.0, 4.0)
        )
        self.assertEqual(
            region.end,
            PointMm(5.0, 8.0)
        )

    def test_region_mm_to_other_origin_1(self):
        region = RegionMm(PointMm(2.0, 4.0), SizeMm(1.0, 2.0))
        # Image x along slide y, Image y along slide x
        origin = ImageOrigin(
            PointMm(1.0, 2.0),
            Orientation([0, 1, 0, 1, 0, 0])
        )
        region = origin.transform_region(region)
        self.assertEqual(
            region.start,
            PointMm(2.0, 1.0)
        )
        self.assertEqual(
            region.end,
            PointMm(4.0, 2.0)
        )

    def test_region_mm_to_other_origin_2(self):
        region = RegionMm(PointMm(1.0, 4.0), SizeMm(2.0, 1.0))
        # Image x reversed to slide y, Image y reversed to slide x
        origin = ImageOrigin(
            PointMm(4.0, 8.0),
            Orientation([0, -1, 0, -1, 0, 0])
        )
        region = origin.transform_region(region)
        self.assertEqual(
            region.start,
            PointMm(3.0, 1.0)
        )
        self.assertEqual(
            region.end,
            PointMm(4.0, 3.0)
        )

    def test_region_mm_to_other_origin_3(self):
        region = RegionMm(PointMm(2.0, 5.0), SizeMm(2.0, 1.0))
        # Image x along slide x, Image y reversed to slide y
        origin = ImageOrigin(
            PointMm(1.0, 8.0),
            Orientation([1, 0, 0, 0, -1, 0])
        )
        region = origin.transform_region(region)
        self.assertEqual(
            region.start,
            PointMm(1.0, 2.0)
        )
        self.assertEqual(
            region.end,
            PointMm(3.0, 3.0)
        )

    def test_region_mm_to_other_origin_4(self):
        region = RegionMm(PointMm(2.0, 3.0), SizeMm(2.0, 3.0))
        # Image x reversed to slide x, Image y along slide y
        origin = ImageOrigin(
            PointMm(5.0, 2.0),
            Orientation([-1, 0, 0, 0, 1, 0])
        )
        region = origin.transform_region(region)
        self.assertEqual(
            region.start,
            PointMm(1.0, 1.0)
        )
        self.assertEqual(
            region.end,
            PointMm(3.0, 4.0)
        )
