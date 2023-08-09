#    Copyright 2022, 2023 SECTRA AB
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

from typing import List, Optional, TypeVar

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.valuerep import DSfloat

from wsidicom.geometry import Orientation, PointMm, RegionMm, SizeMm

GeometryType = TypeVar("GeometryType", PointMm, RegionMm)


class ImageCoordinateSystem:
    def __init__(
        self,
        origin: PointMm,
        orientation: Orientation,
    ):
        self._origin = origin
        self._orientation = orientation

    @property
    def origin(self) -> PointMm:
        return self._origin

    @property
    def orientation(self) -> Orientation:
        return self._orientation

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> Optional["ImageCoordinateSystem"]:
        try:
            origin = PointMm(
                dataset.TotalPixelMatrixOriginSequence[
                    0
                ].XOffsetInSlideCoordinateSystem,
                dataset.TotalPixelMatrixOriginSequence[
                    0
                ].YOffsetInSlideCoordinateSystem,
            )
        except (AttributeError, IndexError):
            return None
        try:
            orientation = Orientation(dataset.ImageOrientationSlide)
        except AttributeError:
            return None
        return cls(origin, orientation)

    @property
    def total_pixel_matrix_origin_sequence(self) -> DicomSequence:
        """Return formatted TotalPixelMatrixOriginSequence."""
        offset_item = Dataset()
        offset_item.XOffsetInSlideCoordinateSystem = DSfloat(self.origin.x, True)
        offset_item.YOffsetInSlideCoordinateSystem = DSfloat(self.origin.y, True)
        return DicomSequence([offset_item])

    @property
    def image_orientation_slide(
        self,
    ) -> List[float]:
        """Return formatted ImageOrientationSlide."""
        return [DSfloat(value, True) for value in self.orientation.values]

    @property
    def rotation(self) -> float:
        """The rotation of the image in relation to the slide coordinate system in degrees."""
        return self._orientation.rotation

    def image_to_slide(self, image: GeometryType) -> GeometryType:
        if isinstance(image, PointMm):
            offset = self._orientation.apply_transform(image)
            return self._origin + offset
        start = self.image_to_slide(image.start)
        end = self.image_to_slide(image.end)
        return RegionMm(start, SizeMm(end.x - start.x, end.y - start.y))

    def slide_to_image(self, slide: GeometryType) -> GeometryType:
        if isinstance(slide, PointMm):
            offset = slide - self._origin
            return self._orientation.apply_reverse_transform(offset)
        start = self.slide_to_image(slide.start)
        end = self.slide_to_image(slide.end)
        return RegionMm(start, SizeMm(end.x - start.x, end.y - start.y))

    def to_other_corrdinate_system(
        self, other: "ImageCoordinateSystem", image: GeometryType
    ) -> GeometryType:
        slide = self.image_to_slide(image)
        return other.slide_to_image(slide)
