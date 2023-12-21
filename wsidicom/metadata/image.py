#    Copyright 2023 SECTRA AB
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

"""Image model."""
import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypeVar

from wsidicom.geometry import Orientation, PointMm, RegionMm, SizeMm


class FocusMethod(Enum):
    AUTO = "auto"
    MANUAL = "manual"


@dataclass
class ExtendedDepthOfField:
    number_of_focal_planes: int
    distance_between_focal_planes: float


GeometryType = TypeVar("GeometryType", PointMm, RegionMm)


@dataclass
class ImageCoordinateSystem:
    origin: PointMm
    rotation: float

    @property
    def orientation(self) -> Orientation:
        return Orientation.from_rotation(self.rotation)

    def image_to_slide(self, image: GeometryType) -> GeometryType:
        if isinstance(image, PointMm):
            offset = self.orientation.apply_transform(image)
            return self.origin + offset
        start = self.image_to_slide(image.start)
        end = self.image_to_slide(image.end)
        return RegionMm(start, SizeMm(end.x - start.x, end.y - start.y))

    def slide_to_image(self, slide: GeometryType) -> GeometryType:
        if isinstance(slide, PointMm):
            offset = slide - self.origin
            return self.orientation.apply_reverse_transform(offset)
        start = self.slide_to_image(slide.start)
        end = self.slide_to_image(slide.end)
        return RegionMm(start, SizeMm(end.x - start.x, end.y - start.y))

    def to_other_corrdinate_system(
        self, other: "ImageCoordinateSystem", image: GeometryType
    ) -> GeometryType:
        slide = self.image_to_slide(image)
        return other.slide_to_image(slide)


@dataclass
class Image:
    """
    Image metadata.

    Corresponds to the `Required, Empty if Unknown` attributes in the Slide Label
    module:
    https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.12.8.html
    """

    acquisition_datetime: Optional[datetime.datetime] = None
    focus_method: Optional[FocusMethod] = None
    extended_depth_of_field: Optional[ExtendedDepthOfField] = None
    image_coordinate_system: Optional[ImageCoordinateSystem] = None
    pixel_spacing: Optional[SizeMm] = None
    focal_plane_spacing: Optional[float] = None
    depth_of_field: Optional[float] = None
