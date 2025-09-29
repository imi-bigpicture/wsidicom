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
from dataclasses import dataclass, replace
from enum import Enum
from math import sqrt
from typing import Optional, Sequence, TypeVar

from wsidicom.codec import LossyCompressionIsoStandard
from wsidicom.geometry import Orientation, PointMm, RegionMm, SizeMm


class FocusMethod(Enum):
    """Focus methods."""

    AUTO = "auto"
    MANUAL = "manual"


@dataclass(frozen=True)
class ExtendedDepthOfField:
    """Extended depth of field.

    Parameters
    ----------
    number_of_focal_planes : int
        The number of focal planes used for extended depth of field.
    distance_between_focal_planes : float
        The distance between the focal planes used for extended depth of field.
    """

    number_of_focal_planes: int
    distance_between_focal_planes: float


GeometryType = TypeVar("GeometryType", PointMm, RegionMm)


@dataclass(frozen=True)
class ImageCoordinateSystem:
    """Image coordinate system.

    Parameters
    ----------
    origin : PointMm
        The position of the top left pixel in the slide coorindate system.
    rotation : float
        The rotation of the image in degrees in the slide coordinate system.
    z_offset : Optional[float]
        The z offset of the image in the slide coordinate system.
    """

    origin: PointMm
    rotation: float
    z_offset: Optional[float] = None

    @property
    def orientation(self) -> Orientation:
        return Orientation.from_rotation(self.rotation)

    def image_to_slide(self, image: GeometryType) -> GeometryType:
        """Translate a geometry (PointMm or RegionMm) from image coordinate system
        (origin at image top-left corner) to slide coordinate system (origin at
        slide corner).

        Parameters
        ----------
        image : GeometryType
            The geometry to translate.

        Returns
        -------
        GeometryType
            The translate geometry.
        """
        if isinstance(image, PointMm):
            offset = self.orientation.apply_transform(image)
            return self.origin + offset
        start = self.image_to_slide(image.start)
        end = self.image_to_slide(image.end)
        return RegionMm(start, SizeMm(end.x - start.x, end.y - start.y))

    def slide_to_image(self, slide: GeometryType) -> GeometryType:
        """Translate a geometry (PointMm or RegionMm) from slide coordinate system
        (origin at slide corner) to image coordinate system (origin at image top-left
        corner).

        Parameters
        ----------
        slide : GeometryType
            The geometry to translate.

        Returns
        -------
        GeometryType
            The translate geometry.
        """
        if isinstance(slide, PointMm):
            offset = slide - self.origin
            return self.orientation.apply_reverse_transform(offset)
        start = self.slide_to_image(slide.start)
        end = self.slide_to_image(slide.end)
        start, end = (
            PointMm(min(start.x, end.x), min(start.y, end.y)),
            PointMm(max(start.x, end.x), max(start.y, end.y)),
        )
        size = SizeMm(end.x - start.x, end.y - start.y)
        return RegionMm(start, size)

    def to_other_coordinate_system(
        self, other: "ImageCoordinateSystem", image: GeometryType
    ) -> GeometryType:
        """Translate a geometry (PointMm or RegionMm) from this image coordinate system
        to another image coordinate system.

        Parameters
        ----------
        other : ImageCoordinateSystem
            The target image coordinate system.
        image : GeometryType
            The geometry to translate.

        Returns
        -------
        GeometryType
            The translate geometry.
        """
        slide = self.image_to_slide(image)
        return other.slide_to_image(slide)

    @classmethod
    def from_middle_of_slide(
        cls,
        slide_middle: PointMm,
        image_size: SizeMm,
        rotation: float,
        z_offset: Optional[float],
    ) -> "ImageCoordinateSystem":
        """Create an image coordinate system that places the image in the middle of the
        slide.

        Parameters
        ----------
        slide_middle : PointMm
            The middle of the slide.
        image_size : SizeMm
            The size of the image.
        rotation : float
            The rotation of the image in degrees.
        z_offset : Optional[float]
            The z offset of the image.

        Returns
        -------
        ImageCoordinateSystem
            The image coordinate system.
        """
        middle_of_slide_system = ImageCoordinateSystem(
            origin=slide_middle,
            rotation=rotation,
        )
        image_coordinate_system_origin = middle_of_slide_system.image_to_slide(
            -PointMm(image_size.width / 2, image_size.height / 2)
        )
        return ImageCoordinateSystem(
            origin=image_coordinate_system_origin,
            rotation=rotation,
            z_offset=z_offset,
        )

    def origin_and_rotation_match(
        self, other: Optional["ImageCoordinateSystem"], origin_threshold: float
    ) -> bool:
        if other is None:
            return False
        if self.rotation != other.rotation:
            return False
        origin_distance = sqrt(
            (self.origin.x - other.origin.x) ** 2
            + (self.origin.y - other.origin.y) ** 2
        )
        return origin_distance <= origin_threshold


@dataclass(frozen=True)
class LossyCompression:
    """Lossy compression.

    Parameters
    ----------
    method : LossyCompressionIsoStandard
        The method used for lossy compression.
    ratio : float
        The compression ratio.
    """

    method: LossyCompressionIsoStandard
    ratio: float


@dataclass(frozen=True)
class Image:
    """
    Image metadata.

    Parameters
    ----------
    acquisition_datetime : Optional[datetime.datetime] = None
        The acquisition datetime of the image.
    focus_method : Optional[FocusMethod] = None
        The focus method used for imaging.
    extended_depth_of_field : Optional[ExtendedDepthOfField] = None
        Describes if extended depth of field has been used for imaging.
    image_coordinate_system : Optional[ImageCoordinateSystem] = None
        The image coordinate system in relation the slide frame of reference.
    pixel_spacing : Optional[SizeMm] = None
        The pixel spacing in mm per pixel.
    focal_plane_spacing : Optional[float] = None
        The spacing between focal planes if image with multiple focal planes.
    depth_of_field : Optional[float] = None
        The depth of field of the image.
    lossy_compressions : Optional[Sequence[LossyCompression]] = None
        The lossy compressions method that has been applied to the image data.
    """

    acquisition_datetime: Optional[datetime.datetime] = None
    focus_method: Optional[FocusMethod] = None
    extended_depth_of_field: Optional[ExtendedDepthOfField] = None
    image_coordinate_system: Optional[ImageCoordinateSystem] = None
    pixel_spacing: Optional[SizeMm] = None
    focal_plane_spacing: Optional[float] = None
    depth_of_field: Optional[float] = None
    lossy_compressions: Optional[Sequence[LossyCompression]] = None

    def remove_confidential(self) -> "Image":
        return replace(
            self,
            acquisition_datetime=None,
        )
