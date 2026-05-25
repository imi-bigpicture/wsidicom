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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from math import sqrt
from typing import ClassVar, Optional, TypeVar

from wsidicom.codec import LossyCompressionIsoStandard
from wsidicom.geometry import Orientation, PointMm, RegionMm, SizeMm


class FocusMethod(Enum):
    """Focus methods."""

    AUTO = "auto"
    MANUAL = "manual"


class ImageType(Enum):
    """Type of WSI image."""

    VOLUME = "VOLUME"
    LABEL = "LABEL"
    OVERVIEW = "OVERVIEW"
    THUMBNAIL = "THUMBNAIL"


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
    z_offset : float | None
        The z offset of the image in the slide coordinate system.
    """

    origin: PointMm
    rotation: float
    z_offset: float | None = None

    # Standard glass microscope slide geometry. The label area occupies the top 25 mm
    # of the slide, so the imaged (non-label) part is 25 x 50 mm and the full slide is
    # 25 x 75 mm.
    SLIDE_SIZE_WITHOUT_LABEL: ClassVar[SizeMm] = SizeMm(25, 50)
    SLIDE_SIZE_WITH_LABEL: ClassVar[SizeMm] = SizeMm(25, 75)

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
        z_offset: float | None,
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
        z_offset : float | None
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

    @classmethod
    def default_for(
        cls,
        rotation: float,
        image_type: ImageType,
        z_offset: float | None = None,
        slide_size_without_label: SizeMm | None = None,
        slide_size_with_label: SizeMm | None = None,
    ) -> "ImageCoordinateSystem":
        """Create a default image coordinate system for an image of ``image_type``.

        Best effort for images that do not specify a position: the image is assumed to
        fill its area of a standard slide, with its first pixel at the corner matching
        ``rotation``. The origin is therefore a canonical slide corner (e.g. ``(0, 0)``
        or ``(25, 50)``), recognisable as a default rather than a measured position.

        Parameters
        ----------
        rotation : float
            The rotation of the image in degrees. One of 0, 90, 180 or 270.
        image_type : ImageType
            The type of image to place on the slide.
        z_offset : float | None
            The z offset of the image.
        slide_size_without_label : SizeMm | None
            Size of the non-label part of the slide. Defaults to a standard slide
            (``SLIDE_SIZE_WITHOUT_LABEL``), e.g. for large-format scanners.
        slide_size_with_label : SizeMm | None
            Size of the full slide including the label area. Defaults to a standard
            slide (``SLIDE_SIZE_WITH_LABEL``).

        Returns
        -------
        ImageCoordinateSystem
            The default image coordinate system.
        """
        region = cls._default_slide_region(
            image_type,
            slide_size_without_label or cls.SLIDE_SIZE_WITHOUT_LABEL,
            slide_size_with_label or cls.SLIDE_SIZE_WITH_LABEL,
        )
        return cls(
            origin=cls._corner_for_rotation(region, rotation),
            rotation=rotation,
            z_offset=z_offset,
        )

    @staticmethod
    def _default_slide_region(
        image_type: ImageType,
        size_without_label: SizeMm,
        size_with_label: SizeMm,
    ) -> RegionMm:
        """Return the slide area that a default image of ``image_type`` fills."""
        if image_type == ImageType.OVERVIEW:
            return RegionMm(PointMm(0, 0), size_with_label)
        if image_type == ImageType.LABEL:
            return RegionMm(
                PointMm(0, size_without_label.height),
                SizeMm(
                    size_with_label.width,
                    size_with_label.height - size_without_label.height,
                ),
            )
        return RegionMm(PointMm(0, 0), size_without_label)

    @staticmethod
    def _corner_for_rotation(region: RegionMm, rotation: float) -> PointMm:
        """Return the first-pixel position for an image filling ``region`` rotated by
        ``rotation`` degrees, i.e. the region corner the top-left pixel maps to."""
        if rotation == 0:
            return region.start
        if rotation == 90:
            return PointMm(region.end.x, region.start.y)
        if rotation == 180:
            return region.end
        if rotation == 270:
            return PointMm(region.start.x, region.end.y)
        raise ValueError(
            f"Unsupported default image coordinate system rotation {rotation}; "
            "expected 0, 90, 180 or 270."
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
    acquisition_datetime : datetime.datetime | None = None
        The acquisition datetime of the image.
    focus_method : FocusMethod | None = None
        The focus method used for imaging.
    extended_depth_of_field : ExtendedDepthOfField | None = None
        Describes if extended depth of field has been used for imaging.
    image_coordinate_system : ImageCoordinateSystem | None = None
        The image coordinate system in relation the slide frame of reference.
    pixel_spacing : SizeMm | None = None
        The pixel spacing in mm per pixel.
    focal_plane_spacing : float | None = None
        The spacing between focal planes if image with multiple focal planes.
    depth_of_field : float | None = None
        The depth of field of the image.
    lossy_compressions : Sequence[LossyCompression] | None = None
        The lossy compressions method that has been applied to the image data.
    """

    acquisition_datetime: datetime.datetime | None = None
    focus_method: FocusMethod | None = None
    extended_depth_of_field: ExtendedDepthOfField | None = None
    image_coordinate_system: ImageCoordinateSystem | None = None
    pixel_spacing: SizeMm | None = None
    focal_plane_spacing: float | None = None
    depth_of_field: float | None = None
    lossy_compressions: Sequence[LossyCompression] | None = None

    def remove_confidential(self) -> "Image":
        return replace(
            self,
            acquisition_datetime=None,
        )
