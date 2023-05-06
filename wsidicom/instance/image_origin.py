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

import warnings
from typing import List, Optional

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.valuerep import DSfloat

from wsidicom.geometry import Orientation, PointMm, RegionMm


class ImageOrigin:
    def __init__(
        self,
        origin: Optional[PointMm] = None,
        orientation: Optional[Orientation] = None,
    ):
        if origin is None:
            origin = PointMm(0, 0)
        if orientation is None:
            orientation = Orientation([0, 1, 0, 1, 0, 0])
        self._origin = origin
        self._orientation = orientation

    @property
    def origin(self) -> PointMm:
        return self._origin

    @property
    def orientation(self) -> Orientation:
        return self._orientation

    @classmethod
    def from_dataset(cls, dataset: Dataset):
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
            warnings.warn(
                "Using default image origin as TotalPixelMatrixOriginSequence "
                "not set in file"
            )
            origin = None
        try:
            orientation = Orientation(dataset.ImageOrientationSlide)
        except AttributeError:
            warnings.warn(
                "Using default image orientation as ImageOrientationSlide "
                "not set in file"
            )
            orientation = None
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
        return list(self.orientation.orientation)

    @property
    def rotation(self) -> float:
        return self._orientation.rotation

    def transform_region(self, region: RegionMm) -> "RegionMm":
        region.position = region.position - self._origin
        return self._orientation.apply(region)
