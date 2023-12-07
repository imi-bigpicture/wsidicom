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

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from PIL.Image import Image
from pydicom import Dataset
from pydicom.uid import UID

from wsidicom.errors import WsiDicomError, WsiDicomUidDuplicateError
from wsidicom.geometry import Size, SizeMm
from wsidicom.instance.dataset import ImageType, WsiDataset
from wsidicom.instance.image_data import ImageData
from wsidicom.instance.image_coordinate_system import ImageCoordinateSystem
from wsidicom.instance.pillow_image_data import PillowImageData
from wsidicom.uid import SlideUids


class WsiInstance:
    """Represents a level, label, or overview wsi image, containing image data
    and datasets with metadata."""

    def __init__(
        self, datasets: Union[WsiDataset, Sequence[WsiDataset]], image_data: ImageData
    ):
        """Create a WsiInstance from datasets with metadata and image data.

        Parameters
        ----------
        datasets: Union[WsiDataset, Sequence[WsiDataset]]
            Single dataset or list of datasets.
        image_data: ImageData
            Image data.
        """
        if not isinstance(datasets, Sequence):
            datasets = [datasets]
        self._datasets = datasets
        self._image_data = image_data
        self._identifier, self._uids = self._validate_instance(self.datasets)
        self._image_type = self.dataset.image_type

        if self.ext_depth_of_field:
            if self.ext_depth_of_field_planes is None:
                raise WsiDicomError("Instance Missing NumberOfFocalPlanes.")
            if self.ext_depth_of_field_plane_distance is None:
                raise WsiDicomError("Instance Missing DistanceBetweenFocalPlanes.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.dataset}, {self.image_data})"

    def __str__(self) -> str:
        return self.pretty_str()

    def pretty_str(self, indent: int = 0, depth: Optional[int] = None) -> str:
        string = f"default z: {self.default_z} " f"default path: { self.default_path}"
        if depth is not None:
            depth -= 1
            if depth < 0:
                return string
        string += " ImageData " + self.image_data.pretty_str(indent + 1, depth)
        return string

    @property
    def image_type(self) -> ImageType:
        """Return wsi type."""
        return self._image_type

    @property
    def datasets(self) -> Sequence[WsiDataset]:
        return self._datasets

    @property
    def dataset(self) -> WsiDataset:
        return self.datasets[0]

    @property
    def image_data(self) -> ImageData:
        return self._image_data

    @property
    def size(self) -> Size:
        """Return image size in pixels."""
        return self._image_data.image_size

    @property
    def tile_size(self) -> Size:
        """Return tile size in pixels."""
        return self._image_data.tile_size

    @property
    def mpp(self) -> Optional[SizeMm]:
        """Return pixel spacing in um/pixel."""
        if self.pixel_spacing is None:
            return None
        return self.pixel_spacing * 1000.0

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        """Return pixel spacing in mm/pixel."""
        return self._image_data.pixel_spacing

    @property
    def mm_size(self) -> Optional[SizeMm]:
        """Return slide size in mm."""
        return self.dataset.mm_size

    @property
    def mm_depth(self) -> Optional[float]:
        """Return imaged depth in mm."""
        return self.dataset.mm_depth

    @property
    def slice_thickness(self) -> Optional[float]:
        """Return slice thickness."""
        return self.dataset.slice_thickness

    @property
    def slice_spacing(self) -> Optional[float]:
        """Return slice spacing."""
        return self.dataset.spacing_between_slices

    @property
    def focus_method(self) -> str:
        return self.dataset.focus_method

    @property
    def ext_depth_of_field(self) -> bool:
        return self.dataset.ext_depth_of_field

    @property
    def ext_depth_of_field_planes(self) -> Optional[int]:
        return self.dataset.ext_depth_of_field_planes

    @property
    def ext_depth_of_field_plane_distance(self) -> Optional[float]:
        return self.dataset.ext_depth_of_field_plane_distance

    @property
    def identifier(self) -> UID:
        """Return identifier (instance uid for single file instance or
        concatenation uid for multiple file instance)."""
        return self._identifier

    @property
    def default_z(self) -> float:
        return self._image_data.default_z

    @property
    def default_path(self) -> str:
        return self._image_data.default_path

    @property
    def focal_planes(self) -> List[float]:
        return self._image_data.focal_planes

    @property
    def optical_paths(self) -> List[str]:
        return self._image_data.optical_paths

    @property
    def tiled_size(self) -> Size:
        return self._image_data.tiled_size

    @property
    def uids(self) -> SlideUids:
        """Return base uids"""
        return self._uids

    @property
    def image_coordinate_system(self) -> Optional[ImageCoordinateSystem]:
        return self.image_data.image_coordinate_system

    @classmethod
    def create_label(
        cls, image: Union[Image, str, Path], base_dataset: Dataset
    ) -> "WsiInstance":
        """Create a label WsiInstance.

        Parameters
        ----------
        image: Union[Image, str, Path]
            Image or path to image.
        base_dataset: Dataset
            Base dataset to include.

        Returns
        ----------
        WsiInstance
            Created label WsiInstance.
        """
        if isinstance(image, Image):
            image_data = PillowImageData(image)
        else:
            image_data = PillowImageData.from_file(image)
        return cls.create_instance(image_data, base_dataset, ImageType.LABEL)

    @classmethod
    def create_instance(
        cls, image_data: ImageData, base_dataset: Dataset, image_type: ImageType
    ) -> "WsiInstance":
        """Create WsiInstance from ImageData.

        Parameters
        ----------
        image_data: ImageData
            Image data and metadata.
        base_dataset: Dataset
            Base dataset to include.
        image_type: ImageType
            Type of instance to create.

        Returns
        ----------
        WsiInstance
            Created WsiInstance.
        """
        instance_dataset = WsiDataset.create_instance_dataset(
            base_dataset, image_type, image_data
        )

        return cls(instance_dataset, image_data)

    @staticmethod
    def check_duplicate_instance(
        instances: Sequence["WsiInstance"], self: object
    ) -> None:
        """Check for duplicates in list of instances. Instances are duplicate
        if instance identifier (file instance uid or concatenation uid) match.
        Stops at first found duplicate and raises WsiDicomUidDuplicateError.

        Parameters
        ----------
        instances: Sequence['WsiInstance']
            List of instances to check.
        caller: Object
            Object that the instances belongs to.
        """
        instance_identifiers: List[str] = []
        for instance in instances:
            instance_identifier = instance.identifier
            if instance_identifier not in instance_identifiers:
                instance_identifiers.append(instance_identifier)
            else:
                raise WsiDicomUidDuplicateError(str(instance), str(self))

    def _validate_instance(
        self, datasets: Sequence[WsiDataset]
    ) -> Tuple[UID, SlideUids]:
        """Check that no files in instance are duplicate, that all files in
        instance matches (uid, type and size).
        Raises WsiDicomMatchError otherwise.
        Returns the matching file uid.

        Returns
        ----------
        Tuple[UID, SlideUids]
            Instance identifier uid and base uids
        """
        WsiDataset.check_duplicate_dataset(datasets, self)

        base_dataset = datasets[0]
        for dataset in datasets[1:]:
            if not base_dataset.matches_instance(dataset):
                raise WsiDicomError("Datasets in instances does not match")
        return (
            base_dataset.uids.identifier,
            base_dataset.uids.slide,
        )

    def matches(self, other_instance: "WsiInstance") -> bool:
        """Return true if other instance is of the same group as self.

        Parameters
        ----------
        other_instance: WsiInstance
            Instance to check.

        Returns
        ----------
        bool
            True if instances are of same group.

        """
        return (
            self.uids.matches(other_instance.uids)
            and self.size == other_instance.size
            and self.tile_size == other_instance.tile_size
            and self.image_type == other_instance.image_type
        )
