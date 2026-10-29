#    Copyright 2026 SECTRA AB
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

"""Descriptions of the DICOM files written when saving a WSI."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydicom.uid import UID

from wsidicom.geometry import Size, SizeMm
from wsidicom.instance import WsiDataset
from wsidicom.metadata import ImageType, LossyCompression
from wsidicom.uid import FileUids


@dataclass(frozen=True)
class CreatedFileBase:
    """What is shared by all files written when saving a WSI.

    Parameters
    ----------
    image_type: ImageType
        The image type of the file.
    uids: FileUids
        The SOP Instance UID of the file, the SOP Instance UID of the
        concatenation source if the file is part of a concatenation, and the
        study, series and frame of reference UIDs.
    image_size: Size
        Size of the total pixel matrix in pixels.
    tile_size: Size
        Size of the tiles in pixels.
    pixel_spacing: SizeMm | None
        Pixel spacing in mm per pixel, or None if the file has none.
    focal_planes: tuple[float, ...]
        Focal planes in the file, in um.
    optical_paths: tuple[str, ...]
        Identifiers of the optical paths in the file.
    transfer_syntax: UID
        Transfer syntax the pixel data is written with.
    photometric_interpretation: str
        Photometric interpretation of the pixel data.
    lossy_compressions: tuple[LossyCompression, ...]
        The lossy compressions the image has been through, in order. Empty if
        it has not been lossy compressed.
    """

    image_type: ImageType
    uids: FileUids
    image_size: Size
    tile_size: Size
    pixel_spacing: SizeMm | None
    focal_planes: tuple[float, ...]
    optical_paths: tuple[str, ...]
    transfer_syntax: UID
    photometric_interpretation: str
    lossy_compressions: tuple[LossyCompression, ...]

    @property
    def mpp(self) -> SizeMm | None:
        """Pixel spacing in um per pixel, or None if the file has none."""
        if self.pixel_spacing is None:
            return None
        return self.pixel_spacing * 1000.0

    def _check_image_type(self, allowed: Sequence[ImageType]) -> None:
        """Raise if the image type is not one of `allowed`."""
        if self.image_type not in allowed:
            raise ValueError(
                f"Image type {self.image_type} is not allowed for "
                f"{type(self).__name__}, expected one of {list(allowed)}."
            )

    @staticmethod
    def _pixel_spacing(dataset: WsiDataset) -> SizeMm | None:
        """Return the pixel spacing of `dataset` as plain floats, or None.

        The dataset holds pydicom decimal strings, which can carry more digits
        than the at most 16 characters written to the file. The values are
        taken from the written strings, so that they are the ones in the file.
        """
        pixel_spacing = dataset.pixel_spacing
        if pixel_spacing is None:
            return None
        return SizeMm(float(str(pixel_spacing.width)), float(str(pixel_spacing.height)))

    @classmethod
    def _require_pixel_spacing(cls, dataset: WsiDataset) -> SizeMm:
        """Return the pixel spacing of `dataset`, raising if it has none."""
        pixel_spacing = cls._pixel_spacing(dataset)
        if pixel_spacing is None:
            raise ValueError(
                f"Pixel spacing is required for {dataset.image_type} images."
            )
        return pixel_spacing


@dataclass(frozen=True)
class CreatedLevelFile(CreatedFileBase):
    """A written file of a pyramid level.

    Parameters
    ----------
    pyramid_index: int
        Index of the pyramid the level belongs to, among the pyramids of the
        saved WSI.
    level_index: int
        Pyramid level index of the level, where 0 is the base level and each
        step halves the resolution.
    """

    image_type: Literal[ImageType.VOLUME]
    pixel_spacing: SizeMm
    pyramid_index: int
    level_index: int

    def __post_init__(self) -> None:
        self._check_image_type([ImageType.VOLUME])

    @property
    def mpp(self) -> SizeMm:
        """Pixel spacing in um per pixel."""
        return self.pixel_spacing * 1000.0

    @classmethod
    def from_dataset(
        cls,
        dataset: WsiDataset,
        transfer_syntax: UID,
        focal_planes: Sequence[float],
        optical_paths: Sequence[str],
        pyramid_index: int,
        level_index: int,
    ) -> "CreatedLevelFile":
        """Describe a written level file from the dataset it was written with.

        Parameters
        ----------
        dataset: WsiDataset
            The dataset as written to the file.
        transfer_syntax: UID
            Transfer syntax the pixel data is written with.
        focal_planes: Sequence[float]
            Focal planes in the file, in um.
        optical_paths: Sequence[str]
            Identifiers of the optical paths in the file.
        pyramid_index: int
            Index of the pyramid the level belongs to.
        level_index: int
            Pyramid level index of the level.

        Returns
        -------
        CreatedLevelFile
            Description of the written file.
        """
        if dataset.image_type != ImageType.VOLUME:
            raise ValueError(f"Expected a VOLUME image, got {dataset.image_type}.")
        return cls(
            image_type=ImageType.VOLUME,
            uids=dataset.uids,
            image_size=dataset.image_size,
            tile_size=dataset.tile_size,
            pixel_spacing=cls._require_pixel_spacing(dataset),
            focal_planes=tuple(focal_planes),
            optical_paths=tuple(optical_paths),
            transfer_syntax=transfer_syntax,
            photometric_interpretation=dataset.photometric_interpretation,
            lossy_compressions=tuple(dataset.lossy_compressions or ()),
            pyramid_index=pyramid_index,
            level_index=level_index,
        )


@dataclass(frozen=True)
class CreatedThumbnailFile(CreatedFileBase):
    """A written thumbnail file.

    Parameters
    ----------
    pyramid_index: int
        Index of the pyramid the thumbnail belongs to, among the pyramids of
        the saved WSI.
    """

    image_type: Literal[ImageType.THUMBNAIL]
    pixel_spacing: SizeMm
    pyramid_index: int

    def __post_init__(self) -> None:
        self._check_image_type([ImageType.THUMBNAIL])

    @property
    def mpp(self) -> SizeMm:
        """Pixel spacing in um per pixel."""
        return self.pixel_spacing * 1000.0

    @classmethod
    def from_dataset(
        cls,
        dataset: WsiDataset,
        transfer_syntax: UID,
        focal_planes: Sequence[float],
        optical_paths: Sequence[str],
        pyramid_index: int,
    ) -> "CreatedThumbnailFile":
        """Describe a written thumbnail file from the dataset it was written with.

        Parameters
        ----------
        dataset: WsiDataset
            The dataset as written to the file.
        transfer_syntax: UID
            Transfer syntax the pixel data is written with.
        focal_planes: Sequence[float]
            Focal planes in the file, in um.
        optical_paths: Sequence[str]
            Identifiers of the optical paths in the file.
        pyramid_index: int
            Index of the pyramid the thumbnail belongs to.

        Returns
        -------
        CreatedThumbnailFile
            Description of the written file.
        """
        if dataset.image_type != ImageType.THUMBNAIL:
            raise ValueError(f"Expected a THUMBNAIL image, got {dataset.image_type}.")
        return cls(
            image_type=ImageType.THUMBNAIL,
            uids=dataset.uids,
            image_size=dataset.image_size,
            tile_size=dataset.tile_size,
            pixel_spacing=cls._require_pixel_spacing(dataset),
            focal_planes=tuple(focal_planes),
            optical_paths=tuple(optical_paths),
            transfer_syntax=transfer_syntax,
            photometric_interpretation=dataset.photometric_interpretation,
            lossy_compressions=tuple(dataset.lossy_compressions or ()),
            pyramid_index=pyramid_index,
        )


@dataclass(frozen=True)
class CreatedAssociatedFile(CreatedFileBase):
    """A written label or overview file."""

    image_type: Literal[ImageType.LABEL, ImageType.OVERVIEW]

    def __post_init__(self) -> None:
        self._check_image_type([ImageType.LABEL, ImageType.OVERVIEW])

    @classmethod
    def from_dataset(
        cls,
        dataset: WsiDataset,
        transfer_syntax: UID,
        focal_planes: Sequence[float],
        optical_paths: Sequence[str],
    ) -> "CreatedAssociatedFile":
        """Describe a written label or overview file from the dataset it was
        written with.

        Parameters
        ----------
        dataset: WsiDataset
            The dataset as written to the file.
        transfer_syntax: UID
            Transfer syntax the pixel data is written with.
        focal_planes: Sequence[float]
            Focal planes in the file, in um.
        optical_paths: Sequence[str]
            Identifiers of the optical paths in the file.

        Returns
        -------
        CreatedAssociatedFile
            Description of the written file.
        """
        image_type = dataset.image_type
        if image_type is not ImageType.LABEL and image_type is not ImageType.OVERVIEW:
            raise ValueError(f"Expected a LABEL or OVERVIEW image, got {image_type}.")
        return cls(
            image_type=image_type,
            uids=dataset.uids,
            image_size=dataset.image_size,
            tile_size=dataset.tile_size,
            pixel_spacing=cls._pixel_spacing(dataset),
            focal_planes=tuple(focal_planes),
            optical_paths=tuple(optical_paths),
            transfer_syntax=transfer_syntax,
            photometric_interpretation=dataset.photometric_interpretation,
            lossy_compressions=tuple(dataset.lossy_compressions or ()),
        )


CreatedFile: TypeAlias = CreatedLevelFile | CreatedThumbnailFile | CreatedAssociatedFile
"""A file written when saving a WSI. Narrow it with `match`, `isinstance`, or by
comparing `image_type`."""
