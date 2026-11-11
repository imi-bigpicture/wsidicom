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

"""Descriptions of the DICOM files of a WSI, as opened or as written."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydicom.uid import UID

from wsidicom.geometry import Size, SizeMm
from wsidicom.instance import WsiDataset
from wsidicom.metadata import ImageType, LossyCompression
from wsidicom.uid import FileUids


@dataclass(frozen=True)
class WsiFileBase:
    """What is shared by all files of a WSI.

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
        Focal planes of the instance the file belongs to, in um.
    optical_paths: tuple[str, ...]
        Identifiers of the optical paths of the instance the file belongs to.
    transfer_syntax: UID
        Transfer syntax of the pixel data.
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
        than the at most 16 characters stored in the file. The values are taken
        from the strings, so that they are the ones in the file.
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
class WsiLevelFile(WsiFileBase):
    """A file of a pyramid level.

    Parameters
    ----------
    pyramid_index: int
        Index of the pyramid the level belongs to, among the pyramids of the
        WSI.
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
    ) -> "WsiLevelFile":
        """Describe a level file from its dataset.

        Parameters
        ----------
        dataset: WsiDataset
            The dataset of the file, as written or as read.
        transfer_syntax: UID
            Transfer syntax of the pixel data.
        focal_planes: Sequence[float]
            Focal planes of the instance the file belongs to, in um.
        optical_paths: Sequence[str]
            Identifiers of the optical paths of the instance the file belongs to.
        pyramid_index: int
            Index of the pyramid the level belongs to.
        level_index: int
            Pyramid level index of the level.

        Returns
        -------
        WsiLevelFile
            Description of the file.
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
class WsiThumbnailFile(WsiFileBase):
    """A thumbnail file.

    Parameters
    ----------
    pyramid_index: int
        Index of the pyramid the thumbnail belongs to, among the pyramids of
        the WSI.
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
    ) -> "WsiThumbnailFile":
        """Describe a thumbnail file from its dataset.

        Parameters
        ----------
        dataset: WsiDataset
            The dataset of the file, as written or as read.
        transfer_syntax: UID
            Transfer syntax of the pixel data.
        focal_planes: Sequence[float]
            Focal planes of the instance the file belongs to, in um.
        optical_paths: Sequence[str]
            Identifiers of the optical paths of the instance the file belongs to.
        pyramid_index: int
            Index of the pyramid the thumbnail belongs to.

        Returns
        -------
        WsiThumbnailFile
            Description of the file.
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
class WsiAssociatedFile(WsiFileBase):
    """A label or overview file."""

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
    ) -> "WsiAssociatedFile":
        """Describe a label or overview file from its dataset.

        Parameters
        ----------
        dataset: WsiDataset
            The dataset of the file, as written or as read.
        transfer_syntax: UID
            Transfer syntax of the pixel data.
        focal_planes: Sequence[float]
            Focal planes of the instance the file belongs to, in um.
        optical_paths: Sequence[str]
            Identifiers of the optical paths of the instance the file belongs to.

        Returns
        -------
        WsiAssociatedFile
            Description of the file.
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


WsiFile: TypeAlias = WsiLevelFile | WsiThumbnailFile | WsiAssociatedFile
"""A file of a WSI. Narrow it with `match`, `isinstance`, or by comparing
`image_type`."""
