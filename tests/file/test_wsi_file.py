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

"""Unit tests for the descriptions of written files."""

from typing import Any

import pytest
from pydicom.uid import JPEGBaseline8Bit, generate_uid

from wsidicom.file import (
    WsiAssociatedFile,
    WsiLevelFile,
    WsiThumbnailFile,
)
from wsidicom.geometry import Size, SizeMm
from wsidicom.metadata import ImageType
from wsidicom.uid import FileUids, SlideUids


@pytest.fixture
def shared_fields() -> dict[str, Any]:
    """Fields shared by all kinds of written files."""
    return {
        "uids": FileUids(
            generate_uid(),
            None,
            SlideUids(generate_uid(), generate_uid(), generate_uid()),
        ),
        "image_size": Size(1024, 512),
        "tile_size": Size(256, 256),
        "focal_planes": (0.0,),
        "optical_paths": ("0",),
        "transfer_syntax": JPEGBaseline8Bit,
        "photometric_interpretation": "YBR_FULL_422",
        "lossy_compressions": (),
    }


@pytest.mark.unittest
class TestCreatedFile:
    def test_level_file_gives_mpp_from_pixel_spacing(
        self, shared_fields: dict[str, Any]
    ) -> None:
        # Arrange
        level_file = WsiLevelFile(
            image_type=ImageType.VOLUME,
            pixel_spacing=SizeMm(0.0005, 0.00025),
            pyramid_index=0,
            level_index=0,
            **shared_fields,
        )

        # Act
        mpp = level_file.mpp

        # Assert
        assert mpp.width == pytest.approx(0.5)
        assert mpp.height == pytest.approx(0.25)

    def test_associated_file_without_pixel_spacing_has_no_mpp(
        self, shared_fields: dict[str, Any]
    ) -> None:
        # Arrange
        associated_file = WsiAssociatedFile(
            image_type=ImageType.LABEL, pixel_spacing=None, **shared_fields
        )

        # Act
        mpp = associated_file.mpp

        # Assert
        assert mpp is None

    @pytest.mark.parametrize(
        "image_type", [ImageType.THUMBNAIL, ImageType.LABEL, ImageType.OVERVIEW]
    )
    def test_level_file_rejects_other_image_type(
        self, shared_fields: dict[str, Any], image_type: ImageType
    ) -> None:
        # Arrange

        # Act & Assert
        with pytest.raises(ValueError, match="is not allowed for"):
            WsiLevelFile(
                image_type=image_type,  # pyright: ignore[reportArgumentType]
                pixel_spacing=SizeMm(0.0005, 0.0005),
                pyramid_index=0,
                level_index=0,
                **shared_fields,
            )

    @pytest.mark.parametrize(
        "image_type", [ImageType.VOLUME, ImageType.LABEL, ImageType.OVERVIEW]
    )
    def test_thumbnail_file_rejects_other_image_type(
        self, shared_fields: dict[str, Any], image_type: ImageType
    ) -> None:
        # Arrange

        # Act & Assert
        with pytest.raises(ValueError, match="is not allowed for"):
            WsiThumbnailFile(
                image_type=image_type,  # pyright: ignore[reportArgumentType]
                pixel_spacing=SizeMm(0.0005, 0.0005),
                pyramid_index=0,
                **shared_fields,
            )

    @pytest.mark.parametrize("image_type", [ImageType.VOLUME, ImageType.THUMBNAIL])
    def test_associated_file_rejects_other_image_type(
        self, shared_fields: dict[str, Any], image_type: ImageType
    ) -> None:
        # Arrange

        # Act & Assert
        with pytest.raises(ValueError, match="is not allowed for"):
            WsiAssociatedFile(
                image_type=image_type,  # pyright: ignore[reportArgumentType]
                pixel_spacing=None,
                **shared_fields,
            )
