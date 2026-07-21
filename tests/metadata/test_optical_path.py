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

import numpy as np
import pytest
from PIL import ImageCms

from wsidicom.metadata.optical_path import (
    ConstantLutSegment,
    DiscreteLutSegment,
    LinearLutSegment,
    Lut,
    OpticalPath,
)


class TestAddColorSpaceFromIcc:
    def test_sets_color_space_from_stripped_profile_description(self):
        # Arrange
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        expected = ImageCms.getProfileDescription(profile).strip()
        optical_path = OpticalPath(icc_profile=profile.tobytes())

        # Act
        updated = optical_path.add_color_space_from_icc()

        # Assert
        assert updated.color_space == expected
        assert updated.color_space == expected.strip()

    def test_does_not_overwrite_existing_color_space(self):
        # Arrange
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        optical_path = OpticalPath(
            icc_profile=profile.tobytes(), color_space="EXISTING"
        )

        # Act
        updated = optical_path.add_color_space_from_icc()

        # Assert
        assert updated is optical_path

    def test_force_overwrites_existing_color_space(self):
        # Arrange
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        expected = ImageCms.getProfileDescription(profile).strip()
        optical_path = OpticalPath(
            icc_profile=profile.tobytes(), color_space="EXISTING"
        )

        # Act
        updated = optical_path.add_color_space_from_icc(force=True)

        # Assert
        assert updated.color_space == expected

    def test_force_keeps_existing_color_space_when_profile_unreadable(self):
        # Arrange
        optical_path = OpticalPath(
            icc_profile=b"not an icc profile", color_space="EXISTING"
        )

        # Act
        updated = optical_path.add_color_space_from_icc(force=True)

        # Assert
        assert updated is optical_path
        assert updated.color_space == "EXISTING"

    @pytest.mark.parametrize("icc_profile", [None, b"", b"not an icc profile"])
    def test_returns_unchanged_for_missing_or_invalid_profile(
        self, icc_profile: bytes | None
    ):
        # Arrange
        optical_path = OpticalPath(icc_profile=icc_profile)

        # Act
        updated = optical_path.add_color_space_from_icc()

        # Assert
        assert updated is optical_path
        assert updated.color_space is None


class TestValidateIccProfile:
    @staticmethod
    def _profile(device_class: bytes, color_space: bytes, pcs: bytes) -> bytes:
        profile = bytearray(
            ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        )
        profile[12:16] = device_class
        profile[16:20] = color_space
        profile[20:24] = pcs
        return bytes(profile)

    def test_no_problems_for_conformant_profile(self):
        # Arrange
        optical_path = OpticalPath(icc_profile=self._profile(b"scnr", b"RGB ", b"XYZ "))

        # Act
        problems = optical_path.validate_icc_profile()

        # Assert
        assert problems == []

    @pytest.mark.parametrize(
        ["device_class", "color_space", "pcs", "expected_count"],
        [
            (b"mntr", b"RGB ", b"XYZ ", 1),
            (b"scnr", b"CMYK", b"Lab ", 1),
            (b"scnr", b"RGB ", b"GRAY", 1),
            (b"mntr", b"CMYK", b"GRAY", 3),
        ],
    )
    def test_reports_one_problem_per_violated_constraint(
        self, device_class: bytes, color_space: bytes, pcs: bytes, expected_count: int
    ):
        # Arrange
        optical_path = OpticalPath(
            icc_profile=self._profile(device_class, color_space, pcs)
        )

        # Act
        problems = optical_path.validate_icc_profile()

        # Assert
        assert len(problems) == expected_count

    @pytest.mark.parametrize("icc_profile", [None, b""])
    def test_no_problems_for_missing_profile(self, icc_profile: bytes | None):
        # Arrange
        optical_path = OpticalPath(icc_profile=icc_profile)

        # Act
        problems = optical_path.validate_icc_profile()

        # Assert
        assert problems == []

    def test_reports_truncated_profile(self):
        # Arrange
        optical_path = OpticalPath(icc_profile=b"too short")

        # Act
        problems = optical_path.validate_icc_profile()

        # Assert
        assert problems == ["ICC profile header is truncated."]


class TestDicomLut:
    @pytest.mark.parametrize(
        ["lut", "expected_table_component"],
        [
            (
                Lut(
                    [ConstantLutSegment(0, 256)],
                    [ConstantLutSegment(0, 256)],
                    [LinearLutSegment(0, 255, 256)],
                    np.uint16,
                ),
                [
                    np.full(256, 0, dtype=np.uint16),
                    np.full(256, 0, dtype=np.uint16),
                    np.linspace(0, 255, 256, dtype=np.uint16),
                ],
            ),
            (
                Lut(
                    [ConstantLutSegment(0, 256)],
                    [ConstantLutSegment(0, 256)],
                    [
                        ConstantLutSegment(0, 100),
                        LinearLutSegment(0, 255, 100),
                        ConstantLutSegment(255, 56),
                    ],
                    np.uint16,
                ),
                [
                    np.full(256, 0, dtype=np.uint16),
                    np.full(256, 0, dtype=np.uint16),
                    np.concatenate(
                        [
                            np.full(100, 0, dtype=np.uint16),
                            np.linspace(0, 255, 100, dtype=np.uint16),
                            np.full(56, 255, dtype=np.uint16),
                        ]
                    ),
                ],
            ),
            (
                Lut(
                    [DiscreteLutSegment([index for index in range(256)])],
                    [DiscreteLutSegment([0 for _ in range(256)])],
                    [DiscreteLutSegment([255 for _ in range(256)])],
                    np.uint16,
                ),
                [
                    np.array([index for index in range(256)], dtype=np.uint16),
                    np.full(256, 0, dtype=np.uint16),
                    np.full(256, 255, dtype=np.uint16),
                ],
            ),
        ],
    )
    def test_parse_to_table(self, lut: Lut, expected_table_component: np.ndarray):
        # Arrange

        # Act
        table = lut.table

        # Assert
        for component, expected_component in zip(
            table, expected_table_component, strict=True
        ):
            assert len(component) == len(expected_component)
            assert np.array_equal(component, expected_component)
