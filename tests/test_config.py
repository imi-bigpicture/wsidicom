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

import pytest

from wsidicom.config import Settings
from wsidicom.options import ResampleFilterOption


@pytest.mark.unittest
class TestResamplingFilterDefaults:
    """Read and pyramid downsampling use separate filters with separate defaults.

    Reads want a quality antialiasing filter; pyramid generation wants BOX, which
    is exact, commutes with tiling, and is the standard mipmap construction.
    """

    def test_read_filter_defaults_to_bilinear(self):
        # Arrange
        settings = Settings()
        expected_default = ResampleFilterOption.BILINEAR

        # Act
        resampling_filter = settings.resampling_filter

        # Assert
        assert resampling_filter == expected_default

    def test_pyramid_filter_defaults_to_box(self):
        # Arrange
        settings = Settings()
        expected_default = ResampleFilterOption.BOX

        # Act
        pyramid_resampling_filter = settings.pyramid_resampling_filter

        # Assert
        assert pyramid_resampling_filter == expected_default

    def test_filters_are_independent(self):
        # Arrange
        settings = Settings()
        expected_default = ResampleFilterOption.BOX

        # Act — changing the read filter must not change the pyramid filter
        settings.resampling_filter = ResampleFilterOption.LANCZOS

        # Assert
        assert settings.pyramid_resampling_filter == expected_default
